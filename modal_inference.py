"""
Modal Inference Service for Qwen2 1.5B using llama-cpp-python
Deploy with: modal deploy modal_inference.py
"""

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount local model files
model_volume = modal.Volume.from_name("model-storage", create_if_missing=True)

# Container Image with GPU-accelerated llama-cpp-python
# Using Modal's GPU image with CUDA 12.1 pre-installed
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands(
        # Install latest llama-cpp-python with CUDA 12.1 support (includes Qwen3)
        "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
    )
    .pip_install(
        "fastapi==0.115.6",
        "pydantic==2.10.3",
        "huggingface-hub==0.25.1",
    )
)

# Modal App
app = modal.App("gigsama-backend", image=image)

# Request/Response Schemas
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for inference")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, gt=0, le=2048)
    top_p: float = Field(0.9, ge=0.0, le=1.0)

class InferenceResponse(BaseModel):
    text: str
    tokens: int
    finish_reason: str
    error: bool = False
    message: str = ""

# ============================================================================
# Inference Function
# ============================================================================

@app.function(
    gpu="T4",
    timeout=600,
)
def run_inference(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    top_p: float = 0.9,
) -> dict:
    """Run inference with the model."""
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    import os
    
    try:
        logger.info(f"Generating: temp={temperature}, max_tokens={max_tokens}")
        
        # Download model from HuggingFace
        logger.info("Downloading model...")
        model_path = hf_hub_download(
            repo_id="skaffum/gigsama-tool-call-v2-gguf",
            filename="Merged-Model-1.7B-Q4_K_M.gguf",
        )
        logger.info(f"Model ready: {model_path}")
        
        # Debug: Check file exists and size
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"✓ Model file exists: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Try to load with verbose=True for debugging
        logger.info("Loading model into llama-cpp-python...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=99,
            n_ctx=2048,
            verbose=True,  # Enable verbose logging
        )
        logger.info("✓ Model loaded successfully!")
        
        response = llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=["<|im_end|>", "User:", "Assistant:"],
            echo=False,
        )
        
        result = {
            "text": response["choices"][0]["text"].strip(),
            "tokens": response["usage"]["completion_tokens"],
            "finish_reason": response["choices"][0]["finish_reason"],
            "error": False,
            "message": "",
        }
        
        logger.info(f"Generated {result['tokens']} tokens")
        return result
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return {
            "text": "",
            "tokens": 0,
            "finish_reason": "error",
            "error": True,
            "message": str(e),
        }

# ============================================================================
# FastAPI Application
# ============================================================================

web_app = FastAPI(
    title="Gigsama Inference API",
    version="1.0",
    description="Qwen2 1.5B inference service"
)

@web_app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Generate text using model."""
    logger.info(f"Received inference request: {request.prompt[:50]}...")
    
    try:
        result = run_inference.remote(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        return result
        
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gigsama-inference",
        "model": "Qwen2-1.5B-Instruct",
        "quantization": "Q4_K_M",
        "gpu": "T4",
        "version": "2.0",
    }

@web_app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Gigsama Inference API",
        "endpoints": {
            "inference": "POST /inference",
            "health": "GET /health",
        },
        "docs": "/docs",
    }

# ============================================================================
# Mount FastAPI to Modal
# ============================================================================

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Expose FastAPI app to Modal."""
    return web_app

# ============================================================================
# Local Entrypoint: Test Inference
# ============================================================================

@app.local_entrypoint()
def test_inference():
    """Test inference with HuggingFace model."""
    
    print("\n" + "="*80)
    print("🧪 TESTING INFERENCE WITH HUGGINGFACE MODEL")
    print("="*80)
    
    test_prompt = "What is artificial intelligence? Answer in one sentence."
    print(f"\n📝 Test prompt: {test_prompt}")
    print("\n⏳ Running inference (first run may take 1-2 min to download model)...")
    
    try:
        result = run_inference.remote(
            prompt=test_prompt,
            temperature=0.0,
            max_tokens=100,
            top_p=0.9,
        )
        
        if result["error"]:
            print(f"❌ Error: {result['message']}")
        else:
            print(f"\n✅ SUCCESS!")
            print(f"📄 Response: {result['text']}")
            print(f"📊 Tokens: {result['tokens']}")
            print(f"⏹️  Finish reason: {result['finish_reason']}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Deploy: modal deploy modal_inference.py")
    print("2. Test backend: python test_backend.py")
    print("="*80 + "\n")