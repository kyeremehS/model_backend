"""
Modal Inference Service for Qwen3 1.7B using llama-cpp-python (Optimized)
Deploy with: modal deploy modal_inference.py
"""

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import time

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Container Image with GPU-accelerated llama-cpp-python
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands(
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
    inference_time: float
    error: bool = False
    message: str = ""


# Global model cache
class ModelCache:
    def __init__(self):
        self.llm = None
        self.loaded = False

model_cache = ModelCache()


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
    """Run inference with the model (cached in memory)."""
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    
    start_time = time.time()
    
    try:
        # Load model once and cache it
        if not model_cache.loaded:
            logger.info("First request - loading model into memory...")
            load_start = time.time()
            
            model_path = hf_hub_download(
                repo_id="skaffum/gigsama-tool-call-v2-gguf",
                filename="Merged-Model-1.7B-Q4_K_M.gguf",
            )
            logger.info(f"Model downloaded: {model_path}")
            
            # Load model with optimized settings
            model_cache.llm = Llama(
                model_path=model_path,
                n_gpu_layers=99,
                n_ctx=1024,  # Reduced from 2048 for faster inference
                verbose=False,  # Disabled verbose logging for production
            )
            model_cache.loaded = True
            
            load_time = time.time() - load_start
            logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
        
        logger.info(f"Generating: temp={temperature}, max_tokens={max_tokens}")
        
        # Run inference with cached model
        response = model_cache.llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=["<|im_end|>", "User:", "Assistant:"],
            echo=False,
        )
        
        inference_time = time.time() - start_time
        
        result = {
            "text": response["choices"][0]["text"].strip(),
            "tokens": response["usage"]["completion_tokens"],
            "finish_reason": response["choices"][0]["finish_reason"],
            "inference_time": round(inference_time, 2),
            "error": False,
            "message": "",
        }
        
        logger.info(f"Generated {result['tokens']} tokens in {inference_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        inference_time = time.time() - start_time
        return {
            "text": "",
            "tokens": 0,
            "finish_reason": "error",
            "inference_time": round(inference_time, 2),
            "error": True,
            "message": str(e),
        }


# FastAPI Application
web_app = FastAPI(
    title="Gigsama Inference API",
    version="2.0",
    description="Qwen3 1.7B inference service (Optimized)"
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
        "model_cached": model_cache.loaded,
        "version": "2.0",
        "optimizations": [
            "Model caching in memory",
            "Reduced context window (1024)",
            "Verbose logging disabled",
            "Inference timing included"
        ]
    }


@web_app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Gigsama Inference API (Optimized)",
        "endpoints": {
            "inference": "POST /inference",
            "health": "GET /health",
        },
        "docs": "/docs",
    }


# Mount FastAPI to Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Expose FastAPI app to Modal."""
    return web_app