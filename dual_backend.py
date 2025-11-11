from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import httpx
import logging
from datetime import datetime
from uuid import uuid4
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dual Backend Service with Modal Integration")

# Configuration & System Prompts

DEFAULT_DETERMINISTIC_PROMPT = """You are a recruitment assistant that converts user requests into structured tool calls.

IMPORTANT: Always respond in ENGLISH only.

ALWAYS respond with ONLY valid JSON in this exact format:
{
  "tool_calls": [
    {
      "name": "search_candidates",
      "arguments": {
        "role": "job title",
        "industry": "industry name",
        "location": "location name",
        "skills": ["optional", "skills", "list"],
        "years_experience": 0
      }
    }
  ]
}

Rules:
1. Extract role, industry, and location from user message
2. Extract any mentioned skills
3. Extract years of experience if mentioned
4. Return ONLY the JSON, no other text
5. Use "search_candidates" as the tool name
6. All responses must be in ENGLISH

Examples:
- User: "I want frontend developers in FinTech in Kenya"
  Response: {"tool_calls": [{"name": "search_candidates", "arguments": {"role": "frontend developer", "industry": "FinTech", "location": "Kenya"}}]}

- User: "Hire backend devs with 5+ years in HealthTech in Nairobi"
  Response: {"tool_calls": [{"name": "search_candidates", "arguments": {"role": "backend developer", "industry": "HealthTech", "location": "Nairobi", "years_experience": 5}}]}
"""

DEFAULT_CREATIVE_PROMPT = """You are a helpful and creative assistant. 

IMPORTANT: Always respond in ENGLISH only.

Write helpful, engaging, and original responses in English.
Be thoughtful, conversational, and natural in your writing.
Provide rich, detailed, and well-structured responses.
Use simple, clear language that anyone can understand."""

DETERMINISTIC_CONFIG = {
    "name": "deterministic",
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 500,
    "engine": "modal",
    "system_prompt": DEFAULT_DETERMINISTIC_PROMPT,
}

CREATIVE_CONFIG = {
    "name": "creative",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 1000,
    "engine": "modal",
    "system_prompt": DEFAULT_CREATIVE_PROMPT,
}

# Environment variables
MODAL_INFERENCE_URL = os.getenv(
    "MODAL_INFERENCE_URL",
    "https://affum3331--gigsama-backend-fastapi-app.modal.run"
)

# HTTP client configuration
INFERENCE_TIMEOUT = 60.0  # 60 seconds for inference
INFERENCE_RETRIES = 3
INFERENCE_RETRY_DELAY = 2  # seconds between retries

# Keep-alive to prevent cold starts
KEEP_ALIVE_INTERVAL = 300  # Ping Modal every 5 minutes

# Request/Response Models

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    system: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    tokens_input: int
    tokens_output: int
    finish_reason: str
    backend_used: str
    routing_reason: str
    latency_ms: float
    request_id: str
    cold_start: bool = False

class ErrorResponse(BaseModel):
    error: bool
    code: str
    message: str
    request_id: str

class InferenceRequest(BaseModel):
    prompt: str
    temperature: float
    max_tokens: int
    top_p: float

# HTTP Client with Retry Logic
class ModalInferenceClient:
    """Handles communication with Modal inference endpoint with retries and error handling."""
    
    def __init__(self, base_url: str, timeout: float = INFERENCE_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = INFERENCE_RETRIES
        self.retry_delay = INFERENCE_RETRY_DELAY
        self.session: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize HTTP client."""
        self.session = httpx.AsyncClient(timeout=self.timeout)
    
    async def close(self):
        """Close HTTP client."""
        if self.session:
            await self.session.aclose()
    
    async def infer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> dict:
        """
        Call Modal inference endpoint with retry logic.
        Returns: {text, tokens_output, finish_reason}
        """
        if not self.session:
            await self.initialize()
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        for attempt in range(self.retries):
            try:
                logger.info(f"Inference attempt {attempt + 1}/{self.retries}")
                
                response = await self.session.post(
                    f"{self.base_url}/inference",
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Inference successful on attempt {attempt + 1}")
                return {
                    "text": data.get("text", ""),
                    "tokens_output": data.get("tokens", 0),
                    "finish_reason": data.get("finish_reason", "stop"),
                }
            
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise ValueError(f"Inference timeout after {self.retries} retries")
            
            except httpx.HTTPError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise ValueError(f"Inference failed after {self.retries} retries: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise


# Global Inference Client

inference_client = ModalInferenceClient(MODAL_INFERENCE_URL)

# Routing Logic
def analyze_intent(system_message: str, user_message: str) -> tuple[str, str]:
    """
    Analyzes system message and user message to determine backend routing.
    Returns: (backend_type, reasoning)
    """
    
    system_lower = system_message.lower()
    message_lower = user_message.lower()
    
    # Tool-calling keywords
    tool_keywords = [
        "search", "find", "list", "fetch", "query", "call",
        "execute", "function", "retrieve", "look up", "get",
        "extract", "parse", "structure", "validate", "filter"
    ]
    
    # Conversation keywords
    conversation_keywords = [
        "explain", "how", "why", "advice", "best practice",
        "tell me", "discuss", "what is", "help me", "think about",
        "write", "create", "generate", "compose", "describe", "imagine"
    ]
    
    # Check if system explicitly defines mode
    if "tool" in system_lower or "function" in system_lower or "extract" in system_lower:
        if any(kw in message_lower for kw in tool_keywords):
            return "deterministic", "System defines tool-calling mode + user message matches tool pattern"
    
    if "creative" in system_lower or "write" in system_lower or "generate" in system_lower:
        if any(kw in message_lower for kw in conversation_keywords):
            return "creative", "System defines creative mode + user message matches conversation pattern"
    
    # Count keyword matches
    tool_count = sum(1 for kw in tool_keywords if kw in message_lower)
    conversation_count = sum(1 for kw in conversation_keywords if kw in message_lower)
    
    if tool_count > conversation_count and tool_count > 0:
        return "deterministic", "User message contains tool-calling keywords"
    
    if conversation_count > tool_count and conversation_count > 0:
        return "creative", "User message contains conversation keywords"
    
    # Default: safe deterministic
    return "deterministic", "No clear signal, defaulting to deterministic (safe default)"

# Prompt Assembly
def build_prompt(system: str, messages: List[Message]) -> tuple[str, int]:
    """
    Assembles final prompt from system message and conversation.
    Uses Qwen chat format with explicit English instruction.
    Returns: (prompt, estimated_input_tokens)
    """
    # Use Qwen's chat template format with language control
    prompt = f"<|im_start|>system\n{system}\n\nIMPORTANT: You MUST respond in English language only. Never use Chinese, Japanese, or any other language.<|im_end|>\n"
    
    for msg in messages:
        role = "user" if msg.role == "user" else "assistant"
        prompt += f"<|im_start|>{role}\n{msg.content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    
    # Rough token estimation (1 token ≈ 4 characters)
    estimated_tokens = len(prompt) // 4
    
    return prompt, estimated_tokens


# Response Normalization
def normalize_response(raw_output: str, backend_type: str) -> str:
    """
    Normalizes model output for consistency.
    - Strips markdown code fences
    - Validates JSON for deterministic backends
    - Cleans whitespace
    """
    text = raw_output.strip()
    
    logger.info(f"Raw model output: {text[:200]}...")
    
    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Validate JSON for deterministic backend
    if backend_type == "deterministic":
        try:
            # First try: parse the text as-is
            parsed = json.loads(text)
            logger.info("✓ Output is valid JSON (as-is)")
            
            # Check if it needs wrapping in tool_calls array
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                # Model returned single tool call, wrap it in tool_calls array
                wrapped = {"tool_calls": [parsed]}
                logger.info("ℹ️ Wrapped single tool call in tool_calls array")
                return json.dumps(wrapped)
            
            return text
            
        except json.JSONDecodeError:
            logger.warning(f"Initial JSON parse failed, attempting extraction...")
            
            # Second try: extract JSON object from text
            if "{" in text and "}" in text:
                # Find the first { and last }
                start = text.find("{")
                end = text.rfind("}") + 1
                extracted = text[start:end]
                
                try:
                    parsed = json.loads(extracted)
                    logger.info(f"✓ Successfully extracted valid JSON: {extracted[:100]}...")
                    
                    # Check if it needs wrapping
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        wrapped = {"tool_calls": [parsed]}
                        logger.info("ℹ️ Wrapped extracted tool call in tool_calls array")
                        return json.dumps(wrapped)
                    
                    return extracted
                    
                except json.JSONDecodeError:
                    logger.warning(f"Extracted JSON still invalid: {extracted[:100]}...")
            
            # Third try: Look for nested JSON (handle multiple braces)
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Verify it has the expected structure for tool calls
                    if "tool_calls" in parsed or isinstance(parsed, dict):
                        logger.info(f"✓ Found valid JSON via regex: {match[:100]}...")
                        
                        # Check if it needs wrapping
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            wrapped = {"tool_calls": [parsed]}
                            logger.info("ℹ️ Wrapped regex-found tool call in tool_calls array")
                            return json.dumps(wrapped)
                        
                        return match
                except json.JSONDecodeError:
                    continue
            
            # If all extraction attempts fail, log the full output and return as-is
            # (let the client handle it)
            logger.error(f"⚠️ Could not extract valid JSON. Full output:\n{text}")
            logger.info("Returning raw output for client-side handling")
            return text
    
    return text

# Keep-Alive Function (Prevent Cold Starts)
async def keep_alive_ping():
    """Periodically ping Modal to keep the function warm."""
    while True:
        try:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            logger.info("Sending keep-alive ping to Modal...")
            
            if not inference_client.session:
                await inference_client.initialize()
            
            response = await inference_client.session.post(
                f"{inference_client.base_url}/inference",
                json={
                    "prompt": "test",
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "top_p": 0.9,
                },
                timeout=10.0
            )
            response.raise_for_status()
            logger.info("Keep-alive ping successful")
        
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {str(e)}")

# Main Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with smart routing to deterministic or creative backend.
    Calls Modal for inference.
    """
    request_id = str(uuid4())[:8]
    start_time = time.time()
    cold_start = False
    
    try:
        # Step 1: Determine routing
        backend_type, routing_reason = analyze_intent(
            request.system,
            request.messages[-1].content if request.messages else ""
        )
        logger.info(f"[{request_id}] Routing decision: {backend_type} - {routing_reason}")
        
        # Step 2: Select configuration
        config = DETERMINISTIC_CONFIG if backend_type == "deterministic" else CREATIVE_CONFIG
        
        # Step 3: Use provided system prompt or fallback to config default
        system_prompt = request.system or config["system_prompt"]
        logger.info(f"[{request_id}] Using system prompt: {system_prompt[:80]}...")
        
        # Step 4: Override params if provided in request
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["max_tokens"] = request.max_tokens
        
        # Step 5: Build prompt
        prompt, tokens_input = build_prompt(system_prompt, request.messages)
        logger.info(f"[{request_id}] Prompt built: ~{tokens_input} tokens")
        
        # Step 6: Run inference on Modal
        try:
            inference_result = await inference_client.infer(
                prompt=prompt,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                top_p=config["top_p"]
            )
        except ValueError as e:
            if "timeout" in str(e).lower():
                cold_start = True
                logger.warning(f"[{request_id}] Likely cold start: {str(e)}")
                raise
            raise
        
        # Step 7: Normalize output
        normalized_text = normalize_response(
            inference_result["text"],
            backend_type
        )
        
        # Step 7.5: Check for Chinese characters and log warning
        if any('\u4e00' <= char <= '\u9fff' for char in normalized_text):
            logger.warning(f"[{request_id}] ⚠️ Response contains Chinese characters despite English instruction")
            logger.warning(f"[{request_id}] This may indicate the model is ignoring language instructions")
        
        # Step 8: Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"[{request_id}] Completed in {latency_ms:.2f}ms")
        
        return ChatResponse(
            text=normalized_text,
            tokens_input=tokens_input,
            tokens_output=inference_result["tokens_output"],
            finish_reason=inference_result["finish_reason"],
            backend_used=backend_type,
            routing_reason=routing_reason,
            latency_ms=latency_ms,
            request_id=request_id,
            cold_start=cold_start
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "code": "INFERENCE_ERROR",
                "message": str(e),
                "request_id": request_id,
                "latency_ms": latency_ms
            }
        )

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "service": "Dual Backend Service with Modal Integration",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "POST /chat": "Main chat endpoint with smart routing",
            "GET /health": "Health check",
            "POST /debug/routing": "Debug routing logic",
            "GET /stats": "Backend statistics"
        },
        "modal_url": MODAL_INFERENCE_URL
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "modal_url": MODAL_INFERENCE_URL,
        "backends": {
            "deterministic": DETERMINISTIC_CONFIG,
            "creative": CREATIVE_CONFIG
        }
    }

@app.post("/debug/routing")
async def debug_routing(system: str, user_message: str):
    """Debug endpoint to test routing logic without running inference."""
    backend_type, reasoning = analyze_intent(system, user_message)
    return {
        "backend_type": backend_type,
        "reasoning": reasoning,
        "system_message": system,
        "user_message": user_message,
        "config": (DETERMINISTIC_CONFIG if backend_type == "deterministic" else CREATIVE_CONFIG)
    }

@app.get("/stats")
async def stats():
    """Returns backend statistics."""
    return {
        "modal_url": MODAL_INFERENCE_URL,
        "inference_timeout": INFERENCE_TIMEOUT,
        "inference_retries": INFERENCE_RETRIES,
        "keep_alive_interval": KEEP_ALIVE_INTERVAL,
    }

# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """Initialize backend on startup."""
    logger.info("=" * 80)
    logger.info("Backend service starting up...")
    logger.info(f"Modal URL: {MODAL_INFERENCE_URL}")
    logger.info(f"Deterministic config: {DETERMINISTIC_CONFIG}")
    logger.info(f"Creative config: {CREATIVE_CONFIG}")
    logger.info("=" * 80)
    
    # Initialize inference client
    await inference_client.initialize()
    
    # Start keep-alive background task
    asyncio.create_task(keep_alive_ping())

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Backend service shutting down...")
    await inference_client.close()

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )