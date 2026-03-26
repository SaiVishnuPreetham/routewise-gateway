"""
RouteWise — Gateway Server
============================
FastAPI application with a single POST /chat endpoint that routes prompts
to either the Fast model or the Capable model using LiteLLM as the
LLM abstraction layer.

Architecture
------------
    POST /chat  →  SemanticCache.lookup()
                        ↓ miss
                   RoutingModel.classify()
                        ↓
                   litellm.completion()  →  Fast or Capable model
                        ↓
                   GatewayLogger captures request
                        ↓
                   SemanticCache.store()
                        ↓
                   Return response + metadata
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import litellm
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gateway.cache import SemanticCache, CacheLookupResult
from gateway.routing_model import RoutingModel, RoutingDecision, FAST_LABEL, CAPABLE_LABEL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # load .env file

# Model identifiers — LiteLLM resolves provider from the prefix
FAST_MODEL = "groq/llama-3.1-8b-instant"
CAPABLE_MODEL = "gemini/gemini-2.0-flash"

MODEL_MAP = {
    FAST_LABEL: FAST_MODEL,
    CAPABLE_LABEL: CAPABLE_MODEL,
}

MODEL_DISPLAY = {
    FAST_LABEL: "Fast (Groq Llama3.1-8B)",
    CAPABLE_LABEL: "Capable (Gemini 2.0 Flash)",
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("routewise.server")


# ---------------------------------------------------------------------------
# Gateway Logger — Custom LiteLLM callback for in-memory log capture
# ---------------------------------------------------------------------------

class GatewayLogger(litellm.integrations.custom_logger.CustomLogger):
    """
    Captures every LLM request into a persistent log list.
    Registered via litellm.callbacks = [gateway_logger].
    The Streamlit dashboard reads gateway_logger.logs for display.
    Logs are saved to logs.json to survive restarts.
    """

    def __init__(self):
        super().__init__()
        self.logs: list[dict] = []
        self._load_logs()

    def _load_logs(self):
        if os.path.exists('logs.json'):
            try:
                with open('logs.json', 'r') as f:
                    self.logs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load logs.json: {e}")

    def _save_logs(self):
        try:
            with open('logs.json', 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save logs.json: {e}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called synchronously after a successful LLM call."""
        self._record(kwargs, response_obj, start_time, end_time, success=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called synchronously after a failed LLM call."""
        self._record(kwargs, response_obj, start_time, end_time, success=False)

    def _record(self, kwargs, response_obj, start_time, end_time, success: bool):
        try:
            messages = kwargs.get("messages", [])
            prompt = messages[-1].get("content", "") if messages else ""
            model = kwargs.get("model", "unknown")

            # Calculate cost using LiteLLM's built-in cost calculator
            cost = 0.0
            try:
                if response_obj and success:
                    cost = litellm.completion_cost(completion_response=response_obj)
            except Exception:
                pass

            latency_ms = 0.0
            if start_time and end_time:
                latency_ms = (end_time - start_time).total_seconds() * 1000

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                "model_used": model,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": round(cost, 8),
                "success": success,
                "cache_hit": False,
                "routing_reason": "",
                "confidence": 0.0,
            }
            self.logs.append(entry)
            self._save_logs()
        except Exception as e:
            logger.warning(f"GatewayLogger._record error: {e}")

    def enrich_last_log(self, **fields):
        """Enrich the most recent log entry with routing metadata."""
        if self.logs:
            self.logs[-1].update(fields)
            self._save_logs()


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

gateway_logger = GatewayLogger()
routing_model = RoutingModel()
semantic_cache = SemanticCache()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt to process")
    system_prompt: Optional[str] = Field(None, description="Optional system instruction")


class ChatResponse(BaseModel):
    response: str
    model_used: str
    model_label: str
    routing_reason: str
    confidence_score: float
    raw_score: float
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    cache_similarity: float


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    # Register the custom logger with LiteLLM
    litellm.callbacks = [gateway_logger]
    litellm.set_verbose = False

    # Set API keys from env
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    os.environ.setdefault("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    logger.info("RouteWise Gateway started")
    logger.info(f"  Fast model   : {FAST_MODEL}")
    logger.info(f"  Capable model: {CAPABLE_MODEL}")
    logger.info(f"  Cache threshold: {semantic_cache.threshold}")
    yield
    logger.info("RouteWise Gateway shutting down")


app = FastAPI(
    title="RouteWise AI Gateway",
    description="Smart AI gateway that routes prompts to the optimal LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main gateway endpoint. Accepts a prompt, routes to the optimal model,
    and returns the response with full metadata.
    """
    t0 = time.perf_counter()
    prompt = request.prompt.strip()

    # ----- Step 1: Cache lookup -----
    cache_result: CacheLookupResult = semantic_cache.lookup(prompt)

    if cache_result.hit:
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Log the cache hit
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "model_used": cache_result.cached_model or "cache",
            "routing_reason": f"Cache hit (similarity={cache_result.similarity:.4f})",
            "confidence": 1.0,
            "latency_ms": round(elapsed_ms, 1),
            "cost_usd": 0.0,
            "cache_hit": True,
            "cache_similarity": cache_result.similarity,
            "success": True,
        }
        gateway_logger.logs.append(log_entry)
        gateway_logger._save_logs()

        return ChatResponse(
            response=cache_result.cached_response or "",
            model_used=cache_result.cached_model or "cache",
            model_label=MODEL_DISPLAY.get(
                FAST_LABEL if cache_result.cached_model == FAST_MODEL else CAPABLE_LABEL,
                "Cached"
            ),
            routing_reason=f"Cache hit (similarity={cache_result.similarity:.4f})",
            confidence_score=1.0,
            raw_score=0.0,
            latency_ms=round(elapsed_ms, 1),
            cost_usd=0.0,
            cache_hit=True,
            cache_similarity=cache_result.similarity,
        )

    # ----- Step 2: Route the prompt -----
    decision: RoutingDecision = routing_model.classify(prompt)
    model_id = MODEL_MAP[decision.label]
    model_display = MODEL_DISPLAY[decision.label]

    # ----- Step 3: Call the LLM via LiteLLM -----
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = litellm.completion(
            model=model_id,
            messages=messages,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"LLM call failed: {e}")

        # LiteLLM callback already logged this failure — enrich it with routing metadata
        gateway_logger.enrich_last_log(
            routing_reason=decision.reason,
            confidence=decision.confidence,
            latency_ms=round(elapsed_ms, 1),
            cache_hit=False,
            cache_similarity=cache_result.similarity,
            success=False,
        )

        raise HTTPException(
            status_code=502,
            detail=f"LLM call to {model_display} failed: {str(e)}",
        )

    # ----- Step 4: Extract response text and cost -----
    response_text = response.choices[0].message.content or ""

    # Estimate tokens (rough approximation)
    cost = 0.0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # For Llama/Gemini
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(response_text))
        total_tokens = input_tokens + output_tokens
        print(f"Debug: model_id={model_id}, input_tokens={input_tokens}, output_tokens={output_tokens}, total_tokens={total_tokens}")  # Debug
        # Approximate rates (fixed demo costs)
        if "groq" in model_id:
            cost = 0.05  # Fixed demo cost for Groq
        elif "gemini" in model_id:
            cost = 0.08  # Fixed demo cost for Gemini
        else:
            cost = 0.0
        print(f"Debug: Estimated cost: {cost}")  # Debug
        print(f"Debug: model_id contains groq: {'groq' in model_id}")  # Debug
    except Exception as e:
        print(f"Debug: Cost estimation failed: {e}")  # Debug
        cost = 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ----- Step 5: Store in cache -----
    semantic_cache.store(
        prompt=prompt,
        response=response_text,
        model_used=model_id,
        metadata={
            "routing_label": decision.label,
            "routing_reason": decision.reason,
            "confidence": decision.confidence,
        },
    )

    # ----- Step 6: Enrich the log entry with routing metadata -----
    gateway_logger.enrich_last_log(
        routing_reason=decision.reason,
        confidence=decision.confidence,
        cache_hit=False,
        cache_similarity=cache_result.similarity,
    )

    # Also add a manual log entry if LiteLLM's callback didn't fire (safety net)
    if not gateway_logger.logs or "routing_reason" not in gateway_logger.logs[-1]:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "model_used": model_id,
            "routing_reason": decision.reason,
            "confidence": decision.confidence,
            "latency_ms": round(elapsed_ms, 1),
            "cost_usd": round(cost, 8),
            "cache_hit": False,
            "cache_similarity": cache_result.similarity,
            "success": True,
        }
        gateway_logger.logs.append(log_entry)
        gateway_logger._save_logs()

    return ChatResponse(
        response=response_text,
        model_used=model_id,
        model_label=model_display,
        routing_reason=decision.reason,
        confidence_score=decision.confidence,
        raw_score=decision.raw_score,
        latency_ms=round(elapsed_ms, 1),
        cost_usd=round(cost, 8),
        cache_hit=False,
        cache_similarity=cache_result.similarity,
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/logs")
async def get_logs(limit: int = 100):
    """Return the most recent gateway logs for the dashboard."""
    return {"logs": gateway_logger.logs[-limit:], "total": len(gateway_logger.logs)}


@app.get("/cache/stats")
async def cache_stats():
    """Return cache statistics."""
    return semantic_cache.stats


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "fast": FAST_MODEL,
            "capable": CAPABLE_MODEL,
        },
        "cache": semantic_cache.stats,
        "total_requests": len(gateway_logger.logs),
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn gateway.server:app --reload --port 8000
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway.server:app", host="0.0.0.0", port=8000, reload=True)
