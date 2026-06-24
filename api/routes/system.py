from __future__ import annotations
import os
import time
from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/api/health")
async def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.predictor is not None,
            "version": "0.1.0",
            "uptime_seconds": int(time.time() - request.app.state.start_time)}

@router.get("/api/metrics")
async def metrics(request: Request):
    n = request.app.state.request_count
    return {"total_requests": n, "error_requests": request.app.state.error_count,
            "avg_latency_ms": round(request.app.state.total_latency_ms / max(n, 1), 1)}

@router.get("/api/config")
async def config():
    """Returns runtime config the frontend needs, injected via env."""
    return {
        "api_base_url": os.getenv("PUBLIC_API_BASE_URL", ""),
        "version": "0.1.0",
    }
