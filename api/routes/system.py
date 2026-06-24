from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

from api.db import get_all

router = APIRouter()


@router.get("/api/health")
async def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.predictor is not None,
            "version": "0.1.0",
            "uptime_seconds": int(time.time() - request.app.state.start_time)}

@router.get("/api/metrics")
async def metrics():
    m = get_all()
    n = int(m.get("total_requests", 0))
    errors = int(m.get("error_requests", 0))
    latency = m.get("total_latency_ms", 0.0)
    return {"total_requests": n, "error_requests": errors,
            "avg_latency_ms": round(latency / max(n, 1), 1)}

@router.get("/api/config")
async def config():
    """Returns runtime config the frontend needs, injected via env."""
    return {
        "api_base_url": os.getenv("PUBLIC_API_BASE_URL", ""),
        "version": "0.1.0",
    }
