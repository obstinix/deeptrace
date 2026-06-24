"""API middleware for tracking request metrics."""
from __future__ import annotations
import time
from fastapi import Request

async def track_metrics(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    ms = (time.time() - t0) * 1000
    request.app.state.total_latency_ms += ms
    request.app.state.request_count += 1
    if response.status_code >= 400:
        request.app.state.error_count += 1
    response.headers["X-Processing-Time-Ms"] = f"{ms:.1f}"
    return response
