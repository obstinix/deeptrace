"""API middleware for tracking request metrics."""
from __future__ import annotations
import time
from fastapi import Request

from api.db import increment

async def track_metrics(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    ms = (time.time() - t0) * 1000
    increment("total_latency_ms", ms)
    increment("total_requests", 1.0)
    if response.status_code >= 400:
        increment("error_requests", 1.0)
    response.headers["X-Processing-Time-Ms"] = f"{ms:.1f}"
    return response
