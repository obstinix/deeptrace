"""FastAPI inference server for deepfake detection."""
from __future__ import annotations
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.routes.model import _load_predictor
from api.routes import predict_router, model_router, system_router
from api.routes.predict import limiter
from api.middleware import track_metrics
from api.db import init_db
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.predictor = _load_predictor()
    app.state.request_count = 0
    app.state.error_count = 0
    app.state.total_latency_ms = 0.0
    app.state.start_time = time.time()
    yield

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")

app = FastAPI(title="Deepfake Recognition API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.middleware("http")(track_metrics)

app.include_router(predict_router)
app.include_router(model_router)
app.include_router(system_router)
app.mount("/static", StaticFiles(directory="stitch_veritas_ai_detection_platform"), name="static")

@app.get("/")
def serve_frontend():
    if not Path("index.html").exists():
        return "index.html not found. Check deployment.", 404
    return FileResponse("index.html")
