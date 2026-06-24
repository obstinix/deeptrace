"""FastAPI inference server for deepfake detection."""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api.db import init_db
from api.middleware import track_metrics
from api.routes import model_router, predict_router, system_router
from api.routes.model import _load_predictor
from api.routes.predict import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.predictor = _load_predictor()
    yield

app = FastAPI(title="Deepfake Recognition API", version="0.1.0", lifespan=lifespan)
for k, v in [("predictor", None), ("request_count", 0), ("error_count", 0), ("total_latency_ms", 0.0), ("start_time", time.time()), ("limiter", limiter)]:
    setattr(app.state, k, v)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","), allow_methods=["GET", "POST"], allow_headers=["*"])
app.middleware("http")(track_metrics)
for router in [predict_router, model_router, system_router]:
    app.include_router(router)
app.mount("/static", StaticFiles(directory="stitch_veritas_ai_detection_platform"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html") if Path("index.html").exists() else ("index.html not found. Check deployment.", 404)
