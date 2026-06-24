"""FastAPI inference server for deepfake detection."""
from __future__ import annotations
import io, sys, time
from contextlib import asynccontextmanager
from pathlib import Path
import os
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def _load_predictor(ckpt_path=None):
    try:
        from deepfake_recognition.inference.predictor import Predictor
        if ckpt_path is None:
            ckpt_path = os.getenv("MODEL_CHECKPOINT", "checkpoints/resnet18/best.pth")
        
        path = Path(ckpt_path)
        if path.exists():
            print(f"Loading checkpoint: {path}")
            return Predictor.from_checkpoint(path)
        
        # Fallback list
        for ckpt in [Path("checkpoints/resnet18/best.pth"),
                     Path("checkpoints/efficientnet_b3/best.pth")]:
            if ckpt.exists():
                print(f"Loading fallback checkpoint: {ckpt}")
                return Predictor.from_checkpoint(ckpt)
                
        print("WARNING: No checkpoint found. Train a model first.")
        return None
    except Exception as e:
        print(f"WARNING: Failed to load predictor: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = _load_predictor()
    app.state.request_count = 0
    app.state.error_count = 0
    app.state.total_latency_ms = 0.0
    app.state.start_time = time.time()
    yield


app = FastAPI(title="Deepfake Recognition API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="stitch_veritas_ai_detection_platform"), name="static")

@app.get("/")
def serve_frontend():
    if not Path("index.html").exists():
        return "index.html not found. Check deployment.", 404
    return FileResponse("index.html")

class ReloadRequest(BaseModel):
    checkpoint_path: str



@app.middleware("http")
async def track_metrics(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    ms = (time.time() - t0) * 1000
    app.state.total_latency_ms += ms
    app.state.request_count += 1
    if response.status_code >= 400:
        app.state.error_count += 1
    response.headers["X-Processing-Time-Ms"] = f"{ms:.1f}"
    return response


@app.post("/api/model/reload")
async def reload_model(req: ReloadRequest):
    new_predictor = _load_predictor(req.checkpoint_path)
    if new_predictor is None:
        raise HTTPException(status_code=400, detail="Failed to load checkpoint")
    app.state.predictor = new_predictor
    return {"status": "reloaded", "model_version": req.checkpoint_path, "val_accuracy": 0.0}

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None,
            "version": "0.1.0",
            "uptime_seconds": int(time.time() - app.state.start_time)}

@app.get("/api/config")
async def config():
    """Returns runtime config the frontend needs, injected via env."""
    return {
        "api_base_url": os.getenv("PUBLIC_API_BASE_URL", ""),
        "version": "0.1.0",
    }


@app.get("/api/metrics")
async def metrics():
    n = app.state.request_count
    return {"total_requests": n, "error_requests": app.state.error_count,
            "avg_latency_ms": round(app.state.total_latency_ms / max(n, 1), 1)}


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...), use_tta: bool = False):
    if app.state.predictor is None:
        raise HTTPException(503, "No model loaded. Run training/train.py first.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large ({len(data)//1024}KB). Max 10MB.")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Cannot decode image: {e}")
    t0 = time.time()
    r = app.state.predictor.predict_pil(img, use_tta=use_tta)
    return {"label": r["label"], "confidence": round(r["confidence"], 4),
            "probabilities": {"real": round(r["prob_real"], 4),
                              "fake": round(r["prob_fake"], 4)},
            "processing_ms": round((time.time() - t0) * 1000, 1),
            "gradcam_image": r.get("gradcam_image")}


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), sample_frames: int = 16):
    if app.state.predictor is None:
        raise HTTPException(503, "No model loaded.")
    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video too large. Max 100MB.")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name
    try:
        t0 = time.time()
        r = app.state.predictor.predict_video(tmp_path, n_frames=min(sample_frames, 32))
        return {**r, "processing_ms": round((time.time() - t0) * 1000, 1)}
    finally:
        os.unlink(tmp_path)
