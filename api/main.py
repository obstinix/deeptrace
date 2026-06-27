"""FastAPI inference server for deepfake detection."""
import os
import sys
import time
import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api.db import init_db, get_all
from api.middleware import track_metrics
from api.routes.predict import limiter

from deepfake_recognition.utils.model_factory import (
    build_model as factory_build,
    get_gradcam_target_layer,
    SUPPORTED_ARCHITECTURES,
    IMAGE_SIZES,
    count_parameters,
)
from deepfake_recognition.utils.gradcam import generate_gradcam_base64
from deepfake_recognition.inference.predictor import Predictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Registry of supported architectures and their active states
MODEL_REGISTRY: dict = {
    arch: {"model": None, "loaded": False, "checkpoint": ckpt, "params": 0}
    for arch, ckpt in SUPPORTED_ARCHITECTURES.items()
}

DEFAULT_MODEL = "resnet18"

from typing import Optional

def _load_arch(architecture: str, checkpoint_path: Optional[str] = None) -> bool:
    """Load or reload a single architecture into the registry. Returns True on success."""
    arch  = architecture.lower()
    ckpt  = checkpoint_path or SUPPORTED_ARCHITECTURES.get(arch)
    if not ckpt or not Path(ckpt).exists():
        MODEL_REGISTRY[arch]["loaded"] = False
        return False
    try:
        net = factory_build(architecture=arch, num_classes=2, dropout=0.5)
        state = torch.load(ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state)
        net.to(DEVICE)
        net.eval()
        MODEL_REGISTRY[arch]["model"]     = net
        MODEL_REGISTRY[arch]["loaded"]    = True
        MODEL_REGISTRY[arch]["checkpoint"] = ckpt
        MODEL_REGISTRY[arch]["params"]    = count_parameters(net)
        print(f"[registry] loaded {arch} from {ckpt}")
        return True
    except Exception as e:
        print(f"[registry] failed to load {arch}: {e}")
        MODEL_REGISTRY[arch]["loaded"] = False
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load all available checkpoints
    for arch in SUPPORTED_ARCHITECTURES:
        _load_arch(arch)
    
    # Sync predictor for backward compatibility/tests
    if MODEL_REGISTRY[DEFAULT_MODEL]["loaded"]:
        app.state.predictor = Predictor(MODEL_REGISTRY[DEFAULT_MODEL]["model"], str(DEVICE))
    else:
        app.state.predictor = None
        
    yield

app = FastAPI(title="Deepfake Recognition API", version="0.1.0", lifespan=lifespan)

for k, v in [("predictor", None), ("request_count", 0), ("error_count", 0), ("total_latency_ms", 0.0), ("start_time", time.time()), ("limiter", limiter)]:
    setattr(app.state, k, v)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","), allow_methods=["GET", "POST"], allow_headers=["*"])
app.middleware("http")(track_metrics)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024

# ── Inference / Prediction Endpoints ──────────────────────────────────────────

@app.post("/api/predict/image")
@limiter.limit("30/minute")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
    use_tta: bool = False
):
    # Backward compatibility with test_predict_no_model_503
    if request.app.state.predictor is None:
        raise HTTPException(503, "No model loaded. Run training/train.py first.")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large ({len(data)//1024}KB). Max 10MB.")

    arch = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded. Check checkpoint at {SUPPORTED_ARCHITECTURES.get(arch, '?')}"
        )

    net = entry["model"]
    target_layer = get_gradcam_target_layer(net, arch)

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Cannot decode image: {e}") from e

    t0 = time.time()

    from torchvision import transforms as T
    import torch.nn.functional as F

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = tf(img).unsqueeze(0).to(DEVICE)

    if use_tta:
        from deepfake_recognition.data.transforms import get_tta_transforms
        tta_tfs = get_tta_transforms(224)
        with torch.no_grad():
            logits = torch.stack([net(t_tf(img).unsqueeze(0).to(DEVICE)) for t_tf in tta_tfs]).mean(0)
    else:
        with torch.no_grad():
            logits = net(input_tensor)

    probs = F.softmax(logits, dim=1)[0]
    prob_fake = probs[0].item()
    prob_real = probs[1].item()

    # Generate Grad-CAM
    gradcam_b64 = None
    try:
        with torch.enable_grad():
            input_tensor_grad = input_tensor.clone().requires_grad_(True)
            gradcam_b64 = generate_gradcam_base64(net, input_tensor_grad, img, target_layer=target_layer)
    except Exception as e:
        print(f"WARNING: GradCAM generation failed: {e}")

    verdict = "fake" if prob_fake > 0.5 else "real"
    confidence = max(prob_real, prob_fake)

    return {
        "label": verdict,
        "confidence": round(confidence, 4),
        "probabilities": {
            "real": round(prob_real, 4),
            "fake": round(prob_fake, 4)
        },
        "processing_ms": round((time.time() - t0) * 1000, 1),
        "gradcam_image": gradcam_b64
    }


@app.post("/api/predict/video")
@limiter.limit("5/minute")
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
    sample_frames: int = 16
):
    if request.app.state.predictor is None:
        raise HTTPException(503, "No model loaded.")

    arch = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded. Check checkpoint at {SUPPORTED_ARCHITECTURES.get(arch, '?')}"
        )

    net = entry["model"]

    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video too large. Max 100MB.")

    import tempfile
    import cv2
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        preds = []
        
        from torchvision import transforms as T
        import torch.nn.functional as F

        tf = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        n_frames = min(sample_frames, 32)
        for i in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n_frames))
            ret, frame = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = net(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
            
            prob_fake = probs[0].item()
            prob_real = probs[1].item()
            verdict = "fake" if prob_fake > 0.5 else "real"
            
            preds.append({
                "frame_idx": int(i * total / n_frames),
                "label": verdict,
                "prob_fake": round(prob_fake, 4)
            })
            
        cap.release()
        
        if not preds:
            return {"label": "unknown", "confidence": 0.0, "frames_analyzed": 0, "processing_ms": round((time.time() - t0) * 1000, 1)}

        weights = [p["prob_fake"] if p["prob_fake"] > 0.5 else (1 - p["prob_fake"]) for p in preds]
        total_w = sum(weights) or 1.0
        avg_fake = sum(p["prob_fake"] * w for p, w in zip(preds, weights)) / total_w
        
        avg_label = "fake" if avg_fake > 0.5 else "real"
        avg_conf = max(avg_fake, 1 - avg_fake)

        return {
            "label": avg_label,
            "confidence": round(avg_conf, 4),
            "frames_analyzed": len(preds),
            "frame_predictions": preds,
            "processing_ms": round((time.time() - t0) * 1000, 1)
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ── Model Registry & Compare Endpoints ───────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return the status of every registered model."""
    return {
        arch: {
            "loaded":     entry["loaded"],
            "checkpoint": entry["checkpoint"],
            "params":     entry["params"],
        }
        for arch, entry in MODEL_REGISTRY.items()
    }


@app.post("/api/model/load")
async def load_model(body: dict):
    arch = body.get("architecture", "").lower()
    if arch not in SUPPORTED_ARCHITECTURES:
        raise HTTPException(status_code=400, detail=f"Unknown architecture: '{arch}'")
    ok = _load_arch(arch)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoint missing for '{arch}': {SUPPORTED_ARCHITECTURES[arch]}"
        )
    return {"status": "loaded", "architecture": arch,
            "params": MODEL_REGISTRY[arch]["params"]}


@app.post("/api/model/reload")
async def reload_model(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    arch = body.get("architecture")
    ckpt = body.get("checkpoint_path")
    
    if not arch:
        if ckpt and "efficientnet" in ckpt.lower():
            arch = "efficientnet_b0"
        else:
            arch = DEFAULT_MODEL
            
    ckpt = ckpt or SUPPORTED_ARCHITECTURES.get(arch)
    ok = _load_arch(arch, ckpt)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for '{arch}'")
        
    # Sync app.state.predictor for compatibility
    if arch == DEFAULT_MODEL and MODEL_REGISTRY[arch]["loaded"]:
        app.state.predictor = Predictor(MODEL_REGISTRY[arch]["model"], str(DEVICE))
        
    return {
        "status":     "reloaded",
        "architecture": arch,
        "checkpoint": MODEL_REGISTRY[arch]["checkpoint"],
        "params":     MODEL_REGISTRY[arch]["params"],
    }


@app.post("/api/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Run all loaded models on the same image.
    Returns per-model prediction + confidence + inference time.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
        
    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large. Max 10MB.")

    results = {}
    for arch, entry in MODEL_REGISTRY.items():
        if not entry["loaded"]:
            results[arch] = {"error": "not loaded"}
            continue

        net = entry["model"]
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            
            from torchvision import transforms as T
            import torch.nn.functional as F
            
            tf = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            tensor = tf(img).unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = net(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
            ms = (time.perf_counter() - t0) * 1000

            fake_idx = 0
            real_idx = 1
            fake_prob = probs[fake_idx].item()
            real_prob = probs[real_idx].item()
            verdict   = "fake" if fake_prob > 0.5 else "real"

            results[arch] = {
                "prediction":    verdict,
                "confidence":    round(max(fake_prob, real_prob), 4),
                "probabilities": {
                    "fake": round(fake_prob, 4),
                    "real": round(real_prob, 4),
                },
                "inference_time_ms": round(ms, 2),
                "params": entry["params"],
            }
        except Exception as e:
            results[arch] = {"error": str(e)}

    verdicts = [v["prediction"] for v in results.values() if "prediction" in v]
    consensus = max(set(verdicts), key=verdicts.count) if verdicts else "unknown"

    return {
        "models":    results,
        "consensus": consensus,
        "agreement": len(set(verdicts)) == 1 if verdicts else False,
    }

# ── System / Health / Config Endpoints ────────────────────────────────────────

@app.get("/api/health")
async def health(model: str = Query(default=DEFAULT_MODEL)):
    arch = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    loaded = entry is not None and entry["loaded"]
    return {
        "status": "ok",
        "model_loaded": loaded,
        "version": "0.1.0",
        "uptime_seconds": int(time.time() - app.state.start_time),
    }


@app.get("/api/metrics")
async def metrics():
    m = get_all()
    n = int(m.get("total_requests", 0))
    errors = int(m.get("error_requests", 0))
    latency = m.get("total_latency_ms", 0.0)
    return {"total_requests": n, "error_requests": errors,
            "avg_latency_ms": round(latency / max(n, 1), 1)}


@app.get("/api/config")
async def config():
    return {
        "api_base_url": os.getenv("PUBLIC_API_BASE_URL", ""),
        "version": "0.1.0",
    }

# ── Static / Frontend Serving ──────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="stitch_veritas_ai_detection_platform"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html") if Path("index.html").exists() else ("index.html not found. Check deployment.", 404)


@app.get("/webhooks", response_class=FileResponse)
@app.get("/webhooks.html", response_class=FileResponse)
async def docs():
    return FileResponse("webhooks.html")
