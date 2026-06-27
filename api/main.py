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

import base64
from deepfake_recognition.utils.model_factory import (
    build_model as factory_build,
    get_gradcam_target_layer,
    SUPPORTED_ARCHITECTURES,
    IMAGE_SIZES,
    count_parameters,
    supports_gradcam,
)
from deepfake_recognition.utils.gradcam import generate_gradcam_base64
from deepfake_recognition.inference.predictor import Predictor
from deepfake_recognition.utils.attention_rollout import AttentionRollout
from deepfake_recognition.utils.face_pipeline import FacePipeline, FaceDetection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Registry of supported architectures and their active states
MODEL_REGISTRY: dict = {
    arch: {"model": None, "loaded": False, "checkpoint": ckpt, "params": 0}
    for arch, ckpt in SUPPORTED_ARCHITECTURES.items()
}

DEFAULT_MODEL = "resnet18"

# ── Face pipeline (singleton — loaded once at startup) ──────────────────────
# model_selection=1 (full-range) handles photos, not just close-up selfies
FACE_PIPELINE = FacePipeline(
    model_selection=1,
    min_detection_confidence=0.5,
    margin=0.30,
    output_size=224,
    align=True,
    max_faces=10,   # prevent DoS on images with many faces
)

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
        net.load_state_dict(state, strict=False)
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

# Helper to generate explainability map (either Grad-CAM or Attention Rollout)
def get_explainability_map(
    net: torch.nn.Module,
    arch: str,
    input_tensor: torch.Tensor,
    original_image
) -> Optional[str]:
    """
    Returns a base64-encoded PNG of the explainability overlay,
    or None if generation fails.
    """
    try:
        if supports_gradcam(arch):
            # Existing Grad-CAM path (ResNet-18 / EfficientNet-B0)
            target_layer = get_gradcam_target_layer(net, arch)
            with torch.enable_grad():
                input_tensor_grad = input_tensor.clone().requires_grad_(True)
                return generate_gradcam_base64(net, input_tensor_grad, original_image, target_layer=target_layer)
        else:
            # Attention Rollout path (ViT-B/16)
            rollout = AttentionRollout(
                net,
                head_fusion="mean",
                discard_ratio=0.9,
            )
            mask    = rollout(input_tensor)           # (14, 14) numpy array
            overlay = AttentionRollout.overlay(
                original_image,
                mask,
                alpha=0.5,
                colormap="viridis",                   # cooler palette than jet
            )
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            rollout.remove_hooks()
            return "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode()
    except Exception as e:
        print(f"[explainability] {arch} failed: {e}")
        return None


import torchvision.transforms as T
from PIL import Image as PILImage

# Standard inference transform — same normalisation as training
_INFER_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def pil_to_tensor(img: PILImage.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised (1, 3, 224, 224) tensor on DEVICE."""
    return _INFER_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def image_to_base64(img: PILImage.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 data-URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _run_inference_on_crop(
    net: torch.nn.Module,
    arch: str,
    crop: PILImage.Image,
    face_det: "FaceDetection",
) -> dict:
    """
    Run inference on a single pre-cropped face image.

    Returns a dict with prediction, confidence, probabilities,
    explainability map, and timing.
    """
    from src.deepfake_recognition.utils.model_factory import (
        supports_gradcam, get_gradcam_target_layer
    )

    tensor = pil_to_tensor(crop)

    t0     = time.perf_counter()
    with torch.no_grad():
        logits = net(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000

    # Determine class indices — align with training class_to_idx
    # ImageFolder sorts alphabetically: fake=0, real=1
    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    verdict   = "fake" if fake_prob > 0.5 else "real"
    confidence = round(max(fake_prob, real_prob), 4)

    # Explainability
    explainability_method = "grad_cam" if supports_gradcam(arch) else "attention_rollout"
    explainability = None
    try:
        explainability = get_explainability_map(
            net, arch, tensor, crop
        )
    except Exception as e:
        print(f"[explainability] {arch} face#{face_det.face_idx}: {e}")

    return {
        "face_idx":    face_det.face_idx,
        "bbox":        face_det.to_dict()["bbox"],
        "keypoints":   face_det.to_dict()["keypoints"],
        "detection_confidence": face_det.confidence,
        "prediction":  verdict,
        "confidence":  confidence,
        "probabilities": {
            "fake": round(fake_prob, 4),
            "real": round(real_prob, 4),
        },
        "explainability":        explainability,
        "explainability_method": explainability_method,
        "inference_time_ms":     round(ms, 2),
        "crop_b64":              image_to_base64(crop),
    }


@app.post("/api/predict/image")
@limiter.limit("30/minute")
async def predict_image(
    request: Request,
    file:  UploadFile = File(...),
    model: str        = Query(default=DEFAULT_MODEL),
    face_detect: bool = Query(default=True,
        description="Run face detection before inference. "
                    "Falls back to whole-frame if no face found."),
    max_faces: int    = Query(default=5,
        description="Maximum number of faces to analyse per image."),
):
    # Backward compatibility with test_predict_no_model_503
    if request.app.state.predictor is None:
        raise HTTPException(503, "No model loaded. Run training/train.py first.")
    t0_total = time.perf_counter()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large ({len(contents)//1024}KB). Max 10MB.")

    arch  = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded. "
                   f"Check checkpoint at: {SUPPORTED_ARCHITECTURES.get(arch, '?')}"
        )
    net = entry["model"]

    # Read and decode the uploaded image
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # ── Face detection path ────────────────────────────────────────────────
    if face_detect:
        # Temporarily override max_faces for this request
        FACE_PIPELINE.max_faces = max_faces
        detections, crops = FACE_PIPELINE.process(pil_image)

        if detections:
            # Draw boxes on the original image for the frontend
            annotated = FACE_PIPELINE.draw_boxes(pil_image, detections)
            annotated_b64 = image_to_base64(annotated)

            # Run inference on each face crop
            face_results = [
                _run_inference_on_crop(net, arch, crop, det)
                for crop, det in zip(crops, detections)
            ]

            # Aggregate verdict: fake if ANY face is fake
            any_fake     = any(r["prediction"] == "fake" for r in face_results)
            agg_verdict  = "fake" if any_fake else "real"
            agg_conf     = max(r["confidence"] for r in face_results)
            fake_faces   = [r["face_idx"] for r in face_results
                            if r["prediction"] == "fake"]

            elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
            return {
                # Top-level aggregate — backward compatible
                "label":       agg_verdict,
                "prediction":  agg_verdict,
                "confidence":  agg_conf,
                "architecture": arch,
                "processing_ms": elapsed_ms,
                "inference_time_ms": elapsed_ms,

                # Face-level detail
                "mode":            "face_detect",
                "faces_detected":  len(detections),
                "faces_analysed":  len(face_results),
                "fake_face_indices": fake_faces,
                "face_results":    face_results,
                "annotated_image": annotated_b64,

                # Backward-compat: first face explainability at top level
                "gradcam_image":         face_results[0]["explainability"],
                "explainability":        face_results[0]["explainability"],
                "explainability_method": face_results[0]["explainability_method"],
                "probabilities":         face_results[0]["probabilities"],
            }

    # ── Whole-frame fallback ───────────────────────────────────────────────
    # Used when face_detect=False OR when no face was detected
    tensor = pil_to_tensor(pil_image)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = net(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    verdict   = "fake" if fake_prob > 0.5 else "real"

    exp_method = "grad_cam" if supports_gradcam(arch) else "attention_rollout"
    explainability = None
    try:
        explainability = get_explainability_map(
            net, arch, tensor, pil_image
        )
    except Exception as e:
        print(f"[explainability] whole-frame {arch}: {e}")

    elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
    return {
        "label":        verdict,
        "prediction":  verdict,
        "confidence":  round(max(fake_prob, real_prob), 4),
        "architecture": arch,
        "mode":         "whole_frame",
        "faces_detected": 0,
        "probabilities": {
            "fake": round(fake_prob, 4),
            "real": round(real_prob, 4),
        },
        "explainability":        explainability,
        "gradcam_image":         explainability,
        "explainability_method": exp_method,
        "processing_ms":         elapsed_ms,
        "inference_time_ms":     elapsed_ms,
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
            "explainability_method": "grad_cam" if supports_gradcam(arch) else "attention_rollout",
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
async def compare_models(
    file: UploadFile = File(...),
    face_detect: bool = Query(default=True),
    max_faces: int    = Query(default=3),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
        
    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large. Max 10MB.")

    # Read and decode the uploaded image
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Detect once — share crops across all models
    detections, crops = [], []
    if face_detect:
        FACE_PIPELINE.max_faces = max_faces
        detections, crops = FACE_PIPELINE.process(pil_image)

    use_face_crops = bool(detections)
    inputs = crops if use_face_crops else [pil_image]

    results = {}
    for arch, entry in MODEL_REGISTRY.items():
        if not entry["loaded"]:
            results[arch] = {"error": "not loaded"}
            continue

        net = entry["model"]
        model_faces = []

        for i, inp in enumerate(inputs):
            det = detections[i] if use_face_crops else None
            res = _run_inference_on_crop(net, arch, inp,
                      det if det else FaceDetection(0,0,0,0,1.0,face_idx=0))
            model_faces.append(res)

        # Aggregate per-model
        any_fake  = any(r["prediction"] == "fake" for r in model_faces)
        agg_conf  = max(r["confidence"] for r in model_faces)
        results[arch] = {
            "prediction":           "fake" if any_fake else "real",
            "confidence":           agg_conf,
            "face_results":         model_faces,
            "params":               entry["params"],
            "explainability_method": ("grad_cam" if supports_gradcam(arch)
                                       else "attention_rollout"),
        }

    # Consensus
    verdicts = [v["prediction"] for v in results.values() if "prediction" in v]
    consensus = max(set(verdicts), key=verdicts.count) if verdicts else "unknown"

    # Annotated image (shared across models)
    annotated_b64 = None
    if detections:
        annotated = FACE_PIPELINE.draw_boxes(pil_image, detections)
        annotated_b64 = image_to_base64(annotated)

    return {
        "models":          results,
        "consensus":       consensus,
        "agreement":       len(set(verdicts)) == 1 if verdicts else False,
        "faces_detected":  len(detections),
        "mode":            "face_detect" if use_face_crops else "whole_frame",
        "annotated_image": annotated_b64,
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
