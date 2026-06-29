"""FastAPI inference server for deepfake detection."""
import os
import sys
import time
import io
import uuid
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Literal, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.db import init_db, get_all
from api.middleware import track_metrics
from api.routes.predict import limiter

# ── API auth imports ─────────────────────────────────────────────────────────
from api.auth.keys       import init_db as _init_key_db, create_key, list_keys, revoke_key, get_usage
from api.auth.middleware import require_auth, require_admin, require_feature, AuthContext, PUBLIC_PATHS
from api.auth.tiers      import TIERS, get_tier

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
from deepfake_recognition.utils.multi_face import (
    aggregate_verdict,
    build_group_response,
    VerdictStrategy,
)
from src.deepfake_recognition.audio.audio_pipeline import AudioPipeline, AudioResult
from src.deepfake_recognition.audio.audio_fusion   import fuse_verdicts, FusionStrategy
from src.deepfake_recognition.utils.explainability.router import (
    get_explanation,
    FAST_METHODS,
    SLOW_METHODS,
    ALL_METHODS,
    ExplainMethod,
)
from src.deepfake_recognition.utils.explainability.explainability_cache import ExplainCache
from src.deepfake_recognition.utils.explainability.shap_explainer import ShapExplainer
from src.deepfake_recognition.utils.calibration import TemperatureScaler
from src.deepfake_recognition.utils.ensemble import EnsembleScorer

# ── Celery + async job queue imports ─────────────────────────────────────────
from celery_app   import celery_app as _celery_app
from worker.tasks import analyse_video as _analyse_video_task
from worker       import storage as _job_store
from celery.result import AsyncResult

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ── Audio pipeline (singleton) ───────────────────────────────────────────────
AUDIO_PIPELINE = AudioPipeline(
    checkpoint_path="checkpoints/aasist/best.pth",
    device=torch.device("cpu"),   # CPU is fast enough — saves GPU VRAM
    aggregate="mean",
)

# ── Ensemble scorer (singleton) ──────────────────────────────────────────────
ENSEMBLE = EnsembleScorer(
    strategy="weighted_average",    # overridden at startup if weights.json exists
    weights_path="checkpoints/ensemble/weights.json",
)

# ── Explainability infrastructure ────────────────────────────────────────────
SHAP_EXPLAINER = ShapExplainer(
    n_background=50,
    n_evals=50,
    device=torch.device("cpu"),
)
EXPLAIN_CACHE = ExplainCache(max_workers=2, ttl_seconds=300)


# Shared directory between API and Celery worker for uploaded video files
VIDEO_UPLOAD_DIR = Path(os.environ.get("DEEPTRACE_UPLOAD_DIR",
                                        "/tmp/deeptrace_jobs"))
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Registry of supported architectures and their active states
MODEL_REGISTRY: dict = {
    arch: {"model": None, "loaded": False, "checkpoint": ckpt, "params": 0}
    for arch, ckpt in SUPPORTED_ARCHITECTURES.items()
}

# ── Calibration registry — per-architecture temperature scalers ──────────────
CALIBRATION_REGISTRY: dict = {
    arch: None for arch in SUPPORTED_ARCHITECTURES
}

def _load_calibration(arch: str) -> bool:
    """Attempt to load temperature.json for the given architecture. Returns True on success."""
    ckpt = MODEL_REGISTRY.get(arch, {}).get("checkpoint", "")
    if not ckpt:
        return False
    temp_path = Path(ckpt).parent / "temperature.json"
    if not temp_path.exists():
        print(f"[calibration] {arch}: no temperature.json found — using T=1.0")
        CALIBRATION_REGISTRY[arch] = None
        return False
    try:
        scaler = TemperatureScaler.load(str(temp_path))
        CALIBRATION_REGISTRY[arch] = scaler
        print(f"[calibration] {arch}: loaded T={scaler.temperature:.4f} from {temp_path}")
        return True
    except Exception as e:
        print(f"[calibration] {arch}: failed to load — {e}")
        CALIBRATION_REGISTRY[arch] = None
        return False

def _calibrated_softmax(logits: torch.Tensor, arch: str) -> torch.Tensor:
    """Apply calibrated softmax (temperature scaling) if available, else raw softmax."""
    scaler = CALIBRATION_REGISTRY.get(arch)
    if scaler is not None:
        return scaler.calibrate(logits)
    return torch.softmax(logits, dim=1)

def _collect_member_probs(
    tensor: torch.Tensor,               # (1, 3, 224, 224) normalised, on DEVICE
) -> Dict[str, float]:
    """
    Run all loaded MODEL_REGISTRY members on the same tensor.
    Returns {arch: calibrated_fake_prob} for every loaded model.
    """
    results = {}
    for arch, entry in MODEL_REGISTRY.items():
        if not entry.get("loaded"):
            continue
        net = entry["model"]
        try:
            with torch.no_grad():
                logits = net(tensor)
            probs = _calibrated_softmax(logits, arch)   # existing helper
            results[arch] = round(float(probs[0, 0].item()), 4)   # fake_idx=0
        except Exception as e:
            print(f"[ensemble] {arch} inference failed: {e}")
    return results

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
    await _init_key_db()
    print("[auth] key database ready")
    init_db()
    # Load all available checkpoints
    for arch in SUPPORTED_ARCHITECTURES:
        _load_arch(arch)
    
    # Load temperature calibration files for all architectures
    for arch in SUPPORTED_ARCHITECTURES:
        _load_calibration(arch)
    
    # Load audio pipeline
    AUDIO_PIPELINE.load()

    # Populate SHAP background from real training/validation frames (if available)
    import glob
    from PIL import Image as _PIL
    real_frames = glob.glob("data/frames*/val/real/*.jpg")[:50]
    if real_frames:
        try:
            bg_images = [_PIL.open(p) for p in real_frames]
            SHAP_EXPLAINER.set_background(bg_images)
            print(f"[shap] background populated with {len(bg_images)} images")
        except Exception as bg_e:
            print(f"[shap] background population failed: {bg_e}")
    else:
        print("[shap] no real frames found — will use synthetic noise background")

    # Sync predictor for backward compatibility/tests
    if MODEL_REGISTRY[DEFAULT_MODEL]["loaded"]:
        app.state.predictor = Predictor(MODEL_REGISTRY[DEFAULT_MODEL]["model"], str(DEVICE))
    else:
        app.state.predictor = None

    # Ensemble — auto-detects learned vs weighted_average from weights.json
    if Path("checkpoints/ensemble/weights.json").exists():
        try:
            global ENSEMBLE
            ENSEMBLE = EnsembleScorer(
                weights_path="checkpoints/ensemble/weights.json"
            )
            print(f"[ensemble] loaded — strategy={ENSEMBLE.strategy}")
        except Exception as e:
            print(f"[ensemble] load failed: {e} — using defaults")
    else:
        print("[ensemble] weights.json not found — using DEFAULT_WEIGHTS")
        
    yield

app = FastAPI(title="Deepfake Recognition API", version="0.1.0", lifespan=lifespan)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """
    Inject X-RateLimit-* headers into every authenticated response.
    The headers are set by reading the rl_state stored on the request state
    by the require_auth dependency.
    """
    async def dispatch(self, request, call_next):
        route = request.scope.get("route")
        if route:
            print(f"DEBUG MIDDLEWARE: path={request.url.path} matched route={route.path} endpoint={route.endpoint.__name__}")
        response = await call_next(request)
        # Middleware cannot access FastAPI dependency return values directly.
        # We use request.state to pass headers set in the dependency.
        rl = getattr(request.state, "rl_state", None)
        if rl:
            for window, info in rl.items():
                pfx = f"X-RateLimit-{window.capitalize()}"
                response.headers[f"{pfx}-Limit"]   = str(info["limit"])
                response.headers[f"{pfx}-Used"]    = str(info["used"])
                response.headers[f"{pfx}-Reset-In"] = str(info["reset_in"])
        return response

app.add_middleware(RateLimitHeadersMiddleware)

for k, v in [("predictor", None), ("request_count", 0), ("error_count", 0), ("total_latency_ms", 0.0), ("start_time", time.time()), ("limiter", limiter)]:
    setattr(app.state, k, v)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","), allow_methods=["GET", "POST"], allow_headers=["*"])
app.middleware("http")(track_metrics)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024

# ── Inference / Prediction Endpoints ──────────────────────────────────────────



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
        probs  = _calibrated_softmax(logits, arch)[0]
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
    explainability_meta = {}
    try:
        explainability, explainability_meta = get_explanation(
            model=net,
            arch=arch,
            input_tensor=tensor,
            original=crop,
            method="auto",
            device=DEVICE,
            shap_explainer_instance=SHAP_EXPLAINER,
        )
        explainability_method = explainability_meta.get("method", explainability_method)
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
        "explainability_meta":   explainability_meta,
        "inference_time_ms":     round(ms, 2),
        "crop_b64":              image_to_base64(crop),
    }


def _run_batched_inference(
    net: torch.nn.Module,
    arch: str,
    crops: list,
    detections: list,
    include_explainability: bool = False,
) -> list:
    """
    Run inference on all face crops in a SINGLE forward pass (batched).
    Dramatically faster than serial inference for 4+ faces.

    Falls back to serial inference for explainability maps (Grad-CAM /
    Attention Rollout require individual inputs).
    """
    if not crops:
        return []

    # Build batch tensor
    tensors = torch.cat([pil_to_tensor(c) for c in crops], dim=0)  # (N, 3, 224, 224)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits_batch = net(tensors)                       # (N, 2)
        probs_batch  = _calibrated_softmax(logits_batch, arch) # (N, 2)
    total_ms = (time.perf_counter() - t0) * 1000

    results = []
    for i, (det, crop) in enumerate(zip(detections, crops)):
        fake_prob = probs_batch[i, 0].item()
        real_prob = probs_batch[i, 1].item()
        verdict   = "fake" if fake_prob > 0.5 else "real"
        per_face_ms = round(total_ms / len(crops), 2)

        explainability        = None
        explainability_method = "grad_cam" if supports_gradcam(arch) else "attention_rollout"
        explainability_meta   = {}
        if include_explainability:
            # Explainability requires individual tensor — run separately
            try:
                t = pil_to_tensor(crop)
                explainability, explainability_meta = get_explanation(
                    model=net,
                    arch=arch,
                    input_tensor=t,
                    original=crop,
                    method="auto",
                    device=DEVICE,
                    shap_explainer_instance=SHAP_EXPLAINER,
                )
                explainability_method = explainability_meta.get("method", explainability_method)
            except Exception as e:
                print(f"[explainability] face#{det.face_idx}: {e}")

        results.append({
            "face_idx":    det.face_idx,
            "bbox":        det.to_dict()["bbox"],
            "keypoints":   det.to_dict()["keypoints"],
            "detection_confidence": det.confidence,
            "prediction":  verdict,
            "confidence":  round(max(fake_prob, real_prob), 4),
            "probabilities": {
                "fake": round(fake_prob, 4),
                "real": round(real_prob, 4),
            },
            "explainability":        explainability,
            "explainability_method": explainability_method,
            "explainability_meta":   explainability_meta,
            "inference_time_ms":     per_face_ms,
            "crop_b64":              image_to_base64(crop),
        })

    return results


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
    explain_method: str = Query(
        default="auto",
        description=(
            "Explainability method: auto | grad_cam | attention_rollout | "
            "lime | shap. "
            "lime and shap are slow (3–20s) — use /api/explain for async. "
            "auto selects grad_cam for CNNs, attention_rollout for ViT."
        ),
    ),
    auth: AuthContext = Depends(require_auth),
):
    # Backward compatibility with test_predict_no_model_503
    if request.app.state.predictor is None:
        raise HTTPException(503, "No model loaded. Run training/train.py first.")
    t0_total = time.perf_counter()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    contents = await file.read()
    if len(contents) > auth.tier.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large ({len(contents)//1024}KB). Max {auth.tier.max_image_mb}MB.")

    arch  = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded. "
                   f"Check checkpoint at: {SUPPORTED_ARCHITECTURES.get(arch, '?')}"
        )
    net = entry["model"]

    _explain_method = explain_method.lower()
    if _explain_method in SLOW_METHODS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{_explain_method}' is too slow for synchronous prediction. "
                "Submit via POST /api/explain instead and poll GET /api/explain/{job_id}."
            ),
        )

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

            # Run batched inference on face crops
            face_results = _run_batched_inference(
                net, arch, crops, detections,
                include_explainability=True,
            )

            # Aggregate verdict: fake if ANY face is fake
            any_fake     = any(r["prediction"] == "fake" for r in face_results)
            agg_verdict  = "fake" if any_fake else "real"
            agg_conf     = max(r["confidence"] for r in face_results)
            fake_faces   = [r["face_idx"] for r in face_results
                            if r["prediction"] == "fake"]

            elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)

            # Calibration metadata
            cal = CALIBRATION_REGISTRY.get(arch)
            cal_meta = {
                "calibrated": cal is not None,
                "temperature": round(cal.temperature, 4) if cal else 1.0,
                "ece_improvement": cal.ece_improvement if cal and cal.ece_improvement is not None else None,
            }

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

                # Calibration
                "calibration":           cal_meta,

                # Backward-compat: first face explainability at top level
                "gradcam_image":         face_results[0]["explainability"],
                "explainability":        face_results[0]["explainability"],
                "explainability_method": face_results[0]["explainability_method"],
                "explainability_meta":   face_results[0].get("explainability_meta", {}),
                "probabilities":         face_results[0]["probabilities"],
                "_auth": {
                    "key_id": auth.key_id,
                    "tier":   auth.tier.name,
                },
            }

    # ── Whole-frame fallback ───────────────────────────────────────────────
    # Used when face_detect=False OR when no face was detected
    tensor = pil_to_tensor(pil_image)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = net(tensor)
        probs  = _calibrated_softmax(logits, arch)[0]
    ms = (time.perf_counter() - t0) * 1000

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    verdict   = "fake" if fake_prob > 0.5 else "real"

    explainability = None
    explainability_meta = {}
    exp_method = _explain_method
    try:
        explainability, explainability_meta = get_explanation(
            model=net,
            arch=arch,
            input_tensor=tensor,
            original=pil_image,
            method=_explain_method,
            device=DEVICE,
            shap_explainer_instance=SHAP_EXPLAINER,
        )
        exp_method = explainability_meta.get("method", _explain_method)
    except Exception as e:
        print(f"[explainability] whole-frame {arch}: {e}")

    elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
    # Build calibration metadata for the response
    cal = CALIBRATION_REGISTRY.get(arch)
    cal_meta = {
        "calibrated": cal is not None,
        "temperature": round(cal.temperature, 4) if cal else 1.0,
        "ece_improvement": cal.ece_improvement if cal and cal.ece_improvement is not None else None,
    }

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
        "calibration":           cal_meta,
        "explainability":        explainability,
        "gradcam_image":         explainability,
        "explainability_method": exp_method,
        "explainability_meta":   explainability_meta,
        "processing_ms":         elapsed_ms,
        "inference_time_ms":     elapsed_ms,
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }


@app.post("/api/predict/ensemble")
async def predict_ensemble(
    file:      UploadFile = File(...),
    strategy:  str        = Query(
        default="auto",
        description=(
            "auto       — use whatever strategy is in weights.json\n"
            "weighted_average — use fixed per-arch weights\n"
            "learned    — use fitted logistic regression meta-classifier"
        ),
    ),
    threshold: float = Query(
        default=0.5,
        description="Decision threshold for fake vs real (default 0.5)",
    ),
    auth: AuthContext = Depends(require_feature("can_use_ensemble")),
):
    """
    Run all loaded models on the image and return a single fused verdict.
    Uses calibrated probabilities and either weighted averaging or a
    trained meta-classifier to combine per-model predictions.
    """
    # Decode image
    contents = await file.read()
    if len(contents) > auth.tier.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {auth.tier.max_image_mb}MB.")
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tensor = pil_to_tensor(pil_image)   # existing helper — (1,3,224,224) on DEVICE

    # Collect per-model calibrated fake probs
    member_probs = _collect_member_probs(tensor)

    if not member_probs:
        raise HTTPException(
            status_code=503,
            detail="No models loaded in registry. "
                   "Check /api/models and model checkpoints.",
        )

    # Override ensemble strategy for this request if specified
    if strategy == "auto" or strategy == ENSEMBLE.strategy:
        scorer = ENSEMBLE
    else:
        # Build a temporary scorer with the requested strategy
        scorer = EnsembleScorer(
            strategy=strategy,
            weights_path="checkpoints/ensemble/weights.json",
        )

    result = scorer.score(member_probs)

    # Apply custom threshold
    ensemble_p  = result["ensemble_fake_prob"]
    verdict     = "fake" if ensemble_p >= threshold else "real"
    confidence  = max(ensemble_p, 1 - ensemble_p)

    # Per-model detail (for transparency)
    per_model = {}
    for arch, fake_p in member_probs.items():
        entry = MODEL_REGISTRY.get(arch, {})
        per_model[arch] = {
            "fake_prob":   fake_p,
            "real_prob":   round(1 - fake_p, 4),
            "verdict":     "fake" if fake_p >= threshold else "real",
            "temperature": entry.get("temperature", 1.0),
            "calibrated":  CALIBRATION_REGISTRY.get(arch) is not None,
            "params":      entry.get("params", 0),
        }

    return {
        # Top-level verdict — backward compatible
        "prediction":  verdict,
        "confidence":  round(float(confidence), 4),
        "mode":        "ensemble",

        # Ensemble detail
        "ensemble": {
            **result,
            "threshold": threshold,
            "verdict":   verdict,           # re-apply threshold
            "confidence": round(float(confidence), 4),
        },

        # Per-model breakdown
        "per_model": per_model,

        # Metadata
        "n_models_loaded": len(member_probs),
        "calibration_applied": all(
            CALIBRATION_REGISTRY.get(a) is not None
            for a in member_probs
        ),
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }


@app.post("/api/predict/group")
async def predict_group(
    file: UploadFile = File(...),

    # Model selection
    model: str = Query(default=DEFAULT_MODEL),

    # Face detection params
    min_confidence: float = Query(
        default=0.5,
        description="Minimum MediaPipe face detection confidence (0-1).",
    ),
    min_face_fraction: float = Query(
        default=0.03,
        description="Minimum face area as a fraction of image area. "
                    "Smaller faces are ignored. 0.03 = 3% of image.",
    ),
    iou_threshold: float = Query(
        default=0.45,
        description="IoU threshold for NMS duplicate removal.",
    ),
    max_faces: int = Query(
        default=20,
        description="Hard cap on faces to analyse per image.",
    ),

    # Verdict aggregation
    strategy: str = Query(
        default="any_fake",
        description="Verdict strategy: any_fake | majority | weighted | confident",
    ),
    confidence_threshold: float = Query(
        default=0.70,
        description="Used only by 'confident' strategy: fake_prob threshold.",
    ),

    # Response size controls
    include_crops: bool = Query(
        default=True,
        description="Include base64 face crop images in the response.",
    ),
    include_explainability: bool = Query(
        default=False,
        description="Include Grad-CAM / Attention Rollout maps. "
                    "Adds significant response size for many faces.",
    ),
    auth: AuthContext = Depends(require_auth),
):
    """
    Analyse a group photo. Detects all faces, runs deepfake inference on each,
    and returns a per-face verdict plus a configurable image-level aggregate.
    """
    # ── Model lookup ────────────────────────────────────────────────────────
    arch  = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded. "
                   f"Check checkpoint: {SUPPORTED_ARCHITECTURES.get(arch, '?')}"
        )
    net = entry["model"]

    # ── Image decode ────────────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > auth.tier.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {auth.tier.max_image_mb}MB.")
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_w, img_h = pil_image.size

    # ── Face detection with per-request params ───────────────────────────────
    pipeline = FacePipeline(
        model_selection=1,
        min_detection_confidence=min_confidence,
        margin=0.30,
        output_size=224,
        align=True,
        max_faces=max_faces,
        iou_threshold=iou_threshold,
        min_face_fraction=min_face_fraction,
        min_face_pixels=40,
    )

    try:
        detections, crops = pipeline.process(pil_image)
    finally:
        pipeline.close()

    if not detections:
        # No-face fallback: whole-frame inference
        tensor = pil_to_tensor(pil_image)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = net(tensor)
            probs  = _calibrated_softmax(logits, arch)[0]
        ms = (time.perf_counter() - t0) * 1000

        fake_p = probs[0].item()
        real_p = probs[1].item()
        verdict = "fake" if fake_p > 0.5 else "real"

        return {
            "prediction":     verdict,
            "confidence":     round(max(fake_p, real_p), 4),
            "architecture":   arch,
            "mode":           "whole_frame_fallback",
            "faces_detected": 0,
            "faces_analysed": 0,
            "face_results":   [],
            "aggregate": {
                "verdict":             verdict,
                "strategy":            strategy,
                "total_faces":         0,
                "fake_count":          0,
                "real_count":          0,
                "fake_face_indices":   [],
                "aggregate_confidence": round(max(fake_p, real_p), 4),
                "dissenting_faces":    [],
                "unanimity":           True,
            },
            "annotated_image": None,
            "image_width":  img_w,
            "image_height": img_h,
            "warning": "No faces detected. Whole-frame inference used as fallback.",
            "_auth": {
                "key_id": auth.key_id,
                "tier":   auth.tier.name,
            },
        }

    # ── Batched inference ───────────────────────────────────────────────────
    face_results = _run_batched_inference(
        net, arch, crops, detections,
        include_explainability=include_explainability,
    )

    # ── Annotated image with verdict-coloured boxes ─────────────────────────
    verdicts_map = {r["face_idx"]: r["prediction"] for r in face_results}
    annotated    = FacePipeline.draw_boxes(pil_image, detections, verdicts=verdicts_map)
    annotated_b64 = image_to_base64(annotated)

    # ── Build response ──────────────────────────────────────────────────────
    res = build_group_response(
        face_results=face_results,
        detections=detections,
        annotated_b64=annotated_b64,
        arch=arch,
        image_width=img_w,
        image_height=img_h,
        strategy=strategy,
        confidence_threshold=confidence_threshold,
        include_crops=include_crops,
        include_explainability=include_explainability,
    )
    res["_auth"] = {
        "key_id": auth.key_id,
        "tier":   auth.tier.name,
    }
    return res


@app.post("/api/predict/video/sync")
@limiter.limit("5/minute")
async def predict_video_sync(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
    sample_frames: int = 16,
    auth: AuthContext = Depends(require_auth),
):
    """
    Synchronous video analysis. Blocks until complete.
    Use only for files < 50 MB or in development.
    For large files use POST /api/predict/video (async).
    """
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
    if len(data) > auth.tier.max_video_mb * 1024 * 1024:
        raise HTTPException(413, f"Video too large. Max {auth.tier.max_video_mb}MB.")

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
                probs = _calibrated_softmax(logits, arch)[0]
            
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

        # ── Audio analysis ─────────────────────────────────────────────────
        audio_result: Optional[AudioResult] = None
        if AUDIO_PIPELINE.loaded:
            try:
                audio_result = AUDIO_PIPELINE.analyse_file(tmp_path)
            except Exception as e:
                print(f"[audio] analysis failed: {e}")

        # ── Fusion ──────────────────────────────────────────────────────────
        audio_spoof_prob = audio_result.spoof_prob if (
            audio_result and audio_result.has_audio and not audio_result.error
        ) else None

        fusion = fuse_verdicts(
            visual_fake_prob = avg_fake,
            audio_spoof_prob = audio_spoof_prob,
            strategy         = "weighted",
        )

        fake_frames = sum(1 for p in preds if p["label"] == "fake")
        real_frames = sum(1 for p in preds if p["label"] == "real")

        # ── Response ─────────────────────────────────────────────────────────
        return {
            # Top-level fused verdict — backward compatible field names
            "prediction":   fusion["verdict"],
            "label":        fusion["verdict"],
            "confidence":   fusion["confidence"],
            "architecture": arch,
            "mode":         "video_multimodal",
            "processing_ms": round((time.time() - t0) * 1000, 1),
            "frame_predictions": preds,

            # Visual sub-result (existing fields, now nested)
            "visual_result": {
                "prediction":       avg_label,
                "fake_prob":        round(avg_fake, 4),
                "real_prob":        round(1 - avg_fake, 4),
                "frames_analysed":  len(preds),
                "fake_frames":      fake_frames,
                "real_frames":      real_frames,
            },

            # Audio sub-result
            "audio_result": audio_result.to_dict() if audio_result else {
                "has_audio": False,
                "error": "Audio pipeline not loaded",
            },

            # Fusion metadata
            "fusion": fusion,
            "_auth": {
                "key_id": auth.key_id,
                "tier":   auth.tier.name,
            },
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ── Async Video Job Endpoints ─────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
VIDEO_MAX_SIZE_MB = int(os.environ.get("DEEPTRACE_VIDEO_MAX_MB", "2000"))

@app.post("/api/predict/video")
async def predict_video_async(
    file:               UploadFile = File(...),
    model:              str        = Query(default=DEFAULT_MODEL),
    sample_every_n_sec: float      = Query(default=5.0,
        description="Sample one frame every N seconds"),
    max_frames:         int        = Query(default=120,
        description="Hard cap on frames to analyse per video"),
    fusion_strategy:    str        = Query(default="weighted",
        description="Audio/visual fusion: weighted | conflict_flag | visual_only"),
    run_ensemble:       bool       = Query(default=True,
        description="Run all models + ensemble scoring"),
    auth: AuthContext = Depends(require_auth),
):
    """
    Submit a video for async deepfake analysis.
    Returns immediately with a job_id.
    Poll GET /api/jobs/{job_id} to check status and retrieve results.
    """
    # Validate extension
    ext = Path(file.filename or "upload.mp4").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format: '{ext}'. "
                   f"Supported: {sorted(VIDEO_EXTENSIONS)}",
        )

    # Validate model
    arch  = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{arch}' not loaded.",
        )

    # Stream upload to shared temp directory
    safe_name  = f"{uuid.uuid4()}{ext}"
    video_path = VIDEO_UPLOAD_DIR / safe_name

    try:
        with open(video_path, "wb") as dst:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                dst.write(chunk)
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    file_size = video_path.stat().st_size
    if file_size > auth.tier.max_video_mb * 1024 * 1024:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size // (1024*1024)} MB "
                   f"(max {auth.tier.max_video_mb} MB)",
        )

    # Register job in Redis
    job_id = _job_store.create_job(
        filename=file.filename or safe_name,
        model=arch,
        file_size=file_size,
        options={
            "sample_every_n_sec": sample_every_n_sec,
            "max_frames":         max_frames,
            "fusion_strategy":    fusion_strategy,
            "run_ensemble":       run_ensemble,
            "submitted_by":       auth.key_id,
        },
    )

    # Submit Celery task
    task = _analyse_video_task.apply_async(
        kwargs={
            "job_id":             job_id,
            "video_path":         str(video_path),
            "model":              arch,
            "sample_every_n_sec": sample_every_n_sec,
            "max_frames":         max_frames,
            "fusion_strategy":    fusion_strategy,
            "run_ensemble":       run_ensemble,
        },
        queue="video",
    )
    _job_store.set_celery_id(job_id, task.id)

    return {
        "job_id":      job_id,
        "celery_id":   task.id,
        "status":      "pending",
        "filename":    file.filename,
        "file_size":   file_size,
        "model":       arch,
        "poll_url":    f"/api/jobs/{job_id}",
        "cancel_url":  f"/api/jobs/{job_id}/cancel",
        "submitted_at": time.time(),
        "options": {
            "sample_every_n_sec": sample_every_n_sec,
            "max_frames":         max_frames,
            "fusion_strategy":    fusion_strategy,
            "run_ensemble":       run_ensemble,
        },
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, auth: AuthContext = Depends(require_auth)):
    """Poll the status and result of an async video analysis job."""
    job = _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or expired (TTL: {_job_store.JOB_TTL_SECONDS}s)",
        )
    return job


@app.get("/api/jobs")
async def list_jobs(limit: int = Query(default=20, le=100), auth: AuthContext = Depends(require_auth)):
    """List the most recently submitted jobs."""
    return {"jobs": _job_store.list_jobs(limit=limit)}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, auth: AuthContext = Depends(require_auth)):
    """
    Cancel a pending or running job.
    Running jobs receive a SoftTimeLimitExceeded exception and clean up.
    """
    job = _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] in ("done", "error", "cancelled"):
        return {"job_id": job_id, "status": job["status"],
                "message": "Job already completed"}

    celery_id = job.get("celery_id")
    if celery_id:
        AsyncResult(celery_id, app=_celery_app).revoke(terminate=True)

    _job_store.set_status(job_id, "cancelled")
    return {"job_id": job_id, "status": "cancelled"}


# ── Model Registry & Compare Endpoints ───────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return the status of every registered model."""
    result = {}
    for arch, entry in MODEL_REGISTRY.items():
        cal = CALIBRATION_REGISTRY.get(arch)
        result[arch] = {
            "loaded":     entry["loaded"],
            "checkpoint": entry["checkpoint"],
            "params":     entry["params"],
            "explainability_method": "grad_cam" if supports_gradcam(arch) else "attention_rollout",
            "calibrated": cal is not None,
            "temperature": round(cal.temperature, 4) if cal else None,
        }
    return result


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
async def reload_model(request: Request, auth: AuthContext = Depends(require_admin)):
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


@app.post("/api/model/calibrate")
async def reload_calibration(request: Request, auth: AuthContext = Depends(require_admin)):
    """
    Hot-reload temperature calibration for one or all architectures.

    Body (JSON):
        {"architecture": "resnet18"}       — reload one
        {} or {"architecture": "all"}      — reload all

    Returns the updated calibration state.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    arch = body.get("architecture", "all").lower()

    if arch == "all":
        results = {}
        for a in SUPPORTED_ARCHITECTURES:
            ok = _load_calibration(a)
            cal = CALIBRATION_REGISTRY.get(a)
            results[a] = {
                "calibrated": cal is not None,
                "temperature": round(cal.temperature, 4) if cal else None,
            }
        return {"status": "reloaded", "architectures": results}

    if arch not in SUPPORTED_ARCHITECTURES:
        raise HTTPException(400, f"Unknown architecture: '{arch}'")

    ok = _load_calibration(arch)
    cal = CALIBRATION_REGISTRY.get(arch)
    return {
        "status": "reloaded" if ok else "no_calibration",
        "architecture": arch,
        "calibrated": cal is not None,
        "temperature": round(cal.temperature, 4) if cal else None,
    }


@app.post("/api/compare")
async def compare_models(
    file: UploadFile = File(...),
    face_detect: bool = Query(default=True),
    max_faces: int    = Query(default=3),
    auth: AuthContext = Depends(require_auth),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
        
    contents = await file.read()
    if len(contents) > auth.tier.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {auth.tier.max_image_mb}MB.")

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

        if use_face_crops:
            model_faces = _run_batched_inference(
                net, arch, crops, detections,
                include_explainability=False,
            )
        else:
            # Single whole-frame inference
            model_faces = _run_batched_inference(
                net, arch, [pil_image],
                [FaceDetection(0, 0, float(pil_image.size[0]), float(pil_image.size[1]), 1.0, face_idx=0)],
                include_explainability=False,
            )

        # Aggregate per-model
        any_fake  = any(r["prediction"] == "fake" for r in model_faces)
        agg_conf  = max(r["confidence"] for r in model_faces) if model_faces else 0.0
        # Calculate P(fake) and P(real) for model aggregation
        fake_prob = sum(r["probabilities"]["fake"] for r in model_faces) / len(model_faces) if model_faces else 0.5
        results[arch] = {
            "prediction":           "fake" if any_fake else "real",
            "confidence":           agg_conf,
            "probabilities": {
                "fake": round(fake_prob, 4),
                "real": round(1 - fake_prob, 4),
            },
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

    # ── Ensemble verdict ─────────────────────────────────────────────────────
    # Collect calibrated fake probs from compare results
    member_probs_compare = {
        arch: r["probabilities"]["fake"]
        for arch, r in results.items()
        if "probabilities" in r and "fake" in r.get("probabilities", {})
    }
    ensemble_result = {}
    if member_probs_compare:
        ensemble_result = ENSEMBLE.score(member_probs_compare)

    # Return
    return {
        "models":          results,
        "consensus":       consensus,
        "agreement":       len(set(verdicts)) == 1 if verdicts else False,
        "faces_detected":  len(detections),
        "mode":            "face_detect" if use_face_crops else "whole_frame",
        "annotated_image": annotated_b64,
        "ensemble":        ensemble_result,
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }

AUDIO_MIME_TYPES = {
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    ".aac", ".wma", ".opus",
}

@app.post("/api/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    aggregate: str   = Query(
        default="mean",
        description="Segment aggregation: mean | majority | max_spoof",
    ),
    auth: AuthContext = Depends(require_auth),
):
    """
    Analyse an audio-only file for voice synthesis / voice conversion deepfakes.
    Accepts: MP3, WAV, FLAC, M4A, OGG, AAC, WMA, OPUS.
    """
    if not AUDIO_PIPELINE.loaded:
        raise HTTPException(
            status_code=503,
            detail="Audio model (AASIST-L) not loaded. "
                   f"Check checkpoint: {AUDIO_PIPELINE.checkpoint_path}",
        )

    ext = Path(file.filename or "upload.wav").suffix.lower()
    if ext not in AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: '{ext}'. "
                   f"Supported: {sorted(AUDIO_MIME_TYPES)}",
        )

    contents = await file.read()

    # Temporarily override aggregate mode if requested
    original_aggregate       = AUDIO_PIPELINE.aggregate
    AUDIO_PIPELINE.aggregate = aggregate
    try:
        result = AUDIO_PIPELINE.analyse_bytes(contents, filename=file.filename or "upload.wav")
    finally:
        AUDIO_PIPELINE.aggregate = original_aggregate

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return {
        "prediction":    result.prediction,    # "spoof" | "bonafide"
        "confidence":    result.confidence,
        "mode":          "audio_only",
        "audio_result":  result.to_dict(),
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }


# ── Async Explainability Endpoints ────────────────────────────────────────────

@app.post("/api/explain")
async def submit_explain_job(
    file:           UploadFile = File(...),
    model:          str        = Query(default=DEFAULT_MODEL),
    method:         str        = Query(default="lime",
                                       description="lime | shap"),
    # LIME params
    lime_samples:   int        = Query(default=1000),
    lime_segments:  int        = Query(default=75),
    lime_top_k:     int        = Query(default=10),
    lime_pos_only:  bool       = Query(default=False),
    # SHAP params
    shap_n_bg:      int        = Query(default=50),
    shap_n_evals:   int        = Query(default=50),
    auth: AuthContext = Depends(require_feature("can_use_explain")),
):
    """
    Submit an async LIME or SHAP explanation job.
    Returns immediately with a job_id. Poll GET /api/explain/{job_id} for result.
    """
    method = method.lower()
    if method not in SLOW_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"'{method}' is a fast method — use GET /api/predict/image?explain_method={method} instead.",
        )

    arch  = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    if not entry or not entry["loaded"]:
        raise HTTPException(status_code=503,
                            detail=f"Model '{arch}' not loaded.")

    contents = await file.read()
    if len(contents) > auth.tier.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {auth.tier.max_image_mb}MB.")
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    net   = entry["model"]
    dev   = DEVICE

    # Build normalised tensor for the job
    tensor = _INFER_TRANSFORM(pil_image).unsqueeze(0).to(dev)

    # Capture everything the job needs — avoid closure over request objects
    _arch       = arch
    _pil        = pil_image.copy()
    _net        = net
    _method     = method
    _lime_kw    = dict(
        lime_num_samples=lime_samples,
        lime_num_superpixels=lime_segments,
        lime_top_k=lime_top_k,
        lime_positive_only=lime_pos_only,
    )
    _shap_kw    = dict(
        shap_n_background=shap_n_bg,
        shap_n_evals=shap_n_evals,
        shap_explainer_instance=SHAP_EXPLAINER,
    )

    def _job():
        kw = _lime_kw if _method == "lime" else _shap_kw
        b64, meta = get_explanation(
            model=_net, arch=_arch,
            input_tensor=tensor, original=_pil,
            method=_method, device=dev, **kw,
        )
        return {"explainability": b64, "explainability_meta": meta}

    job_id = EXPLAIN_CACHE.submit(_job)

    return {
        "job_id":    job_id,
        "status":    "pending",
        "method":    method,
        "model":     arch,
        "poll_url":  f"/api/explain/{job_id}",
        "estimated_seconds": 8 if method == "lime" else 18,
        "_auth": {
            "key_id": auth.key_id,
            "tier":   auth.tier.name,
        },
    }


@app.get("/api/explain/{job_id}")
async def poll_explain_job(job_id: str, auth: AuthContext = Depends(require_auth)):
    """Poll the status of an async LIME or SHAP explanation job."""
    entry = EXPLAIN_CACHE.get(job_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or expired (TTL: 300s).",
        )
    return entry



@app.post("/api/ensemble/reload")
async def reload_ensemble(auth: AuthContext = Depends(require_admin)):
    """Reload ensemble weights from checkpoints/ensemble/weights.json."""
    global ENSEMBLE
    path = "checkpoints/ensemble/weights.json"
    if not Path(path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"weights.json not found at {path}. "
                   "Run: python training/fit_ensemble.py"
        )
    try:
        ENSEMBLE = EnsembleScorer(weights_path=path)
        return {
            "status":   "reloaded",
            "strategy": ENSEMBLE.strategy,
            "weights":  ENSEMBLE.active_weights
                        if ENSEMBLE.strategy == "weighted_average" else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


# ── System / Health / Config Endpoints ────────────────────────────────────────

@app.get("/api/health")
async def health(model: str = Query(default=DEFAULT_MODEL)):
    arch = model.lower()
    entry = MODEL_REGISTRY.get(arch)
    loaded = entry is not None and entry["loaded"]

    # Redis connectivity check
    try:
        import redis as _redis_lib
        _r = _redis_lib.Redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
        )
        _r.ping()
        redis_connected = True
        video_queue_len = _r.llen("video")
    except Exception:
        redis_connected = False
        video_queue_len = -1

    return {
        "status": "ok",
        "model_loaded": loaded,
        "audio_model_loaded": AUDIO_PIPELINE.loaded,
        "audio_checkpoint":   AUDIO_PIPELINE.checkpoint_path,
        "explain_queue_pending": EXPLAIN_CACHE.pending_count(),
        "ensemble_strategy":  ENSEMBLE.strategy,
        "ensemble_fitted":    ENSEMBLE.is_fitted,
        "ensemble_members":   list(ENSEMBLE.active_weights.keys())
                              if ENSEMBLE.strategy == "weighted_average" else [],
        "redis_connected":    redis_connected,
        "video_queue_depth":  video_queue_len,
        "auth_enabled":       True,
        "key_db_path":        os.environ.get("DEEPTRACE_DB_PATH", "data/deeptrace.db"),
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

# ── Key management ─────────────────────────────────────────────────────────

@app.post("/api/keys")
async def create_api_key(
    body: dict,
    auth: AuthContext = Depends(require_admin),
):
    """Create a new API key. Admin only."""
    name  = body.get("name", "").strip()
    tier  = body.get("tier", "free")
    notes = body.get("notes", "")

    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")
    if tier not in TIERS:
        raise HTTPException(status_code=400,
                            detail=f"Invalid tier '{tier}'. Valid: {list(TIERS)}")

    result = await create_key(name=name, tier=tier,
                               created_by=auth.key_id, notes=notes)
    return result   # includes raw_key — shown once


@app.get("/api/keys")
async def list_api_keys(
    include_inactive: bool = Query(default=False),
    auth: AuthContext = Depends(require_admin),
):
    """List all API keys (hashes redacted). Admin only."""
    keys = await list_keys(include_inactive=include_inactive)
    return {"keys": keys, "total": len(keys)}


@app.delete("/api/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    auth:   AuthContext = Depends(require_admin),
):
    """Revoke an API key. Admin only."""
    if key_id == auth.key_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot revoke your own key. Use another admin key.",
        )
    ok = await revoke_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Key '{key_id}' not found")
    return {"key_id": key_id, "status": "revoked"}


@app.get("/api/keys/me/usage")
async def get_my_usage(
    days: int = Query(default=30, le=90),
    auth: AuthContext = Depends(require_auth),
):
    """Get your own usage stats."""
    usage = await get_usage(auth.key_id, days=days)
    return {
        "key_id": auth.key_id,
        "name":   auth.name,
        "tier":   auth.tier.name,
        "days":   days,
        "usage":  usage,
    }


@app.get("/api/keys/{key_id}/usage")
async def get_key_usage(
    key_id: str,
    days:   int = Query(default=30, le=90),
    auth:   AuthContext = Depends(require_admin),
):
    """Get per-endpoint usage stats for a key. Admin only."""
    usage = await get_usage(key_id, days=days)
    return {"key_id": key_id, "days": days, "usage": usage}


# ── Static / Frontend Serving ──────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="stitch_veritas_ai_detection_platform"), name="static")

@app.get("/", response_class=FileResponse)
@app.get("/index.html", response_class=FileResponse)
def serve_frontend():
    if Path("index.html").exists():
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found. Check deployment.")


@app.get("/webhooks", response_class=FileResponse)
@app.get("/webhooks.html", response_class=FileResponse)
async def docs():
    return FileResponse("webhooks.html")
