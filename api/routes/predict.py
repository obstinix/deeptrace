from __future__ import annotations
import io
import time
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024

@router.post("/api/predict/image")
@limiter.limit("30/minute")
async def predict_image(request: Request, file: UploadFile = File(...), use_tta: bool = False):
    if request.app.state.predictor is None:
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
    r = request.app.state.predictor.predict_pil(img, use_tta=use_tta)
    return {"label": r["label"], "confidence": round(r["confidence"], 4),
            "probabilities": {"real": round(r["prob_real"], 4),
                              "fake": round(r["prob_fake"], 4)},
            "processing_ms": round((time.time() - t0) * 1000, 1),
            "gradcam_image": r.get("gradcam_image")}

@router.post("/api/predict/video")
@limiter.limit("5/minute")
async def predict_video(request: Request, file: UploadFile = File(...), sample_frames: int = 16):
    if request.app.state.predictor is None:
        raise HTTPException(503, "No model loaded.")
    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video too large. Max 100MB.")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name
    try:
        t0 = time.time()
        r = request.app.state.predictor.predict_video(tmp_path, n_frames=min(sample_frames, 32))
        return {**r, "processing_ms": round((time.time() - t0) * 1000, 1)}
    finally:
        os.unlink(tmp_path)
