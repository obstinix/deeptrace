"""
worker/tasks.py

DeepTrace Celery tasks.

analyse_video(job_id, video_path, model, options)
  → Extracts frames, runs visual + audio + ensemble inference,
    reports progress to Redis, stores result in Redis.

The task receives a file path (not raw bytes) — the API endpoint saves
the upload to the shared /tmp/deeptrace_jobs/ directory before submitting.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is importable in worker process
sys.path.insert(0, str(Path(__file__).parent.parent))

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery_app import celery_app
from worker.storage import set_result, set_error
from worker.progress import ProgressReporter


# ---------------------------------------------------------------------------
# Worker-local model registry
# Each worker process initialises models once on first use (lazy).
# ---------------------------------------------------------------------------

_worker_models: Dict[str, Any] = {}
_worker_audio_pipeline          = None
_worker_ensemble_scorer         = None


def _get_model(arch: str):
    """Lazy-load and cache a visual model in the worker process."""
    global _worker_models
    if arch not in _worker_models:
        import torch
        from src.deepfake_recognition.utils.model_factory import (
            build_model, SUPPORTED_ARCHITECTURES,
        )
        from src.deepfake_recognition.utils.calibration import TemperatureScaler

        ckpt_path = SUPPORTED_ARCHITECTURES.get(arch)
        if not ckpt_path or not Path(ckpt_path).exists():
            raise RuntimeError(f"Checkpoint not found for '{arch}': {ckpt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_model(arch, num_classes=2, dropout=0.0)
        state  = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device).eval()

        temp_path = Path(ckpt_path).parent / "temperature.json"
        calibrator = (TemperatureScaler.load(str(temp_path))
                      if temp_path.exists() else None)

        _worker_models[arch] = {
            "model":      model,
            "calibrator": calibrator,
            "device":     device,
        }
    return _worker_models[arch]


def _get_audio_pipeline():
    global _worker_audio_pipeline
    if _worker_audio_pipeline is None:
        from src.deepfake_recognition.audio.audio_pipeline import AudioPipeline
        p = AudioPipeline(
            checkpoint_path="checkpoints/aasist/best.pth",
            device=__import__("torch").device("cpu"),
        )
        p.load()
        _worker_audio_pipeline = p
    return _worker_audio_pipeline


def _get_ensemble():
    global _worker_ensemble_scorer
    if _worker_ensemble_scorer is None:
        from src.deepfake_recognition.utils.ensemble import EnsembleScorer
        _worker_ensemble_scorer = EnsembleScorer(
            weights_path="checkpoints/ensemble/weights.json"
        )
    return _worker_ensemble_scorer


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(
    video_path: str,
    sample_every_n_sec: float,
    max_frames:         int,
    reporter:           ProgressReporter,
) -> List[Any]:
    """
    Extract frames from a video file using OpenCV.
    Returns a list of PIL Images.
    """
    import cv2
    from PIL import Image as PILImage

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hop        = max(1, int(fps * sample_every_n_sec))
    duration_s = total_frames / fps

    reporter.update(
        "extracting_frames", pct=5,
        message=f"Extracting frames ({duration_s:.0f}s video, "
                f"sampling every {sample_every_n_sec}s)…",
        data={"duration_sec": round(duration_s, 1), "fps": round(fps, 1)},
    )

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = PILImage.fromarray(rgb)
        frames.append(frame)
        frame_idx += hop

    cap.release()
    reporter.update(
        "extracting_frames", pct=15,
        message=f"Extracted {len(frames)} frames",
        data={"frames_extracted": len(frames)},
    )
    return frames


# ---------------------------------------------------------------------------
# Visual inference
# ---------------------------------------------------------------------------

def _run_visual_inference(
    frames:   List[Any],
    arch:     str,
    reporter: ProgressReporter,
    fake_class_idx: int = 0,
) -> Dict[str, Any]:
    """Run batched visual inference on extracted frames."""
    import torch
    import torchvision.transforms as T

    entry  = _get_model(arch)
    model  = entry["model"]
    cal    = entry["calibrator"]
    device = entry["device"]

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    batch_size  = 16
    all_fake_p  = []
    n_frames    = len(frames)

    for i in range(0, n_frames, batch_size):
        batch_imgs = frames[i : i + batch_size]
        tensors    = torch.stack([tf(f.convert("RGB")) for f in batch_imgs]).to(device)

        with torch.no_grad():
            logits = model(tensors)
            if cal is not None:
                probs = cal.calibrate(logits)
            else:
                probs = torch.softmax(logits, dim=1)

        all_fake_p.extend(probs[:, fake_class_idx].cpu().tolist())

        pct = 15 + int((i / n_frames) * 45)
        reporter.update(
            "visual_inference", pct=pct,
            message=f"Visual inference: {min(i + batch_size, n_frames)}/{n_frames} frames",
        )

    mean_fake = sum(all_fake_p) / len(all_fake_p) if all_fake_p else 0.5
    fake_frames = sum(1 for p in all_fake_p if p > 0.5)
    real_frames = n_frames - fake_frames

    reporter.update(
        "visual_inference", pct=60,
        message=f"Visual: {fake_frames}/{n_frames} frames fake "
                f"(mean P={mean_fake:.3f})",
        data={
            "frames_analysed": n_frames,
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "mean_fake_prob": round(mean_fake, 4),
        },
    )
    return {
        "prediction":    "fake" if mean_fake > 0.5 else "real",
        "fake_prob":     round(mean_fake, 4),
        "real_prob":     round(1 - mean_fake, 4),
        "frames_analysed": n_frames,
        "fake_frames":   fake_frames,
        "real_frames":   real_frames,
        "per_frame_probs": [round(p, 4) for p in all_fake_p],
    }


# ---------------------------------------------------------------------------
# The Celery task
# ---------------------------------------------------------------------------

class VideoAnalysisTask(Task):
    """Base class — ensures worker-local resources are cleaned up on shutdown."""
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = kwargs.get("job_id") or (args[0] if args else None)
        if job_id:
            set_error(job_id, str(exc))


@celery_app.task(
    bind=True,
    base=VideoAnalysisTask,
    name="worker.tasks.analyse_video",
    queue="video",
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=600,
    time_limit=660,
)
def analyse_video(
    self,
    job_id:             str,
    video_path:         str,
    model:              str    = "resnet18",
    sample_every_n_sec: float  = 5.0,
    max_frames:         int    = 120,
    fusion_strategy:    str    = "weighted",
    run_ensemble:       bool   = True,
    fake_class_idx:     int    = 0,
) -> Dict[str, Any]:
    """
    Celery task: full multimodal video analysis.

    Args:
        job_id:             DeepTrace job ID (for Redis progress/result store)
        video_path:         Absolute path to the video file on shared storage
        model:              Primary visual model architecture
        sample_every_n_sec: Frame sampling interval in seconds
        max_frames:         Hard cap on frames to analyse
        fusion_strategy:    Audio/visual fusion strategy
        run_ensemble:       Whether to run all models + ensemble scoring
        fake_class_idx:     Index of the fake class in model output

    Returns:
        Result dict (also stored in Redis via set_result)
    """
    reporter = ProgressReporter(job_id)
    t_start  = time.perf_counter()

    try:
        reporter.update("started", pct=0, message="Job started")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ── Frame extraction ──────────────────────────────────────────────
        frames = _extract_frames(
            video_path, sample_every_n_sec, max_frames, reporter
        )
        if not frames:
            raise RuntimeError("No frames could be extracted from video")

        # ── Visual inference ──────────────────────────────────────────────
        visual = _run_visual_inference(frames, model, reporter, fake_class_idx)

        # ── Ensemble: run all other loaded models ─────────────────────────
        member_probs: Dict[str, float] = {model: visual["fake_prob"]}
        ensemble_result: Dict = {}

        if run_ensemble:
            from src.deepfake_recognition.utils.model_factory import SUPPORTED_ARCHITECTURES
            other_archs = [a for a in SUPPORTED_ARCHITECTURES if a != model]

            for i, arch in enumerate(other_archs):
                try:
                    reporter.update(
                        "ensemble_inference", pct=60 + i * 5,
                        message=f"Ensemble: running {arch}…",
                    )
                    other_visual = _run_visual_inference(
                        frames, arch, reporter, fake_class_idx
                    )
                    member_probs[arch] = other_visual["fake_prob"]
                except Exception as e:
                    print(f"[task] ensemble member {arch} failed: {e}")

            try:
                scorer         = _get_ensemble()
                ensemble_result = scorer.score(member_probs)
                reporter.update(
                    "ensemble_inference", pct=72,
                    message=(
                        f"Ensemble: {ensemble_result['ensemble_verdict']} "
                        f"(P={ensemble_result['ensemble_fake_prob']:.3f})"
                    ),
                )
            except Exception as e:
                print(f"[task] ensemble scoring failed: {e}")

        # ── Audio analysis ────────────────────────────────────────────────
        reporter.update("audio_analysis", pct=75,
                        message="Running audio analysis…")
        audio_result_dict: Dict = {"has_audio": False, "error": "Not run"}
        audio_spoof_prob: Optional[float] = None

        try:
            pipeline     = _get_audio_pipeline()
            audio_result = pipeline.analyse_file(video_path)
            audio_result_dict = audio_result.to_dict()
            if audio_result.has_audio and not audio_result.error:
                audio_spoof_prob = audio_result.spoof_prob
            reporter.update(
                "audio_analysis", pct=85,
                message=(
                    f"Audio: {audio_result.prediction} "
                    f"({audio_result.segments_analysed} segments)"
                    if audio_result.has_audio
                    else "No audio stream"
                ),
            )
        except Exception as e:
            audio_result_dict = {"has_audio": False, "error": str(e)}
            print(f"[task] audio analysis failed: {e}")

        # ── Fusion ───────────────────────────────────────────────────────
        reporter.update("fusion", pct=90, message="Fusing visual + audio…")
        from src.deepfake_recognition.audio.audio_fusion import fuse_verdicts

        visual_fake_prob = (
            ensemble_result.get("ensemble_fake_prob", visual["fake_prob"])
            if ensemble_result else visual["fake_prob"]
        )
        fusion = fuse_verdicts(
            visual_fake_prob=visual_fake_prob,
            audio_spoof_prob=audio_spoof_prob,
            strategy=fusion_strategy,
        )

        # ── Build result ──────────────────────────────────────────────────
        elapsed = round((time.perf_counter() - t_start), 2)
        result  = {
            "prediction":  fusion["verdict"],
            "confidence":  fusion["confidence"],
            "mode":        "video_multimodal_async",
            "architecture": model,
            "visual_result": visual,
            "audio_result":  audio_result_dict,
            "ensemble":      ensemble_result,
            "member_probs":  {k: round(v, 4) for k, v in member_probs.items()},
            "fusion":        fusion,
            "job_id":        job_id,
            "elapsed_sec":   elapsed,
        }

        reporter.update("done", pct=100,
                        message=f"Complete in {elapsed:.1f}s — "
                                f"{fusion['verdict'].upper()}")
        set_result(job_id, result)

        # Clean up temp file
        try:
            os.unlink(video_path)
        except OSError:
            pass

        return result

    except SoftTimeLimitExceeded:
        msg = f"Job exceeded 10-minute time limit"
        reporter.update("error", pct=0, message=msg)
        set_error(job_id, msg)
        try:
            os.unlink(video_path)
        except OSError:
            pass
        raise

    except Exception as exc:
        tb  = traceback.format_exc()
        msg = f"{type(exc).__name__}: {exc}"
        reporter.update("error", pct=0, message=msg, data={"traceback": tb[:500]})
        set_error(job_id, msg)
        try:
            os.unlink(video_path)
        except OSError:
            pass
        # Retry up to max_retries for transient errors
        raise self.retry(exc=exc)
