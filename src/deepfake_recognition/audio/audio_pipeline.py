"""
src/deepfake_recognition/audio/audio_pipeline.py

Full audio deepfake detection pipeline for DeepTrace:
  1. Extract audio track from video (FFmpeg)
  2. Resample to 16 kHz mono (librosa)
  3. Segment into overlapping 4-second windows
  4. Run AASIST inference on each segment
  5. Aggregate segment scores into a clip-level verdict

Standalone usage:
    pipeline = AudioPipeline("checkpoints/aasist/best.pth")
    result   = pipeline.analyse_file("path/to/video.mp4")
    print(result)
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

from src.deepfake_recognition.audio.audio_model import AASIST


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE     = 16_000          # AASIST expects 16 kHz
SEGMENT_SECS    = 4.0             # segment length (seconds)
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECS)   # 64,000 samples
HOP_SECS        = 2.0             # overlap: 50% hop
HOP_SAMPLES     = int(SAMPLE_RATE * HOP_SECS)
MIN_AUDIO_SECS  = 1.0             # skip files shorter than this


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """Prediction for a single 4-second audio segment."""
    segment_idx:   int
    start_sec:     float
    end_sec:       float
    prediction:    str             # "spoof" | "bonafide"
    spoof_prob:    float
    bonafide_prob: float
    confidence:    float


@dataclass
class AudioResult:
    """Aggregate result for a full audio/video file."""
    prediction:        str         # "spoof" | "bonafide"
    confidence:        float
    spoof_prob:        float
    bonafide_prob:     float
    duration_sec:      float
    segments_analysed: int
    spoof_segments:    int
    bonafide_segments: int
    segment_results:   List[SegmentResult] = field(default_factory=list)
    inference_time_ms: float = 0.0
    has_audio:         bool  = True
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "prediction":        self.prediction,
            "confidence":        round(self.confidence, 4),
            "spoof_prob":        round(self.spoof_prob, 4),
            "bonafide_prob":     round(self.bonafide_prob, 4),
            "duration_sec":      round(self.duration_sec, 2),
            "segments_analysed": self.segments_analysed,
            "spoof_segments":    self.spoof_segments,
            "bonafide_segments": self.bonafide_segments,
            "has_audio":         self.has_audio,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "error":             self.error,
            "segment_results": [
                {
                    "segment_idx":   s.segment_idx,
                    "start_sec":     round(s.start_sec, 2),
                    "end_sec":       round(s.end_sec, 2),
                    "prediction":    s.prediction,
                    "spoof_prob":    round(s.spoof_prob, 4),
                    "bonafide_prob": round(s.bonafide_prob, 4),
                    "confidence":    round(s.confidence, 4),
                }
                for s in self.segment_results
            ],
        }


# ---------------------------------------------------------------------------
# AudioPipeline
# ---------------------------------------------------------------------------

class AudioPipeline:
    """
    Singleton audio deepfake detection pipeline.
    Instantiate once at server startup; reuse across requests.

    Args:
        checkpoint_path:  Path to AASIST-L best.pth
        device:           torch device (cpu recommended — fast enough)
        aggregate:        How to combine segment scores:
                          "majority" — spoof if majority of segments are spoof
                          "max_spoof" — spoof if any segment spoof_prob > threshold
                          "mean" — spoof if mean spoof_prob > 0.5
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/aasist/best.pth",
        device: Optional[torch.device] = None,
        aggregate: str = "mean",
    ):
        if not _LIBROSA_AVAILABLE:
            raise ImportError("librosa is required: pip install librosa")

        self.checkpoint_path = checkpoint_path
        self.device          = device or torch.device("cpu")
        self.aggregate       = aggregate
        self._model: Optional[AASIST] = None
        self._loaded         = False

    # ------------------------------------------------------------------ load

    def load(self) -> bool:
        """Load the AASIST model from checkpoint. Returns True on success."""
        ckpt = Path(self.checkpoint_path)
        if not ckpt.exists():
            print(f"[audio] checkpoint not found: {ckpt}")
            return False
        try:
            self._model = AASIST()
            state       = torch.load(ckpt, map_location=self.device)

            # The HuggingFace checkpoint may be wrapped in a dict
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # Strip "module." prefix if trained with DataParallel
            state = {k.replace("module.", ""): v for k, v in state.items()}

            self._model.load_state_dict(state, strict=False)
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            print(f"[audio] AASIST-L loaded from {ckpt}")
            return True
        except Exception as e:
            print(f"[audio] failed to load AASIST: {e}")
            self._loaded = False
            return False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # --------------------------------------------------------------- extract

    @staticmethod
    def extract_audio_from_video(
        video_path: str,
        output_path: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> bool:
        """
        Use FFmpeg to extract the audio track from a video file.
        Outputs a 16 kHz mono WAV file.
        Returns True if audio was successfully extracted.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i",          video_path,
            "-vn",                         # no video
            "-acodec",     "pcm_s16le",    # 16-bit PCM
            "-ar",         str(sample_rate),
            "-ac",         "1",            # mono
            "-loglevel",   "error",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            # Check if the error is "no audio stream" vs a real error
            if "Output file #0 does not contain any stream" in stderr \
               or "no streams" in stderr.lower():
                return False   # video has no audio — not an error
            print(f"[audio] FFmpeg error: {stderr[:300]}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    @staticmethod
    def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, float]:
        """
        Load an audio file, resample to target_sr Hz, convert to mono.
        Returns (waveform_float32, duration_seconds).
        """
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        duration = len(wav) / target_sr
        return wav.astype(np.float32), duration

    # ------------------------------------------------------------ segment

    @staticmethod
    def segment_waveform(
        wav: np.ndarray,
        segment_samples: int = SEGMENT_SAMPLES,
        hop_samples:     int = HOP_SAMPLES,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Slice a waveform into overlapping fixed-length segments.
        Short clips shorter than segment_samples are zero-padded to length.

        Returns list of (start_sample_index, segment_array).
        """
        n = len(wav)

        # Short clip — zero pad to full segment length
        if n < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[:n] = wav
            return [(0, padded)]

        segments = []
        start    = 0
        while start + segment_samples <= n:
            segments.append((start, wav[start : start + segment_samples].copy()))
            start += hop_samples

        # Include the tail if it's at least half a segment long
        tail_start = start
        tail_len   = n - tail_start
        if tail_len >= segment_samples // 2:
            tail = np.zeros(segment_samples, dtype=np.float32)
            tail[:tail_len] = wav[tail_start : tail_start + tail_len]
            segments.append((tail_start, tail))

        return segments

    # ---------------------------------------------------------------- infer

    def _infer_segments(
        self,
        segments: List[Tuple[int, np.ndarray]],
    ) -> List[SegmentResult]:
        """
        Run AASIST on a list of waveform segments.
        Batches all segments into a single forward pass.
        """
        assert self._model is not None

        batch = torch.tensor(
            np.stack([s[1] for s in segments]),  # (N, segment_samples)
            dtype=torch.float32,
        ).unsqueeze(1).to(self.device)             # (N, 1, segment_samples)

        with torch.no_grad():
            logits = self._model(batch)            # (N, 2)
            probs  = torch.softmax(logits, dim=1)  # (N, 2)

        results = []
        for i, (start_sample, _) in enumerate(segments):
            bonafide_p = probs[i, 0].item()
            spoof_p    = probs[i, 1].item()
            prediction = "spoof" if spoof_p > 0.5 else "bonafide"
            confidence = max(bonafide_p, spoof_p)

            results.append(SegmentResult(
                segment_idx   = i,
                start_sec     = start_sample / SAMPLE_RATE,
                end_sec       = (start_sample + SEGMENT_SAMPLES) / SAMPLE_RATE,
                prediction    = prediction,
                spoof_prob    = round(spoof_p, 4),
                bonafide_prob = round(bonafide_p, 4),
                confidence    = round(confidence, 4),
            ))
        return results

    # --------------------------------------------------------------- aggregate

    def _aggregate(self, segment_results: List[SegmentResult]) -> Tuple[str, float, float, float]:
        """
        Combine segment-level predictions into a clip-level verdict.
        Returns (prediction, confidence, mean_spoof_prob, mean_bonafide_prob).
        """
        if not segment_results:
            return "unknown", 0.0, 0.0, 0.0

        spoof_probs    = [s.spoof_prob    for s in segment_results]
        bonafide_probs = [s.bonafide_prob for s in segment_results]
        mean_spoof     = float(np.mean(spoof_probs))
        mean_bonafide  = float(np.mean(bonafide_probs))

        if self.aggregate == "mean":
            verdict    = "spoof" if mean_spoof > 0.5 else "bonafide"
            confidence = max(mean_spoof, mean_bonafide)

        elif self.aggregate == "majority":
            n_spoof   = sum(1 for s in segment_results if s.prediction == "spoof")
            n_total   = len(segment_results)
            verdict   = "spoof" if n_spoof > n_total / 2 else "bonafide"
            confidence = n_spoof / n_total if verdict == "spoof" \
                         else (n_total - n_spoof) / n_total

        elif self.aggregate == "max_spoof":
            max_spoof  = max(spoof_probs)
            threshold  = 0.70
            verdict    = "spoof" if max_spoof >= threshold else "bonafide"
            confidence = max_spoof if verdict == "spoof" else mean_bonafide

        else:
            raise ValueError(f"Unknown aggregate mode: '{self.aggregate}'")

        return verdict, round(confidence, 4), round(mean_spoof, 4), round(mean_bonafide, 4)

    # ------------------------------------------------------- public interface

    def analyse_file(self, file_path: str) -> AudioResult:
        """
        Full pipeline: extract audio → segment → infer → aggregate.
        Handles both video files (extracts audio) and audio-only files.
        """
        if not self._loaded or self._model is None:
            return AudioResult(
                prediction="unknown", confidence=0.0,
                spoof_prob=0.0, bonafide_prob=0.0,
                duration_sec=0.0, segments_analysed=0,
                spoof_segments=0, bonafide_segments=0,
                has_audio=False,
                error="AASIST model not loaded",
            )

        t0  = time.perf_counter()
        ext = Path(file_path).suffix.lower()
        audio_path = file_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # If it's a video, extract the audio track first
            is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
            if is_video:
                audio_path = os.path.join(tmpdir, "extracted.wav")
                ok = self.extract_audio_from_video(file_path, audio_path)
                if not ok:
                    ms = (time.perf_counter() - t0) * 1000
                    return AudioResult(
                        prediction="unknown", confidence=0.0,
                        spoof_prob=0.0, bonafide_prob=0.0,
                        duration_sec=0.0, segments_analysed=0,
                        spoof_segments=0, bonafide_segments=0,
                        has_audio=False,
                        inference_time_ms=round(ms, 2),
                        error="No audio stream found in video",
                    )

            # Load + resample
            try:
                wav, duration = self.load_audio(audio_path)
            except Exception as e:
                return AudioResult(
                    prediction="unknown", confidence=0.0,
                    spoof_prob=0.0, bonafide_prob=0.0,
                    duration_sec=0.0, segments_analysed=0,
                    spoof_segments=0, bonafide_segments=0,
                    has_audio=True,
                    error=f"Audio decode failed: {e}",
                )

            if duration < MIN_AUDIO_SECS:
                return AudioResult(
                    prediction="unknown", confidence=0.0,
                    spoof_prob=0.0, bonafide_prob=0.0,
                    duration_sec=duration, segments_analysed=0,
                    spoof_segments=0, bonafide_segments=0,
                    has_audio=True,
                    error=f"Audio too short: {duration:.2f}s (min {MIN_AUDIO_SECS}s)",
                )

            # Segment
            segments = self.segment_waveform(wav)

            # Infer
            segment_results = self._infer_segments(segments)

        # Aggregate
        verdict, conf, mean_spoof, mean_bonafide = self._aggregate(segment_results)
        ms = (time.perf_counter() - t0) * 1000

        return AudioResult(
            prediction        = verdict,
            confidence        = conf,
            spoof_prob        = mean_spoof,
            bonafide_prob     = mean_bonafide,
            duration_sec      = duration,
            segments_analysed = len(segment_results),
            spoof_segments    = sum(1 for s in segment_results if s.prediction == "spoof"),
            bonafide_segments = sum(1 for s in segment_results if s.prediction == "bonafide"),
            segment_results   = segment_results,
            inference_time_ms = round(ms, 2),
            has_audio         = True,
            error             = None,
        )

    def analyse_bytes(self, audio_bytes: bytes, filename: str = "upload.mp4") -> AudioResult:
        """Write bytes to a temp file then analyse. Used by the API endpoint."""
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix or ".mp4",
            delete=False,
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.analyse_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
