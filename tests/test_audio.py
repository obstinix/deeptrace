"""
tests/test_audio.py

Unit tests for audio deepfake detection pipeline, model, and fusion logic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import pytest

from src.deepfake_recognition.audio.audio_model import AASIST
from src.deepfake_recognition.audio.audio_pipeline import AudioPipeline, SegmentResult
from src.deepfake_recognition.audio.audio_fusion import fuse_verdicts


def test_aasist_architecture():
    """Verify AASIST model initializes and forward passes matching shape."""
    model = AASIST()
    # Batch size 2, 1 channel, 16000 samples/sec * 4 sec = 64000 samples
    x = torch.randn(2, 1, 64000)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_waveform_segmentation():
    """Test slicing waveforms into overlapping fixed-length windows."""
    # 10 seconds of audio at 16000 Hz = 160000 samples
    wav = np.zeros(160000, dtype=np.float32)
    segments = AudioPipeline.segment_waveform(
        wav, segment_samples=64000, hop_samples=32000
    )
    # Expected:
    # Seg 0: 0 -> 64000
    # Seg 1: 32000 -> 96000
    # Seg 2: 64000 -> 128000
    # Seg 3: 96000 -> 160000
    # Seg 4: 128000 -> 192000 (zero-padded tail segment)
    # Total = 5 segments
    assert len(segments) == 5
    assert segments[0][0] == 0
    assert segments[1][0] == 32000
    assert segments[2][0] == 64000
    assert segments[3][0] == 96000
    assert segments[4][0] == 128000


def test_waveform_segmentation_short():
    """Verify short clips are zero-padded to full segment length."""
    # 2 seconds of audio = 32000 samples
    wav = np.ones(32000, dtype=np.float32)
    segments = AudioPipeline.segment_waveform(
        wav, segment_samples=64000, hop_samples=32000
    )
    assert len(segments) == 1
    assert segments[0][0] == 0
    assert len(segments[0][1]) == 64000
    assert np.all(segments[0][1][:32000] == 1.0)
    assert np.all(segments[0][1][32000:] == 0.0)


def test_audio_pipeline_aggregation():
    """Test clip-level verdict aggregation from segment scores."""
    pipeline = AudioPipeline(aggregate="mean")
    segs = [
        SegmentResult(0, 0.0, 4.0, "bonafide", 0.1, 0.9, 0.9),
        SegmentResult(1, 2.0, 6.0, "spoof",    0.8, 0.2, 0.8),
    ]
    # Mean spoof: (0.1 + 0.8) / 2 = 0.45 <= 0.5 -> bonafide
    verdict, conf, mean_s, mean_b = pipeline._aggregate(segs)
    assert verdict == "bonafide"
    assert conf == 0.55
    assert mean_s == 0.45

    pipeline.aggregate = "majority"
    # 1 spoof, 1 bonafide -> tie. majority definition: n_spoof > n_total / 2 -> bonafide
    verdict, conf, _, _ = pipeline._aggregate(segs)
    assert verdict == "bonafide"

    pipeline.aggregate = "max_spoof"
    # Max spoof = 0.8 >= 0.70 threshold -> spoof
    verdict, conf, _, _ = pipeline._aggregate(segs)
    assert verdict == "spoof"
    assert conf == 0.8


def test_verdict_fusion_weighted():
    """Test weighted visual and audio prediction fusion."""
    # Standard weighting: visual=0.6, audio=0.4
    # Visual P(fake) = 0.8, Audio P(spoof) = 0.2
    # Fused P(fake) = 0.6 * 0.8 + 0.4 * 0.2 = 0.48 + 0.08 = 0.56 > 0.5 -> fake
    res = fuse_verdicts(
        visual_fake_prob=0.8,
        audio_spoof_prob=0.2,
        strategy="weighted",
    )
    assert res["verdict"] == "fake"
    assert res["confidence"] == 0.56
    assert res["conflict"] is True
    assert "Visual says fake" in res["conflict_description"]


def test_verdict_fusion_conflict_flag():
    """Test conflict_flag fusion strategy."""
    # Visual = fake (0.9), Audio = bonafide (0.1) -> disagree -> uncertain
    res = fuse_verdicts(
        visual_fake_prob=0.9,
        audio_spoof_prob=0.1,
        strategy="conflict_flag",
    )
    assert res["verdict"] == "uncertain"
    assert res["confidence"] == 0.0
    assert res["conflict"] is True


def test_verdict_fusion_fallback():
    """Verify fallback behavior when one modality is missing."""
    # Only visual available
    res = fuse_verdicts(
        visual_fake_prob=0.7,
        audio_spoof_prob=None,
        strategy="weighted",
    )
    assert res["verdict"] == "fake"
    assert res["confidence"] == 0.7
    assert res["strategy"] == "visual_only"

    # Only audio available
    res2 = fuse_verdicts(
        visual_fake_prob=None,
        audio_spoof_prob=0.3,
        strategy="weighted",
    )
    assert res2["verdict"] == "bonafide"
    assert res2["confidence"] == 0.7
    assert res2["strategy"] == "audio_only"
