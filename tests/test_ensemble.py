"""
tests/test_ensemble.py

Unit tests for EnsembleScorer:
- weighted_average strategy scoring, agreement, and disagreement
- weight normalisation utility
- model contributions and disagreement metrics
- serialization (save and load weights)
"""
import tempfile
import os
import numpy as np
import pytest

from src.deepfake_recognition.utils.ensemble import EnsembleScorer, _normalise_weights


def test_weight_normalisation():
    """Verify weight normalisation sums to 1.0."""
    w = _normalise_weights({"a": 2.0, "b": 1.0, "c": 1.0})
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["a"] == 0.5
    assert w["b"] == 0.25
    assert w["c"] == 0.25

    # Edge case: sum to 0
    w2 = _normalise_weights({"a": 0.0, "b": 0.0})
    assert abs(sum(w2.values()) - 1.0) < 1e-6
    assert w2["a"] == 0.5


def test_weighted_average_agreement_fake():
    """Verify weighted_average strategy when members agree on fake."""
    scorer = EnsembleScorer(strategy="weighted_average", weights={
        "resnet18": 0.20, "efficientnet_b0": 0.30, "vit_b16": 0.50
    })
    r = scorer.score({"resnet18": 0.85, "efficientnet_b0": 0.88, "vit_b16": 0.91})
    assert r["ensemble_verdict"] == "fake"
    assert r["ensemble_fake_prob"] == round(0.20 * 0.85 + 0.30 * 0.88 + 0.50 * 0.91, 4)
    assert r["disagreement"] < 0.05
    assert r["n_members"] == 3


def test_weighted_average_agreement_real():
    """Verify weighted_average strategy when members agree on real."""
    scorer = EnsembleScorer(strategy="weighted_average", weights={
        "resnet18": 0.20, "efficientnet_b0": 0.30, "vit_b16": 0.50
    })
    r = scorer.score({"resnet18": 0.10, "efficientnet_b0": 0.08, "vit_b16": 0.12})
    assert r["ensemble_verdict"] == "real"
    assert r["ensemble_fake_prob"] == round(0.20 * 0.10 + 0.30 * 0.08 + 0.50 * 0.12, 4)
    assert r["n_members"] == 3


def test_weighted_average_disagreement():
    """Verify weighted_average strategy when members disagree."""
    scorer = EnsembleScorer(strategy="weighted_average", weights={
        "resnet18": 0.20, "efficientnet_b0": 0.30, "vit_b16": 0.50
    })
    r = scorer.score({"resnet18": 0.22, "efficientnet_b0": 0.18, "vit_b16": 0.81})
    # ViT has high weight, so it will shift score above 0.5 (fake)
    assert r["ensemble_verdict"] == "fake"
    assert r["disagreement"] > 0.20


def test_graceful_degradation():
    """Verify scoring degrades gracefully when some members are missing."""
    scorer = EnsembleScorer(strategy="weighted_average", weights={
        "resnet18": 0.20, "efficientnet_b0": 0.30, "vit_b16": 0.50
    })
    r = scorer.score({"vit_b16": 0.73})
    assert r["ensemble_verdict"] == "fake"
    assert r["n_members"] == 1
    assert r["ensemble_fake_prob"] == 0.73


def test_ensemble_save_load():
    """Verify ensemble weights can be saved and restored correctly."""
    scorer = EnsembleScorer(strategy="weighted_average",
                            weights={"resnet18": 0.2, "efficientnet_b0": 0.3, "vit_b16": 0.5})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        scorer.save(path)
        loaded = EnsembleScorer(weights_path=path)

        r1 = scorer.score({"resnet18": 0.7, "efficientnet_b0": 0.8, "vit_b16": 0.9})
        r2 = loaded.score({"resnet18": 0.7, "efficientnet_b0": 0.8, "vit_b16": 0.9})
        assert abs(r1["ensemble_fake_prob"] - r2["ensemble_fake_prob"]) < 1e-4
    finally:
        if os.path.exists(path):
            os.unlink(path)
