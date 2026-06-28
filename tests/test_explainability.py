"""
tests/test_explainability.py

Unit tests for LIME & SHAP explainability modules, router, and job cache.
"""
from __future__ import annotations

import time
import pytest
import numpy as np
import torch
from PIL import Image

from src.deepfake_recognition.utils.explainability.explainability_cache import ExplainCache
from src.deepfake_recognition.utils.explainability.lime_explainer import LimeExplainer
from src.deepfake_recognition.utils.explainability.shap_explainer import ShapExplainer
from src.deepfake_recognition.utils.explainability.router import get_explanation, supports_gradcam


def test_explain_cache_success():
    """Verify ExplainCache submits, executes in threadpool, and retrieves results."""
    cache = ExplainCache(max_workers=2, ttl_seconds=10)

    def job_fn(x, y):
        time.sleep(0.1)
        return x + y

    job_id = cache.submit(job_fn, 5, 7)
    status = cache.get(job_id)
    assert status["status"] in ("pending", "done")

    # Wait for completion
    time.sleep(0.3)
    status = cache.get(job_id)
    assert status["status"] == "done"
    assert status["result"] == 12
    assert status["error"] is None


def test_explain_cache_error():
    """Verify ExplainCache handles exceptions raised within jobs."""
    cache = ExplainCache(max_workers=1, ttl_seconds=5)

    def fail_fn():
        raise ValueError("Job crashed!")

    job_id = cache.submit(fail_fn)
    time.sleep(0.2)
    status = cache.get(job_id)
    assert status["status"] == "error"
    assert "Job crashed!" in status["error"]


def test_lime_explainer_prediction_wrapper():
    """Verify LimeExplainer builds batch wrapper and processes predictions."""
    explainer = LimeExplainer(num_samples=10, num_superpixels=5, batch_size=2)

    # Mock model function that returns constant probabilities
    def mock_model_fn(batch):
        # batch: (B, H, W, C)
        assert len(batch.shape) == 4
        return np.array([[0.9, 0.1]] * len(batch))

    wrapped = explainer._build_predict_fn(mock_model_fn)
    dummy_input = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    res = wrapped(dummy_input)
    assert res.shape == (5, 2)
    assert np.allclose(res[:, 0], 0.9, atol=0.01)
    assert np.allclose(res[:, 1], 0.1, atol=0.01)


def test_shap_explainer_synthetic_background():
    """Verify ShapExplainer generates Gaussian noise background when unpopulated."""
    explainer = ShapExplainer(n_background=20)
    query_tensor = torch.randn(1, 3, 224, 224)
    bg = explainer._get_or_build_background(query_tensor)
    assert bg.shape == (20, 3, 224, 224)


def test_shap_explainer_rendering():
    """Verify SHAP signed overlays blend correctly and output PIL images."""
    img = Image.new("RGB", (100, 100), color="white")
    magnitude = np.ones((224, 224), dtype=np.float32)
    signed = np.ones((224, 224), dtype=np.float32)

    overlay = ShapExplainer._render_overlay(img, magnitude, signed)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (100, 100)


def test_router_invalid_methods():
    """Verify explainability router raises errors on invalid method/arch pairs."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 2))
    t = torch.randn(1, 3, 224, 224)
    img = Image.new("RGB", (224, 224))

    # Grad-CAM on ViT should raise ValueError
    with pytest.raises(ValueError) as exc:
        get_explanation(model, "vit_b16", t, img, method="grad_cam")
    assert "grad_cam is not supported for 'vit_b16'" in str(exc.value)

    # Attention Rollout on ResNet-18 should raise ValueError
    with pytest.raises(ValueError) as exc2:
        get_explanation(model, "resnet18", t, img, method="attention_rollout")
    assert "attention_rollout is only supported for ViT" in str(exc2.value)
