"""
src/deepfake_recognition/utils/explainability/router.py

Unified explainability router for DeepTrace.
All four methods are exposed through a single interface.

Method selection logic:
  - grad_cam:          Fast (~50ms). CNN-only (ResNet-18, EfficientNet-B0).
  - attention_rollout: Fast (~80ms). ViT-B/16 only.
  - lime:              Slow (3–8s). Model-agnostic.
  - shap:              Slow (8–20s). Model-agnostic.
  - auto (default):    grad_cam for CNNs, attention_rollout for ViT.
                       Equivalent to the pre-this-doc behaviour.
"""
from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.deepfake_recognition.utils.model_factory import (
    supports_gradcam,
    get_gradcam_target_layer,
)

ExplainMethod = str   # "auto" | "grad_cam" | "attention_rollout" | "lime" | "shap"

FAST_METHODS = {"grad_cam", "attention_rollout", "auto"}
SLOW_METHODS = {"lime", "shap"}
ALL_METHODS  = FAST_METHODS | SLOW_METHODS


# ---------------------------------------------------------------------------
# PIL → base64 helper
# ---------------------------------------------------------------------------

def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# numpy array model_fn builder for LIME
# (LIME passes uint8 numpy arrays; we need to normalise and batch them)
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_numpy_model_fn(
    model:  nn.Module,
    device: torch.device,
):
    """
    Return a function that LIME can call:
        fn(images: np.ndarray[N, H, W, C] uint8) → np.ndarray[N, 2] float
    """
    def fn(images: np.ndarray) -> np.ndarray:
        model.eval()
        batch_list = []
        for img_arr in images:
            arr  = img_arr.astype(np.float32) / 255.0
            arr  = (arr - _MEAN) / _STD
            t    = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            batch_list.append(t)
        batch  = torch.stack(batch_list).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return fn


# ---------------------------------------------------------------------------
# Fast synchronous explainers
# ---------------------------------------------------------------------------

def _run_gradcam(
    model:        nn.Module,
    arch:         str,
    input_tensor: torch.Tensor,
    original:     Image.Image,
) -> Tuple[Image.Image, dict]:
    from src.deepfake_recognition.utils.gradcam import GradCAM
    target_layer = get_gradcam_target_layer(model, arch)
    cam          = GradCAM(model, target_layer=target_layer)

    # GradCAM returns a (H, W) numpy heatmap in [0,1]
    heatmap = cam.generate(input_tensor)

    # Render overlay using existing colourmap logic
    import matplotlib.cm as cm
    cmap    = cm.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize(original.size, Image.BILINEAR)
    overlay  = Image.blend(original.convert("RGB"), heat_img, alpha=0.5)

    meta = {
        "method": "grad_cam",
        "architecture": arch,
        "heatmap_min": round(float(heatmap.min()), 4),
        "heatmap_max": round(float(heatmap.max()), 4),
    }
    return overlay, meta


def _run_attention_rollout(
    model:        nn.Module,
    input_tensor: torch.Tensor,
    original:     Image.Image,
) -> Tuple[Image.Image, dict]:
    from src.deepfake_recognition.utils.attention_rollout import AttentionRollout
    rollout = AttentionRollout(model, head_fusion="mean", discard_ratio=0.9)
    mask    = rollout(input_tensor)
    overlay = AttentionRollout.overlay(original, mask, alpha=0.5, colormap="viridis")
    meta    = {
        "method": "attention_rollout",
        "mask_min": round(float(mask.min()), 4),
        "mask_max": round(float(mask.max()), 4),
    }
    return overlay, meta


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def get_explanation(
    model:        nn.Module,
    arch:         str,
    input_tensor: torch.Tensor,
    original:     Image.Image,
    method:       ExplainMethod = "auto",
    device:       Optional[torch.device] = None,
    # LIME params (used when method="lime")
    lime_num_samples:     int  = 1000,
    lime_num_superpixels: int  = 75,
    lime_top_k:           int  = 10,
    lime_positive_only:   bool = False,
    # SHAP params (used when method="shap")
    shap_n_background:    int  = 50,
    shap_n_evals:         int  = 50,
    shap_explainer_instance = None,   # pass cached ShapExplainer if available
) -> Tuple[Optional[str], dict]:
    """
    Run the requested explainability method and return a base64 PNG + metadata.

    Args:
        model:        DeepTrace nn.Module, on the correct device
        arch:         Architecture name ("resnet18", "efficientnet_b0", "vit_b16")
        input_tensor: (1, 3, 224, 224) normalised tensor on model's device
        original:     Original PIL Image (for overlay rendering)
        method:       Which method to use (see module docstring)
        device:       torch.device — used by SHAP/LIME wrappers

    Returns:
        (base64_png_or_none, metadata_dict)
    """
    dev = device or next(model.parameters()).device

    # ── Resolve "auto" ───────────────────────────────────────────────────────
    if method == "auto":
        method = "grad_cam" if supports_gradcam(arch) else "attention_rollout"

    # ── Validate method × architecture compatibility ─────────────────────────
    if method == "grad_cam" and not supports_gradcam(arch):
        raise ValueError(
            f"grad_cam is not supported for '{arch}'. "
            f"Use 'attention_rollout', 'lime', or 'shap' instead."
        )
    if method == "attention_rollout" and supports_gradcam(arch):
        raise ValueError(
            f"attention_rollout is only supported for ViT. "
            f"Use 'grad_cam', 'lime', or 'shap' for '{arch}'."
        )

    # ── Fast methods ─────────────────────────────────────────────────────────
    if method == "grad_cam":
        overlay, meta = _run_gradcam(model, arch, input_tensor, original)
        return _encode(overlay), meta

    if method == "attention_rollout":
        overlay, meta = _run_attention_rollout(model, input_tensor, original)
        return _encode(overlay), meta

    # ── LIME ─────────────────────────────────────────────────────────────────
    if method == "lime":
        from src.deepfake_recognition.utils.explainability.lime_explainer import LimeExplainer
        explainer = LimeExplainer(
            num_samples=lime_num_samples,
            num_superpixels=lime_num_superpixels,
            top_k_features=lime_top_k,
            positive_only=lime_positive_only,
        )
        model_fn = _build_numpy_model_fn(model, dev)
        overlay, meta = explainer.explain(original, model_fn, fake_class_idx=0)
        return _encode(overlay), meta

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if method == "shap":
        from src.deepfake_recognition.utils.explainability.shap_explainer import ShapExplainer
        if shap_explainer_instance is not None:
            explainer = shap_explainer_instance
        else:
            explainer = ShapExplainer(
                n_background=shap_n_background,
                n_evals=shap_n_evals,
                device=dev,
            )
        overlay, meta = explainer.explain(original, model, device=dev)
        return _encode(overlay), meta

    raise ValueError(f"Unknown explainability method: '{method}'. "
                     f"Valid: {sorted(ALL_METHODS)}")
