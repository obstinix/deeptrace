"""
src/deepfake_recognition/utils/explainability/shap_explainer.py

SHAP (SHapley Additive exPlanations) for DeepTrace.
Lundberg & Lee, 2017 — https://arxiv.org/abs/1705.07874

Uses GradientExplainer (fast, gradient-based SHAP for neural nets)
rather than KernelSHAP (which is identical to LIME in the limit).
GradientExplainer requires a background dataset of representative images
to estimate E[f(X)] — the baseline the model predicts in the absence of
each feature.

Output: a per-pixel SHAP value array (H, W) showing which pixels most
increased the model's fake probability. Visualised as a red/blue heatmap
overlaid on the original image.
"""
from __future__ import annotations

import io
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# Standard ImageNet normalisation — must match training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pil_to_normalised_tensor(
    img: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Convert PIL Image → normalised (1, 3, 224, 224) tensor."""
    arr  = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr  = (arr - _MEAN) / _STD
    return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32) \
               .unsqueeze(0).to(device)


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (1, 3, 224, 224) tensor → uint8 (224, 224, 3) array."""
    arr  = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    arr  = arr * _STD + _MEAN
    arr  = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


class ShapExplainer:
    """
    SHAP GradientExplainer for DeepTrace models.

    Args:
        n_background:   Number of background images to use for baseline.
                        More = more accurate; 50–100 is typical.
        n_evals:        Number of SHAP evaluations (forward + backward passes).
                        50 is a good default; higher is more accurate but slower.
        fake_class_idx: Index of the fake class in the model output (default 0).
        device:         torch device. Should match the model's device.
    """

    def __init__(
        self,
        n_background:   int           = 50,
        n_evals:        int           = 50,
        fake_class_idx: int           = 0,
        device:         Optional[torch.device] = None,
    ):
        if not _SHAP_AVAILABLE:
            raise ImportError("shap is required: pip install shap")

        self.n_background   = n_background
        self.n_evals        = n_evals
        self.fake_class_idx = fake_class_idx
        self.device         = device or torch.device("cpu")

        # Background dataset — populated lazily from the first few
        # images seen at runtime, or from a provided sample set
        self._background: Optional[torch.Tensor] = None

    def set_background(self, images: List[Image.Image]) -> None:
        """
        Provide a background dataset. Should be a representative sample
        of real images from the training distribution (not deepfakes).
        Call this once after server startup with a fixed sample set
        from data/frames/val/real/.
        """
        tensors = [
            _pil_to_normalised_tensor(img, self.device)
            for img in images[:self.n_background]
        ]
        self._background = torch.cat(tensors, dim=0)  # (N, 3, 224, 224)
        print(f"[shap] background set: {self._background.shape}")

    def _get_or_build_background(
        self,
        query_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the background tensor, building a synthetic one from
        Gaussian noise if set_background() was never called.
        Using noise is acceptable for visualisation — it shifts the
        baseline to E[model(noise)] which is roughly uniform, so SHAP
        values represent absolute feature importance rather than
        importance relative to a specific reference image.
        """
        if self._background is not None:
            return self._background

        # Synthetic background: Gaussian noise in normalised space
        bg = torch.randn(
            self.n_background,
            *query_tensor.shape[1:],
            device=self.device,
        ) * 0.1   # small std to stay near the normalised image manifold
        return bg

    def explain(
        self,
        image:  Image.Image,
        model:  nn.Module,
        device: Optional[torch.device] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Run SHAP GradientExplainer on a single PIL image.

        Args:
            image:  PIL Image (resized to 224×224 internally)
            model:  The DeepTrace nn.Module (must support gradients)
            device: Override device (uses self.device if None)

        Returns:
            (overlay_image, metadata)
            overlay_image: PIL Image with SHAP value heatmap overlaid
            metadata: dict with timing and per-channel SHAP statistics
        """
        t0  = time.perf_counter()
        dev = device or self.device

        model.eval()

        query_tensor = _pil_to_normalised_tensor(image, dev)   # (1, 3, 224, 224)
        background   = self._get_or_build_background(query_tensor)

        # GradientExplainer — uses expected gradients
        # (interpolation between query and background)
        explainer = shap.GradientExplainer(
            model,
            background,
        )

        # shap_values: list of length num_classes, each element (1, 3, 224, 224)
        shap_values = explainer.shap_values(
            query_tensor,
            nsamples=self.n_evals,
        )

        # Focus on the fake class
        # shap_values[fake_class_idx]: (1, 3, 224, 224) — per-pixel per-channel
        sv = shap_values[self.fake_class_idx]   # (1, 3, 224, 224) or (3, 224, 224)
        if sv.ndim == 4:
            sv = sv[0]                           # → (3, 224, 224)
        sv = sv.transpose(1, 2, 0)              # → (224, 224, 3)

        # Collapse channels: mean absolute SHAP across RGB
        sv_magnitude = np.abs(sv).mean(axis=-1)  # (224, 224)

        # Signed sum for direction: positive = pushes toward fake
        sv_signed = sv.sum(axis=-1)               # (224, 224)

        overlay = self._render_overlay(image, sv_magnitude, sv_signed)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        metadata = {
            "method":             "shap",
            "n_background":       int(background.shape[0]),
            "n_evals":            self.n_evals,
            "fake_class_idx":     self.fake_class_idx,
            "max_shap_magnitude": round(float(sv_magnitude.max()), 6),
            "mean_shap_magnitude": round(float(sv_magnitude.mean()), 6),
            "positive_pixels":    int((sv_signed > 0).sum()),
            "negative_pixels":    int((sv_signed < 0).sum()),
            "inference_time_ms":  round(elapsed_ms, 1),
        }

        return overlay, metadata

    @staticmethod
    def _render_overlay(
        original:     Image.Image,
        magnitude:    np.ndarray,   # (224, 224) absolute SHAP values
        signed:       np.ndarray,   # (224, 224) signed SHAP values
        alpha:        float = 0.55,
    ) -> Image.Image:
        """
        Render the SHAP heatmap as a red/blue overlay.
        Red = pixels that push the prediction toward fake.
        Blue = pixels that push the prediction toward real.
        """
        # Normalise magnitude to [0, 1]
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude = magnitude / mag_max

        # Build RGB heatmap
        heatmap = np.zeros((224, 224, 3), dtype=np.float32)
        pos_mask = signed > 0
        neg_mask = signed < 0

        # Red channel: positive SHAP (fake-pushing pixels)
        heatmap[:, :, 0] = np.where(pos_mask, magnitude, 0)
        # Blue channel: negative SHAP (real-pushing pixels)
        heatmap[:, :, 2] = np.where(neg_mask, magnitude, 0)

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_img   = Image.fromarray(heatmap_uint8).resize(
            original.size, Image.BILINEAR
        )

        # Blend with original
        orig_rgb = original.convert("RGB")
        blended  = Image.blend(orig_rgb, heatmap_img, alpha=alpha)
        return blended
