"""
src/deepfake_recognition/utils/explainability/lime_explainer.py

LIME (Local Interpretable Model-agnostic Explanations) for DeepTrace.
Ribeiro et al., 2016 — https://arxiv.org/abs/1602.04938

How it works:
  1. Segment the image into superpixels (SLIC algorithm)
  2. Generate N random binary masks — each mask keeps or blacks out each superpixel
  3. Run the model on all N masked versions of the image
  4. Fit a weighted linear model: prediction ~ sum(w_i * mask_i)
  5. The weights w_i are the "importance" of each superpixel to the fake verdict

Output: a PIL Image overlay showing the most-important superpixels,
coloured green (supports fake) or red (contradicts fake).
"""
from __future__ import annotations

import io
import time
from typing import Callable, Tuple, Optional

import numpy as np
from PIL import Image

try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

try:
    from skimage.segmentation import mark_boundaries
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


class LimeExplainer:
    """
    LIME image explainer, parameterised for deepfake detection.

    Args:
        num_samples:        Number of perturbed images to generate.
                            More = more accurate but slower.
                            500 is fast (~2s); 1000 is standard (~4–6s).
        num_superpixels:    Number of SLIC superpixels to segment image into.
                            50–100 is a good range for 224×224 faces.
        top_k_features:     Number of highest-weight superpixels to highlight.
        hide_rest:          If True, grey-out non-highlighted superpixels.
        positive_only:      If True, show only features that push toward "fake".
                            If False, show both positive and negative.
        batch_size:         How many perturbed images to pass to the model at once.
                            Tune to fit available VRAM.
    """

    def __init__(
        self,
        num_samples:     int  = 1000,
        num_superpixels: int  = 75,
        top_k_features:  int  = 10,
        hide_rest:       bool = True,
        positive_only:   bool = False,
        batch_size:      int  = 32,
    ):
        if not _LIME_AVAILABLE:
            raise ImportError(
                "lime is required: pip install lime"
            )
        if not _SKIMAGE_AVAILABLE:
            raise ImportError(
                "scikit-image is required: pip install scikit-image"
            )

        self.num_samples     = num_samples
        self.num_superpixels = num_superpixels
        self.top_k_features  = top_k_features
        self.hide_rest       = hide_rest
        self.positive_only   = positive_only
        self.batch_size      = batch_size

        self._explainer = lime_image.LimeImageExplainer(verbose=False)

    def _build_predict_fn(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrap a model callable so LIME can call it on batches of uint8 numpy arrays.

        Args:
            model_fn: Callable that accepts a (B, H, W, C) uint8 numpy array
                      and returns (B, 2) float probability array [bonafide, fake]
                      OR [fake, bonafide] — see note on class ordering below.

        Note on class ordering:
            LIME expects predict_fn to return a probability array where the
            *target class* is at a known index. DeepTrace uses ImageFolder
            which sorts classes alphabetically: fake=0, real=1.
            The model_fn wrapper must return probs in that order.
        """
        batch_size = self.batch_size

        def predict_fn(images: np.ndarray) -> np.ndarray:
            # images: (N, H, W, C) uint8
            all_probs = []
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                probs = model_fn(batch)   # (batch, 2)
                all_probs.append(probs)
            return np.vstack(all_probs)

        return predict_fn

    def explain(
        self,
        image:      Image.Image,
        model_fn:   Callable[[np.ndarray], np.ndarray],
        fake_class_idx: int = 0,
    ) -> Tuple[Image.Image, dict]:
        """
        Run LIME on a single PIL image.

        Args:
            image:          PIL Image (will be resized to 224×224 if not already)
            model_fn:       See _build_predict_fn docstring
            fake_class_idx: Index of the "fake" class in the model output (default 0)

        Returns:
            (overlay_image, metadata)
            overlay_image: PIL Image with highlighted superpixels
            metadata: dict with timing, num_superpixels used, top_features weights
        """
        t0 = time.perf_counter()

        # Ensure 224×224 RGB numpy array
        img_rgb = np.array(image.convert("RGB").resize((224, 224)))

        predict_fn = self._build_predict_fn(model_fn)

        segmenter = SegmentationAlgorithm(
            "slic",
            n_segments=self.num_superpixels,
            compactness=10,
            sigma=1,
        )

        explanation = self._explainer.explain_instance(
            img_rgb,
            classifier_fn=predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=self.num_samples,
            segmentation_fn=segmenter,
            batch_size=self.batch_size,
        )

        # Get image + mask for the fake class
        temp_img, mask = explanation.get_image_and_mask(
            label=fake_class_idx,
            positive_only=self.positive_only,
            num_features=self.top_k_features,
            hide_rest=self.hide_rest,
        )

        # Draw superpixel boundaries
        if _SKIMAGE_AVAILABLE:
            bounded = mark_boundaries(
                temp_img.astype(np.uint8),
                explanation.segments,
                color=(0, 0.83, 1.0),   # DeepTrace cyan
                mode="outer",
            )
            overlay = Image.fromarray((bounded * 255).astype(np.uint8))
        else:
            overlay = Image.fromarray(temp_img.astype(np.uint8))

        # Resize overlay back to original image size if needed
        if image.size != (224, 224):
            overlay = overlay.resize(image.size, Image.BILINEAR)

        # Extract feature weights for the fake class
        local_exp      = explanation.local_exp.get(fake_class_idx, [])
        top_features   = [
            {"superpixel_id": int(sp_id), "weight": round(float(w), 5)}
            for sp_id, w in sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
            [:self.top_k_features]
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        metadata   = {
            "method":           "lime",
            "num_samples":      self.num_samples,
            "num_superpixels":  int(explanation.segments.max()) + 1,
            "top_k_features":   self.top_k_features,
            "positive_only":    self.positive_only,
            "top_features":     top_features,
            "inference_time_ms": round(elapsed_ms, 1),
        }

        return overlay, metadata
