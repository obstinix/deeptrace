"""
src/deepfake_recognition/utils/multi_face.py

Verdict aggregation and response shaping for multi-face DeepTrace results.

Supported verdict strategies:
  any_fake   — fake if ANY face is fake (strictest; use for high-stakes forensics)
  majority   — fake if MORE THAN HALF of analysed faces are fake
  weighted   — fake if size-weighted fake score > 0.5 (larger faces count more)
  confident  — fake if ANY face has fake confidence > `threshold` (default 0.7)
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
import math

VerdictStrategy = Literal["any_fake", "majority", "weighted", "confident"]


# ---------------------------------------------------------------------------
# Per-face result normalisation
# ---------------------------------------------------------------------------

def normalise_face_result(raw: dict, include_crop: bool, include_explainability: bool) -> dict:
    """
    Strip large base64 fields from a face result dict based on caller preferences.
    This keeps response payloads manageable for multi-face images.
    """
    out = {k: v for k, v in raw.items() if k not in ("crop_b64", "explainability")}

    if include_crop and "crop_b64" in raw:
        out["crop_b64"] = raw["crop_b64"]

    if include_explainability and "explainability" in raw:
        out["explainability"]        = raw["explainability"]
        out["explainability_method"] = raw.get("explainability_method")

    return out


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

def aggregate_verdict(
    face_results: List[dict],
    strategy: VerdictStrategy = "any_fake",
    confidence_threshold: float = 0.70,
) -> dict:
    """
    Combine per-face predictions into a single image-level verdict.

    Args:
        face_results:         List of per-face result dicts (from _run_inference_on_crop)
        strategy:             Aggregation strategy (see module docstring)
        confidence_threshold: Used only by the 'confident' strategy

    Returns:
        dict with keys: verdict, strategy, fake_count, real_count,
                        total_faces, fake_face_indices, aggregate_confidence,
                        dissenting_faces, unanimity
    """
    if not face_results:
        return {
            "verdict":             "unknown",
            "strategy":            strategy,
            "fake_count":          0,
            "real_count":          0,
            "total_faces":         0,
            "fake_face_indices":   [],
            "aggregate_confidence": 0.0,
            "dissenting_faces":    [],
            "unanimity":           True,
        }

    fake_indices = [r["face_idx"] for r in face_results if r["prediction"] == "fake"]
    real_indices = [r["face_idx"] for r in face_results if r["prediction"] == "real"]
    n_fake = len(fake_indices)
    n_real = len(real_indices)
    n_total = len(face_results)

    # ── Strategy implementations ────────────────────────────────────────────

    if strategy == "any_fake":
        verdict = "fake" if n_fake > 0 else "real"

    elif strategy == "majority":
        verdict = "fake" if n_fake > n_total / 2 else "real"

    elif strategy == "weighted":
        # Weight each face by the square root of its bbox area
        # (avoids tiny faces dominating; diminishing returns on large faces)
        weighted_fake_score = 0.0
        total_weight        = 0.0
        for r in face_results:
            bbox   = r.get("bbox", {})
            w      = bbox.get("width",  1)
            h      = bbox.get("height", 1)
            weight = math.sqrt(w * h)
            fake_p = r.get("probabilities", {}).get("fake", 0.0)
            weighted_fake_score += weight * fake_p
            total_weight        += weight
        normalised = weighted_fake_score / (total_weight + 1e-6)
        verdict    = "fake" if normalised > 0.5 else "real"

    elif strategy == "confident":
        # Fake if any face has fake probability above threshold
        verdict = "real"
        for r in face_results:
            fake_p = r.get("probabilities", {}).get("fake", 0.0)
            if fake_p >= confidence_threshold:
                verdict = "fake"
                break

    else:
        raise ValueError(f"Unknown verdict strategy: '{strategy}'. "
                         f"Valid: any_fake, majority, weighted, confident")

    # ── Derived fields ───────────────────────────────────────────────────────

    # Faces that dissent from the majority prediction
    majority_label = "fake" if n_fake >= n_real else "real"
    dissenting     = [r["face_idx"] for r in face_results
                      if r["prediction"] != majority_label]

    # Aggregate confidence: average of the winning-class confidence scores
    conf_scores = [r["confidence"] for r in face_results]
    agg_conf    = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0

    return {
        "verdict":              verdict,
        "strategy":             strategy,
        "fake_count":           n_fake,
        "real_count":           n_real,
        "total_faces":          n_total,
        "fake_face_indices":    fake_indices,
        "real_face_indices":    real_indices,
        "aggregate_confidence": agg_conf,
        "dissenting_faces":     dissenting,
        "unanimity":            len(set(r["prediction"] for r in face_results)) == 1,
    }


# ---------------------------------------------------------------------------
# Spatial metadata for the UI canvas
# ---------------------------------------------------------------------------

def build_spatial_index(
    face_results: List[dict],
    image_width: int,
    image_height: int,
) -> List[dict]:
    """
    Produce normalised bounding box coordinates (0–1) for each face,
    plus a size category tag used by the UI to decide label font size.

    Returns a list parallel to face_results with keys:
      face_idx, nx1, ny1, nx2, ny2, size_category ("large"|"medium"|"small")
    """
    spatial = []
    for r in face_results:
        bbox = r.get("bbox", {})
        x1   = bbox.get("x1", 0)
        y1   = bbox.get("y1", 0)
        x2   = bbox.get("x2", 0)
        y2   = bbox.get("y2", 0)
        w    = bbox.get("width",  1)
        h    = bbox.get("height", 1)

        # Normalised to [0, 1]
        nx1 = x1 / (image_width  + 1e-6)
        ny1 = y1 / (image_height + 1e-6)
        nx2 = x2 / (image_width  + 1e-6)
        ny2 = y2 / (image_height + 1e-6)

        # Size category by fraction of image area
        frac = (w * h) / (image_width * image_height + 1e-6)
        if frac >= 0.10:
            size_cat = "large"
        elif frac >= 0.03:
            size_cat = "medium"
        else:
            size_cat = "small"

        spatial.append({
            "face_idx":     r["face_idx"],
            "nx1": round(nx1, 4), "ny1": round(ny1, 4),
            "nx2": round(nx2, 4), "ny2": round(ny2, 4),
            "size_category": size_cat,
        })

    return spatial


# ---------------------------------------------------------------------------
# Full response builder
# ---------------------------------------------------------------------------

def build_group_response(
    face_results: List[dict],
    detections: list,               # List[FaceDetection] from face_pipeline
    annotated_b64: Optional[str],
    arch: str,
    image_width: int,
    image_height: int,
    strategy: VerdictStrategy = "any_fake",
    confidence_threshold: float = 0.70,
    include_crops: bool = True,
    include_explainability: bool = False,  # off by default — too large for groups
) -> dict:
    """
    Build the complete /api/predict/group response dict.
    """
    aggregate  = aggregate_verdict(face_results, strategy, confidence_threshold)
    spatial    = build_spatial_index(face_results, image_width, image_height)
    normalised = [
        normalise_face_result(r, include_crops, include_explainability)
        for r in face_results
    ]

    # Merge spatial metadata into face results
    spatial_by_idx = {s["face_idx"]: s for s in spatial}
    for r in normalised:
        s = spatial_by_idx.get(r["face_idx"], {})
        r["spatial"] = {k: v for k, v in s.items() if k != "face_idx"}

    return {
        # Top-level verdict — backward compatible with single-face predict
        "prediction":  aggregate["verdict"],
        "confidence":  aggregate["aggregate_confidence"],
        "architecture": arch,
        "mode":         "multi_face",

        # Aggregate detail
        "aggregate":   aggregate,

        # Per-face results
        "faces_detected":  len(detections),
        "faces_analysed":  len(face_results),
        "face_results":    normalised,

        # Annotated image (coloured boxes: red=fake, teal=real)
        "annotated_image": annotated_b64,

        # Image dimensions (used by canvas renderer)
        "image_width":  image_width,
        "image_height": image_height,
    }
