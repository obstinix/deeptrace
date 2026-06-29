"""
src/deepfake_recognition/utils/ensemble.py

Ensemble scoring for DeepTrace — combines per-architecture calibrated
probabilities into a single fused fake probability.

Two strategies:
  weighted_average — linear combination of per-model fake probabilities.
                     Weights are fixed (default) or loaded from weights.json.
  learned          — logistic regression meta-classifier trained on val-set
                     per-model probabilities as features.

Terminology:
  "member probability" — P(fake) from one member model, after calibration.
  "ensemble probability" — fused P(fake) across all loaded members.
  "weight"             — per-architecture contribution to the ensemble (sum to 1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler as SklearnScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

# These reflect the typical relative accuracy on Celeb-DF v2:
# ViT-B/16 ≈ 92%, EfficientNet-B0 ≈ 91%, ResNet-18 ≈ 90%
# Weights are proportional to expected AUC improvement over random.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "resnet18":        0.20,
    "efficientnet_b0": 0.30,
    "vit_b16":         0.50,
}

WEIGHTS_PATH = "checkpoints/ensemble/weights.json"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalise weights to sum to 1.0 over present keys."""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# EnsembleScorer
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """
    Fuses per-model calibrated fake probabilities into a single prediction.

    Args:
        strategy:      "weighted_average" | "learned"
        weights:       Dict mapping arch → weight (weighted_average only).
                       If None, loads from weights.json or uses DEFAULT_WEIGHTS.
        weights_path:  Path to weights.json (loaded at __init__ if strategy != "learned")
    """

    STRATEGIES = {"weighted_average", "learned"}

    def __init__(
        self,
        strategy:     Literal["weighted_average", "learned"] = "weighted_average",
        weights:      Optional[Dict[str, float]] = None,
        weights_path: str = WEIGHTS_PATH,
    ):
        self.strategy     = strategy
        self.weights_path = weights_path

        # Meta-classifier state (learned strategy)
        self._meta_clf:    Optional[object]       = None   # fitted LogisticRegression
        self._meta_scaler: Optional[object]       = None   # fitted StandardScaler
        self._feature_order: Optional[List[str]]  = None   # arch order for feature vector

        # Weights state (weighted_average strategy)
        self._weights: Dict[str, float] = {}
        self._is_fitted = False

        if weights is not None:
            self._weights  = _normalise_weights(weights)
            self._is_fitted = True
        elif Path(weights_path).exists():
            self._load(weights_path)
        else:
            # Fall back to defaults — still usable without weights.json
            self._weights  = dict(DEFAULT_WEIGHTS)
            self._is_fitted = True

    # ------------------------------------------------------------------ load

    def _load(self, path: str) -> None:
        """Load fitted weights / meta-classifier from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.strategy = data.get("strategy", self.strategy)

        if self.strategy == "weighted_average":
            self._weights  = data.get("weights", DEFAULT_WEIGHTS)
            self._is_fitted = True

        elif self.strategy == "learned":
            if not _SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for the learned ensemble")

            self._feature_order = data.get("feature_order", [])
            coef   = np.array(data["coef"])
            intercept = float(data["intercept"])

            # Reconstruct classifier from saved parameters
            clf = LogisticRegression()
            clf.coef_      = coef
            clf.intercept_ = np.array([intercept])
            clf.classes_   = np.array([0, 1])
            self._meta_clf = clf

            # Reconstruct scaler
            if "scaler_mean" in data and "scaler_std" in data:
                sc         = SklearnScaler()
                sc.mean_   = np.array(data["scaler_mean"])
                sc.scale_  = np.array(data["scaler_std"])
                sc.var_    = sc.scale_ ** 2
                sc.n_features_in_ = len(sc.mean_)
                self._meta_scaler = sc

            self._is_fitted = True

    # ----------------------------------------------------------------- score

    def score(
        self,
        member_probs: Dict[str, float],
    ) -> dict:
        """
        Compute the ensemble fake probability from per-model calibrated probabilities.

        Args:
            member_probs: Dict mapping arch → P(fake), e.g.
                          {"resnet18": 0.82, "efficientnet_b0": 0.77, "vit_b16": 0.91}
                          Missing architectures are excluded from scoring.

        Returns:
            dict with keys:
              ensemble_fake_prob  float in [0, 1]
              ensemble_verdict    "fake" | "real"
              ensemble_confidence float
              strategy            str
              weights_used        dict (arch → effective weight)
              member_contributions dict (arch → weighted contribution)
              disagreement        float — std of member fake probs (0 = perfect agreement)
              n_members           int
        """
        if not member_probs:
            return self._empty_result()

        if self.strategy == "weighted_average":
            return self._score_weighted(member_probs)
        elif self.strategy == "learned":
            return self._score_learned(member_probs)
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'")

    def _score_weighted(self, member_probs: Dict[str, float]) -> dict:
        # Only use architectures present in both member_probs and weights
        present = {k: v for k, v in self._weights.items() if k in member_probs}
        if not present:
            # Fallback: equal weights over whatever is present
            present = {k: 1.0 for k in member_probs}

        effective = _normalise_weights(present)

        ensemble_p = sum(effective[arch] * member_probs[arch]
                         for arch in effective)

        contributions = {
            arch: round(effective[arch] * member_probs[arch], 4)
            for arch in effective
        }
        disagree = float(np.std(list(member_probs.values()))) if len(member_probs) > 1 else 0.0

        verdict    = "fake" if ensemble_p > 0.5 else "real"
        confidence = max(ensemble_p, 1 - ensemble_p)

        return {
            "ensemble_fake_prob":   round(float(ensemble_p), 4),
            "ensemble_verdict":     verdict,
            "ensemble_confidence":  round(float(confidence), 4),
            "strategy":             "weighted_average",
            "weights_used":         {k: round(v, 4) for k, v in effective.items()},
            "member_contributions": contributions,
            "disagreement":         round(disagree, 4),
            "n_members":            len(effective),
        }

    def _score_learned(self, member_probs: Dict[str, float]) -> dict:
        if self._meta_clf is None or self._feature_order is None:
            raise RuntimeError(
                "Learned ensemble not fitted. Run training/fit_ensemble.py first."
            )
        # Build feature vector in the stored order
        # Missing members → impute with 0.5 (maximum uncertainty)
        x = np.array([
            member_probs.get(arch, 0.5)
            for arch in self._feature_order
        ], dtype=np.float32).reshape(1, -1)

        if self._meta_scaler is not None:
            x = self._meta_scaler.transform(x)

        ensemble_p  = float(self._meta_clf.predict_proba(x)[0, 1])
        verdict     = "fake" if ensemble_p > 0.5 else "real"
        confidence  = max(ensemble_p, 1 - ensemble_p)
        disagree    = float(np.std(list(member_probs.values()))) if len(member_probs) > 1 else 0.0

        # For learned strategy, contributions are the coef × feature values
        coef = self._meta_clf.coef_[0]   # shape: (n_features,)
        raw_x = np.array([member_probs.get(a, 0.5) for a in self._feature_order])
        contribs = {
            arch: round(float(abs(coef[i]) * raw_x[i]), 4)
            for i, arch in enumerate(self._feature_order)
            if arch in member_probs
        }

        return {
            "ensemble_fake_prob":   round(ensemble_p, 4),
            "ensemble_verdict":     verdict,
            "ensemble_confidence":  round(float(confidence), 4),
            "strategy":             "learned",
            "weights_used":         {},   # not applicable for LR meta-clf
            "member_contributions": contribs,
            "disagreement":         round(disagree, 4),
            "n_members":            len([k for k in self._feature_order
                                         if k in member_probs]),
        }

    # ------------------------------------------------------------------ fit

    def fit_weighted(
        self,
        weights: Dict[str, float],
        save: bool = True,
    ) -> "EnsembleScorer":
        """
        Set fixed weights (weighted_average strategy).
        Weights are normalised to sum to 1.
        """
        self.strategy  = "weighted_average"
        self._weights  = _normalise_weights(weights)
        self._is_fitted = True
        if save:
            self.save()
        return self

    def fit_learned(
        self,
        member_probs_matrix: np.ndarray,   # (N, n_models)
        labels:              np.ndarray,   # (N,) int {0, 1}
        feature_order:       List[str],
        save: bool = True,
    ) -> "EnsembleScorer":
        """
        Fit a logistic regression meta-classifier.
        Called from training/fit_ensemble.py.

        Args:
            member_probs_matrix: (N, n_models) — per-model P(fake) on val set
            labels:              (N,) ground truth (fake=1, real=0)
            feature_order:       List of arch names (column order for matrix)
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        self.strategy        = "learned"
        self._feature_order  = list(feature_order)

        # Scale features
        scaler = SklearnScaler()
        X_scaled = scaler.fit_transform(member_probs_matrix)
        self._meta_scaler = scaler

        # Fit logistic regression with L2 regularisation
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_scaled, labels)
        self._meta_clf  = clf
        self._is_fitted = True

        if save:
            self.save()
        return self

    # ------------------------------------------------------------------ save

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.weights_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data: dict = {"strategy": self.strategy}

        if self.strategy == "weighted_average":
            data["weights"] = self._weights

        elif self.strategy == "learned":
            if self._meta_clf is None:
                raise RuntimeError("Cannot save — meta-classifier not fitted")
            data["feature_order"] = self._feature_order
            data["coef"]          = self._meta_clf.coef_.tolist()
            data["intercept"]     = float(self._meta_clf.intercept_[0])
            if self._meta_scaler is not None:
                data["scaler_mean"] = self._meta_scaler.mean_.tolist()
                data["scaler_std"]  = self._meta_scaler.scale_.tolist()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ensemble] saved → {path}  (strategy={self.strategy})")

    # ------------------------------------------------------------------ util

    @staticmethod
    def _empty_result() -> dict:
        return {
            "ensemble_fake_prob":   0.5,
            "ensemble_verdict":     "unknown",
            "ensemble_confidence":  0.0,
            "strategy":             "none",
            "weights_used":         {},
            "member_contributions": {},
            "disagreement":         0.0,
            "n_members":            0,
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def active_weights(self) -> Dict[str, float]:
        """Return the current effective weights (weighted_average only)."""
        return dict(self._weights)
