"""
src/deepfake_recognition/utils/calibration.py

Temperature Scaling calibration for DeepTrace binary classifiers.
Guo et al., 2017 — "On Calibration of Modern Neural Networks"
https://arxiv.org/abs/1706.04599

Core idea:
    calibrated_prob = softmax(logits / T)

    T is a single scalar fit on the validation set by minimising
    negative log-likelihood (NLL) using L-BFGS.

    T > 1.0  →  softer distribution  →  reduces overconfidence
    T < 1.0  →  sharper distribution →  increases confidence (rarely needed)
    T = 1.0  →  no change (uncalibrated baseline)

Usage:
    # Fit
    calibrator = TemperatureScaler()
    calibrator.fit(logits_val, labels_val)   # numpy arrays
    calibrator.save("checkpoints/resnet18/temperature.json")

    # Apply at inference
    calibrator = TemperatureScaler.load("checkpoints/resnet18/temperature.json")
    calibrated_probs = calibrator.calibrate(logits)   # torch tensor in, tensor out
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs:  np.ndarray,   # (N,) predicted probability for the positive class
    labels: np.ndarray,   # (N,) binary ground truth {0, 1}
    n_bins: int = 15,
) -> Tuple[float, list]:
    """
    Compute the Expected Calibration Error and per-bin statistics.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) − conf(B_b)|

    where B_b is the set of samples whose predicted confidence falls in
    bin b, acc(B_b) is the fraction of correct predictions in that bin,
    and conf(B_b) is the mean confidence in that bin.

    Returns:
        (ece_score, bins)
        ece_score: float in [0, 1]. Lower is better. Perfect = 0.0.
        bins: list of dicts — one per non-empty bin, for reliability diagram.
    """
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples   = len(probs)
    ece         = 0.0
    bin_records = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue

        bin_probs  = probs[mask]
        bin_labels = labels[mask]
        bin_conf   = float(bin_probs.mean())
        bin_acc    = float(bin_labels.mean())
        bin_n      = int(mask.sum())

        ece += (bin_n / n_samples) * abs(bin_acc - bin_conf)

        bin_records.append({
            "bin_lo":       round(lo, 4),
            "bin_hi":       round(hi, 4),
            "n_samples":    bin_n,
            "mean_conf":    round(bin_conf, 4),
            "mean_acc":     round(bin_acc,  4),
            "gap":          round(abs(bin_acc - bin_conf), 4),
        })

    return round(float(ece), 6), bin_records


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """
    Single-parameter post-hoc calibrator using temperature scaling.

    Fit once on the validation set after training. Adds zero parameters
    to the model and preserves accuracy exactly.
    """

    DEFAULT_T = 1.0   # uncalibrated baseline

    def __init__(self, temperature: float = DEFAULT_T):
        self.temperature = float(temperature)
        self._fit_meta: dict = {}

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        logits: np.ndarray,   # (N, 2) raw logits from the model, on CPU
        labels: np.ndarray,   # (N,)   integer ground truth {0, 1}
        fake_class_idx: int = 0,
        verbose: bool = True,
    ) -> "TemperatureScaler":
        """
        Fit T on a held-out validation set by minimising NLL.

        Args:
            logits:         Raw pre-softmax logits. Shape (N, 2).
                            Must NOT be post-softmax probabilities.
            labels:         Ground-truth class indices. 0=fake, 1=real (or vice
                            versa — consistent with your ImageFolder ordering).
            fake_class_idx: Index of the fake class in logits (default 0).
            verbose:        Print fitting progress.

        Returns:
            self (for chaining)
        """
        assert logits.ndim == 2 and logits.shape[1] == 2, \
            f"Expected logits of shape (N, 2), got {logits.shape}"
        assert len(logits) == len(labels), \
            f"logits/labels length mismatch: {len(logits)} vs {len(labels)}"

        t0 = time.perf_counter()
        n  = len(logits)

        # Pre-calibration ECE
        pre_probs = self._apply_temperature(logits, self.DEFAULT_T)[:, fake_class_idx]
        pre_ece, pre_bins = expected_calibration_error(pre_probs, labels == fake_class_idx)

        if verbose:
            print(f"[calibration] fitting on {n} validation samples …")
            print(f"[calibration] pre-calibration  ECE = {pre_ece:.5f}")

        # Objective: NLL of the validation set under logits/T
        # Optimise log(T) for unconstrained search (T must be positive)
        def nll_grad(log_t: np.ndarray) -> Tuple[float, np.ndarray]:
            T_val   = float(np.exp(log_t[0]))
            t_torch = torch.tensor([T_val], requires_grad=True, dtype=torch.float64)
            lg      = torch.tensor(logits, dtype=torch.float64)
            lb      = torch.tensor(labels, dtype=torch.long)

            scaled  = lg / t_torch
            loss    = F.cross_entropy(scaled.float(), lb)
            loss.backward()

            grad_T   = t_torch.grad.item()
            # Chain rule: d/d(log T) = d/dT * T
            grad_logT = grad_T * T_val
            return float(loss.item()), np.array([grad_logT])

        result = minimize(
            nll_grad,
            x0=np.array([0.0]),   # log(T=1.0) = 0.0
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        self.temperature = float(np.exp(result.x[0]))

        # Post-calibration ECE
        post_probs = self._apply_temperature(logits, self.temperature)[:, fake_class_idx]
        post_ece, post_bins = expected_calibration_error(post_probs, labels == fake_class_idx)

        elapsed = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[calibration] temperature      T = {self.temperature:.4f}")
            print(f"[calibration] post-calibration ECE = {post_ece:.5f}  "
                  f"(diff = {post_ece - pre_ece:+.5f})")
            print(f"[calibration] fit time: {elapsed:.0f}ms")

        self._fit_meta = {
            "n_val_samples":   n,
            "temperature":     round(self.temperature, 6),
            "pre_ece":         pre_ece,
            "post_ece":        post_ece,
            "ece_improvement": round(pre_ece - post_ece, 6),
            "pre_bins":        pre_bins,
            "post_bins":       post_bins,
            "fit_time_ms":     round(elapsed, 1),
            "converged":       bool(result.success),
            "nll_final":       round(float(result.fun), 6),
        }

        return self

    # ------------------------------------------------------------- apply

    @staticmethod
    def _apply_temperature(
        logits: np.ndarray,
        T:      float,
    ) -> np.ndarray:
        """Apply temperature scaling and return softmax probabilities."""
        scaled = logits / T
        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp     = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def calibrate(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temperature scaling to a logits tensor.

        Args:
            logits: (B, 2) raw pre-softmax logits on any device.

        Returns:
            (B, 2) calibrated probability tensor on the same device.
        """
        return torch.softmax(logits / self.temperature, dim=1)

    # --------------------------------------------------- save / load

    def save(self, path: str) -> None:
        """Save temperature and fit metadata to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "fit_meta":    self._fit_meta,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[calibration] saved to {path}  (T={self.temperature:.4f})")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load a saved TemperatureScaler from JSON."""
        with open(path) as f:
            data = json.load(f)
        scaler = cls(temperature=data["temperature"])
        scaler._fit_meta = data.get("fit_meta", {})
        return scaler

    @property
    def is_fitted(self) -> bool:
        return bool(self._fit_meta)

    @property
    def ece_improvement(self) -> Optional[float]:
        return self._fit_meta.get("ece_improvement")
