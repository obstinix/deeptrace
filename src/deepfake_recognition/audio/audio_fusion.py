"""
src/deepfake_recognition/audio/audio_fusion.py

Fuse visual (frame-level) and audio (clip-level) deepfake predictions
into a single multimodal verdict.

Fusion strategies:
  visual_only   — ignore audio (fallback when no audio stream)
  audio_only    — ignore visual (fallback when no visual model loaded)
  weighted      — weighted average of spoof/fake probabilities
  conflict_flag — flag explicitly when visual and audio disagree
"""
from __future__ import annotations
from typing import Literal, Optional


FusionStrategy = Literal["visual_only", "audio_only", "weighted", "conflict_flag"]

# Default weights: visual carries more signal for face-swap deepfakes;
# audio carries equal signal for voice-synthesis deepfakes.
# In practice: visual=0.6, audio=0.4 works well for Celeb-DF + ASVspoof.
DEFAULT_VISUAL_WEIGHT = 0.6
DEFAULT_AUDIO_WEIGHT  = 0.4


def fuse_verdicts(
    visual_fake_prob:    Optional[float],    # P(fake)  from visual model, 0–1
    audio_spoof_prob:    Optional[float],    # P(spoof) from audio model,  0–1
    strategy:            FusionStrategy = "weighted",
    visual_weight:       float = DEFAULT_VISUAL_WEIGHT,
    audio_weight:        float = DEFAULT_AUDIO_WEIGHT,
    conflict_threshold:  float = 0.3,        # used by conflict_flag strategy
) -> dict:
    """
    Fuse visual and audio deepfake probabilities into a multimodal verdict.

    Args:
        visual_fake_prob:   Visual model P(fake). None if not available.
        audio_spoof_prob:   Audio model P(spoof). None if not available.
        strategy:           Fusion strategy.
        visual_weight:      Weight for visual signal (weighted strategy).
        audio_weight:       Weight for audio signal (weighted strategy).
        conflict_threshold: Min difference between probs to flag as conflict.

    Returns:
        dict with keys: verdict, confidence, strategy,
                        visual_fake_prob, audio_spoof_prob,
                        conflict (bool), conflict_description (str|None)
    """
    # ── Fallback to available modality ─────────────────────────────────────
    if visual_fake_prob is None and audio_spoof_prob is None:
        return {
            "verdict":             "unknown",
            "confidence":          0.0,
            "strategy":            strategy,
            "visual_fake_prob":    None,
            "audio_spoof_prob":    None,
            "conflict":            False,
            "conflict_description": None,
        }

    if visual_fake_prob is None:
        strategy = "audio_only"
    elif audio_spoof_prob is None:
        strategy = "visual_only"

    # ── Fusion strategies ───────────────────────────────────────────────────
    conflict            = False
    conflict_description = None

    if strategy == "visual_only":
        p       = visual_fake_prob
        verdict = "fake" if p > 0.5 else "real"
        conf    = round(max(p, 1 - p), 4)

    elif strategy == "audio_only":
        p       = audio_spoof_prob
        verdict = "spoof" if p > 0.5 else "bonafide"
        conf    = round(max(p, 1 - p), 4)

    elif strategy == "weighted":
        # Normalise weights
        total   = visual_weight + audio_weight
        vw      = visual_weight / total
        aw      = audio_weight  / total
        fused_p = vw * visual_fake_prob + aw * audio_spoof_prob
        verdict = "fake" if fused_p > 0.5 else "real"
        conf    = round(max(fused_p, 1 - fused_p), 4)

        # Flag conflict if modalities strongly disagree
        delta = abs(visual_fake_prob - audio_spoof_prob)
        if delta >= conflict_threshold:
            conflict = True
            v_label  = "fake" if visual_fake_prob > 0.5 else "real"
            a_label  = "spoof" if audio_spoof_prob > 0.5 else "bonafide"
            # Note: visual models predict "fake"/"real", audio models predict "spoof"/"bonafide"
            v_is_fake = (v_label == "fake")
            a_is_spoof = (a_label == "spoof")
            if v_is_fake != a_is_spoof:
                conflict_description = (
                    f"Visual says {v_label} ({visual_fake_prob:.0%}), "
                    f"audio says {a_label} ({audio_spoof_prob:.0%}). "
                    f"Manual review recommended."
                )

    elif strategy == "conflict_flag":
        # Hard conflict: if both modalities disagree, verdict = "uncertain"
        v_fake     = visual_fake_prob > 0.5
        a_spoof    = audio_spoof_prob > 0.5
        if v_fake == a_spoof:
            verdict = "fake" if v_fake else "real"
            conf    = round((visual_fake_prob + audio_spoof_prob) / 2, 4)
        else:
            verdict = "uncertain"
            conf    = 0.0
            conflict = True
            conflict_description = (
                f"Visual: {'fake' if v_fake else 'real'} "
                f"({visual_fake_prob:.0%}) vs "
                f"Audio: {'spoof' if a_spoof else 'bonafide'} "
                f"({audio_spoof_prob:.0%}). "
                "Modalities disagree — further investigation required."
            )
    else:
        raise ValueError(f"Unknown fusion strategy: '{strategy}'")

    return {
        "verdict":              verdict,
        "confidence":           conf,
        "strategy":             strategy,
        "visual_fake_prob":     round(visual_fake_prob,  4) if visual_fake_prob  is not None else None,
        "audio_spoof_prob":     round(audio_spoof_prob,  4) if audio_spoof_prob  is not None else None,
        "conflict":             conflict,
        "conflict_description": conflict_description,
    }
