"""
api/auth/tiers.py

Tier definitions — controls per-key request quotas and file size limits.
Edit this file to change tier allowances; no code changes needed elsewhere.

Tier names: "free" | "pro" | "admin"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Tier:
    name:               str
    # Requests per window
    requests_per_minute: int
    requests_per_hour:   int
    requests_per_day:    int
    # File size limits
    max_image_mb:        int        # per-request image size limit
    max_video_mb:        int        # per-request video size limit
    # Feature access
    can_submit_video:    bool       # access to async video queue
    can_use_ensemble:    bool       # access to /api/predict/ensemble
    can_use_explain:     bool       # access to /api/explain (LIME/SHAP)
    can_manage_keys:     bool       # access to /api/keys management
    # Concurrency
    max_concurrent_jobs: int        # max queued+running Celery jobs per key


TIERS: Dict[str, Tier] = {
    "free": Tier(
        name                = "free",
        requests_per_minute = 10,
        requests_per_hour   = 100,
        requests_per_day    = 500,
        max_image_mb        = 10,
        max_video_mb        = 100,
        can_submit_video    = True,
        can_use_ensemble    = False,
        can_use_explain     = False,
        can_manage_keys     = False,
        max_concurrent_jobs = 1,
    ),
    "pro": Tier(
        name                = "pro",
        requests_per_minute = 60,
        requests_per_hour   = 1_000,
        requests_per_day    = 10_000,
        max_image_mb        = 50,
        max_video_mb        = 2_000,
        can_submit_video    = True,
        can_use_ensemble    = True,
        can_use_explain     = True,
        can_manage_keys     = False,
        max_concurrent_jobs = 5,
    ),
    "admin": Tier(
        name                = "admin",
        requests_per_minute = 600,
        requests_per_hour   = 100_000,
        requests_per_day    = 1_000_000,
        max_image_mb        = 500,
        max_video_mb        = 10_000,
        can_submit_video    = True,
        can_use_ensemble    = True,
        can_use_explain     = True,
        can_manage_keys     = True,
        max_concurrent_jobs = 50,
    ),
}

DEFAULT_TIER = "free"


def get_tier(name: str) -> Tier:
    return TIERS.get(name, TIERS[DEFAULT_TIER])
