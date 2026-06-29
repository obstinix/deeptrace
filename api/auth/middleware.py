"""
api/auth/middleware.py

FastAPI dependency that validates X-API-Key and enforces rate limits.

Usage in endpoint:
    @app.post("/api/predict/image")
    async def predict_image(
        ...,
        auth: AuthContext = Depends(require_auth),
    ):
        # auth.key_id, auth.tier, auth.rate_limit_state available here

Public endpoints (no auth required) are listed in PUBLIC_PATHS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from api.auth import keys as key_store
from api.auth import ratelimit as rl
from api.auth.tiers import Tier, get_tier

# Header name — industry standard
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Endpoints that do not require auth
PUBLIC_PATHS = {
    "/",
    "/api/health",
    "/api/config",
    "/docs",
    "/openapi.json",
    "/redoc",
}


@dataclass
class AuthContext:
    """Injected into authenticated endpoints."""
    key_id:           str
    name:             str
    tier:             Tier
    rate_limit_state: dict   # window counts — used to set response headers


async def require_auth(
    request: Request,
    raw_key: Optional[str] = Depends(API_KEY_HEADER),
) -> AuthContext:
    """
    FastAPI dependency. Validates the API key and checks rate limits.
    Raises HTTP 401 / 403 / 429 on failure.
    """
    # Skip auth for public paths
    if request.url.path in PUBLIC_PATHS:
        # Return a no-op context for public endpoints
        return AuthContext(
            key_id="anonymous",
            name="anonymous",
            tier=get_tier("free"),
            rate_limit_state={},
        )

    # Require key
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Add header: X-API-Key: <your_key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate — try cache first, fall back to DB + bcrypt
    key_id = rl.get_cached_key_id(raw_key)
    key_record = None

    if key_id:
        # Cache hit — still need tier info from DB (cheap SELECT by PK)
        key_record = await key_store.get_key_by_id(key_id)
        if not key_record or not key_record.get("is_active"):
            rl.invalidate_cached_key(raw_key)
            key_record = None
            key_id     = None

    if key_record is None:
        # Cache miss — full bcrypt validation
        key_record = await key_store.validate_raw_key(raw_key)
        if key_record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        key_id = key_record["id"]
        rl.cache_key_id(raw_key, key_id, ttl=60)

    tier = get_tier(key_record.get("tier", "free"))

    # Rate limit check
    try:
        rl_state = rl.check_and_increment(
            key_id              = key_id,
            requests_per_minute = tier.requests_per_minute,
            requests_per_hour   = tier.requests_per_hour,
            requests_per_day    = tier.requests_per_day,
        )
    except rl.RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded ({e.window}: {e.limit} req). "
                f"Resets in {e.reset_in}s."
            ),
            headers={
                "Retry-After":            str(e.reset_in),
                "X-RateLimit-Window":     e.window,
                "X-RateLimit-Limit":      str(e.limit),
                "X-RateLimit-Reset-In":   str(e.reset_in),
            },
        )

    # Background: update last_seen + usage (non-blocking, fire and forget)
    import asyncio
    endpoint  = request.url.path
    bytes_in  = int(request.headers.get("content-length", 0))
    asyncio.create_task(key_store.update_last_seen(key_id))
    asyncio.create_task(key_store.record_usage(key_id, endpoint, bytes_in))

    # In require_auth(), after computing rl_state:
    request.state.rl_state = rl_state

    return AuthContext(
        key_id           = key_id,
        name             = key_record.get("name", ""),
        tier             = tier,
        rate_limit_state = rl_state,
    )


async def require_admin(auth: AuthContext = Depends(require_auth)) -> AuthContext:
    """Dependency for admin-only endpoints."""
    if not auth.tier.can_manage_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin access required. Your tier: {auth.tier.name}",
        )
    return auth


def require_feature(feature: str):
    """
    Factory for feature-gated endpoints.

    Usage:
        @app.post("/api/predict/ensemble")
        async def predict_ensemble(
            ...,
            auth: AuthContext = Depends(require_feature("can_use_ensemble")),
        ):
    """
    async def _check(auth: AuthContext = Depends(require_auth)) -> AuthContext:
        if not getattr(auth.tier, feature, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Feature '{feature}' not available on tier '{auth.tier.name}'. "
                    f"Upgrade to 'pro' or 'admin'."
                ),
            )
        return auth
    return _check
