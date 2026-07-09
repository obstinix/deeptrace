"""
api/auth/ratelimit.py

Redis-backed token-bucket rate limiter for DeepTrace API keys.

Uses a sliding-window counter per (key_id, window) — simpler than a
true token bucket but sufficient for API rate limiting at this scale.
Three windows are checked independently: per-minute, per-hour, per-day.

All rate limit keys are prefixed "dt:rl:" and expire automatically.

Auth cache: bcrypt is slow (~100ms). We cache validated key IDs in Redis
for 60 seconds so each key only pays the bcrypt cost once per minute.

Keys:
  dt:rl:{key_id}:min:{epoch_minute}    TTL: 120s
  dt:rl:{key_id}:hour:{epoch_hour}     TTL: 7200s
  dt:rl:{key_id}:day:{epoch_day}       TTL: 172800s
  dt:auth:cache:{key_prefix}:{suffix}  TTL: 60s   (bcrypt result cache)
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Optional, Tuple

import redis

def _sanitize_redis_url(val: str) -> str:
    raw_val = val
    val = (val or "").strip()
    if not val:
        return ""
    if "redis://" in val:
        val = val[val.find("redis://"):]
    elif "rediss://" in val:
        val = val[val.find("rediss://"):]
    val = val.split()[0]
    if "rediss://" not in val and ("--tls" in raw_val or "upstash.io" in val):
        val = val.replace("redis://", "rediss://")
    return val.strip()

REDIS_URL = _sanitize_redis_url(os.environ.get("CELERY_BROKER_URL")) or "redis://localhost:6379/0"

# Use DB 2 for auth/rate-limit state (0=broker, 1=celery results)
_AUTH_REDIS_URL = REDIS_URL.rsplit("/", 1)[0] + "/2"


def _r() -> redis.Redis:
    return redis.Redis.from_url(_AUTH_REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    def __init__(self, window: str, limit: int, reset_in: int):
        self.window   = window    # "minute" | "hour" | "day"
        self.limit    = limit
        self.reset_in = reset_in  # seconds until window resets
        super().__init__(f"Rate limit exceeded ({window}: {limit} req)")


_IN_MEMORY_LIMITS = {}  # key: count


def check_and_increment(
    key_id:              str,
    requests_per_minute: int,
    requests_per_hour:   int,
    requests_per_day:    int,
) -> dict:
    """
    Check all three rate limit windows and increment counters atomically.
    Falls back gracefully to an in-memory dictionary if Redis is down.

    Returns a dict of current window counts (for response headers).
    Raises RateLimitExceeded if any window is exceeded.
    """
    now         = time.time()
    epoch_min   = int(now // 60)
    epoch_hour  = int(now // 3600)
    epoch_day   = int(now // 86400)

    k_min  = f"dt:rl:{key_id}:min:{epoch_min}"
    k_hour = f"dt:rl:{key_id}:hour:{epoch_hour}"
    k_day  = f"dt:rl:{key_id}:day:{epoch_day}"

    r = None
    use_redis = False
    cur_min, cur_hour, cur_day = 0, 0, 0

    try:
        r    = _r()
        pipe = r.pipeline()
        pipe.get(k_min)
        pipe.get(k_hour)
        pipe.get(k_day)
        res = pipe.execute()
        cur_min  = int(res[0] or 0)
        cur_hour = int(res[1] or 0)
        cur_day  = int(res[2] or 0)
        use_redis = True
    except Exception:
        # Fallback to in-memory dict
        cur_min  = _IN_MEMORY_LIMITS.get(k_min, 0)
        cur_hour = _IN_MEMORY_LIMITS.get(k_hour, 0)
        cur_day  = _IN_MEMORY_LIMITS.get(k_day, 0)

    # Check before incrementing
    if cur_min >= requests_per_minute:
        reset_in = 60 - int(now % 60)
        raise RateLimitExceeded("minute", requests_per_minute, reset_in)
    if cur_hour >= requests_per_hour:
        reset_in = 3600 - int(now % 3600)
        raise RateLimitExceeded("hour", requests_per_hour, reset_in)
    if cur_day >= requests_per_day:
        reset_in = 86400 - int(now % 86400)
        raise RateLimitExceeded("day", requests_per_day, reset_in)

    # Increment atomically
    if use_redis and r:
        try:
            pipe = r.pipeline()
            pipe.incr(k_min);  pipe.expire(k_min,  120)
            pipe.incr(k_hour); pipe.expire(k_hour, 7_200)
            pipe.incr(k_day);  pipe.expire(k_day,  172_800)
            pipe.execute()
        except Exception:
            pass
    else:
        # Save in-memory
        _IN_MEMORY_LIMITS[k_min] = cur_min + 1
        _IN_MEMORY_LIMITS[k_hour] = cur_hour + 1
        _IN_MEMORY_LIMITS[k_day] = cur_day + 1
        
        # Simple cleanup of old keys to avoid memory leak
        cutoff = now - 172800
        for k in list(_IN_MEMORY_LIMITS.keys()):
            try:
                parts = k.split(":")
                if len(parts) >= 5:
                    window = parts[3]
                    epoch = int(parts[4])
                    if window == "min" and epoch < int(cutoff // 60):
                        _IN_MEMORY_LIMITS.pop(k, None)
                    elif window == "hour" and epoch < int(cutoff // 3600):
                        _IN_MEMORY_LIMITS.pop(k, None)
                    elif window == "day" and epoch < int(cutoff // 86400):
                        _IN_MEMORY_LIMITS.pop(k, None)
            except Exception:
                pass

    return {
        "minute":  {"used": cur_min  + 1, "limit": requests_per_minute,
                    "reset_in": 60 - int(now % 60)},
        "hour":    {"used": cur_hour + 1, "limit": requests_per_hour,
                    "reset_in": 3600 - int(now % 3600)},
        "day":     {"used": cur_day  + 1, "limit": requests_per_day,
                    "reset_in": 86400 - int(now % 86400)},
    }


# ---------------------------------------------------------------------------
# Auth cache
# ---------------------------------------------------------------------------

def _cache_key(raw_key: str) -> str:
    """Hash the raw key for cache storage — never store the key itself."""
    h = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
    return f"dt:auth:cache:{h}"


def get_cached_key_id(raw_key: str) -> Optional[str]:
    """Return cached key_id string if auth was recently validated."""
    try:
        return _r().get(_cache_key(raw_key))
    except Exception:
        return None


def cache_key_id(raw_key: str, key_id: str, ttl: int = 60) -> None:
    """Cache a validated key_id for `ttl` seconds."""
    try:
        _r().set(_cache_key(raw_key), key_id, ex=ttl)
    except Exception:
        pass   # cache miss is non-fatal


def invalidate_cached_key(raw_key: str) -> None:
    """Remove from cache (call on revocation)."""
    try:
        _r().delete(_cache_key(raw_key))
    except Exception:
        pass
