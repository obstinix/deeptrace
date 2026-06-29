"""
tests/test_auth.py

Unit tests for DeepTrace API key management, database operations,
tier restrictions, and in-memory rate limit fallbacks.
"""
import pytest
import os
import sqlite3
import time
from pathlib import Path

# Override DB path for tests to avoid writing to production/development DB
os.environ["DEEPTRACE_DB_PATH"] = "tests/test_deeptrace.db"

import api.auth.keys as key_store
import api.auth.ratelimit as rl
from api.auth.tiers import get_tier

@pytest.fixture(autouse=True)
async def setup_test_db():
    """Initialises a clean temporary test database before every test."""
    db_path = Path("tests/test_deeptrace.db")
    db_path.unlink(missing_ok=True)
    await key_store.init_db()
    yield
    db_path.unlink(missing_ok=True)

@pytest.mark.anyio
async def test_key_lifecycle():
    """Tests creating, listing, validating, and revoking API keys."""
    # 1. Create Key
    key_info = await key_store.create_key(name="test-key", tier="free", notes="Unit test key")
    assert "raw_key" in key_info
    assert key_info["tier"] == "free"
    
    raw_key = key_info["raw_key"]
    key_id = key_info["id"]

    # 2. Validate Key
    record = await key_store.validate_raw_key(raw_key)
    assert record is not None
    assert record["id"] == key_id
    assert record["name"] == "test-key"
    assert record["tier"] == "free"
    
    # 3. List Keys
    keys = await key_store.list_keys()
    assert len(keys) >= 1
    assert any(k["id"] == key_id for k in keys)

    # 4. Revoke Key
    ok = await key_store.revoke_key(key_id)
    assert ok is True

    # 5. Confirm Revoked
    record_after = await key_store.validate_raw_key(raw_key)
    assert record_after is None

@pytest.mark.anyio
async def test_rate_limiting_fallback():
    """Tests that rate limiter falls back gracefully to in-memory limits when Redis is down."""
    # Reset in-memory counter state
    rl._IN_MEMORY_LIMITS.clear()
    
    key_id = "test-client-id"
    
    # 1. Run 3 requests under a limit of 5
    for i in range(3):
        state = rl.check_and_increment(
            key_id=key_id,
            requests_per_minute=5,
            requests_per_hour=100,
            requests_per_day=500,
        )
        assert state["minute"]["used"] == i + 1
        assert state["minute"]["limit"] == 5

    # 2. Trigger 2 more to hit limit
    rl.check_and_increment(key_id, 5, 100, 500)
    rl.check_and_increment(key_id, 5, 100, 500)

    # 3. Assert next one raises RateLimitExceeded
    with pytest.raises(rl.RateLimitExceeded) as exc_info:
        rl.check_and_increment(key_id, 5, 100, 500)
        
    assert exc_info.value.window == "minute"
    assert exc_info.value.limit == 5

@pytest.mark.anyio
async def test_tier_resolutions():
    """Tests tier retrieval and capability gating properties."""
    free_tier = get_tier("free")
    assert free_tier.name == "free"
    assert free_tier.requests_per_minute == 10
    assert free_tier.can_submit_video is True
    assert free_tier.can_use_ensemble is False

    pro_tier = get_tier("pro")
    assert pro_tier.name == "pro"
    assert pro_tier.requests_per_minute == 60
    assert pro_tier.can_use_ensemble is True

    admin_tier = get_tier("admin")
    assert admin_tier.name == "admin"
    assert admin_tier.can_manage_keys is True
