"""
api/auth/keys.py

API key store backed by SQLite (async via aiosqlite).

Keys are stored as bcrypt hashes — the raw key is shown once on creation
and never stored. Validation requires a full bcrypt compare (~100ms),
cached in Redis to avoid checking the DB + bcrypt on every request.

Schema:
  api_keys(
    id          TEXT PRIMARY KEY,     -- UUID
    name        TEXT NOT NULL,        -- human label ("production", "ci-bot")
    key_hash    TEXT NOT NULL,        -- bcrypt hash of the raw key
    key_prefix  TEXT NOT NULL,        -- first 8 chars of raw key (for display)
    tier        TEXT NOT NULL,        -- "free" | "pro" | "admin"
    created_at  REAL NOT NULL,        -- unix timestamp
    last_seen   REAL,                 -- unix timestamp of last successful auth
    is_active   INTEGER DEFAULT 1,    -- 0 = revoked
    created_by  TEXT,                 -- key_id of admin who created this key
    notes       TEXT                  -- free-text notes
  )

  key_usage(
    key_id      TEXT NOT NULL,
    date        TEXT NOT NULL,        -- YYYY-MM-DD UTC
    endpoint    TEXT NOT NULL,
    requests    INTEGER DEFAULT 0,
    bytes_in    INTEGER DEFAULT 0,
    PRIMARY KEY (key_id, date, endpoint)
  )
"""
from __future__ import annotations

import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import bcrypt

DB_PATH = os.environ.get("DEEPTRACE_DB_PATH", "data/deeptrace.db")

# Raw key format: dt_ prefix + 32 hex chars = 35 chars total
# Example: dt_a3f8c2e1b4d9f0e7a2c5b8d3e6f1a4b7
KEY_PREFIX_LEN = 8   # chars shown in key listings (after "dt_")
KEY_BYTES      = 16  # 16 random bytes = 32 hex chars


def _generate_raw_key() -> str:
    return "dt_" + secrets.token_hex(KEY_BYTES)


def _hash_key(raw_key: str) -> str:
    return bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt(rounds=12)).decode()


def _verify_key(raw_key: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw_key.encode(), hashed.encode())
    except Exception:
        return False


class AsyncDBContext:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = None

    def __await__(self):
        async def _async_init():
            return self
        return _async_init().__await__()

    async def __aenter__(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA journal_mode=WAL")
        await self.db.execute("PRAGMA foreign_keys=ON")
        return self.db

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            await self.db.close()


def _get_db() -> AsyncDBContext:
    return AsyncDBContext(DB_PATH)


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    async with await _get_db() as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                key_hash    TEXT NOT NULL,
                key_prefix  TEXT NOT NULL,
                tier        TEXT NOT NULL DEFAULT 'free',
                created_at  REAL NOT NULL,
                last_seen   REAL,
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_by  TEXT,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS key_usage (
                key_id      TEXT NOT NULL,
                date        TEXT NOT NULL,
                endpoint    TEXT NOT NULL,
                requests    INTEGER NOT NULL DEFAULT 0,
                bytes_in    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_id, date, endpoint)
            );

            CREATE INDEX IF NOT EXISTS idx_key_usage_key_id
                ON key_usage (key_id);
        """)
        await db.commit()


# ---------------------------------------------------------------------------
# Key CRUD
# ---------------------------------------------------------------------------

async def create_key(
    name:       str,
    tier:       str = "free",
    created_by: Optional[str] = None,
    notes:      str = "",
) -> Dict[str, Any]:
    """
    Create a new API key. Returns the record including the raw key (shown once).
    """
    raw_key    = _generate_raw_key()
    key_id     = str(uuid.uuid4())
    key_hash   = _hash_key(raw_key)
    key_prefix = raw_key[3 : 3 + KEY_PREFIX_LEN]   # after "dt_"
    now        = time.time()

    async with await _get_db() as db:
        await db.execute(
            """INSERT INTO api_keys
               (id, name, key_hash, key_prefix, tier, created_at, created_by, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key_id, name, key_hash, key_prefix, tier, now, created_by, notes),
        )
        await db.commit()

    return {
        "id":         key_id,
        "name":       name,
        "raw_key":    raw_key,        # ← shown ONCE; not stored
        "key_prefix": key_prefix,
        "tier":       tier,
        "created_at": now,
        "created_by": created_by,
        "notes":      notes,
        "warning":    "Save this key now — it will not be shown again.",
    }


async def get_key_by_id(key_id: str) -> Optional[Dict[str, Any]]:
    async with await _get_db() as db:
        row = await db.execute_fetchone(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        )
    return dict(row) if row else None


async def list_keys(include_inactive: bool = False) -> List[Dict[str, Any]]:
    async with await _get_db() as db:
        q   = "SELECT * FROM api_keys" + ("" if include_inactive else " WHERE is_active = 1")
        cur = await db.execute(q + " ORDER BY created_at DESC")
        rows = await cur.fetchall()
    return [
        {k: v for k, v in dict(row).items() if k != "key_hash"}
        for row in rows
    ]


async def revoke_key(key_id: str) -> bool:
    async with await _get_db() as db:
        cur = await db.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
        )
        await db.commit()
    return cur.rowcount > 0


async def update_last_seen(key_id: str) -> None:
    async with await _get_db() as db:
        await db.execute(
            "UPDATE api_keys SET last_seen = ? WHERE id = ?",
            (time.time(), key_id),
        )
        await db.commit()


async def record_usage(
    key_id:   str,
    endpoint: str,
    bytes_in: int = 0,
) -> None:
    """Upsert usage stats for today."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with await _get_db() as db:
        await db.execute(
            """INSERT INTO key_usage (key_id, date, endpoint, requests, bytes_in)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT (key_id, date, endpoint)
               DO UPDATE SET
                 requests = requests + 1,
                 bytes_in = bytes_in + excluded.bytes_in""",
            (key_id, today, endpoint, bytes_in),
        )
        await db.commit()


async def get_usage(key_id: str, days: int = 30) -> List[Dict[str, Any]]:
    async with await _get_db() as db:
        cur = await db.execute(
            """SELECT date, endpoint, requests, bytes_in
               FROM key_usage
               WHERE key_id = ?
               ORDER BY date DESC, requests DESC
               LIMIT ?""",
            (key_id, days * 50),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Validation (used by middleware)
# ---------------------------------------------------------------------------

async def validate_raw_key(raw_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate a raw API key string. Returns the key record (without hash)
    if valid and active, else None.

    bcrypt compare is ~100ms. Cache the result in Redis — see ratelimit.py.
    """
    async with await _get_db() as db:
        # Find candidate rows by prefix (avoids scanning all keys with bcrypt)
        prefix = raw_key[3 : 3 + KEY_PREFIX_LEN] if raw_key.startswith("dt_") else ""
        cur    = await db.execute(
            "SELECT * FROM api_keys WHERE key_prefix = ? AND is_active = 1",
            (prefix,),
        )
        rows = await cur.fetchall()

    for row in rows:
        d = dict(row)
        if _verify_key(raw_key, d["key_hash"]):
            return {k: v for k, v in d.items() if k != "key_hash"}

    return None
