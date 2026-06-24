"""Minimal SQLite-backed metrics store."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("METRICS_DB_PATH", "/tmp/deeptrace_metrics.db"))

def init_db():
    # Ensure directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL DEFAULT 0
            )
        """)
        for key in ("total_requests", "error_requests", "total_latency_ms"):
            conn.execute(
                "INSERT OR IGNORE INTO metrics (key, value) VALUES (?, 0)", (key,)
            )

def increment(key: str, amount: float = 1.0):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE metrics SET value = value + ? WHERE key = ?", (amount, key)
        )

def get_all() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT key, value FROM metrics").fetchall()
    return {row[0]: row[1] for row in rows}

# Initialize database on import
init_db()
