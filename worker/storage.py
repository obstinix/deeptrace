"""
worker/storage.py

Redis-backed job result store for DeepTrace video jobs.

Separate from Celery's result backend (which stores task state).
This store holds structured progress events and the final result dict,
accessible by job_id from both the API and the worker.

Key schema:
  deeptrace:job:{job_id}:meta     — JSON: job metadata (submitted, filename, model, etc.)
  deeptrace:job:{job_id}:progress — Redis list (LPUSH): JSON progress event objects
  deeptrace:job:{job_id}:result   — JSON: final result dict (set on completion)
  deeptrace:jobs                  — Redis sorted set: job_id → submitted_ts (for listing)

TTL: all keys expire after JOB_TTL_SECONDS (default 3600 = 1 hour).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

REDIS_URL       = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
JOB_TTL_SECONDS = int(os.environ.get("DEEPTRACE_JOB_TTL", "3600"))

_KEY_META     = "deeptrace:job:{job_id}:meta"
_KEY_PROGRESS = "deeptrace:job:{job_id}:progress"
_KEY_RESULT   = "deeptrace:job:{job_id}:result"
_KEY_JOBS     = "deeptrace:jobs"


def _r() -> redis.Redis:
    """Return a Redis connection. Called per-operation — no persistent connection held."""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def create_job(
    filename:  str,
    model:     str,
    file_size: int,
    options:   Optional[Dict[str, Any]] = None,
) -> str:
    """
    Register a new job in Redis. Returns the job_id.
    Called from the FastAPI endpoint before submitting to Celery.
    """
    job_id  = str(uuid.uuid4())
    now     = time.time()
    meta    = {
        "job_id":    job_id,
        "status":    "pending",
        "submitted": now,
        "filename":  filename,
        "file_size": file_size,
        "model":     model,
        "options":   options or {},
        "celery_id": None,
    }
    r = _r()
    pipe = r.pipeline()
    pipe.set(_KEY_META.format(job_id=job_id), json.dumps(meta), ex=JOB_TTL_SECONDS)
    pipe.zadd(_KEY_JOBS, {job_id: now})
    pipe.expire(_KEY_JOBS, JOB_TTL_SECONDS)
    pipe.execute()
    return job_id


def set_celery_id(job_id: str, celery_task_id: str) -> None:
    """Record the Celery task ID on the job meta — used for cancellation."""
    r   = _r()
    raw = r.get(_KEY_META.format(job_id=job_id))
    if raw:
        meta = json.loads(raw)
        meta["celery_id"] = celery_task_id
        r.set(_KEY_META.format(job_id=job_id), json.dumps(meta), ex=JOB_TTL_SECONDS)


def set_status(job_id: str, status: str) -> None:
    """Update the job status field in meta."""
    r   = _r()
    key = _KEY_META.format(job_id=job_id)
    raw = r.get(key)
    if raw:
        meta = json.loads(raw)
        meta["status"] = status
        if status in ("done", "error", "cancelled"):
            meta["completed"] = time.time()
        r.set(key, json.dumps(meta), ex=JOB_TTL_SECONDS)


def push_progress(job_id: str, event: Dict[str, Any]) -> None:
    """
    Append a progress event to the job's progress list.
    Events are stored newest-first (LPUSH) for efficient recent-N reads.
    """
    r   = _r()
    key = _KEY_PROGRESS.format(job_id=job_id)
    r.lpush(key, json.dumps({**event, "ts": time.time()}))
    r.expire(key, JOB_TTL_SECONDS)


def set_result(job_id: str, result: Dict[str, Any]) -> None:
    """Store the final result dict and mark the job as done."""
    r   = _r()
    r.set(
        _KEY_RESULT.format(job_id=job_id),
        json.dumps(result),
        ex=JOB_TTL_SECONDS,
    )
    set_status(job_id, "done")


def set_error(job_id: str, error: str) -> None:
    """Mark the job as failed with an error message."""
    r   = _r()
    r.set(
        _KEY_RESULT.format(job_id=job_id),
        json.dumps({"error": error}),
        ex=JOB_TTL_SECONDS,
    )
    set_status(job_id, "error")


# ---------------------------------------------------------------------------
# Job retrieval
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the full job state dict for polling.
    Returns None if the job is unknown or expired.
    """
    r    = _r()
    raw  = r.get(_KEY_META.format(job_id=job_id))
    if raw is None:
        return None

    meta     = json.loads(raw)
    progress = [
        json.loads(e)
        for e in (r.lrange(_KEY_PROGRESS.format(job_id=job_id), 0, 49) or [])
    ]   # last 50 events, newest first
    result_raw = r.get(_KEY_RESULT.format(job_id=job_id))
    result     = json.loads(result_raw) if result_raw else None

    return {
        "job_id":    job_id,
        "status":    meta.get("status"),
        "submitted": meta.get("submitted"),
        "completed": meta.get("completed"),
        "filename":  meta.get("filename"),
        "file_size": meta.get("file_size"),
        "model":     meta.get("model"),
        "celery_id": meta.get("celery_id"),
        "progress":  progress,
        "result":    result,
    }


def list_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recently submitted jobs (up to limit)."""
    r       = _r()
    job_ids = r.zrevrange(_KEY_JOBS, 0, limit - 1)
    jobs    = []
    for jid in job_ids:
        raw = r.get(_KEY_META.format(job_id=jid))
        if raw:
            meta = json.loads(raw)
            jobs.append({
                "job_id":    jid,
                "status":    meta.get("status"),
                "submitted": meta.get("submitted"),
                "filename":  meta.get("filename"),
                "model":     meta.get("model"),
            })
    return jobs


def get_celery_id(job_id: str) -> Optional[str]:
    r   = _r()
    raw = r.get(_KEY_META.format(job_id=job_id))
    if raw:
        return json.loads(raw).get("celery_id")
    return None
