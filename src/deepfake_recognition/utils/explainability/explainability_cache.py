"""
src/deepfake_recognition/utils/explainability/explainability_cache.py

In-process async job store for LIME / SHAP explain jobs.
Jobs are submitted, run in a ThreadPoolExecutor, and polled via job ID.

Usage:
    cache = ExplainCache(max_workers=2, ttl_seconds=300)
    job_id = cache.submit(fn, args)
    result = cache.get(job_id)   # {"status": "pending"|"done"|"error", ...}
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, Optional


class ExplainCache:
    """
    Thread-safe in-process job store for async explainability jobs.

    Args:
        max_workers: Max concurrent LIME/SHAP jobs.
                     Keep at 1–2 — these are CPU-bound and memory-heavy.
        ttl_seconds: How long to keep completed results in memory.
                     After TTL, results are evicted on the next get() call.
    """

    def __init__(self, max_workers: int = 2, ttl_seconds: int = 300):
        self._executor   = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, dict] = {}
        self._ttl        = ttl_seconds

    def submit(
        self,
        fn:    Callable,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Submit a job. Returns a job_id string immediately.
        fn is called in a background thread with *args, **kwargs.
        """
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status":     "pending",
            "submitted":  time.time(),
            "completed":  None,
            "result":     None,
            "error":      None,
        }

        def _run():
            try:
                result = fn(*args, **kwargs)
                self._jobs[job_id].update({
                    "status":    "done",
                    "completed": time.time(),
                    "result":    result,
                })
            except Exception as e:
                self._jobs[job_id].update({
                    "status":    "error",
                    "completed": time.time(),
                    "error":     str(e),
                })

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        """
        Get the status/result of a job. Returns None if job_id is unknown.
        Evicts expired completed jobs.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        # Evict expired completed jobs
        if job["status"] in ("done", "error"):
            if job["completed"] and (time.time() - job["completed"]) > self._ttl:
                del self._jobs[job_id]
                return None

        return {
            "job_id":    job_id,
            "status":    job["status"],
            "submitted": job["submitted"],
            "completed": job.get("completed"),
            "result":    job.get("result"),
            "error":     job.get("error"),
        }

    def pending_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j["status"] == "pending")

    def shutdown(self):
        self._executor.shutdown(wait=False)
