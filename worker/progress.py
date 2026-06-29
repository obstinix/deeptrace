"""
worker/progress.py

Lightweight progress reporting from within a Celery task.
Pushes structured events to Redis so the API poll endpoint
can stream them to the client.

Usage (from inside a Celery task):
    reporter = ProgressReporter(job_id)
    reporter.update("extracting_frames", pct=10, message="Extracting frames…")
    reporter.update("visual_inference",  pct=40, message="Running ResNet-18…")
    reporter.update("audio_analysis",    pct=75, message="Running AASIST…")
    reporter.update("done",              pct=100, message="Complete")
"""
from __future__ import annotations
import time
from worker.storage import push_progress, set_status


class ProgressReporter:
    def __init__(self, job_id: str):
        self.job_id  = job_id
        self._last_pct = -1

    def update(
        self,
        stage:   str,
        pct:     int,
        message: str = "",
        data:    dict | None = None,
    ) -> None:
        """
        Push a progress event.

        Args:
            stage:   Machine-readable stage name
                     (extracting_frames | visual_inference | audio_analysis |
                      ensemble | fusion | done | error)
            pct:     Completion percentage 0–100
            message: Human-readable description for the UI
            data:    Optional extra payload (frame counts, etc.)
        """
        # Clamp and deduplicate — don't push the same pct twice
        pct = max(0, min(100, pct))
        if pct == self._last_pct and stage not in ("done", "error"):
            return
        self._last_pct = pct

        event = {
            "stage":   stage,
            "pct":     pct,
            "message": message,
        }
        if data:
            event["data"] = data

        push_progress(self.job_id, event)

        # Mirror status to meta for fast poll checks
        if stage == "done":
            set_status(self.job_id, "done")
        elif stage == "error":
            set_status(self.job_id, "error")
        elif self._last_pct == 0:
            set_status(self.job_id, "running")
