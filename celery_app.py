"""
celery_app.py

DeepTrace Celery application instance.
Imported by both the FastAPI app (to submit tasks) and the worker
(to execute them). Must be importable from both contexts.
"""
import os
print(f"DEBUG: celery_app.py loading... os.environ CELERY_BROKER_URL={os.environ.get('CELERY_BROKER_URL')!r}", flush=True)

# Prevent Celery auto-discovery from overriding defaults with empty strings
for key in ["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]:
    if key in os.environ and not os.environ[key].strip():
        print(f"DEBUG: Deleting empty/whitespace env var {key}", flush=True)
        del os.environ[key]

from celery import Celery

BROKER_URL  = (os.environ.get("CELERY_BROKER_URL") or "").strip() or "redis://localhost:6379/0"
RESULT_URL  = (os.environ.get("CELERY_RESULT_BACKEND") or "").strip() or "redis://localhost:6379/1"

print(f"DEBUG: BROKER_URL={BROKER_URL!r}", flush=True)
print(f"DEBUG: RESULT_URL={RESULT_URL!r}", flush=True)

celery_app = Celery(
    "deeptrace",
    broker=BROKER_URL,
    backend=RESULT_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    # Force use of resolved fallback URLs, overriding empty environment variables parsed at boot
    broker_url               = BROKER_URL,
    result_backend           = RESULT_URL,

    # Serialisation
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],

    # Timeouts
    task_soft_time_limit     = 600,    # 10 min soft limit — task receives SoftTimeLimitExceeded
    task_time_limit          = 660,    # 11 min hard kill
    result_expires           = 3600,   # results live in Redis for 1 hour

    # Reliability
    task_acks_late           = True,   # ack only after completion (not on pickup)
    task_reject_on_worker_lost = True, # re-queue if worker dies mid-task
    worker_prefetch_multiplier = 1,    # one task at a time per worker slot

    # Routing: large video jobs go to the "video" queue
    task_routes = {
        "worker.tasks.analyse_video": {"queue": "video"},
    },

    # Result chord / group support
    result_chord_join_timeout = 30,
)
