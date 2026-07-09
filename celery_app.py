"""
celery_app.py

DeepTrace Celery application instance.
Imported by both the FastAPI app (to submit tasks) and the worker
(to execute them). Must be importable from both contexts.
"""
import os
def sanitize_redis_url(val: str) -> str:
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
    if val.startswith("rediss://") and "ssl_cert_reqs=" not in val:
        separator = "&" if "?" in val else "?"
        val = f"{val}{separator}ssl_cert_reqs=none"
    return val.strip()

# Clean up raw dashboard inputs (e.g. if user pasted a full command like redis-cli -u redis://...)
for key in ["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]:
    if key in os.environ:
        clean_val = sanitize_redis_url(os.environ[key])
        if clean_val:
            os.environ[key] = clean_val
        else:
            del os.environ[key]

# Prevent Celery auto-discovery from overriding defaults with empty strings
for key in ["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]:
    if key in os.environ and not os.environ[key].strip():
        del os.environ[key]

from celery import Celery

BROKER_URL  = (os.environ.get("CELERY_BROKER_URL") or "").strip() or "redis://localhost:6379/0"
RESULT_URL  = (os.environ.get("CELERY_RESULT_BACKEND") or "").strip() or "redis://localhost:6379/1"

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
