#!/usr/bin/env bash
set -e
python -c "import os; print('DEBUG env CELERY_BROKER_URL at start-deploy:', repr(os.environ.get('CELERY_BROKER_URL')), flush=True)"
celery -A celery_app worker --loglevel=info --concurrency=2 \
  --queues=video,default &
CELERY_PID=$!
trap 'kill -TERM $CELERY_PID 2>/dev/null; exit' TERM INT
uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 &
API_PID=$!
wait -n "$CELERY_PID" "$API_PID"
code=$?
kill -TERM "$CELERY_PID" "$API_PID" 2>/dev/null
exit $code
