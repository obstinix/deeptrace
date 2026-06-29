"""
tests/test_async_jobs.py

Unit tests for the Redis job storage backend used by the async Celery tasks.
Mocks redis connection to verify key schemas, serialization, and pipeline calls.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
import worker.storage as storage

@pytest.fixture
def mock_redis():
    with patch("worker.storage._r") as mock_conn:
        client = MagicMock()
        mock_conn.return_value = client
        yield client

def test_create_job(mock_redis):
    # Setup mocks
    pipe = MagicMock()
    mock_redis.pipeline.return_value = pipe

    # Call
    job_id = storage.create_job(
        filename="test_video.mp4",
        model="resnet18",
        file_size=1024 * 1024 * 5,
        options={"sample_every_n_sec": 5.0}
    )

    # Verify ID is a valid string UUID
    assert len(job_id) == 36

    # Verify Redis pipeline interactions
    mock_redis.pipeline.assert_called_once()
    
    # Check that set and zadd are called on pipeline
    pipe.set.assert_called_once()
    pipe.zadd.assert_called_once()
    pipe.execute.assert_called_once()

    # Verify content of metadata saved
    set_args, set_kwargs = pipe.set.call_args
    key = set_args[0]
    data = json.loads(set_args[1])
    
    assert key == f"deeptrace:job:{job_id}:meta"
    assert data["filename"] == "test_video.mp4"
    assert data["model"] == "resnet18"
    assert data["status"] == "pending"

def test_set_celery_id(mock_redis):
    job_id = "test-job-id"
    meta_json = json.dumps({
        "job_id": job_id,
        "status": "pending",
        "submitted": 12345.0,
        "filename": "test.mp4",
        "file_size": 100,
        "model": "resnet18",
        "options": {},
        "celery_id": None
    })
    mock_redis.get.return_value = meta_json

    storage.set_celery_id(job_id, "celery-uuid-123")

    mock_redis.get.assert_called_with(f"deeptrace:job:{job_id}:meta")
    
    # Check that metadata is updated with celery task id
    set_args, set_kwargs = mock_redis.set.call_args
    assert set_args[0] == f"deeptrace:job:{job_id}:meta"
    data = json.loads(set_args[1])
    assert data["celery_id"] == "celery-uuid-123"

def test_push_progress(mock_redis):
    job_id = "test-job-id"
    event = {"stage": "extracting_frames", "pct": 20, "message": "Extracting frames..."}

    storage.push_progress(job_id, event)

    # lpush pushes to list
    mock_redis.lpush.assert_called_once()
    lpush_args = mock_redis.lpush.call_args[0]
    assert lpush_args[0] == f"deeptrace:job:{job_id}:progress"
    
    event_data = json.loads(lpush_args[1])
    assert event_data["stage"] == "extracting_frames"
    assert event_data["pct"] == 20
    assert "ts" in event_data

def test_set_result(mock_redis):
    job_id = "test-job-id"
    result = {"verdict": "fake", "confidence": 0.95, "processing_ms": 1200}
    
    meta_json = json.dumps({
        "job_id": job_id,
        "status": "pending",
        "submitted": 12345.0,
        "filename": "test.mp4"
    })
    mock_redis.get.return_value = meta_json

    storage.set_result(job_id, result)

    # Set final results key
    mock_redis.set.assert_any_call(
        f"deeptrace:job:{job_id}:result",
        json.dumps(result),
        ex=3600
    )

    # Verify status is changed to "done"
    status_set_args = [call[0] for call in mock_redis.set.call_args_list if f"deeptrace:job:{job_id}:meta" in call[0][0]]
    assert len(status_set_args) == 1
    meta_data = json.loads(status_set_args[0][1])
    assert meta_data["status"] == "done"
    assert "completed" in meta_data
