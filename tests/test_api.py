import time
from pathlib import Path

from fastapi.testclient import TestClient

from chess_video.api import create_app


TOKEN = "unit-test-local-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def test_api_requires_token(tmp_path):
    with TestClient(create_app(job_root=tmp_path, api_token=TOKEN, pipeline=lambda **kwargs: {})) as client:
        assert client.get("/v1/jobs").status_code == 401
        assert client.get("/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/jobs", headers=HEADERS).json() == {"jobs": []}


def test_api_executes_job_and_serves_only_allowlisted_files(tmp_path):
    def pipeline(pgn, output_dir, settings, **kwargs):
        (output_dir / "video.mp4").write_bytes(b"video")
        (output_dir / "private-provider.json").write_text("private")
        return {"video_path": str(output_dir / "video.mp4"), "duration": 1.0}

    with TestClient(create_app(job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)) as client:
        response = client.post("/v1/jobs", headers=HEADERS, json={"pgn": "1. f3 e5 2. g4 Qh4# 0-1"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert "video_path" not in str(job)
        assert client.get(f"/v1/jobs/{job_id}/files/video.mp4", headers=HEADERS).content == b"video"
        assert client.get(f"/v1/jobs/{job_id}/files/private-provider.json", headers=HEADERS).status_code == 404
        assert client.get("/v1/jobs/not-a-uuid", headers=HEADERS).status_code == 404


def test_api_rejects_invalid_dimensions(tmp_path):
    with TestClient(create_app(job_root=tmp_path, api_token=TOKEN, pipeline=lambda **kwargs: {})) as client:
        response = client.post("/v1/jobs", headers=HEADERS, json={"pgn": "1. e4 *", "width": 607})
        assert response.status_code == 422


def test_restart_marks_interrupted_job_failed_without_rebilling(tmp_path):
    import json

    job_id = "b828dc6a-3328-423e-aa22-1fd703e28fda"
    root = tmp_path / job_id
    root.mkdir()
    (root / "job.json").write_text(json.dumps({"id": job_id, "status": "running"}))
    calls = []
    with TestClient(create_app(job_root=tmp_path, api_token=TOKEN, pipeline=lambda **kwargs: calls.append(1))) as client:
        result = client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
        assert result["status"] == "failed"
        assert "interrupted" in result["error"].lower()
    assert not calls


def test_api_never_exposes_provider_exception_text(tmp_path):
    def pipeline(**kwargs):
        raise RuntimeError("xi-api-key: SECRET_SHOULD_NOT_ESCAPE")

    with TestClient(create_app(job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)) as client:
        job_id = client.post("/v1/jobs", headers=HEADERS, json={"pgn": "1. e4 *"}).json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
            if result["status"] == "failed":
                break
            time.sleep(0.01)
        assert "SECRET_SHOULD_NOT_ESCAPE" not in str(result)
        assert result["status"] == "failed"
