import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from chess_video.api import create_app
from chess_video.cli import main, parser

TOKEN = "dotenv-integration-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def dotenv_configuration(tmp_path, monkeypatch):
    for key in ("CHESS_VIDEO_API_TOKEN", "API_TOKEN", "CHESS_VIDEO_JOB_ROOT", "JOB_ROOT"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    configured_root = tmp_path / "configured-jobs"
    contents = example.replace("CHESS_VIDEO_API_TOKEN=", f"CHESS_VIDEO_API_TOKEN={TOKEN}")
    contents = contents.replace("CHESS_VIDEO_JOB_ROOT=.jobs", f"CHESS_VIDEO_JOB_ROOT={configured_root}")
    (tmp_path / ".env").write_text(contents, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return configured_root


def assert_authenticated_job_persists(app, root):
    with TestClient(app) as client:
        assert client.get("/v1/jobs").status_code == 401
        assert client.get("/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/jobs", headers=HEADERS).json() == {"jobs": []}
        response = client.post("/v1/jobs", headers=HEADERS, json={"pgn": "1. e4 *"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert json.loads((root / job_id / "request.json").read_text(encoding="utf-8"))["pgn"] == "1. e4 *"
        assert json.loads((root / job_id / "job.json").read_text(encoding="utf-8"))["id"] == job_id
        assert client.get(f"/v1/jobs/{job_id}", headers=HEADERS).status_code == 200
    assert (root / ".owner.lock").is_file()


def test_serve_parser_defers_job_root_to_settings():
    assert parser().parse_args(["serve"]).jobs is None


def test_api_factory_loads_authentication_and_job_root_from_example_dotenv(dotenv_configuration):
    assert_authenticated_job_persists(create_app(pipeline=Mock(return_value={})), dotenv_configuration)
    assert not Path(".jobs").exists()


@pytest.mark.parametrize("override", [False, True])
def test_serve_uses_dotenv_configuration_with_optional_job_root_override(
    dotenv_configuration, monkeypatch, override,
):
    real_create_app = create_app
    pipeline = Mock(return_value={})
    expected_root = Path.cwd() / "explicit-jobs" if override else dotenv_configuration
    observed = []

    def factory(*args, **kwargs):
        return real_create_app(*args, **kwargs, pipeline=pipeline)

    def run(app, **kwargs):
        observed.append(kwargs)
        assert_authenticated_job_persists(app, expected_root)

    argv = ["chess-video", "serve"]
    if override:
        argv.extend(["--jobs", str(expected_root)])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr("chess_video.api.create_app", factory)
    monkeypatch.setattr("uvicorn.run", run)
    assert main() == 0
    assert observed == [{"host": "127.0.0.1", "port": 8787, "workers": 1}]
    assert not Path(".jobs").exists()
    if override:
        assert not dotenv_configuration.exists()
