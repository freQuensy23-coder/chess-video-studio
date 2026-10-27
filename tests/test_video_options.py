from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from chess_video.api import create_app
from chess_video.cli import main
from chess_video.pipeline import generate_video
from chess_video.timeline import compile_timeline
from chess_video.video_options import validate_video_options

INVALID_OPTIONS = [
    (608, 720, 30),
    (238, 1080, 30),
    (608, 1081, 30),
    (607, 1080, 30),
    (1080, 608, 30),
    (608, 608, 30),
    (608, 3842, 30),
    (608, 1080, 0),
    (608, 1080, 121),
    (True, 1080, 30),
    (608, True, 30),
    (608, 1080, True),
    (608.0, 1080, 30),
    (608, 1080.0, 30),
    (608, 1080, 30.0),
    ("608", 1080, 30),
    (608, "1080", 30),
    (608, 1080, "30"),
    (240, 322, 30),
]


@pytest.mark.parametrize(("width", "height", "fps"), INVALID_OPTIONS)
def test_shared_video_options_reject_unsupported_frames(width, height, fps):
    with pytest.raises(ValueError, match="Output"):
        validate_video_options(width, height, fps)


@pytest.mark.parametrize(("width", "height", "fps"), [
    (240, 324, 1),
    (240, 324, 120),
    (608, 1080, 30),
    (2800, 3780, 30),
    (2800, 3840, 30),
])
def test_shared_video_options_accept_supported_boundaries(width, height, fps):
    assert validate_video_options(width, height, fps) is None


@pytest.mark.parametrize(("width", "height", "fps"), INVALID_OPTIONS)
def test_renderer_uses_shared_output_contract(width, height, fps):
    from chess_video.renderer import Renderer

    with pytest.raises(ValueError, match="Output"):
        Renderer({"width": width, "height": height, "fps": fps, "duration": 1.0, "events": []})


@pytest.mark.parametrize(("width", "height", "fps"), INVALID_OPTIONS)
def test_pipeline_rejects_options_before_files_or_paid_stages(tmp_path, width, height, fps):
    output = tmp_path / "not-created"
    with (
        patch("chess_video.pipeline.generate_script") as llm,
        patch("chess_video.pipeline.synthesize") as speech,
        patch("chess_video.pipeline.load_game") as parse,
        pytest.raises(ValueError, match="Output"),
    ):
        generate_video("1. e4 *", output, object(), width=width, height=height, fps=fps)
    assert not output.exists()
    parse.assert_not_called()
    llm.assert_not_called()
    speech.assert_not_called()


def test_invalid_options_preserve_existing_generation_artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Output"):
        generate_video("1. e4 *", tmp_path, object(), width=608, height=720)
    assert manifest.read_text(encoding="utf-8") == '{"previous": true}'
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize(("width", "height", "fps"), INVALID_OPTIONS)
def test_compiler_rejects_options_before_script_validation(width, height, fps):
    with (
        patch("chess_video.timeline.validate_script") as script,
        pytest.raises(ValueError, match="Output"),
    ):
        compile_timeline(None, None, None, width=width, height=height, fps=fps)
    script.assert_not_called()


@pytest.mark.parametrize(("width", "height", "fps"), INVALID_OPTIONS)
def test_api_rejects_unsupported_options_without_queueing(tmp_path, width, height, fps):
    pipeline = Mock(return_value={})
    headers = {"Authorization": "Bearer admission-token"}
    with TestClient(create_app(job_root=tmp_path, api_token="admission-token", pipeline=pipeline)) as client:
        response = client.post("/v1/jobs", headers=headers, json={
            "pgn": "1. e4 *", "width": width, "height": height, "fps": fps,
        })
        assert response.status_code == 422
        assert client.get("/v1/jobs", headers=headers).json() == {"jobs": []}
        assert not list(tmp_path.glob("*/request.json"))
    pipeline.assert_not_called()


@pytest.mark.parametrize(("width", "height", "fps"), [
    (320, 480, 12),
    (400, 540, 30),
    (2160, 3840, 60),
])
def test_api_admits_supported_shape_boundaries(tmp_path, width, height, fps):
    pipeline = Mock(return_value={})
    headers = {"Authorization": "Bearer admission-token"}
    with TestClient(create_app(job_root=tmp_path, api_token="admission-token", pipeline=pipeline)) as client:
        response = client.post("/v1/jobs", headers=headers, json={
            "pgn": "1. e4 *", "width": width, "height": height, "fps": fps,
        })
        assert response.status_code == 202
        assert len(client.get("/v1/jobs", headers=headers).json()["jobs"]) == 1


@pytest.mark.parametrize("command", ["generate", "showcase"])
def test_cli_rejects_unsupported_shape_before_generation(tmp_path, monkeypatch, capsys, command):
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. e4 *", encoding="utf-8")
    output = tmp_path / "not-created"
    argv = ["chess-video", command, "--output", str(output), "--width", "608", "--height", "720"]
    if command == "generate":
        argv.extend(["--pgn", str(pgn)])
    monkeypatch.setattr("sys.argv", argv)
    with (
        patch("chess_video.pipeline.generate_script") as llm,
        patch("chess_video.pipeline.synthesize") as speech,
        patch("chess_video.renderer.write_showcase") as showcase,
    ):
        assert main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Output" in captured.err
    assert not output.exists()
    llm.assert_not_called()
    speech.assert_not_called()
    showcase.assert_not_called()
