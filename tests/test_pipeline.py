import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chess_video.pipeline import generate_video


GAME = {"initial_fen": "start", "positions": [{"ply": 0}], "headers": {}, "result": "*"}
SCRIPT = {"title": "Mate", "language": "en", "perspective": "white", "beats": []}
SPEECH = {"audio_path": "speech.mp3", "duration": 2.0}
TIMELINE = {"duration": 3.0, "events": []}


def test_generation_exports_all_intermediate_artifacts(tmp_path):
    def render(timeline, path):
        Path(path).write_bytes(b"generated-mp4")
        return Path(path)

    with (
        patch("chess_video.pipeline.load_game", return_value=GAME),
        patch("chess_video.pipeline.generate_script", return_value=SCRIPT),
        patch("chess_video.pipeline.validate_script", return_value=SCRIPT),
        patch("chess_video.pipeline.synthesize", return_value=SPEECH),
        patch("chess_video.pipeline.compile_timeline", return_value=TIMELINE),
        patch("chess_video.pipeline.render_video", side_effect=render),
    ):
        result = generate_video("PGN", tmp_path, object())

    assert Path(result["video_path"]).read_bytes() == b"generated-mp4"
    assert json.loads((tmp_path / "script.json").read_text()) == SCRIPT
    assert json.loads((tmp_path / "timeline.json").read_text()) == TIMELINE
    assert (tmp_path / "game.pgn").read_text() == "PGN"
    assert (tmp_path / "manifest.json").is_file()


def test_provider_failure_does_not_create_success_manifest(tmp_path):
    with (
        patch("chess_video.pipeline.load_game", return_value=GAME),
        patch("chess_video.pipeline.generate_script", side_effect=RuntimeError("provider unavailable")),
        pytest.raises(RuntimeError, match="provider unavailable"),
    ):
        generate_video("PGN", tmp_path, object())
    assert not (tmp_path / "manifest.json").exists()


def test_existing_script_is_validated_without_llm_generation(tmp_path):
    script_path = tmp_path / "input.json"
    script_path.write_text(json.dumps(SCRIPT))
    with (
        patch("chess_video.pipeline.load_game", return_value=GAME),
        patch("chess_video.pipeline.generate_script") as llm,
        patch("chess_video.pipeline.validate_script", return_value=SCRIPT),
        patch("chess_video.pipeline.synthesize", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError, match="stop"),
    ):
        generate_video("PGN", tmp_path / "out", object(), script_path=script_path)
    llm.assert_not_called()
