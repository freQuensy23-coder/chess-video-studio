from pathlib import Path
from unittest.mock import patch

import pytest

from chess_video.cli import main, parser


def test_generate_requires_pgn_and_output():
    with pytest.raises(SystemExit):
        parser().parse_args(["generate"])


def test_server_cannot_bind_public_interface():
    with pytest.raises(SystemExit):
        parser().parse_args(["serve", "--host", "0.0.0.0"])


def test_showcase_does_not_require_provider_settings(tmp_path, capsys):
    output = tmp_path / "showcase.mp4"
    with (
        patch("sys.argv", ["chess-video", "showcase", "--output", str(output)]),
        patch("chess_video.renderer.write_showcase", return_value=output) as render,
    ):
        assert main() == 0
    assert render.call_args.kwargs == {"width": 608, "height": 1080, "fps": 30}
    assert str(output) in capsys.readouterr().out


def test_provider_failure_returns_nonzero_without_success_json(tmp_path, capsys):
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. f3 e5 2. g4 Qh4# 0-1")
    with (
        patch("sys.argv", ["chess-video", "generate", "--pgn", str(pgn), "--output", str(tmp_path / "out")]),
        patch("chess_video.pipeline.generate_video", side_effect=RuntimeError("provider unavailable")),
    ):
        assert main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Generation stopped" in captured.err


def test_render_reads_existing_timeline_without_llm(tmp_path, capsys):
    source = tmp_path / "timeline.json"
    source.write_text('{"duration": 2, "events": []}')
    output = tmp_path / "out.mp4"
    with (
        patch("sys.argv", ["chess-video", "render", "--timeline", str(source), "--output", str(output)]),
        patch("chess_video.renderer.render_video", return_value=Path(output)) as render,
    ):
        assert main() == 0
    assert render.call_args.args[0] == {"duration": 2, "events": []}
    assert "video_path" in capsys.readouterr().out
