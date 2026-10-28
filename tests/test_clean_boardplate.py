import copy
import json
import subprocess
from unittest.mock import patch

import chess
import pytest
from PIL import Image

import chess_video.renderer as renderer_module
from chess_video.cli import main
from chess_video.renderer import Renderer, render_frame, render_video


def timeline(scenario="opening", **overrides):
    board = chess.Board()
    if scenario != "opening":
        for uci in ("f2f3", "e7e5", "g2g4"):
            board.push_uci(uci)
    if scenario == "initial-mate":
        board.push_uci("d8h4")
    initial_fen = board.fen()
    events = []
    if scenario != "initial-mate":
        uci = "e2e4" if scenario == "opening" else "d8h4"
        move = {"kind": "move", "start": 0.8, "end": 1.5, "uci": uci, "before_fen": board.fen()}
        board.push_uci(uci)
        move["fen"] = board.fen()
        events.append(move)
    events.extend([
        {"kind": "arrow", "start": 1.6, "end": 3.0, "from_square": "g1", "to_square": "f3"},
        {"kind": "highlight", "start": 1.6, "end": 3.0, "square": "h4"},
        {"kind": "zoom", "start": 0.1, "end": 0.7, "square": "e2", "scale": 1.3},
        {"kind": "face", "start": 0.0, "end": 4.0, "emotion": "surprised"},
    ])
    if scenario != "opening":
        events.append({"kind": "result", "start": 2.0, "end": 3.0, "label": "Checkmate"})
    data = {
        "version": 1, "width": 240, "height": 426, "fps": 10, "duration": 4.0,
        "initial_fen": initial_fen, "events": events,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("scenario", ["opening", "mate", "initial-mate"])
@pytest.mark.parametrize("presenter", ["male", "female"])
@pytest.mark.parametrize("orientation", ["white", "black"])
def test_clean_boardplate_preserves_board_without_any_presenter_work(monkeypatch, scenario, presenter, orientation):
    data = timeline(scenario, presenter=presenter, orientation=orientation, show_presenter=False)
    original = copy.deepcopy(data)
    times = (0.0, 0.5, 1.1, 1.7, 2.5, 3.8)
    with Renderer(dict(data, show_presenter=True)) as reference:
        monkeypatch.setattr(reference, "_face_panel", lambda time: Image.new(
            "RGB", (reference.width, reference.height - reference.face_y), (8, 17, 27)
        ))
        monkeypatch.setattr(reference, "_reaction", lambda time: 0.0)
        expected = [reference.frame(time).tobytes() for time in times]
        cursor_keys = reference._cursor_keys
        camera_sequences = reference._camera_sequences

    def forbidden(*args, **kwargs):
        pytest.fail("Clean boardplate reached the presenter rig or mate reaction")

    for name in ("_face", "_face_base"):
        monkeypatch.setattr(renderer_module, name, forbidden)
    for name in ("_face_panel", "_reaction", "_emotion", "_mouth"):
        monkeypatch.setattr(Renderer, name, forbidden)
    with Renderer(data) as renderer:
        assert renderer.show_presenter is False
        assert renderer._cursor_keys == cursor_keys
        assert renderer._camera_sequences == camera_sequences
        for time, expected_frame in zip(times, expected):
            frame = renderer.frame(time)
            assert frame.size == (240, 426)
            assert frame.tobytes() == expected_frame
            assert frame.crop((0, renderer.panel_height, renderer.width, renderer.height)).getcolors() == [
                (renderer.width * (renderer.height - renderer.panel_height), (8, 17, 27))
            ]
        if scenario != "opening":
            assert renderer.board_at(3.8).is_checkmate()
            assert renderer.result_at(2.5)
            assert renderer.result_at(3.8) is None
    assert data == original


def test_presenter_visibility_defaults_to_existing_frames():
    data = timeline("mate")
    with Renderer(data) as default, Renderer(dict(data, show_presenter=True)) as explicit:
        assert default.show_presenter is True
        for time in (0.0, 1.1, 2.5, 3.8):
            assert default.frame(time).tobytes() == explicit.frame(time).tobytes()
    clean = dict(data, show_presenter=False)
    assert render_frame(clean, 3.8).tobytes() != render_frame(data, 3.8).tobytes()
    assert render_frame(clean, 3.8).tobytes() == render_frame(clean, 3.8).tobytes()


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, 0.0, [], {}])
@pytest.mark.parametrize("entrypoint", ["renderer", "frame", "video"])
def test_presenter_visibility_requires_boolean_before_resource_work(monkeypatch, tmp_path, value, entrypoint):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid presenter visibility reached resources")

    monkeypatch.setattr(renderer_module, "_wood_board", forbidden)
    monkeypatch.setattr(Renderer, "_load_audio", forbidden)
    data = timeline(show_presenter=value, audio_path=str(tmp_path / "absent.wav"))
    output = tmp_path / "absent" / "video.mp4"
    with pytest.raises(ValueError, match="show_presenter.*boolean"):
        if entrypoint == "renderer":
            Renderer(data)
        elif entrypoint == "frame":
            render_frame(data, 0.0)
        else:
            render_video(data, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("existing,flag,expected", [
    (False, [], False), (True, [], True), (True, ["--no-presenter"], False),
])
def test_render_cli_preserves_or_disables_existing_presenter(tmp_path, existing, flag, expected):
    source = tmp_path / "timeline.json"
    data = timeline(show_presenter=existing)
    source.write_text(json.dumps(data))
    output = tmp_path / "out.mp4"
    with (
        patch("sys.argv", ["chess-video", "render", "--timeline", str(source), "--output", str(output), *flag]),
        patch("chess_video.renderer.render_video", return_value=output) as render,
    ):
        assert main() == 0
    assert render.call_args.args[0] == dict(data, show_presenter=expected)
    assert json.loads(source.read_text()) == data


def test_clean_boardplate_encodes_real_video_without_face_rig(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Clean boardplate encoding reached the presenter")

    monkeypatch.setattr(Renderer, "_face_panel", forbidden)
    monkeypatch.setattr(Renderer, "_reaction", forbidden)
    output = render_video(timeline("mate", show_presenter=False), tmp_path / "boardplate.mp4")
    decoded = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(output), "-ss", "3.8", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    frame = Image.frombytes("RGB", (240, 426), decoded.stdout)
    with Renderer(timeline("mate", show_presenter=False)) as renderer:
        colors = frame.crop((0, renderer.face_y, renderer.width, renderer.height)).getextrema()
    assert all(high - low <= 4 for low, high in colors)
