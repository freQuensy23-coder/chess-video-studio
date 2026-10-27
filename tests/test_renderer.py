import copy
import math
import shutil
import subprocess
import wave
from pathlib import Path

import chess
import pytest
from PIL import Image, ImageChops

from chess_video.renderer import Renderer, render_frame, render_video


def timeline(fen=chess.STARTING_FEN, **overrides):
    value = {
        "version": 1,
        "width": 608,
        "height": 1080,
        "fps": 30,
        "duration": 4.0,
        "audio_path": None,
        "initial_fen": fen,
        "orientation": "white",
        "title": "",
        "events": [],
    }
    value.update(overrides)
    return value


def move_event(fen, uci, start=0.8, end=2.0, ply=1):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    event = {
        "kind": "move",
        "start": start,
        "end": end,
        "before_fen": board.fen(),
        "uci": uci,
        "san": board.san(move),
        "ply": ply,
        "is_capture": board.is_capture(move),
        "is_castling": board.is_castling(move),
        "is_en_passant": board.is_en_passant(move),
        "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
    }
    board.push(move)
    event.update(fen=board.fen(), is_check=board.is_check(), is_checkmate=board.is_checkmate())
    return event


def square_crop(renderer, frame, square):
    x, y = renderer.square_center(square)
    size = int(renderer.cell * 0.32)
    return frame.crop((int(x - size), int(y - size), int(x + size), int(y + size)))


@pytest.mark.parametrize("width,height", [(608, 1080), (720, 1280), (320, 568)])
def test_frame_dimensions_and_mode(width, height):
    frame = render_frame(timeline(width=width, height=height), 0)
    assert isinstance(frame, Image.Image)
    assert frame.size == (width, height)
    assert frame.mode == "RGB"


def test_black_orientation_rotates_square_coordinates_not_piece_art():
    with Renderer(timeline()) as white, Renderer(timeline(orientation="black")) as black:
        assert white.square_center("a1") == black.square_center("h8")
        assert white.square_center("e4") == black.square_center("d5")
        assert white.square_center("a1")[1] > white.square_center("a8")[1]
        assert black.square_center("a1")[1] < black.square_center("a8")[1]


def test_renderer_does_not_mutate_timeline_or_depend_on_frame_order():
    event = move_event(chess.STARTING_FEN, "e2e4")
    data = timeline(events=[event])
    original = copy.deepcopy(data)
    with Renderer(data) as renderer:
        first = renderer.frame(1.35).tobytes()
        renderer.frame(3.8)
        renderer.frame(0.1)
        assert renderer.frame(1.35).tobytes() == first
    assert data == original


@pytest.mark.parametrize(
    "fen,uci,expected,empty",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", {"g1": "K", "f1": "R"}, ["e1", "h1"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", {"c8": "k", "d8": "r"}, ["e8", "a8"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", {"d6": "P"}, ["e5", "d5"]),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", {"a8": "N"}, ["a7"]),
        ("4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h2h1r", {"h1": "r"}, ["h2"]),
    ],
)
def test_special_moves_use_actual_piece_maps(fen, uci, expected, empty):
    event = move_event(fen, uci)
    with Renderer(timeline(fen, events=[event])) as renderer:
        state = renderer.board_at(2.1)
        for square, symbol in expected.items():
            assert state.piece_at(chess.parse_square(square)).symbol() == symbol
        for square in empty:
            assert state.piece_at(chess.parse_square(square)) is None
        assert renderer.frame(1.5).size == (608, 1080)
        assert renderer.frame(2.1).size == (608, 1080)


def test_castling_rook_visibly_moves_during_drag():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    with Renderer(timeline(fen, events=[move_event(fen, "e1g1")])) as renderer:
        before = renderer.frame(0.2)
        during = renderer.frame(1.5)
        assert ImageChops.difference(square_crop(renderer, before, "h1"), square_crop(renderer, during, "h1")).getbbox()


def test_en_passant_capture_square_is_visibly_emptied():
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    with Renderer(timeline(fen, events=[move_event(fen, "e5d6")])) as renderer:
        assert ImageChops.difference(square_crop(renderer, renderer.frame(0.2), "d5"), square_crop(renderer, renderer.frame(2.2), "d5")).getbbox()


@pytest.mark.parametrize("promotion", ["q", "r", "b", "n"])
def test_promotion_picker_shows_all_choices_and_uses_selected_piece(promotion):
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    event = move_event(fen, "a7a8" + promotion)
    with Renderer(timeline(fen, events=[event])) as renderer:
        assert renderer.promotion_choices_at(1.7) == ("q", "r", "b", "n")
        assert renderer.promotion_choices_at(2.1) == ()
        assert renderer.board_at(2.1).piece_at(chess.A8).symbol() == promotion.upper()
        assert ImageChops.difference(renderer.frame(1.7), renderer.frame(2.1)).getbbox()


def test_selected_move_dots_come_from_legal_moves():
    fen = "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1"
    with Renderer(timeline(fen, events=[{"kind": "cursor", "start": 0, "end": 2, "square": "e2"}])) as renderer:
        assert set(renderer.legal_destinations("e2", 0.5)) == {"e3", "e4", "e5", "e6", "e7", "e8"}
        assert "d2" not in renderer.legal_destinations("e2", 0.5)


def test_all_annotation_types_clear_and_persistence():
    events = [
        {"kind": "arrow", "start": 0.1, "from_square": "b1", "to_square": "c3", "color": "#ffcc44"},
        {"kind": "arrow", "start": 0.3, "end": 3.8, "from_square": "c1", "to_square": "h6", "color": "orange"},
        {"kind": "highlight", "start": 0.4, "end": 3.8, "square": "e4", "color": "cyan"},
        {"kind": "badge", "start": 0.5, "end": 3.8, "square": "c3", "label": "!!", "color": "cyan"},
        {"kind": "face", "start": 0.5, "end": 3.9, "emotion": "worried"},
        {"kind": "cursor", "start": 0.7, "end": 2, "square": "b1"},
        {"kind": "clear", "start": 2.5, "end": 2.5},
    ]
    with Renderer(timeline(events=events)) as renderer:
        assert len(renderer.annotations_at(2.0)) == 4
        assert renderer.annotations_at(2.6) == []
        assert ImageChops.difference(renderer.frame(0), renderer.frame(2)).getbbox()


def test_clear_does_not_remove_annotations_that_start_after_it():
    events = [
        {"kind": "clear", "start": 1, "end": 1},
        {"kind": "highlight", "start": 1.5, "end": 3, "square": "d4", "color": "red"},
    ]
    with Renderer(timeline(events=events)) as renderer:
        assert len(renderer.annotations_at(2)) == 1


def test_zoom_is_board_only_and_clipped_at_corners():
    plain = timeline()
    zoomed = timeline(events=[{"kind": "zoom", "start": 0.1, "end": 3, "square": "a8", "scale": 2.8}])
    first = render_frame(plain, 1.2)
    second = render_frame(zoomed, 1.2)
    assert first.crop((0, 620, 608, 1080)).tobytes() == second.crop((0, 620, 608, 1080)).tobytes()
    assert ImageChops.difference(first.crop((0, 0, 608, 608)), second.crop((0, 0, 608, 608))).getbbox()
    assert second.getpixel((0, 0)) != (0, 0, 0)


def test_cursor_trajectory_is_continuous_and_random_access():
    event = move_event(chess.STARTING_FEN, "e2e4")
    with Renderer(timeline(events=[event])) as renderer:
        points = [renderer.cursor_at(t / 100) for t in range(0, 221)]
        assert max(math.dist(a, b) for a, b in zip(points, points[1:])) < renderer.cell * 0.3
        assert renderer.cursor_at(1.1) == renderer.cursor_at(1.1)


def test_mate_and_winner_overlay_is_grounded_in_board():
    fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
    event = move_event(fen, "e1e8")
    with Renderer(timeline(fen, events=[event, {"kind": "result", "start": 2, "end": 4, "label": "White wins"}])) as renderer:
        assert renderer.board_at(2.5).is_checkmate()
        assert renderer.result_at(2.5) == "CHECKMATE · WHITE WINS"


def test_nonterminal_result_cannot_claim_checkmate():
    with pytest.raises(ValueError, match="terminal|result|mate"):
        Renderer(timeline(events=[{"kind": "result", "start": 1, "end": 3, "label": "Checkmate"}]))


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_fen": "not a fen"},
        {"width": 607},
        {"height": 0},
        {"width": 2000, "height": 1080},
        {"fps": 0},
        {"fps": float("nan")},
        {"duration": float("inf")},
        {"duration": -1},
        {"orientation": "sideways"},
        {"events": [{"kind": "highlight", "start": -1, "end": 2, "square": "a1"}]},
        {"events": [{"kind": "arrow", "start": 0, "end": 2, "from_square": "a1", "to_square": "z9"}]},
        {"events": [{"kind": "zoom", "start": 0, "end": 2, "square": "a1", "scale": float("nan")}]},
        {"events": [{"kind": "zoom", "start": 0, "end": 2, "square": "a1", "scale": 0.5}]},
        {"events": [{"kind": "bogus", "start": 0, "end": 2}]},
    ],
)
def test_invalid_timeline_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        Renderer(timeline(**changes))


@pytest.mark.parametrize("time", [-0.1, float("nan"), float("inf"), 4.1])
def test_invalid_frame_time_rejected(time):
    with pytest.raises(ValueError):
        render_frame(timeline(), time)


def test_illegal_move_and_fen_mismatch_rejected():
    event = move_event(chess.STARTING_FEN, "e2e4")
    event["fen"] = chess.STARTING_FEN
    with pytest.raises(ValueError):
        Renderer(timeline(events=[event]))
    event["uci"] = "e2e5"
    with pytest.raises(ValueError):
        Renderer(timeline(events=[event]))


def test_overlapping_moves_rejected():
    first = move_event(chess.STARTING_FEN, "e2e4", 0.2, 1.5)
    second = move_event(first["fen"], "e7e5", 1, 2)
    with pytest.raises(ValueError, match="overlap"):
        Renderer(timeline(events=[first, second]))


def test_promotion_cursor_selects_the_underpromotion_row():
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    event = move_event(fen, "a7a8r")
    with Renderer(timeline(fen, events=[event])) as renderer:
        destination = renderer.square_center("a8")
        pointer = renderer.cursor_at(1.99)
        assert pointer[0] == pytest.approx(destination[0], abs=1)
        assert pointer[1] == pytest.approx(destination[1] + renderer.cell, abs=1)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_audio_tail_extends_video_and_encoder_uses_safe_arguments(tmp_path, monkeypatch):
    audio = tmp_path / "voice ; original.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x00\x00" * 6400)
    calls = []
    original_popen = subprocess.Popen

    def record(command, *args, **kwargs):
        calls.append(command)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)
    output = render_video(timeline(duration=0.2, audio_path=str(audio), width=160, height=284, fps=10), tmp_path / "delivered.mp4")
    assert output.stat().st_size > 1000
    command = next(command for command in calls if "rawvideo" in command)
    assert "-shortest" not in command
    assert str(audio) in command
    assert command[command.index("-t") + 1] == "0.800000000"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert "yuv420p" in command
    assert command[command.index("-c:a") + 1] == "aac"


def test_missing_audio_rejected(tmp_path):
    with pytest.raises(ValueError, match="Audio"):
        Renderer(timeline(audio_path=str(tmp_path / "missing.wav")))


def test_ffmpeg_failure_propagates_and_preserves_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "movie.mp4"
    output.write_bytes(b"existing")

    def fail(*args, **kwargs):
        raise FileNotFoundError("ffmpeg is unavailable")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises((RuntimeError, FileNotFoundError), match="ffmpeg"):
        render_video(timeline(duration=0.1, width=320, height=568), output)
    assert output.read_bytes() == b"existing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["movie.mp4"]


def test_encoder_failure_removes_partial_output(tmp_path, monkeypatch):
    class BrokenEncoder:
        stdin = None

        def __init__(self, command, **kwargs):
            Path(command[-1]).write_bytes(b"partial")
            raise OSError("encoder failed")

    monkeypatch.setattr(subprocess, "Popen", BrokenEncoder)
    output = tmp_path / "nested" / "movie.mp4"
    with pytest.raises((RuntimeError, OSError)):
        render_video(timeline(duration=0.1, width=320, height=568), output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


@pytest.mark.parametrize('result,label', [
    ('1-0', 'White wins'), ('0-1', 'Чёрные победили'), ('1/2-1/2', 'Ничья'),
])
def test_declared_nonterminal_result_uses_display_label(result, label):
    event = move_event(chess.STARTING_FEN, 'e2e4')
    result_event = {'kind': 'result', 'start': 2, 'end': 3, 'label': label}
    with Renderer(timeline(game_result=result, events=[event, result_event])) as renderer:
        assert renderer.result_at(1.9) is None
        assert renderer.result_at(2) == label
        assert renderer.result_at(3.1) is None
        assert not renderer.board_at(2).is_checkmate()


def test_undeclared_nonterminal_winner_is_rejected():
    with pytest.raises(ValueError, match='terminal|result'):
        Renderer(timeline(events=[{'kind': 'result', 'start': 1, 'end': 3, 'label': 'White wins'}]))


@pytest.mark.parametrize('label', ['Checkmate!', 'Мат!', 'Black wins', 'Чёрные победили', 'Ничья'])
def test_declared_result_rejects_false_mate_or_contradictory_label(label):
    with pytest.raises(ValueError, match='mate|checkmate|winner|result|Result'):
        Renderer(timeline(game_result='1-0', events=[{'kind': 'result', 'start': 1, 'end': 3, 'label': label}]))


@pytest.mark.parametrize('result', ['0-1', '1/2-1/2'])
def test_declared_result_cannot_override_forced_board_winner(result):
    fen = '6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1'
    with pytest.raises(ValueError, match='result|Result|winner'):
        Renderer(timeline(fen, game_result=result, events=[move_event(fen, 'e1e8')]))


@pytest.mark.parametrize('label', ['Black wins', 'Чёрные победили', 'Победа чёрных', 'Draw', 'Ничья', '0-1'])
def test_terminal_mate_rejects_contradictory_localized_result(label):
    fen = '6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1'
    events = [move_event(fen, 'e1e8'), {'kind': 'result', 'start': 2, 'end': 4, 'label': label}]
    with pytest.raises(ValueError, match='result|Result|winner'):
        Renderer(timeline(fen, events=events))


def test_stalemate_cannot_claim_russian_mate():
    fen = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'
    with pytest.raises(ValueError, match='mate|checkmate'):
        Renderer(timeline(fen, events=[{'kind': 'result', 'start': 0, 'end': 3, 'label': 'Мат'}]))


def test_declared_result_waits_until_all_moves_complete():
    event = move_event(chess.STARTING_FEN, 'e2e4')
    with pytest.raises(ValueError, match='completed|final|result|Result'):
        Renderer(timeline(game_result='1-0', events=[event, {'kind': 'result', 'start': 1, 'end': 3, 'label': 'White wins'}]))


@pytest.mark.parametrize('result', ['white', '', None, 1, []])
def test_invalid_declared_result_is_rejected(result):
    with pytest.raises(ValueError, match='result|Result'):
        Renderer(timeline(game_result=result))


def test_russian_mate_overlay_uses_localized_winner():
    fen = '6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1'
    events = [move_event(fen, 'e1e8'), {'kind': 'result', 'start': 2, 'end': 4, 'label': 'Мат'}]
    with Renderer(timeline(fen, game_result='1-0', language='ru', events=events)) as renderer:
        assert renderer.result_at(1.99) is None
        assert renderer.result_at(2) == 'МАТ · БЕЛЫЕ ПОБЕДИЛИ'


def test_default_camera_starts_zoomed_and_smoothly_pulls_back():
    with Renderer(timeline()) as renderer:
        assert renderer._zoom(0)[0] == pytest.approx(1.45)
        assert renderer._zoom(0.5)[0] == pytest.approx(1.225)
        assert renderer._zoom(1)[0] == pytest.approx(1)
        assert renderer._zoom(0.999)[0] > renderer._zoom(1)[0]
        assert renderer._zoom(0)[1] == (renderer.width / 2, renderer.panel_height / 2)


def test_explicit_zoom_overrides_default_initial_pullback():
    events = [{'kind': 'zoom', 'start': 0, 'end': 2, 'square': 'a8', 'scale': 2}]
    with Renderer(timeline(events=events)) as renderer:
        assert renderer._zoom(0.5) == (2, renderer.square_center('a8'))


def test_initial_pullback_is_board_only_and_random_access():
    with Renderer(timeline()) as renderer:
        panel = renderer._static.copy()
        pulled_back = renderer._camera(panel, 1)
        zoomed = renderer._camera(panel, 0)
        assert ImageChops.difference(zoomed, pulled_back).convert('RGB').getbbox()
        first = renderer.frame(0.2)
        renderer.frame(3)
        assert renderer.frame(0.2).tobytes() == first.tobytes()
        explicit = timeline(events=[{'kind': 'zoom', 'start': 0, 'end': 2, 'square': 'a8', 'scale': 2}])
        with Renderer(explicit) as other:
            bounds = (0, renderer.face_y, renderer.width, renderer.height)
            assert first.crop(bounds).tobytes() == other.frame(0.2).crop(bounds).tobytes()


def test_initial_pullback_blends_continuously_into_first_move_camera():
    event = move_event(chess.STARTING_FEN, 'e2e4')
    with Renderer(timeline(events=[event])) as renderer:
        before_scale, before_focus = renderer._zoom(0.99999)
        after_scale, after_focus = renderer._zoom(1)
        assert abs(before_scale - after_scale) < 0.001
        assert math.dist(before_focus, after_focus) < 0.01


def test_initial_pullback_focus_is_continuous_at_move_camera_onset():
    event = move_event(chess.STARTING_FEN, 'e2e4')
    with Renderer(timeline(events=[event])) as renderer:
        onset = event['start'] - 0.12
        before_scale, before_focus = renderer._zoom(onset - 0.00001)
        after_scale, after_focus = renderer._zoom(onset + 0.00001)
        assert abs(before_scale - after_scale) < 0.001
        assert math.dist(before_focus, after_focus) < 0.01


def test_declared_resignation_does_not_turn_claimable_draw_into_forced_draw():
    fen = '4k3/8/8/8/8/8/8/R3K3 w - - 100 1'
    event = {'kind': 'result', 'start': 1, 'end': 3, 'label': 'White wins'}
    with Renderer(timeline(fen, game_result='1-0', events=[event])) as renderer:
        assert renderer.board_at(1).can_claim_fifty_moves()
        assert renderer.result_at(1) == 'White wins'


@pytest.mark.parametrize('label', ['White wins', 'Белые победили', '1-0'])
def test_forced_draw_rejects_winner_label(label):
    fen = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'
    with pytest.raises(ValueError, match='result|Result|winner'):
        Renderer(timeline(fen, events=[{'kind': 'result', 'start': 0, 'end': 3, 'label': label}]))
