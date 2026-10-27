import math

import chess
import pytest
from PIL import Image

from chess_video.renderer import Renderer


def scene(fen=chess.STARTING_FEN, moves=(), **overrides):
    board = chess.Board(fen)
    events = []
    for uci, start, end in moves:
        move = chess.Move.from_uci(uci)
        before = board.fen()
        board.push(move)
        events.append({"kind": "move", "uci": uci, "start": start, "end": end, "before_fen": before, "fen": board.fen()})
    value = {"width": 320, "height": 568, "fps": 24, "duration": 16, "initial_fen": fen, "orientation": "white", "events": events}
    value.update(overrides)
    return value


@pytest.mark.parametrize("orientation", ["white", "black"])
@pytest.mark.parametrize("player", ["white", "black"])
def test_mouse_moves_only_the_players_actual_legal_piece(player, orientation):
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 4, 5), ("g1f3", 6, 7)], orientation=orientation, player_color=player)
    with Renderer(data) as renderer:
        for event in renderer.moves:
            start, end = event["start"], event["end"]
            own = event["_piece"].color == (player == "white")
            assert renderer._selection(start + 0.1) == (chess.square_name(event["_move"].from_square) if own else None)
            if own:
                assert renderer.cursor_at(start + 0.18) == pytest.approx(renderer.square_center(event["_move"].from_square))
                assert renderer.cursor_at(start + 0.82) == pytest.approx(renderer.square_center(event["_move"].to_square))
            else:
                held = renderer.cursor_at(start - 0.1)
                for time in (start, start + 0.16, start + 0.5, end):
                    assert renderer.cursor_at(time) == pytest.approx(held)
                assert not any(start - 0.4 < time <= end for time, _ in renderer._cursor_keys)


@pytest.mark.parametrize("orientation", ["white", "black"])
def test_orientation_is_the_player_color_fallback(orientation):
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 4, 5)], orientation=orientation)
    with Renderer(data) as renderer:
        assert renderer._selection(2.1) == ("e2" if orientation == "white" else None)
        assert renderer._selection(4.1) == ("e7" if orientation == "black" else None)


@pytest.mark.parametrize("value", ["both", "White", None, True, 1, []])
def test_invalid_player_color_is_rejected(value):
    with pytest.raises(ValueError, match="player_color"):
        Renderer(scene(player_color=value))


def test_mouse_holds_through_back_to_back_opponent_turn():
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 3, 4), ("g1f3", 4, 5)])
    with Renderer(data) as renderer:
        held = renderer.cursor_at(3)
        for time in (3.00001, 3.16, 3.5, 3.99999):
            assert renderer.cursor_at(time) == pytest.approx(held)
            assert renderer._selection(time) is None
        assert math.dist(renderer.cursor_at(4.00001), held) < 0.01
        assert renderer.cursor_at(4.18) == pytest.approx(renderer.square_center("g1"))


@pytest.mark.parametrize("player", ["white", "black"])
def test_opponent_move_has_no_click_pulse(player):
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 4, 5)], player_color=player)
    with Renderer(data) as renderer:
        enemy_start = 4 if player == "white" else 2
        early = Image.new("RGBA", renderer._static.size)
        pressed = Image.new("RGBA", renderer._static.size)
        renderer._draw_cursor(early, enemy_start)
        renderer._draw_cursor(pressed, enemy_start + 0.16)
        assert early.tobytes() == pressed.tobytes()
        own_start = 2 if player == "white" else 4
        early = Image.new("RGBA", renderer._static.size)
        pressed = Image.new("RGBA", renderer._static.size)
        renderer._draw_cursor(early, own_start)
        renderer._draw_cursor(pressed, own_start + 0.16)
        assert early.tobytes() != pressed.tobytes()


@pytest.mark.parametrize("square", ["e7", "e4"])
def test_cursor_cues_do_not_select_opponent_or_empty_squares(square):
    data = scene(events=[{"kind": "cursor", "start": 2, "end": 3, "square": square}])
    with Renderer(data) as renderer:
        assert renderer._selection(2.1) is None
        assert renderer.legal_destinations(square, 2.1) == []


def test_player_cannot_select_own_piece_before_opponent_finishes():
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 3, 4), ("g1f3", 4, 5)])
    with Renderer(data) as renderer:
        assert renderer._selection(3.9) is None
        assert renderer.legal_destinations("g1", 3.9) == []


@pytest.mark.parametrize("kind", ["arrow", "highlight", "badge", "cursor"])
def test_opponent_animation_blocks_optional_pointer_actions(kind):
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 4, 5)])
    cue = {"kind": kind, "start": 4.1, "end": 6, "square": "g1"}
    if kind == "arrow":
        cue.update(from_square="e7", to_square="e5")
    data["events"].append(cue)
    with Renderer(data) as renderer:
        assert renderer._cursor_strokes == []
        held = renderer.cursor_at(3)
        for time in (3.9, 4.1, 4.2, 4.5, 4.9, 5):
            assert renderer.cursor_at(time) == held
            assert renderer._selection(time) is None
            assert renderer._cursor_gesture(time) is None


@pytest.mark.parametrize("fen,uci,promoted", [
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", "N"),
    ("4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h2h1r", "r"),
])
@pytest.mark.parametrize("orientation", ["white", "black"])
def test_opponent_promotion_is_automatic_and_preserves_selected_piece(fen, uci, promoted, orientation, monkeypatch):
    player = "black" if chess.Board(fen).turn else "white"
    data = scene(fen, [(uci, 2, 3)], player_color=player, orientation=orientation)
    with Renderer(data) as renderer:
        held = renderer.cursor_at(1.5)
        for time in (2, 2.5, 2.7, 2.95, 3):
            assert renderer.promotion_choices_at(time) == ()
            assert renderer.cursor_at(time) == held
            assert renderer._selection(time) is None
        pieces = []
        monkeypatch.setattr(renderer, "_put_piece", lambda image, symbol, point, lifted=False: pieces.append((symbol, point, lifted)))
        renderer._draw_move(Image.new("RGBA", renderer._static.size), renderer.moves[0], 2.95)
        assert pieces[-1] == (promoted, renderer.square_center(uci[2:4]), False)
        assert renderer.board_at(3).piece_at(chess.parse_square(uci[2:4])).symbol() == promoted


@pytest.mark.parametrize("orientation", ["white", "black"])
@pytest.mark.parametrize("player,fen,uci", [
    ("white", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
    ("black", "4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h2h1n"),
])
def test_player_promotion_picker_follows_owner_even_on_flipped_board(orientation, player, fen, uci):
    with Renderer(scene(fen, [(uci, 2, 3)], player_color=player, orientation=orientation)) as renderer:
        assert renderer.promotion_choices_at(2.7) == ("q", "r", "b", "n")
        assert renderer.promotion_choices_at(3) == ()
        left, top, down = renderer._promotion_geometry(renderer.moves[0])
        choice = ("q", "r", "b", "n").index(uci[-1])
        row = choice if down else 3 - choice
        assert renderer.cursor_at(2.95) == pytest.approx((left + renderer.cell / 2, top + (row + 0.5) * renderer.cell))
        assert renderer.board_at(3).piece_at(chess.parse_square(uci[2:4])).symbol().lower() == uci[-1]


@pytest.mark.parametrize("fen,uci,expected,empty", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", {"g1": "K", "f1": "R"}, ["e1", "h1"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", {"c8": "k", "d8": "r"}, ["e8", "a8"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", {"d6": "P"}, ["e5", "d5"]),
    ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3", {"e3": "p"}, ["d4", "e4"]),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", {"d5": "P"}, ["e4"]),
    ("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "d5e4", {"e4": "p"}, ["d5"]),
])
def test_opponent_special_moves_fly_without_pickup(fen, uci, expected, empty, monkeypatch):
    player = "black" if chess.Board(fen).turn else "white"
    with Renderer(scene(fen, [(uci, 2, 3)], player_color=player)) as renderer:
        moving = renderer.moves[0]["_piece"].symbol()
        pieces = []
        monkeypatch.setattr(renderer, "_put_piece", lambda image, symbol, point, lifted=False: pieces.append((symbol, point, lifted)))
        renderer._draw_move(Image.new("RGBA", renderer._static.size), renderer.moves[0], 2.1)
        flight = [entry for entry in pieces if entry[0] == moving][-1]
        assert flight[1] != renderer.square_center(uci[:2])
        assert not any(entry[2] for entry in pieces)
        state = renderer.board_at(3)
        for square, symbol in expected.items():
            assert state.piece_at(chess.parse_square(square)).symbol() == symbol
        for square in empty:
            assert state.piece_at(chess.parse_square(square)) is None
        assert renderer._selection(2.1) is None
        assert renderer.cursor_at(2.1) == renderer.cursor_at(2.9)


@pytest.mark.parametrize("fen,uci,captured", [
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", "d5"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "d5e4", "e4"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "d5"),
    ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3", "e4"),
])
def test_automatic_capture_removes_the_actual_captured_piece_during_flight(fen, uci, captured, monkeypatch):
    player = "black" if chess.Board(fen).turn else "white"
    with Renderer(scene(fen, [(uci, 2, 3)], player_color=player)) as renderer:
        positions = []

        def position_layer(board):
            positions.append(board.copy())
            return Image.new("RGBA", renderer._static.size)

        monkeypatch.setattr(renderer, "_position_layer", position_layer)
        renderer._draw_move(Image.new("RGBA", renderer._static.size), renderer.moves[0], 2.1)
        renderer._draw_move(Image.new("RGBA", renderer._static.size), renderer.moves[0], 2.85)
        assert positions[0].piece_at(chess.parse_square(captured)) is not None
        assert positions[1].piece_at(chess.parse_square(captured)) is None


def test_ordinary_moves_leave_camera_stable_after_intro():
    moves = [("e2e4", 2, 3), ("e7e5", 4, 5), ("g1f3", 6, 7), ("b8c6", 8, 9), ("f1c4", 10, 11)]
    with Renderer(scene(moves=moves)) as renderer:
        center = (renderer.width / 2, renderer.panel_height / 2)
        for time in (1, 1.9, 2.5, 3.1, 4.5, 6.5, 8.5, 10.5, 12):
            assert renderer._zoom(time) == (1, center)


@pytest.mark.parametrize("fen,uci", [
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "d5e4"),
])
def test_capture_alone_does_not_trigger_a_camera_accent(fen, uci):
    with Renderer(scene(fen, [(uci, 2, 3)])) as renderer:
        assert renderer.moves[0]["_capture"]
        for time in (1.9, 2, 2.5, 3, 3.5, 4):
            assert renderer._zoom(time) == (1, (renderer.width / 2, renderer.panel_height / 2))


@pytest.mark.parametrize("fen,uci", [
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8"),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
])
def test_semantic_camera_accents_promotion_check_and_mate(fen, uci):
    with Renderer(scene(fen, [(uci, 2, 3)])) as renderer:
        assert renderer._zoom(2)[0] == 1
        assert 1.15 <= renderer._zoom(3.9)[0] <= 1.5
        assert renderer._zoom(7) == (1, (renderer.width / 2, renderer.panel_height / 2))


def test_checks_close_together_do_not_repeat_camera_pulses():
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    moves = [("a1a8", 2, 3), ("e8e7", 3.2, 4), ("a8a7", 4.2, 5)]
    with Renderer(scene(fen, moves)) as renderer:
        assert renderer._zoom(3.5)[0] > 1
        assert renderer._zoom(6)[0] == 1


def test_automatic_camera_accents_have_a_small_total_budget():
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    moves = [("a1a8", 2, 3), ("e8e7", 4, 5), ("a8a7", 10, 11), ("e7e8", 12, 13), ("a7a8", 18, 19), ("e8e7", 20, 21), ("a8a7", 26, 27), ("e7e8", 28, 29), ("a7a8", 34, 35)]
    with Renderer(scene(fen, moves, duration=40)) as renderer:
        scales = [renderer._zoom(time)[0] for time in (3.7, 11.7, 19.7, 27.7, 35.7)]
        assert sum(scale > 1 for scale in scales) == 3


def test_rejected_cursor_cue_cannot_select_a_different_piece_during_player_drag():
    data = scene(moves=[("e2e4", 2, 3)])
    data["events"].append({"kind": "cursor", "start": 2.1, "end": 2.9, "square": "g1"})
    with Renderer(data) as renderer:
        assert renderer._selection(2.1) == "e2"
        assert renderer._selection(2.6) is None


def test_scale_one_zoom_never_leaves_an_off_center_camera_focus():
    events = [{"kind": "zoom", "start": 2, "end": 4, "square": "a8", "scale": 1}]
    with Renderer(scene(events=events)) as renderer:
        assert renderer._zoom(3) == (1, (renderer.width / 2, renderer.panel_height / 2))


def test_purposeful_zoom_sequences_have_different_deterministic_envelopes():
    events = [{"kind": "zoom", "start": start, "end": start + 3, "square": "e4", "scale": 1.5} for start in (2, 7, 12)]
    with Renderer(scene(events=events)) as renderer:
        profiles = [tuple(round(renderer._zoom(start + offset)[0], 5) for offset in (0.2, 0.7, 1.5, 2.5, 2.8)) for start in (2, 7, 12)]
        assert len(set(profiles)) == 3
        assert all(max(profile) == pytest.approx(1.5) for profile in profiles)
        renderer._zoom(15.9)
        assert renderer._zoom(2.2)[0] == pytest.approx(profiles[0][0], abs=0.00001)


def test_neutral_camera_center_and_close_explicit_transitions_are_continuous():
    events = [
        {"kind": "zoom", "start": 2, "end": 4, "square": "a8", "scale": 1.5},
        {"kind": "zoom", "start": 3.8, "end": 6, "square": "h1", "scale": 1.7},
    ]
    with Renderer(scene(events=events)) as renderer:
        center = (renderer.width / 2, renderer.panel_height / 2)
        for time in (1.5, 2, 6, 7):
            assert renderer._zoom(time) == (1, center)
        for time in (2, 3.8, 4, 6):
            left_scale, left_focus = renderer._zoom(time - 0.00001)
            right_scale, right_focus = renderer._zoom(time + 0.00001)
            assert abs(left_scale - right_scale) < 0.001
            assert math.dist(left_focus, right_focus) < 0.01


def tactical_arrows(start, end):
    return [{"kind": "arrow", "start": start, "end": end, "from_square": "f2", "to_square": target, "automatic": True, "tactical_group": "fork-1", "focus_squares": ["f2", "d1", "h1"]} for target in ("d1", "h1")]


@pytest.mark.parametrize("orientation", ["white", "black"])
def test_simultaneous_automatic_threat_arrows_have_no_pointer_stroke(orientation):
    arrows = tactical_arrows(2, 5)
    with Renderer(scene(orientation=orientation, events=arrows)) as renderer:
        assert len(renderer.annotations_at(2.5)) == 2
        assert renderer._cursor_strokes == []
        held = renderer.cursor_at(1)
        for time in (1.8, 2, 2.14, 2.28, 4.9, 5):
            assert renderer.cursor_at(time) == held
            assert renderer._cursor_gesture(time) is None
        for event in renderer.annotations_at(2.5):
            layer = Image.new("RGBA", renderer._static.size)
            renderer._arrow(layer, event, 2.5)
            assert layer.getbbox() is not None


@pytest.mark.parametrize("start,end", [(0, 0.8), (2, 5)])
def test_tactical_group_preserves_full_board_over_intro_and_explicit_zoom(start, end):
    events = [{"kind": "zoom", "start": 0, "end": 8, "square": "d5", "scale": 2}, *tactical_arrows(start, end)]
    with Renderer(scene(events=events)) as renderer:
        center = (renderer.width / 2, renderer.panel_height / 2)
        for time in (start, start + 0.1, (start + end) / 2, end - 0.00001):
            assert renderer._zoom(time) == (1, center)
            assert renderer._camera(renderer._static, time) is renderer._static
        for time in (start, end):
            if time > 0:
                left_scale, left_focus = renderer._zoom(time - 0.00001)
                right_scale, right_focus = renderer._zoom(time + 0.00001)
                assert abs(left_scale - right_scale) < 0.001
                assert math.dist(left_focus, right_focus) < 0.01


def test_cleared_tactical_group_releases_full_board_hold():
    events = [{"kind": "zoom", "start": 1, "end": 10, "square": "d5", "scale": 2}, *tactical_arrows(2, 8), {"kind": "clear", "start": 4, "end": 4}]
    with Renderer(scene(events=events)) as renderer:
        assert renderer._zoom(3)[0] == 1
        assert renderer.annotations_at(4.1) == []
        assert renderer._zoom(5)[0] > 1


def test_zoom_pulls_out_before_delayed_opponent_fork_arrows_start():
    data = scene(moves=[("e2e4", 2, 3), ("e7e5", 3, 4)])
    data["events"].extend([{"kind": "zoom", "start": 1, "end": 8, "square": "f2", "scale": 1.7}, *tactical_arrows(4.08, 6)])
    with Renderer(data) as renderer:
        assert renderer._zoom(3.6)[0] > renderer._zoom(3.9)[0] > renderer._zoom(4.05)[0] > 1
        assert renderer._zoom(4.08) == (1, (renderer.width / 2, renderer.panel_height / 2))
        assert renderer._zoom(4.5)[0] == 1
        assert renderer.cursor_at(3.1) == renderer.cursor_at(4.4)
