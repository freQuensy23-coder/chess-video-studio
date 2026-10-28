import copy

import chess
import pytest

from chess_video.providers import _llm_config, _llm_payload, _script_context
from chess_video.settings import Settings


@pytest.mark.parametrize("perspective", ["white", "black"])
@pytest.mark.parametrize("pgn", [
    "1. e4 e5 2. Nf3 Nc6 *",
    '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"]\n\n1... Kf7 2. e4 *',
])
def test_player_decisions_use_exact_pre_move_board_only_on_own_plies(perspective, pgn):
    game = load_game_with_analysis(pgn)
    original = copy.deepcopy(game)
    context = _script_context(game, Settings(_env_file=None, perspective=perspective))
    decisions = context["player_decisions"]
    expected_plies = [
        position["ply"] for position in game["positions"][1:]
        if chess.Board(position["before_fen"]).turn == (perspective == "white")
    ]
    assert [decision["ply"] for decision in decisions] == expected_plies
    for decision in decisions:
        ply = decision["ply"]
        before = decision["pre_move_position"]
        assert before == game["positions"][ply - 1]
        assert before["fen"] == game["positions"][ply]["before_fen"]
        assert before["analysis"] == {"score_cp": ply - 1, "pv_uci": []}
        assert before["ply"] < ply
        assert "result" not in decision
        assert "continuation" not in decision
    assert context["required_ply_sequence"] == list(range(len(game["positions"])))
    assert context["game"] == original
    assert game == original


def load_game_with_analysis(pgn):
    from chess_video.game import load_game

    game = load_game(pgn)
    for position in game["positions"]:
        position["analysis"] = {"score_cp": position["ply"], "pv_uci": []}
    return game


@pytest.mark.parametrize("provider", ["gemini", "openai", "vertex"])
def test_script_prompt_places_player_choice_before_actual_move(provider):
    settings = Settings(_env_file=None, llm_provider=provider)
    payload = _llm_payload([], settings, position_count=5)
    prompt = (
        payload["messages"][0]["content"] if provider == "openai"
        else payload["systemInstruction"]["parts"][0]["text"]
    )
    assert "A beat describes only that position and its actual preceding move." not in prompt
    for instruction in (
        "pre_move_position", "before the actual move", "candidate", "opponent reply",
        "consequence", "commitment phrase near the end", "move_anchor", "Routine own moves",
        "stock phrases", "later positions", "pre-move cues",
    ):
        assert instruction in prompt
    assert "Never describe a future move as already played." in prompt
    assert "Checkmate requires is_checkmate=true" in prompt


def test_player_decision_prompt_invalidates_old_script_cache():
    settings = Settings(_env_file=None)
    assert _llm_config(settings)["version"] == 3
    assert "prompt_sha256" in _llm_config(settings)
