import pytest

from chess_video.game import load_game, validate_script


def script_for(game):
    return {
        "title": "A difficult position",
        "language": "en",
        "perspective": "white",
        "beats": [
            {"ply": position["ply"], "text": "I need to think.", "emotion": "thinking", "cues": []}
            for position in game["positions"]
        ],
    }


def test_regular_pawn_capture_cannot_be_narrated_as_en_passant():
    game = load_game("1. e4 d5 2. exd5 *")
    script = script_for(game)
    script["beats"][3]["text"] = "I take en passant."
    with pytest.raises(ValueError, match="en passant"):
        validate_script(game, script)


def test_actual_en_passant_narration_is_allowed():
    game = load_game("1. e4 a5 2. e5 d5 3. exd6 *")
    script = script_for(game)
    script["beats"][5]["text"] = "I take en passant."
    assert validate_script(game, script)["beats"][5]["text"] == "I take en passant."


def test_hypothetical_en_passant_requires_a_legal_option_on_current_board():
    game = load_game("1. e4 a5 2. e5 d5 3. exd6 *")
    script = script_for(game)
    script["beats"][4]["text"] = "I could take en passant here."
    assert validate_script(game, script)
    script["beats"][0]["text"] = "I could take en passant here."
    with pytest.raises(ValueError, match="en passant"):
        validate_script(game, script)


def test_false_castling_or_promotion_claim_is_rejected():
    game = load_game("1. e4 e5 *")
    for text in ["I castle kingside.", "They promote to a queen."]:
        script = script_for(game)
        script["beats"][1]["text"] = text
        with pytest.raises(ValueError):
            validate_script(game, script)
