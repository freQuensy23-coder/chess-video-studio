from pathlib import Path

from chess_video.game import load_game
from chess_video.providers import _script_context
from chess_video.settings import Settings


def test_script_context_exposes_exact_opponent_fork_without_mutating_game():
    import copy

    pgn = (Path(__file__).parents[1] / 'examples/bishop-trap.pgn').read_text()
    game = load_game(pgn)
    original = copy.deepcopy(game)
    context = _script_context(game, Settings(_env_file=None, perspective='white'))
    forks = [item for item in context['selected_opponent_threats'] if item['ply'] == 14]
    assert forks
    assert {('f2', 'd1'), ('f2', 'h1')} <= {
        (arrow['from_square'], arrow['to_square']) for arrow in forks[0]['arrows']
    }
    assert context['perspective'] == 'white'
    assert context['game'] == original
    assert game == original


def test_opponent_threat_context_is_relative_to_requested_player():
    game = load_game('1. e4 e5 2. Nf3 Nc6 *')
    context = _script_context(game, Settings(_env_file=None, perspective='black'))
    assert context['perspective'] == 'black'
    assert not context['selected_opponent_threats']
    assert context['required_ply_sequence'] == list(range(len(game['positions'])))
