import copy
import io
from pathlib import Path

import chess
import chess.pgn
import pytest

from chess_video.game import load_game
from chess_video.tactics import select_key_threats

FIXTURE = Path(__file__).parents[1] / 'examples' / 'bishop-trap.pgn'


def mirrored_game(pgn):
    original = chess.pgn.read_game(io.StringIO(pgn))
    mirrored = chess.pgn.Game()
    mirrored.setup(original.board().mirror())
    node = mirrored
    for move in original.mainline_moves():
        node = node.add_variation(chess.Move(
            chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
            promotion=move.promotion))
    mirrored.headers['Result'] = '*'
    return load_game(str(mirrored))


def setup_game(fen, moves):
    return load_game(f'[SetUp "1"]\n[FEN "{fen}"]\n\n{moves} *')


def arrow_pairs(threat):
    return {(arrow['from_square'], arrow['to_square']) for arrow in threat['arrows']}


@pytest.mark.parametrize('perspective', ['white', 'black'])
def test_actual_bishop_trap_fork_selects_both_rooks_for_player(perspective):
    pgn = FIXTURE.read_text()
    game = load_game(pgn) if perspective == 'white' else mirrored_game(pgn)
    threats = select_key_threats(game, perspective)
    fork = next(threat for threat in threats if threat['ply'] == 14)
    assert fork['kind'] == 'fork'
    if perspective == 'white':
        assert arrow_pairs(fork) == {('f2', 'd1'), ('f2', 'h1')}
        assert set(fork['focus_squares']) == {'f2', 'd1', 'h1'}
    else:
        assert arrow_pairs(fork) == {('f7', 'd8'), ('f7', 'h8')}
        assert set(fork['focus_squares']) == {'f7', 'd8', 'h8'}
    assert len(threats) <= 4
    assert all(b['ply'] - a['ply'] >= 4 for a, b in zip(threats, threats[1:]))


def test_opening_does_not_decorate_routine_development_or_pawn_pushes():
    game = load_game('1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. Nc3 d6 *')
    assert select_key_threats(game, 'white') == []
    assert select_key_threats(game, 'black') == []


def test_initial_knight_attack_on_pawn_does_not_invent_future_fork():
    game = load_game(FIXTURE.read_text())
    assert not any(threat['ply'] == 0 for threat in select_key_threats(game, 'white'))


def test_meaningful_initial_fork_is_eligible():
    game = setup_game('6k1/8/8/8/8/8/5n2/2KR3R w - - 0 1', '1. Rd2')
    threats = select_key_threats(game, 'white')
    assert len(threats) == 1
    assert threats[0]['ply'] == 0
    assert arrow_pairs(threats[0]) == {('f2', 'd1'), ('f2', 'h1')}


def test_absolutely_pinned_knight_does_not_emit_false_fork():
    game = setup_game('5k2/8/8/8/8/8/5n2/2KR1R1R w - - 0 1', '1. Rd2')
    assert select_key_threats(game, 'white') == []


def test_new_safe_minor_piece_attack_on_queen_is_significant():
    game = setup_game('7k/8/5n2/8/8/4Q3/8/K7 b - - 0 1', '1... Ng4 2. Qd4')
    threats = select_key_threats(game, 'white')
    assert len(threats) == 1
    assert threats[0]['ply'] == 1
    assert threats[0]['kind'] == 'attack'
    assert arrow_pairs(threats[0]) == {('g4', 'e3')}


def test_new_safe_pawn_attack_on_rook_is_significant():
    game = setup_game('7k/8/8/4p3/8/3R4/8/K7 b - - 0 1', '1... e4 2. Rd4')
    threats = select_key_threats(game, 'white')
    assert len(threats) == 1
    assert threats[0]['kind'] == 'attack'
    assert arrow_pairs(threats[0]) == {('e4', 'd3')}


def test_king_and_rook_fork_contains_only_current_direct_attacks():
    game = setup_game('k7/8/8/8/6n1/8/8/3K3R b - - 0 1', '1... Nf2+ 2. Ke2')
    fork = next(threat for threat in select_key_threats(game, 'white') if threat['ply'] == 1)
    assert fork['kind'] == 'fork'
    assert arrow_pairs(fork) == {('f2', 'd1'), ('f2', 'h1')}


def test_safe_pawn_fork_on_two_minor_pieces_is_eligible():
    game = setup_game('7k/8/8/4p3/8/3N1N2/8/K7 b - - 0 1', '1... e4 2. Nd2')
    threats = select_key_threats(game, 'white')
    assert len(threats) == 1
    assert threats[0]['kind'] == 'fork'
    assert arrow_pairs(threats[0]) == {('e4', 'd3'), ('e4', 'f3')}


def test_capturable_forking_piece_does_not_create_a_spurious_emergency():
    game = setup_game('7k/8/8/4p3/8/3N1B2/8/K7 b - - 0 1', '1... e4 2. Bxe4')
    assert select_key_threats(game, 'white') == []


def test_repeated_same_piece_targets_are_not_reannounced():
    game = setup_game('7k/8/5n2/8/8/4Q3/8/K7 b - - 0 1',
                      '1... Ng4 2. Ka2 Nf6 3. Ka1 Ng4 4. Qd4')
    threats = select_key_threats(game, 'white')
    assert [threat['ply'] for threat in threats] == [1]


def test_selector_is_deterministic_and_does_not_mutate_game():
    game = load_game(FIXTURE.read_text())
    original = copy.deepcopy(game)
    first = select_key_threats(game, 'white')
    assert first == select_key_threats(game, 'white')
    assert game == original
    first[0]['focus_squares'].clear()
    assert select_key_threats(game, 'white') != first


def test_perspective_must_be_explicit_supported_color():
    with pytest.raises(ValueError, match='perspective'):
        select_key_threats(load_game('1. e4 *'), 'sideways')
