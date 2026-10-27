from pathlib import Path

import chess
import pytest

from chess_video.analysis import analyze_game
from chess_video.game import load_game, validate_script
from chess_video.models import narration_text

SHORT = '1. e4 e5 2. Nf3 Nc6 *'
MATE = '1. f3 e5 2. g4 Qh4# 0-1'


def script_for(game, language='en'):
    return {
        'title': 'An original chess story',
        'language': language,
        'perspective': 'white',
        'beats': [
            {'ply': p['ply'], 'text': f"Position {p['ply']} now.",
             'emotion': 'neutral', 'move_anchor': None, 'cues': []}
            for p in game['positions']
        ],
    }


def test_legal_replay_truth():
    game = load_game(SHORT)
    assert len(game['positions']) == 5
    board = chess.Board(game['initial_fen'])
    for position in game['positions'][1:]:
        assert position['before_fen'] == board.fen()
        move = chess.Move.from_uci(position['uci'])
        assert move in board.legal_moves
        assert board.san(move) == position['san']
        board.push(move)
        assert board.fen() == position['fen']
    assert game['positions'][0]['san'] == ''
    assert game['result'] == '*'


@pytest.mark.parametrize('pgn', [
    '', 'not a game', '1. e5 *', '1. e4 e5 2. Bh6 *',
    '[FEN "broken"]\n[SetUp "1"]\n\n1. e4 *',
    '1. e4 *\n\n1. d4 *', '1. e4 e5 * trailing nonsense',
    '[Result "banana"]\n\n1. e4 *',
    '[Variant "Atomic"]\n\n1. e4 *',
    '1. f3 e5 2. g4 Qh4# 1-0',
    'x' * 1_000_001,
])
def test_invalid_pgn_rejected(pgn):
    with pytest.raises(ValueError):
        load_game(pgn)


@pytest.mark.parametrize(('fixture', 'flag'), [
    ('castling.pgn', 'is_castling'),
    ('en-passant.pgn', 'is_en_passant'),
    ('promotion.pgn', 'promotion'),
    ('legal-mate.pgn', 'is_checkmate'),
])
def test_special_move_fixtures(fixture, flag):
    path = Path(__file__).parents[1] / 'examples' / fixture
    game = load_game(path.read_text())
    assert any(position[flag] for position in game['positions'][1:])
    if flag == 'is_en_passant':
        assert any(p['is_capture'] for p in game['positions'] if p[flag])


@pytest.mark.parametrize('piece', ['Q', 'R', 'B', 'N'])
def test_every_promotion_piece(piece):
    game = load_game('[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/7p/4K3 w - - 0 1"]\n\n'
                     f'1. a8={piece} *')
    assert game['positions'][-1]['promotion'] == piece.lower()


def test_queenside_and_black_kingside_castling():
    game = load_game('[SetUp "1"]\n[FEN "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"]\n\n'
                     '1. O-O-O O-O *')
    assert all(position['is_castling'] for position in game['positions'][1:])


def test_exact_bishop_trap_reference_ends_in_legal_mate():
    path = Path(__file__).parents[1] / 'examples' / 'bishop-trap.pgn'
    game = load_game(path.read_text())
    assert game['initial_fen'] == 'r1b2rk1/pp1p1ppp/2p5/4P3/2B1nN2/7P/PPP2PP1/2KR3R w - - 0 1'
    assert game['headers']['Source'] == 'https://youtu.be/cG-0ItfHk8E'
    assert game['positions'][-1]['san'] == 'Re8#'
    assert game['positions'][-1]['is_checkmate'] is True
    assert game['result'] == '1-0'


def test_draw_is_not_checkmate():
    game = load_game('[Result "1/2-1/2"]\n\n1. e4 e5 1/2-1/2')
    assert game['result'] == '1/2-1/2'
    assert not game['positions'][-1]['is_checkmate']


def test_actual_stalemate_is_a_draw_without_mate():
    game = load_game('[SetUp "1"]\n[FEN "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"]\n\n1. Qg6 *')
    assert game['result'] == '1/2-1/2'
    assert game['positions'][-1]['is_check'] is False
    assert game['positions'][-1]['is_checkmate'] is False


def test_board_terminal_result_is_inferred():
    game = load_game('1. f3 e5 2. g4 Qh4# *')
    assert game['result'] == '0-1'


def test_script_roundtrip_and_unicode():
    game = load_game(SHORT)
    script = script_for(game, 'ru')
    script['beats'][0]['text'] = 'Начинаем: конь ♞ и пешка!'
    script['beats'][0]['cues'] = [
        {'kind': 'highlight', 'anchor': 'конь ♞', 'square': 'g1'}
    ]
    validated = validate_script(game, script)
    assert validated['beats'][0]['text'] == script['beats'][0]['text']
    assert narration_text(validated) == '\n\n'.join(b['text'] for b in script['beats'])


@pytest.mark.parametrize('change', [
    lambda s: s['beats'].pop(),
    lambda s: s['beats'].reverse(),
    lambda s: s['beats'][1].update(ply=0),
    lambda s: s['beats'][0].update(text=' '),
    lambda s: s['beats'][0].update(text='x' * 4001),
    lambda s: s['beats'][1].update(move_anchor='absent'),
    lambda s: s.update(language='de'),
    lambda s: s['beats'][0].update(emotion='angry'),
    lambda s: s.update(unexpected=True),
])
def test_invalid_script_rejected(change):
    game = load_game(SHORT)
    script = script_for(game)
    change(script)
    with pytest.raises(ValueError):
        validate_script(game, script)


@pytest.mark.parametrize('cue', [
    {'kind': 'highlight', 'anchor': 'Position', 'square': 'i9'},
    {'kind': 'arrow', 'anchor': 'Position', 'from_square': 'a1'},
    {'kind': 'highlight', 'anchor': 'missing', 'square': 'a1'},
    {'kind': 'highlight', 'anchor': 'Position', 'square': 'a1', 'color': 'red;bad'},
    {'kind': 'zoom', 'anchor': 'Position', 'square': 'a1', 'scale': float('nan')},
    {'kind': 'cursor', 'anchor': 'Position', 'square': 'a1', 'duration': -1},
    {'kind': 'face', 'anchor': 'Position', 'label': 'angry'},
    {'kind': 'result', 'anchor': 'Position', 'label': 'Checkmate'},
])
def test_invalid_cues_rejected(cue):
    game = load_game(SHORT)
    script = script_for(game)
    script['beats'][0]['cues'] = [cue]
    with pytest.raises(ValueError):
        validate_script(game, script)


@pytest.mark.parametrize('text', ['Checkmate!', 'White wins.', 'The game is a draw.', 'Мат!', 'Белые победили.'])
def test_false_terminal_claims_rejected(text):
    game = load_game(SHORT)
    script = script_for(game)
    script['beats'][-1]['text'] = text
    with pytest.raises(ValueError):
        validate_script(game, script)


def test_terminal_result_cue_requires_correct_winner():
    game = load_game(MATE)
    script = script_for(game)
    script['beats'][-1].update(text='Checkmate! Black wins.', cues=[
        {'kind': 'result', 'anchor': 'Checkmate', 'label': 'Checkmate'}
    ])
    assert validate_script(game, script)
    script['beats'][-1]['cues'][0]['label'] = 'White wins'
    with pytest.raises(ValueError):
        validate_script(game, script)


def test_forged_game_position_rejected():
    game = load_game(SHORT)
    game['positions'][1]['uci'] = 'e2e5'
    with pytest.raises(ValueError):
        validate_script(game, script_for(game))


def test_optional_engine_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr('chess_video.analysis.shutil.which', lambda _: None)
    result = analyze_game(load_game(SHORT))
    assert result['analysis']['available'] is False
    assert len(result['positions']) == 5
    assert 'configured' in result['analysis']['reason'].lower()
    assert all(position.get('analysis') is None for position in result['positions'])


def test_real_engine_information_is_preserved(monkeypatch):
    class Engine:
        id = {'name': 'Stockfish test protocol'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def analyse(self, board, limit):
            return {'score': chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE),
                    'pv': [next(iter(board.legal_moves))], 'depth': 8}

    monkeypatch.setattr('chess_video.analysis.shutil.which', lambda _: '/installed/stockfish')
    monkeypatch.setattr('chess.engine.SimpleEngine.popen_uci', lambda path: Engine())
    original = load_game(SHORT)
    result = analyze_game(original, engine_path='stockfish')
    assert result['analysis']['available'] is True
    assert len(result['positions']) == 5
    assert result['positions'][0]['analysis']['score_cp'] == 35
    assert result['positions'][0]['analysis']['pv_uci']
    assert 'analysis' not in original['positions'][0]


def test_check_attacker_truth():
    game = load_game(MATE)
    final = game['positions'][-1]
    assert final['checked_king_square'] == 'e1'
    assert final['checking_squares'] == ['h4']


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_engine_time_limit_is_bounded(seconds):
    with pytest.raises(ValueError):
        analyze_game(load_game(SHORT), time_limit=seconds)
