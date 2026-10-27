import io
import re

import chess
import chess.pgn

from .models import Game, RESULTS, Script


MAX_PGN_BYTES = 1_000_000
MAX_PLIES = 2000
TOKEN = re.compile(
    r'\s+|\{[^}]*\}|;[^\n]*|\[[A-Za-z0-9_]+\s+"(?:[^"\\]|\\.)*"\s*\]'
    r'|\(|\)|\$\d+|\d+\.(?:\.\.)?|\.\.\.|1/2-1/2|1-0|0-1|\*'
    r'|(?:O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?[!?]{0,2}'
    r'|[!?]{1,2}'
)


def _validate_pgn_tokens(pgn):
    offset = depth = 0
    results = []
    ended = False
    while offset < len(pgn):
        match = TOKEN.match(pgn, offset)
        if match is None:
            raise ValueError(f'Unrecognized PGN token near character {offset}')
        token = match.group()
        offset = match.end()
        if token.isspace() or token.startswith(('{', ';', '[')):
            continue
        if ended:
            raise ValueError('Only one PGN game is allowed, with no trailing movetext')
        if token == '(':
            depth += 1
            if depth > 32:
                raise ValueError('PGN variation nesting exceeds 32')
        elif token == ')':
            depth -= 1
            if depth < 0:
                raise ValueError('Unbalanced PGN variation')
        elif token in RESULTS and depth == 0:
            results.append(token)
            ended = True
    if depth:
        raise ValueError('Unbalanced PGN variation')
    return results[-1] if results else None


def _position(board, ply, move=None):
    before_fen = board.fen()
    result = {
        'ply': ply, 'before_fen': before_fen, 'fen': before_fen,
        'san': '', 'uci': '', 'is_capture': False,
        'is_check': board.is_check(), 'is_checkmate': board.is_checkmate(),
        'is_castling': False, 'is_en_passant': False, 'promotion': None,
    }
    if move is not None:
        result.update(
            san=board.san(move), uci=move.uci(), is_capture=board.is_capture(move),
            is_castling=board.is_castling(move), is_en_passant=board.is_en_passant(move),
            promotion=chess.piece_symbol(move.promotion) if move.promotion else None,
        )
        board.push(move)
        result.update(fen=board.fen(), is_check=board.is_check(), is_checkmate=board.is_checkmate())
    king = board.king(board.turn)
    result['checked_king_square'] = chess.square_name(king) if board.is_check() and king is not None else None
    result['checking_squares'] = [chess.square_name(square) for square in board.checkers()]
    return result


def load_game(pgn: str) -> dict:
    if not isinstance(pgn, str) or not pgn.strip():
        raise ValueError('PGN must be nonempty text')
    if len(pgn.encode('utf-8')) > MAX_PGN_BYTES:
        raise ValueError('PGN exceeds 1000000 bytes')
    movetext_result = _validate_pgn_tokens(pgn)
    stream = io.StringIO(pgn)
    try:
        parsed = chess.pgn.read_game(stream)
        if parsed is None or parsed.errors:
            raise ValueError('Invalid PGN or illegal moves')
        if chess.pgn.read_game(stream) is not None:
            raise ValueError('Only one PGN game is allowed')
        if parsed.headers.get('Variant', 'Standard').lower() not in {'standard', 'chess', 'normal'}:
            raise ValueError('Only standard chess PGN is supported')
        board = parsed.board()
        if not board.is_valid() or board.chess960:
            raise ValueError('Invalid or unsupported initial FEN')
        declared = parsed.headers.get('Result', '*')
        if declared not in RESULTS or (movetext_result is not None and declared != movetext_result):
            raise ValueError('Invalid or inconsistent PGN result')
        initial_fen = board.fen()
        positions = [_position(board, 0)]
        for ply, move in enumerate(parsed.mainline_moves(), 1):
            if ply > MAX_PLIES:
                raise ValueError('PGN exceeds 2000 plies')
            if board.is_game_over() or move not in board.legal_moves:
                raise ValueError('Illegal move or move after terminal position')
            positions.append(_position(board, ply, move))
        if len(positions) < 2:
            raise ValueError('PGN must contain at least one legal move')
        actual = board.result()
        if actual != '*' and declared not in {'*', actual}:
            raise ValueError('PGN result contradicts terminal board')
        return {
            'headers': dict(parsed.headers), 'initial_fen': initial_fen,
            'positions': positions, 'result': actual if actual != '*' else declared,
        }
    except (ValueError, TypeError, IndexError, RecursionError) as error:
        raise ValueError(f'Invalid PGN: {error}') from error


def validate_game(game: dict) -> dict:
    validated = Game.model_validate(game).model_dump()
    try:
        board = chess.Board(validated['initial_fen'])
        if not board.is_valid():
            raise ValueError('Invalid initial board')
        for ply, position in enumerate(validated['positions']):
            if position['ply'] != ply:
                raise ValueError('Positions must contain consecutive plies from zero')
            if ply:
                move = chess.Move.from_uci(position['uci'])
                if board.is_game_over() or move not in board.legal_moves:
                    raise ValueError('Game contains an illegal move')
            else:
                move = None
            expected = _position(board, ply, move)
            optional_truth = {'checked_king_square', 'checking_squares'}
            compared = {key: value for key, value in expected.items()
                        if key not in optional_truth or key in game['positions'][ply]}
            if any(position[key] != value for key, value in compared.items()):
                raise ValueError(f'Game truth mismatch at ply {ply}')
        actual = board.result()
        if actual != '*' and validated['result'] != actual:
            raise ValueError('Result contradicts terminal board')
        header_result = validated['headers'].get('Result', '*')
        if header_result not in RESULTS or header_result not in {'*', validated['result']}:
            raise ValueError('Header result contradicts game result')
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f'Invalid game: {error}') from error
    return validated


def result_labels(game: dict) -> set[str]:
    result = game['result']
    labels = {result}
    if result == '1-0':
        labels.update({'White wins', 'White wins.', 'Белые победили', 'Победа белых'})
    elif result == '0-1':
        labels.update({'Black wins', 'Black wins.', 'Чёрные победили', 'Черные победили', 'Победа чёрных'})
    elif result == '1/2-1/2':
        labels.update({'Draw', 'Draw.', 'Ничья'})
    if game['positions'][-1]['is_checkmate']:
        labels.update({'Checkmate', 'Checkmate!', 'Мат', 'Мат!'})
    return labels


def _validate_claims(text, position, result, final, check_special=True):
    lower = text.casefold()
    hypothetical = re.search(r'\b(?:could|would|might|can|consider|considering|threat|threatens|possible|можно|могу|мог|угроза|угрожает)\b', lower)
    board = chess.Board(position['fen'])
    special_claims = [
        (r'\ben\s+passant\b|на проходе', 'is_en_passant', board.is_en_passant, 'en passant'),
        (r'\b(?:castle|castles|castled|castling)\b|рокиров', 'is_castling', board.is_castling, 'castling'),
        (r'\b(?:promote|promotes|promoted|promotion)\b|превращ', 'promotion', lambda move: bool(move.promotion), 'promotion'),
    ]
    for pattern, field, legal_test, label in special_claims if check_special else []:
        if re.search(pattern, lower):
            supported = any(legal_test(move) for move in board.legal_moves) if hypothetical else bool(position[field])
            if not supported:
                raise ValueError(f'Narration claims {label} without matching actual move or legal hypothetical option')
    mate = re.search(r'\b(?:checkmate|checkmated|мат)\b', lower)
    prospective = re.search(r'\b(?:threat|threatens|threatening|avoid|avoids|prevent|could|would|угроза|угрожает|избежать)\b', lower)
    if mate and not prospective and not position['is_checkmate']:
        raise ValueError('Narration or label claims checkmate absent actual mate')
    wins = [
        (r'\b(?:white wins|white has won|white won|белые победили|победа белых)\b', '1-0'),
        (r'\b(?:black wins|black has won|black won|ч[её]рные победили|победа ч[её]рных)\b', '0-1'),
        (r'\b(?:the game is a draw|drawn game|it is a draw|ничья)\b', '1/2-1/2'),
    ]
    for expression, claimed_result in wins:
        if re.search(expression, lower) and (not final or result != claimed_result):
            raise ValueError('Narration or label invents a terminal result')
    if re.search(r'\b(?:gives check|is in check|шах)[!.]?$', lower) and not position['is_check']:
        raise ValueError('Narration or label claims check absent actual check')


def validate_script(game: dict, script: dict) -> dict:
    game = validate_game(game)
    validated = Script.model_validate(script).model_dump(exclude_none=True)
    if [beat['ply'] for beat in validated['beats']] != list(range(len(game['positions']))):
        raise ValueError('Script needs exactly one ordered beat per ply, including initial zero')
    _validate_claims(validated['title'], game['positions'][-1], game['result'], True, check_special=False)
    for beat, position in zip(validated['beats'], game['positions']):
        final = beat['ply'] == len(game['positions']) - 1
        _validate_claims(beat['text'], position, game['result'], final)
        for cue in beat['cues']:
            if cue.get('label'):
                _validate_claims(cue['label'], position, game['result'], final)
            if cue['kind'] == 'result':
                if not final or game['result'] == '*' or cue['label'] not in result_labels(game):
                    raise ValueError('Result cue must match the actual final game result')
    return validated
