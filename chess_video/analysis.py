from copy import deepcopy
import math
import shutil

import chess
import chess.engine

from .game import validate_game


def analyze_game(game: dict, engine_path: str | None = None, *, time_limit: float = 0.08) -> dict:
    validate_game(game)
    result = deepcopy(game)
    result['analysis'] = {'available': False, 'engine': None}
    for position in result['positions']:
        position.pop('analysis', None)
    if (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
            or not math.isfinite(time_limit) or not 0.01 <= time_limit <= 2):
        raise ValueError('Engine time_limit must be finite and between 0.01 and 2 seconds')
    if time_limit * len(game['positions']) > 300:
        raise ValueError('Requested engine analysis exceeds the 300 second budget')

    def unavailable(reason):
        result['analysis']['reason'] = reason
        return result

    if engine_path is None:
        return unavailable('Optional analysis unavailable: no Stockfish engine configured')
    if not isinstance(engine_path, str) or not engine_path.strip():
        raise ValueError('Engine path must be a nonempty string')
    executable = shutil.which(engine_path)
    if executable is None:
        return unavailable('Optional analysis unavailable: configured Stockfish is not installed')
    evaluations = []
    try:
        with chess.engine.SimpleEngine.popen_uci(executable) as engine:
            name = engine.id.get('name', '')
            if 'stockfish' not in name.casefold():
                return unavailable('Configured engine does not identify as Stockfish')
            for position in game['positions']:
                board = chess.Board(position['fen'])
                info = engine.analyse(board, chess.engine.Limit(time=time_limit))
                if 'score' not in info:
                    raise ValueError('Stockfish omitted its evaluation')
                score = info['score'].white()
                pv_board = board.copy()
                pv_uci, pv_san = [], []
                for move in info.get('pv', []):
                    if move not in pv_board.legal_moves:
                        raise ValueError('Stockfish returned an illegal principal variation')
                    pv_uci.append(move.uci())
                    pv_san.append(pv_board.san(move))
                    pv_board.push(move)
                evaluations.append({
                    'perspective': 'white', 'score_cp': score.score(), 'mate': score.mate(),
                    'pv_uci': pv_uci, 'pv_san': pv_san, 'depth': info.get('depth'),
                })
    except (OSError, chess.engine.EngineError, TimeoutError, ValueError) as error:
        return unavailable(f'Optional Stockfish analysis failed: {error}')
    for position, evaluation in zip(result['positions'], evaluations):
        position['analysis'] = evaluation
    result['analysis'] = {'available': True, 'engine': name, 'perspective': 'white'}
    return result
