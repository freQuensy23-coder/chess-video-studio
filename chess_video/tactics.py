import chess

from .game import validate_game

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10}
MAX_THREATS = 4
MIN_PLY_GAP = 4


def _capture_board(board, color):
    result = board.copy(stack=False)
    if result.turn != color:
        result.turn = color
        result.ep_square = None
    return result


def _position_threats(board, player, identities):
    if board.is_checkmate() or board.is_stalemate():
        return []
    opponent = not player
    capture_board = _capture_board(board, opponent)
    player_board = _capture_board(board, player)
    capturable_attackers = {move.to_square for move in player_board.generate_legal_captures()}
    targets_by_attacker = {}
    for move in capture_board.generate_legal_captures():
        target = board.piece_at(move.to_square)
        if target is not None and target.color == player and target.piece_type not in {chess.PAWN, chess.KING}:
            targets_by_attacker.setdefault(move.from_square, set()).add(move.to_square)
    if board.turn == player and board.is_check():
        king = board.king(player)
        for attacker in board.checkers():
            if not board.is_pinned(opponent, attacker) or king in board.pin(opponent, attacker):
                targets_by_attacker.setdefault(attacker, set()).add(king)
    threats = []
    for attacker, targets in sorted(targets_by_attacker.items()):
        piece = board.piece_at(attacker)
        if piece is None or piece.color != opponent or attacker in capturable_attackers:
            continue
        value = PIECE_VALUES[piece.piece_type]
        meaningful = [target for target in sorted(targets)
                      if PIECE_VALUES[board.piece_type_at(target)] > value
                      or not board.is_attacked_by(player, target)]
        major_targets = [target for target in meaningful
                         if board.piece_type_at(target) in {chess.ROOK, chess.QUEEN, chess.KING}]
        if len(meaningful) >= 2:
            kind = 'fork'
            selected_targets = meaningful
            score = (100 if major_targets else 60) + sum(PIECE_VALUES[board.piece_type_at(target)] for target in meaningful)
        else:
            serious = [target for target in major_targets
                       if board.piece_type_at(target) != chess.KING
                       and PIECE_VALUES[board.piece_type_at(target)] > value
                       and (PIECE_VALUES[board.piece_type_at(target)] - value >= 3
                            or not board.is_attacked_by(player, target))]
            if not serious:
                continue
            target = min(serious, key=lambda square: (-PIECE_VALUES[board.piece_type_at(square)], square))
            kind = 'attack'
            selected_targets = [target]
            score = PIECE_VALUES[board.piece_type_at(target)] - value
        signature = (kind, identities[attacker], tuple(sorted(identities[target] for target in selected_targets)))
        attacker_name = chess.square_name(attacker)
        arrows = [{'from_square': attacker_name, 'to_square': chess.square_name(target)}
                  for target in selected_targets]
        threats.append((score, signature, {'kind': kind, 'arrows': arrows,
                                          'focus_squares': [attacker_name] + [arrow['to_square'] for arrow in arrows]}))
    return threats


def _advance(board, identities, move):
    identity = identities.pop(move.from_square)
    identities.pop(move.to_square, None)
    if board.is_en_passant(move):
        identities.pop(move.to_square - 8 if board.turn else move.to_square + 8, None)
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        identities[rook_to] = identities.pop(rook_from)
    identities[move.to_square] = identity
    board.push(move)


def select_key_threats(game, perspective) -> list[dict]:
    if perspective not in {'white', 'black'}:
        raise ValueError('Threat perspective must be white or black')
    game = validate_game(game)
    player = perspective == 'white'
    board = chess.Board(game['initial_fen'])
    identities = {square: square for square in board.piece_map()}
    seen = set()
    previous = set()
    candidates = []
    for position in game['positions']:
        ply = position['ply']
        opponent_move = False
        if ply:
            opponent_move = board.turn != player
            _advance(board, identities, chess.Move.from_uci(position['uci']))
        threats = _position_threats(board, player, identities)
        current = {signature for _, signature, _ in threats}
        if ply == 0 or opponent_move:
            for score, signature, threat in threats:
                if signature not in previous and signature not in seen:
                    candidates.append((score, {'ply': ply, **threat}))
                    seen.add(signature)
        previous = current
    selected = []
    for _, threat in sorted(candidates, key=lambda entry: (-entry[0], entry[1]['ply'], entry[1]['focus_squares'])):
        if all(abs(threat['ply'] - other['ply']) >= MIN_PLY_GAP for other in selected):
            selected.append(threat)
            if len(selected) == MAX_THREATS:
                break
    return sorted(selected, key=lambda threat: threat['ply'])
