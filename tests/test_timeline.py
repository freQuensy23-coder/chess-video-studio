import copy
from pathlib import Path

import chess
import pytest

from chess_video.game import load_game
from chess_video.models import narration_text, validate_speech
from chess_video.timeline import compile_timeline


def inputs(pgn='1. e4 e5 *'):
    game = load_game(pgn)
    script = {'title': 'Chess', 'language': 'ru', 'perspective': 'black', 'beats': [
        {'ply': p['ply'], 'text': f"Ход {p['ply']}: туда, туда ♞!", 'emotion': 'thinking',
         'move_anchor': 'туда, туда', 'cues': []}
        for p in game['positions']
    ]}
    text = narration_text(script)
    speech = {'audio_path': 'speech.mp3', 'duration': len(text) * 0.1,
              'text': text, 'characters': list(text),
              'character_start_times_seconds': [i * 0.1 for i in range(len(text))],
              'character_end_times_seconds': [(i + 1) * 0.1 for i in range(len(text))]}
    return game, script, speech


def test_every_move_aligned_to_unicode_phrase():
    game, script, speech = inputs()
    timeline = compile_timeline(game, script, speech)
    moves = [e for e in timeline['events'] if e['kind'] == 'move']
    assert [e['uci'] for e in moves] == ['e2e4', 'e7e5']
    text = narration_text(script)
    offset = len(script['beats'][0]['text']) + 2
    expected = speech['character_start_times_seconds'][text.index('туда, туда', offset)]
    assert moves[0]['start'] == expected
    assert all(e['end'] - e['start'] >= 0.35 for e in moves)
    assert all(a['end'] <= b['start'] for a, b in zip(moves, moves[1:]))
    assert timeline['orientation'] == 'black'
    assert timeline['duration'] >= speech['duration'] + 1
    assert len([e for e in timeline['events'] if e['kind'] == 'face']) == 3
    assert not any(e['kind'] == 'result' for e in timeline['events'])


def test_repeated_anchor_uses_first_exact_occurrence_in_own_beat():
    game, script, speech = inputs()
    script['beats'][1]['cues'] = [
        {'kind': 'highlight', 'anchor': 'туда', 'square': 'e4'},
        {'kind': 'arrow', 'anchor': 'туда, туда', 'from_square': 'e2', 'to_square': 'e4'},
    ]
    timeline = compile_timeline(game, script, speech)
    cues = [e for e in timeline['events'] if e['kind'] in {'highlight', 'arrow'}]
    move = next(e for e in timeline['events'] if e['kind'] == 'move')
    assert len(cues) == 2
    assert all(e['start'] == move['start'] for e in cues)
    assert all(e['end'] > e['start'] for e in cues)


def test_clear_ends_persistent_annotations():
    game, script, speech = inputs()
    script['beats'][0]['cues'] = [
        {'kind': 'highlight', 'anchor': 'Ход', 'square': 'e2'},
        {'kind': 'clear', 'anchor': '♞'},
    ]
    timeline = compile_timeline(game, script, speech)
    highlight = next(e for e in timeline['events'] if e['kind'] == 'highlight')
    clear = next(e for e in timeline['events'] if e['kind'] == 'clear')
    assert highlight['end'] == clear['start']


def test_explicit_duration_is_respected_until_move():
    game, script, speech = inputs()
    script['beats'][0]['cues'] = [
        {'kind': 'cursor', 'anchor': 'Ход', 'square': 'e2', 'duration': 0.2}
    ]
    event = next(e for e in compile_timeline(game, script, speech)['events'] if e['kind'] == 'cursor')
    assert event['end'] == pytest.approx(0.2)


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(text=s['text'] + '!'),
    lambda s: s['characters'].pop(),
    lambda s: s['character_start_times_seconds'].pop(),
    lambda s: s['character_end_times_seconds'].pop(),
    lambda s: s['character_start_times_seconds'].__setitem__(0, -0.1),
    lambda s: s['character_start_times_seconds'].__setitem__(0, float('nan')),
    lambda s: s['character_end_times_seconds'].__setitem__(0, float('inf')),
    lambda s: s['character_start_times_seconds'].__setitem__(1, 5),
    lambda s: s['character_end_times_seconds'].__setitem__(2, 0),
    lambda s: s['character_end_times_seconds'].__setitem__(-1, s['duration'] + 1),
    lambda s: s['characters'].__setitem__(0, 'XX'),
    lambda s: s.update(duration=-1),
])
def test_invalid_speech_rejected(mutate):
    game, script, speech = inputs()
    mutate(speech)
    with pytest.raises(ValueError):
        compile_timeline(game, script, speech)


def test_validate_speech_public_contract():
    _, script, speech = inputs()
    assert validate_speech(script, speech)['characters'] == list(narration_text(script))


def test_missing_move_anchor_uses_beat_onset():
    game, script, speech = inputs()
    script['beats'][1]['move_anchor'] = None
    timeline = compile_timeline(game, script, speech)
    first_move = next(e for e in timeline['events'] if e['kind'] == 'move')
    assert first_move['start'] == speech['character_start_times_seconds'][len(script['beats'][0]['text']) + 2]


def test_impossibly_close_moves_fail_explicitly():
    game, script, speech = inputs()
    speech['character_start_times_seconds'] = [i * 0.0001 for i in range(len(speech['text']))]
    speech['character_end_times_seconds'] = [(i + 1) * 0.0001 for i in range(len(speech['text']))]
    with pytest.raises(ValueError, match='overlap|synchron|close'):
        compile_timeline(game, script, speech)


@pytest.mark.parametrize(('width', 'height', 'fps'), [(607, 1080, 30), (1080, 608, 30), (608, 1080, 0)])
def test_invalid_render_dimensions(width, height, fps):
    with pytest.raises(ValueError):
        compile_timeline(*inputs(), width=width, height=height, fps=fps)


@pytest.mark.parametrize('pgn', ['1. f3 e5 2. g4 Qh4# 0-1', '1. e4 e5 1/2-1/2', '1. e4 e5 1-0'])
def test_only_actual_terminal_result(pgn):
    game, script, speech = inputs(pgn)
    timeline = compile_timeline(game, script, speech)
    results = [e for e in timeline['events'] if e['kind'] == 'result']
    assert len(results) == 1
    assert results[0]['end'] == timeline['duration']
    assert results[0]['start'] >= max(e['end'] for e in timeline['events'] if e['kind'] == 'move')
    if game['result'] == '1/2-1/2':
        assert 'mate' not in results[0]['label'].lower()


@pytest.mark.parametrize('fixture', ['castling.pgn', 'en-passant.pgn', 'promotion.pgn'])
def test_special_moves_survive_timeline(fixture):
    from pathlib import Path

    pgn = (Path(__file__).parents[1] / 'examples' / fixture).read_text()
    game, script, speech = inputs(pgn)
    moves = [e for e in compile_timeline(game, script, speech)['events'] if e['kind'] == 'move']
    assert len(moves) == len(game['positions']) - 1
    for event, position in zip(moves, game['positions'][1:]):
        assert all(event[key] == value for key, value in position.items())


@pytest.mark.parametrize('result', ['1-0', '0-1', '1/2-1/2', '*'])
def test_timeline_preserves_declared_result_and_language(result):
    game, script, speech = inputs(f'1. e4 e5 {result}')
    timeline = compile_timeline(game, script, speech)
    assert timeline['game_result'] == result
    assert timeline['language'] == 'ru'


@pytest.mark.parametrize('cue_duration', [None, 0.1])
def test_result_at_final_move_anchor_waits_for_completed_animation(cue_duration):
    game, script, speech = inputs('1. f3 e5 2. g4 Qh4# 0-1')
    cue = {'kind': 'result', 'anchor': 'туда, туда', 'label': 'Мат'}
    if cue_duration is not None:
        cue['duration'] = cue_duration
    script['beats'][-1]['cues'] = [
        cue,
        {'kind': 'highlight', 'anchor': 'туда, туда', 'square': 'h4'},
    ]
    timeline = compile_timeline(game, script, speech)
    last_move = [event for event in timeline['events'] if event['kind'] == 'move'][-1]
    result = next(event for event in timeline['events'] if event['kind'] == 'result')
    highlight = next(event for event in timeline['events'] if event['kind'] == 'highlight')
    assert result['start'] == last_move['end']
    assert highlight['start'] == last_move['start']
    if cue_duration is not None:
        assert result['end'] == pytest.approx(result['start'] + cue_duration)


@pytest.mark.parametrize('pgn', ['1. e4 e5 1-0', '1. e4 e5 0-1', '1. e4 e5 1/2-1/2'])
def test_compiled_declared_result_is_renderable_without_terminal_board(pgn):
    from chess_video.renderer import Renderer

    data = compile_timeline(*inputs(pgn))
    data['audio_path'] = None
    result = next(event for event in data['events'] if event['kind'] == 'result')
    with Renderer(data) as renderer:
        assert not renderer.board_at(result['start']).is_game_over()
        assert renderer.result_at(result['start']) == result['label']


def test_compiled_mate_is_not_visible_during_final_move_animation():
    from chess_video.renderer import Renderer

    game, script, speech = inputs('1. f3 e5 2. g4 Qh4# 0-1')
    script['beats'][-1]['cues'] = [
        {'kind': 'result', 'anchor': 'туда, туда', 'label': 'Мат'},
    ]
    data = compile_timeline(game, script, speech)
    data['audio_path'] = None
    final_move = [event for event in data['events'] if event['kind'] == 'move'][-1]
    with Renderer(data) as renderer:
        assert renderer.result_at(final_move['start']) is None
        assert renderer.result_at(final_move['end']) == 'МАТ · ЧЁРНЫЕ ПОБЕДИЛИ'


def bishop_trap_inputs():
    game, script, speech = inputs((Path(__file__).parents[1] / 'examples' / 'bishop-trap.pgn').read_text())
    script['perspective'] = 'white'
    return game, script, speech


def automatic_fork(timeline):
    return [event for event in timeline['events']
            if event.get('tactical_group') == 'opponent-threat-14']


def test_opponent_fork_arrows_are_simultaneous_after_actual_move_before_promotion():
    game, script, speech = bishop_trap_inputs()
    timeline = compile_timeline(game, script, speech)
    group = automatic_fork(timeline)
    arrows = [event for event in group if event['kind'] == 'arrow']
    assert {(event['from_square'], event['to_square']) for event in arrows} == {('f2', 'd1'), ('f2', 'h1')}
    assert len({(event['start'], event['end']) for event in group}) == 1
    moves = [event for event in timeline['events'] if event['kind'] == 'move']
    fork_move, promotion = moves[13:15]
    for event in group:
        assert event['automatic'] is True
        assert set(event['focus_squares']) == {'f2', 'd1', 'h1'}
        assert event['start'] == pytest.approx(fork_move['end'] + 0.08)
        assert event['end'] <= promotion['start']
        assert 1.0 <= event['end'] - event['start'] <= 2.5
        assert event['color'] == '#ef7858'
    assert timeline['orientation'] == timeline['player_color'] == 'white'


def test_generated_threats_are_legal_on_every_visible_board_without_lookahead():
    game, script, speech = bishop_trap_inputs()
    timeline = compile_timeline(game, script, speech)
    generated = [event for event in timeline['events']
                 if event.get('automatic') and event['kind'] == 'arrow']
    assert generated
    moves = [event for event in timeline['events'] if event['kind'] == 'move']
    for event in generated:
        for moment in (event['start'], (event['start'] + event['end']) / 2,
                       event['end'] - 0.000001):
            board = chess.Board(timeline['initial_fen'])
            for move in moves:
                if move['end'] <= moment:
                    board.push_uci(move['uci'])
            attacker = chess.parse_square(event['from_square'])
            target = chess.parse_square(event['to_square'])
            assert board.color_at(attacker) == chess.BLACK
            assert board.color_at(target) == chess.WHITE
            assert target in board.attacks(attacker)
            assert not any(move['start'] < moment < move['end'] for move in moves)


def test_auto_enhancement_preserves_all_audio_anchors_duration_and_inputs():
    game, script, speech = bishop_trap_inputs()
    original = copy.deepcopy((game, script, speech))
    timeline = compile_timeline(game, script, speech)
    assert (game, script, speech) == original
    assert timeline['audio_path'] == speech['audio_path']
    assert timeline['duration'] == speech['duration'] + 1.5
    offset = len(script['beats'][0]['text']) + 2
    moves = [event for event in timeline['events'] if event['kind'] == 'move']
    for move, beat, position in zip(moves, script['beats'][1:], game['positions'][1:]):
        anchor = offset + beat['text'].index(beat['move_anchor'])
        assert move['start'] == speech['character_start_times_seconds'][anchor]
        assert move['uci'] == position['uci']
        assert move['fen'] == position['fen']
        offset += len(beat['text']) + 2


def test_explicit_clear_ends_entire_generated_fork_group():
    game, script, speech = bishop_trap_inputs()
    script['beats'][14]['cues'] = [{'kind': 'clear', 'anchor': '♞'}]
    for field in ('character_start_times_seconds', 'character_end_times_seconds'):
        speech[field] = [time * 2 for time in speech[field]]
    speech['duration'] *= 2
    timeline = compile_timeline(game, script, speech)
    clear = next(event for event in timeline['events'] if event['kind'] == 'clear')
    group = automatic_fork(timeline)
    assert group
    assert all(event['end'] == clear['start'] for event in group)


def test_too_short_fork_window_is_skipped_without_shifting_moves_or_voice():
    game, script, speech = bishop_trap_inputs()
    index = sum(len(beat['text']) + 2 for beat in script['beats'][:14])
    cut = index + script['beats'][14]['text'].index('туда, туда')
    next_index = sum(len(beat['text']) + 2 for beat in script['beats'][:15])
    next_anchor = next_index + script['beats'][15]['text'].index('туда, туда')
    original_gap = speech['character_start_times_seconds'][next_anchor] - speech['character_start_times_seconds'][cut]
    removed = original_gap - 0.7
    for field in ('character_start_times_seconds', 'character_end_times_seconds'):
        origin = speech['character_start_times_seconds'][cut]
        speech[field] = [origin + (time - origin) * 0.7 / original_gap if cut <= i < next_anchor
                         else time - removed if i >= next_anchor else time
                         for i, time in enumerate(speech[field])]
    speech['duration'] -= removed
    timeline = compile_timeline(game, script, speech)
    assert automatic_fork(timeline) == []
    promotion = next(event for event in timeline['events'] if event['kind'] == 'move' and event['ply'] == 15)
    assert promotion['start'] == speech['character_start_times_seconds'][next_anchor]
    assert timeline['duration'] == speech['duration'] + 1.5


def test_authored_equivalent_arrows_are_deduplicated_as_atomic_fork_group():
    game, script, speech = bishop_trap_inputs()
    script['beats'][14]['cues'] = [
        {'kind': 'arrow', 'anchor': 'туда, туда', 'from_square': 'f2', 'to_square': 'd1'},
        {'kind': 'arrow', 'anchor': 'туда, туда', 'from_square': 'f2', 'to_square': 'h1'},
    ]
    timeline = compile_timeline(game, script, speech)
    group = automatic_fork(timeline)
    assert len([event for event in group if event['kind'] == 'arrow']) == 2
    start, end = group[0]['start'], group[0]['end']
    arrows = [event for event in timeline['events'] if event['kind'] == 'arrow'
              and event['start'] < end and event['end'] > start
              and (event['from_square'], event['to_square']) in {('f2', 'd1'), ('f2', 'h1')}]
    assert len(arrows) == 2


def test_black_player_gets_white_opponent_fork_without_orientation_changes():
    pgn = '[SetUp "1"]\n[FEN "3k3r/8/8/6N1/8/8/8/K7 w - - 0 1"]\n\n1. Nf7+ Ke7 *'
    timeline = compile_timeline(*inputs(pgn))
    group = [event for event in timeline['events'] if event.get('tactical_group') == 'opponent-threat-1']
    arrows = [event for event in group if event['kind'] == 'arrow']
    assert {(event['from_square'], event['to_square']) for event in arrows} == {('f7', 'd8'), ('f7', 'h8')}
    assert timeline['player_color'] == timeline['orientation'] == 'black'
    moves = [event for event in timeline['events'] if event['kind'] == 'move']
    assert all(moves[0]['end'] < event['start'] < event['end'] <= moves[1]['start'] for event in group)


def test_meaningful_initial_threat_stops_at_first_move():
    pgn = '[SetUp "1"]\n[FEN "6k1/8/8/8/8/8/5n2/2KR3R w - - 0 1"]\n\n1. Rd2 *'
    game, script, speech = inputs(pgn)
    script['perspective'] = 'white'
    timeline = compile_timeline(game, script, speech)
    group = [event for event in timeline['events'] if event.get('tactical_group') == 'opponent-threat-0']
    assert len([event for event in group if event['kind'] == 'arrow']) == 2
    first_move = next(event for event in timeline['events'] if event['kind'] == 'move')
    assert all(event['start'] == 0 and event['end'] <= first_move['start'] for event in group)


def test_boring_opening_has_no_automatic_overlay_groups():
    timeline = compile_timeline(*inputs('1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *'))
    assert not any(event.get('automatic') for event in timeline['events'])
    assert timeline['player_color'] == timeline['orientation'] == 'black'

