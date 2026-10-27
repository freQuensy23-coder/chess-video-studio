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
