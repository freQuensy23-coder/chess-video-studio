import math

from .game import validate_script
from .models import validate_speech
from .tactics import select_key_threats
from .video_options import validate_video_options

MOVE_DURATION = 0.35
ENDING_HOLD = 1.5
PRIORITY = {'clear': 0, 'move': 1, 'face': 2, 'arrow': 3, 'highlight': 3,
            'zoom': 3, 'cursor': 3, 'badge': 3, 'result': 4}


def _result_label(game, language):
    if game['positions'][-1]['is_checkmate']:
        return 'Мат' if language == 'ru' else 'Checkmate'
    if game['result'] == '1/2-1/2':
        return 'Ничья' if language == 'ru' else 'Draw'
    if language == 'ru':
        return 'Белые победили' if game['result'] == '1-0' else 'Чёрные победили'
    return 'White wins' if game['result'] == '1-0' else 'Black wins'


def _add_threat_events(game, perspective, events, moves, clear_times, initial_start, duration):
    moves_by_ply = {move['ply']: move for move in moves}
    for threat in select_key_threats(game, perspective):
        ply = threat['ply']
        completed = moves_by_ply[ply]['end'] if ply else initial_start
        start = completed + 0.08 if ply else completed
        next_move = next((move['start'] for move in moves if move['ply'] > ply), duration)
        reset = min([next_move] + [time for time in clear_times if time >= completed])
        end = min(start + 2.5, reset, duration)
        if end - start < 1.0:
            continue
        pairs = {(arrow['from_square'], arrow['to_square']) for arrow in threat['arrows']}
        events[:] = [event for event in events
                     if not (event['kind'] == 'arrow'
                             and (event['from_square'], event['to_square']) in pairs
                             and event['start'] < end and event['end'] > start)]
        metadata = {'start': start, 'end': end, 'color': '#ef7858', 'automatic': True,
                    'tactical_group': f'opponent-threat-{ply}', 'focus_squares': threat['focus_squares'],
                    'ply': ply}
        events.extend({'kind': 'arrow', **arrow, **metadata} for arrow in threat['arrows'])
        events.extend({'kind': 'highlight', 'square': square, **metadata} for square in threat['focus_squares'])


def compile_timeline(game, script, speech, width=608, height=1080, fps=30, *, presenter='male') -> dict:
    if presenter not in ('male', 'female'):
        raise ValueError('Presenter must be male or female')
    validate_video_options(width, height, fps)
    script = validate_script(game, script)
    speech = validate_speech(script, speech)
    starts = speech['character_start_times_seconds']
    ends = speech['character_end_times_seconds']
    offsets = []
    offset = 0
    for beat in script['beats']:
        offsets.append(offset)
        offset += len(beat['text']) + 2

    def aligned_start(index, anchor):
        beat = script['beats'][index]
        local = beat['text'].index(anchor) if anchor is not None else 0
        return starts[offsets[index] + local]

    moves = []
    for index, position in enumerate(game['positions'][1:], 1):
        start = aligned_start(index, script['beats'][index].get('move_anchor'))
        end = math.nextafter(start + MOVE_DURATION, math.inf)
        if moves and start < moves[-1]['end']:
            raise ValueError(f'Moves at plies {index - 1} and {index} overlap; speech anchors are too close')
        moves.append({'kind': 'move', 'start': start, 'end': end,
                      **{key: value for key, value in position.items() if key != 'analysis'}})
    last_move_end = moves[-1]['end']
    duration = max(speech['duration'], last_move_end) + ENDING_HOLD
    events = list(moves)
    clear_times = []
    cues = []
    for index, beat in enumerate(script['beats']):
        start = aligned_start(index, None)
        beat_end = (starts[offsets[index + 1]] if index + 1 < len(offsets) else duration)
        if beat_end > start:
            events.append({'kind': 'face', 'start': start, 'end': beat_end, 'emotion': beat['emotion']})
        for cue in beat['cues']:
            start = aligned_start(index, cue['anchor'])
            if cue['kind'] == 'clear':
                clear_times.append(start)
            cues.append((start, index, cue))
    resets = sorted(clear_times + [move['start'] for move in moves])
    for start, index, cue in cues:
        kind = cue['kind']
        if kind == 'result':
            start = max(start, last_move_end)
        end = min(duration, start + cue['duration']) if 'duration' in cue else duration
        if kind not in {'clear', 'result', 'face'}:
            end = min([end] + [reset for reset in resets if reset > start])
        if kind == 'face':
            beat_end = starts[offsets[index + 1]] if index + 1 < len(offsets) else duration
            end = min(end, beat_end)
        payload = {key: value for key, value in cue.items() if key not in {'kind', 'anchor', 'duration'}}
        if kind == 'face':
            payload = {'emotion': payload.pop('label')}
        if kind == 'clear':
            end = min(duration, start + 1 / fps)
        if end <= start:
            raise ValueError('Cue has no visible duration at its aligned phrase')
        if kind in {'arrow', 'highlight', 'badge'}:
            payload.setdefault('color', '#f6c445')
        events.append({'kind': kind, 'start': start, 'end': end, **payload})
    if game['result'] != '*' and not any(event['kind'] == 'result' for event in events):
        spoken_end = ends[offsets[-1] + len(script['beats'][-1]['text']) - 1]
        result_start = max(last_move_end, spoken_end)
        events.append({'kind': 'result', 'start': result_start, 'end': duration,
                       'label': _result_label(game, script['language'])})
    _add_threat_events(game, script['perspective'], events, moves, clear_times,
                       aligned_start(0, None), duration)
    events.sort(key=lambda event: (event['start'], PRIORITY[event['kind']]))
    return {
        'version': 1, 'width': width, 'height': height, 'fps': fps, 'duration': duration,
        'presenter': presenter,
        'audio_path': speech['audio_path'], 'initial_fen': game['initial_fen'],
        'game_result': game['result'], 'language': script['language'],
        'orientation': script['perspective'], 'player_color': script['perspective'],
        'title': script['title'], 'events': events,
    }
