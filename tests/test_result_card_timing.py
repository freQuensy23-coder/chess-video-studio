from chess_video.game import load_game
from chess_video.models import narration_text
from chess_video.renderer import Renderer
from chess_video.timeline import compile_timeline


def compiled_mating_timeline():
    game = load_game('1. f3 e5 2. g4 Qh4# 0-1')
    beats = [
        {'ply': position['ply'], 'text': f"I consider position {position['ply']}.",
         'emotion': 'thinking', 'move_anchor': None, 'cues': []}
        for position in game['positions']
    ]
    beats[-1].update(
        text='I play Qh4. Now checkmate.', move_anchor='Qh4', emotion='happy',
        cues=[{'kind': 'result', 'anchor': 'checkmate', 'label': 'Checkmate', 'duration': 0.4}],
    )
    script = {'title': 'The final attack', 'language': 'en', 'perspective': 'black', 'beats': beats}
    text = narration_text(script)
    speech = {
        'audio_path': 'unused-in-frame-tests.wav', 'duration': len(text) * 0.1,
        'text': text, 'characters': list(text),
        'character_start_times_seconds': [index * 0.1 for index in range(len(text))],
        'character_end_times_seconds': [(index + 1) * 0.1 for index in range(len(text))],
    }
    timeline = compile_timeline(game, script, speech)
    timeline['audio_path'] = None
    return timeline


def test_delayed_mate_card_obeys_compiled_spoken_anchor_and_expiry():
    timeline = compiled_mating_timeline()
    result = next(event for event in timeline['events'] if event['kind'] == 'result')
    final_move = [event for event in timeline['events'] if event['kind'] == 'move'][-1]
    assert final_move['end'] < result['start']
    with Renderer(timeline) as renderer:
        assert renderer.board_at(final_move['end']).is_checkmate()
        assert renderer.result_at((final_move['end'] + result['start']) / 2) is None
        assert renderer.result_at(result['start']) == 'CHECKMATE · BLACK WINS'
        assert renderer.result_at(result['end']) is None
        assert renderer.result_at(result['end'] + 0.1) is None


def test_raw_timeline_without_result_events_keeps_automatic_mate_fallback():
    timeline = compiled_mating_timeline()
    timeline['events'] = [event for event in timeline['events'] if event['kind'] != 'result']
    with Renderer(timeline) as renderer:
        assert renderer.result_at(timeline['duration'] - 0.1) == 'CHECKMATE · BLACK WINS'
