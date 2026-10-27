import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chess_video.api import create_app
from chess_video.cli import main, parser
from chess_video.game import load_game
from chess_video.pipeline import generate_video
from chess_video.settings import Settings
from chess_video.timeline import compile_timeline

MALE_VOICE = 'pNInz6obpgDQGcFmaJgB'
FEMALE_VOICE = 'cgSgspJ2msm6clMCkdW9'
TOKEN = 'presenter-test-token'
HEADERS = {'Authorization': f'Bearer {TOKEN}'}


@pytest.fixture(autouse=True)
def isolated_presenter_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ('PRESENTER', 'ELEVENLABS_VOICE_ID', 'STOCKFISH_PATH'):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)


def aligned_inputs():
    game = load_game('1. e4 *')
    script = {
        'title': 'The center', 'language': 'en', 'perspective': 'white',
        'beats': [
            {'ply': 0, 'text': 'I want the center.', 'emotion': 'thinking', 'move_anchor': None, 'cues': []},
            {'ply': 1, 'text': 'I play e4.', 'emotion': 'neutral', 'move_anchor': 'e4', 'cues': []},
        ],
    }
    text = '\n\n'.join(beat['text'] for beat in script['beats'])
    speech = {
        'audio_path': 'narration.mp3', 'text': text, 'duration': len(text) * 0.1,
        'characters': list(text),
        'character_start_times_seconds': [index * 0.1 for index in range(len(text))],
        'character_end_times_seconds': [(index + 1) * 0.1 for index in range(len(text))],
    }
    return game, script, speech


@pytest.mark.parametrize('presenter,voice', [('male', MALE_VOICE), ('female', FEMALE_VOICE)])
def test_presenter_presets_are_resolved_without_setting_an_override(presenter, voice):
    settings = Settings(_env_file=None, presenter=presenter)
    assert settings.elevenlabs_voice_id is None
    assert settings.resolved_voice_id == voice


def test_default_presenter_preserves_adam_voice():
    settings = Settings(_env_file=None)
    assert settings.presenter == 'male'
    assert settings.resolved_voice_id == MALE_VOICE


@pytest.mark.parametrize('presenter', ['male', 'female'])
def test_explicit_voice_overrides_presenter_preset(presenter):
    settings = Settings(_env_file=None, presenter=presenter, elevenlabs_voice_id='custom-voice_1')
    assert settings.resolved_voice_id == 'custom-voice_1'


def test_model_copy_resolves_changed_presenter_without_running_validators():
    original = Settings(_env_file=None)
    female = original.model_copy(update={'presenter': 'female', 'generation_dir': Path('female')})
    assert female.resolved_voice_id == FEMALE_VOICE
    assert original.resolved_voice_id == MALE_VOICE


@pytest.mark.parametrize('updates', [
    {'presenter': 'other'}, {'presenter': None}, {'elevenlabs_voice_id': ''},
    {'elevenlabs_voice_id': '../voice'}, {'elevenlabs_voice_id': 'voice?token=x'},
    {'elevenlabs_voice_id': 'voice with spaces'}, {'elevenlabs_voice_id': 123},
])
def test_invalid_presenter_configuration_is_rejected(updates):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **updates)


def test_example_dotenv_allows_female_preset_and_explicit_override(tmp_path):
    example = (Path(__file__).parents[1] / '.env.example').read_text(encoding='utf-8')
    assert 'PRESENTER=male\n' in example
    assert 'ELEVENLABS_VOICE_ID=\n' in example
    dotenv = tmp_path / '.env'
    dotenv.write_text(example.replace('PRESENTER=male', 'PRESENTER=female'), encoding='utf-8')
    settings = Settings(_env_file=dotenv)
    assert settings.presenter == 'female'
    assert settings.resolved_voice_id == FEMALE_VOICE
    dotenv.write_text(dotenv.read_text().replace('ELEVENLABS_VOICE_ID=\n', 'ELEVENLABS_VOICE_ID=explicit\n'))
    assert Settings(_env_file=dotenv).resolved_voice_id == 'explicit'


@pytest.mark.parametrize('presenter', ['male', 'female'])
def test_compiler_propagates_presenter_without_changing_events(presenter):
    game, script, speech = aligned_inputs()
    baseline = compile_timeline(game, script, speech)
    timeline = compile_timeline(game, script, speech, presenter=presenter)
    assert baseline['presenter'] == 'male'
    assert timeline['presenter'] == presenter
    assert timeline['events'] == baseline['events']
    assert timeline['audio_path'] == baseline['audio_path']


@pytest.mark.parametrize('presenter', ['other', None, {}, ['female']])
def test_compiler_rejects_invalid_presenter_before_input_processing(presenter):
    with pytest.raises(ValueError, match='presenter|Presenter'):
        compile_timeline(None, None, None, presenter=presenter)


def test_pipeline_retains_presenter_and_resolved_voice_in_manifest(tmp_path):
    _, script, speech = aligned_inputs()
    settings = Settings(_env_file=None, presenter='female', stockfish_path=None)
    source = tmp_path / 'existing-script.json'
    source.write_text(json.dumps(script))
    output = tmp_path / 'generated'
    with (
        patch('chess_video.pipeline.generate_script') as llm,
        patch('chess_video.pipeline.synthesize', return_value=speech) as tts,
        patch('chess_video.pipeline.render_video', return_value=output / 'video.mp4') as render,
    ):
        manifest = generate_video('1. e4 *', output, settings, script_path=source)
    llm.assert_not_called()
    effective = tts.call_args.args[2]
    assert effective.generation_dir == output.resolve()
    assert effective.presenter == 'female'
    assert effective.resolved_voice_id == FEMALE_VOICE
    assert settings.generation_dir is None
    assert render.call_args.args[0]['presenter'] == 'female'
    assert manifest['presenter'] == 'female'
    assert manifest['voice_id'] == FEMALE_VOICE
    assert manifest['providers']['llm'] == 'provided_script'
    assert json.loads((output / 'manifest.json').read_text()) == manifest


def test_pipeline_rejects_invalid_profile_before_paid_work(tmp_path):
    with (
        patch('chess_video.pipeline.generate_script') as llm,
        patch('chess_video.pipeline.synthesize') as tts,
        pytest.raises(ValueError, match='presenter|Presenter'),
    ):
        generate_video('1. e4 *', tmp_path / 'out', SimpleNamespace(presenter='other'))
    llm.assert_not_called()
    tts.assert_not_called()


@pytest.mark.parametrize('command,voice_id', [('generate', None), ('generate', 'explicit-voice'), ('script', None)])
def test_cli_routes_presenter_to_generation_settings(command, voice_id, tmp_path, monkeypatch):
    pgn = tmp_path / 'game.pgn'
    pgn.write_text('1. e4 *')
    argv = ['chess-video', command, '--pgn', str(pgn), '--output', str(tmp_path / 'out'), '--presenter', 'female']
    if voice_id is not None:
        argv.extend(['--voice-id', voice_id])
    monkeypatch.setattr('sys.argv', argv)
    _, script, _ = aligned_inputs()
    with (
        patch('chess_video.pipeline.generate_video', return_value={}) as video,
        patch('chess_video.providers.generate_script', return_value=script) as llm,
    ):
        assert main() == 0
    effective = video.call_args.args[2] if command == 'generate' else llm.call_args.args[1]
    assert effective.presenter == 'female'
    assert effective.resolved_voice_id == (voice_id or FEMALE_VOICE)


@pytest.mark.parametrize('override', [None, 'female'])
def test_cli_render_preserves_or_overrides_timeline_presenter(override, tmp_path, monkeypatch):
    source = tmp_path / 'timeline.json'
    source.write_text(json.dumps({'presenter': 'male', 'duration': 2, 'events': []}))
    output = tmp_path / 'out.mp4'
    argv = ['chess-video', 'render', '--timeline', str(source), '--output', str(output)]
    if override:
        argv.extend(['--presenter', override])
    monkeypatch.setattr('sys.argv', argv)
    with patch('chess_video.renderer.render_video', return_value=output) as render:
        assert main() == 0
    assert render.call_args.args[0]['presenter'] == (override or 'male')


def test_cli_showcase_routes_explicit_presenter(tmp_path, monkeypatch):
    output = tmp_path / 'showcase.mp4'
    monkeypatch.setattr('sys.argv', ['chess-video', 'showcase', '--output', str(output), '--presenter', 'female'])
    with patch('chess_video.renderer.write_showcase', return_value=output) as showcase:
        assert main() == 0
    assert showcase.call_args.kwargs['presenter'] == 'female'


@pytest.mark.parametrize('command', ['generate', 'script', 'render', 'showcase'])
def test_cli_rejects_invalid_presenter(command):
    arguments = [command, '--output', 'out', '--presenter', 'other']
    if command in {'generate', 'script'}:
        arguments.extend(['--pgn', 'game.pgn'])
    elif command == 'render':
        arguments.extend(['--timeline', 'timeline.json'])
    with pytest.raises(SystemExit):
        parser().parse_args(arguments)


def test_cli_rejects_invalid_voice_before_generation(tmp_path, monkeypatch):
    source = tmp_path / 'game.pgn'
    source.write_text('1. e4 *')
    monkeypatch.setattr('sys.argv', ['chess-video', 'generate', '--pgn', str(source), '--output', str(tmp_path / 'out'), '--voice-id', '../voice'])
    with patch('chess_video.pipeline.generate_video') as pipeline:
        assert main() == 1
    pipeline.assert_not_called()


def test_api_job_overrides_are_isolated_from_app_and_other_jobs(tmp_path):
    configured = Settings(_env_file=None, presenter='male', elevenlabs_voice_id=None)
    observed = []
    finished = threading.Event()

    def pipeline(**kwargs):
        observed.append(kwargs['settings'])
        if len(observed) == 4:
            finished.set()
        return {}

    app = create_app(settings=configured, job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)
    with TestClient(app) as client:
        for overrides in ({'presenter': 'female'}, {}, {'presenter': 'female', 'voice_id': 'custom'}, {'voice_id': 'independent'}):
            response = client.post('/v1/jobs', headers=HEADERS, json={'pgn': '1. e4 *', **overrides})
            assert response.status_code == 202
        assert finished.wait(5)
    assert [(settings.presenter, settings.resolved_voice_id) for settings in observed] == [
        ('female', FEMALE_VOICE), ('male', MALE_VOICE), ('female', 'custom'), ('male', 'independent'),
    ]
    assert len({id(settings) for settings in observed}) == 4
    assert configured.presenter == 'male'
    assert configured.elevenlabs_voice_id is None


def test_api_presenter_only_override_preserves_configured_voice(tmp_path):
    observed = []
    finished = threading.Event()

    def pipeline(**kwargs):
        observed.append(kwargs['settings'])
        finished.set()
        return {}

    settings = Settings(_env_file=None, elevenlabs_voice_id='global-custom')
    with TestClient(create_app(settings=settings, job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)) as client:
        assert client.post('/v1/jobs', headers=HEADERS, json={'pgn': '1. e4 *', 'presenter': 'female'}).status_code == 202
        assert finished.wait(5)
    assert observed[0].presenter == 'female'
    assert observed[0].resolved_voice_id == 'global-custom'


@pytest.mark.parametrize('overrides', [
    {'presenter': 'other'}, {'presenter': 1}, {'voice_id': ''}, {'voice_id': '../voice'},
    {'voice_id': 'voice?key=value'}, {'voice_id': 'voice with spaces'}, {'voice_id': 123},
])
def test_api_invalid_presenter_configuration_never_enters_queue(tmp_path, overrides):
    pipeline = Mock(return_value={})
    with TestClient(create_app(settings=Settings(_env_file=None), job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)) as client:
        response = client.post('/v1/jobs', headers=HEADERS, json={'pgn': '1. e4 *', **overrides})
        assert response.status_code == 422
        assert client.get('/v1/jobs', headers=HEADERS).json() == {'jobs': []}
    pipeline.assert_not_called()


def test_api_presenter_options_do_not_bypass_authentication(tmp_path):
    pipeline = Mock(return_value={})
    with TestClient(create_app(settings=Settings(_env_file=None), job_root=tmp_path, api_token=TOKEN, pipeline=pipeline)) as client:
        payload = {'pgn': '1. e4 *', 'presenter': 'female', 'voice_id': 'custom'}
        assert client.post('/v1/jobs', json=payload).status_code == 401
        assert client.post('/v1/jobs', headers={'Authorization': 'Bearer wrong'}, json=payload).status_code == 401
        assert client.get('/v1/jobs', headers=HEADERS).json() == {'jobs': []}
    pipeline.assert_not_called()


def test_serve_loads_presenter_from_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PRESENTER', 'female')
    monkeypatch.setenv('CHESS_VIDEO_API_TOKEN', TOKEN)
    monkeypatch.setattr('sys.argv', ['chess-video', 'serve'])
    with patch('chess_video.api.create_app') as factory, patch('uvicorn.run'):
        assert main() == 0
    assert factory.call_args.kwargs['settings'].presenter == 'female'
    assert factory.call_args.kwargs['settings'].resolved_voice_id == FEMALE_VOICE
