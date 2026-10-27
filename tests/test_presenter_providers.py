import base64
import json

import httpx
import pytest

from chess_video import providers
from chess_video.game import load_game
from chess_video.settings import Settings

MALE_VOICE = 'pNInz6obpgDQGcFmaJgB'
FEMALE_VOICE = 'cgSgspJ2msm6clMCkdW9'


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for key in ('PRESENTER', 'ELEVENLABS_VOICE_ID'):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
    return Settings(
        _env_file=None, presenter='male', elevenlabs_voice_id=None,
        elevenlabs_api_key='test-eleven-key', gemini_api_key='test-gemini-key',
        cache_dir=tmp_path / 'cache',
    )


@pytest.fixture
def script():
    return {
        'title': 'The center', 'language': 'en', 'perspective': 'white',
        'beats': [
            {'ply': 0, 'text': 'I want the center.', 'emotion': 'thinking', 'move_anchor': None, 'cues': []},
            {'ply': 1, 'text': 'I play e4.', 'emotion': 'neutral', 'move_anchor': 'e4', 'cues': []},
        ],
    }


def mock_speech(monkeypatch, requests):
    def handler(request):
        requests.append(request)
        text = json.loads(request.content)['text']
        return httpx.Response(200, json={
            'audio_base64': base64.b64encode(b'mock-speech-audio').decode(),
            'alignment': {
                'characters': list(text),
                'character_start_times_seconds': [index * 0.01 for index in range(len(text))],
                'character_end_times_seconds': [(index + 1) * 0.01 for index in range(len(text))],
            },
        })

    monkeypatch.setattr(providers, '_http_client', lambda configured: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(providers, '_audio_duration', lambda path, configured: 5.0)


@pytest.mark.parametrize('presenter,voice', [('male', MALE_VOICE), ('female', FEMALE_VOICE)])
def test_speech_routes_preset_to_timestamp_endpoint_and_cache_config(settings, script, tmp_path, monkeypatch, presenter, voice):
    requests = []
    mock_speech(monkeypatch, requests)
    configured = settings.model_copy(update={'presenter': presenter})
    result = providers.synthesize(script, tmp_path / presenter, configured)
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f'https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128'
    assert request.headers['xi-api-key'] == 'test-eleven-key'
    assert providers._speech_config(configured)['voice_id'] == voice
    assert result['text'] == '\n\n'.join(beat['text'] for beat in script['beats'])
    assert (tmp_path / presenter / 'narration.mp3').read_bytes() == b'mock-speech-audio'


def test_preset_voice_changes_isolate_speech_cache(settings, script, tmp_path, monkeypatch):
    requests = []
    mock_speech(monkeypatch, requests)
    female = settings.model_copy(update={'presenter': 'female'})
    providers.synthesize(script, tmp_path / 'male', settings)
    providers.synthesize(script, tmp_path / 'female', female)
    providers.synthesize(script, tmp_path / 'female-reused', female)
    providers.synthesize(script, tmp_path / 'male-reused', settings)
    assert len(requests) == 2
    assert [request.url.path.split('/')[-2] for request in requests] == [MALE_VOICE, FEMALE_VOICE]
    assert len(list((settings.cache_dir / 'speech').glob('*/speech.json'))) == 2


def test_same_explicit_voice_reuses_speech_for_different_avatars(settings, script, tmp_path, monkeypatch):
    requests = []
    mock_speech(monkeypatch, requests)
    male = settings.model_copy(update={'elevenlabs_voice_id': 'custom-voice'})
    female = male.model_copy(update={'presenter': 'female'})
    providers.synthesize(script, tmp_path / 'male', male)
    providers.synthesize(script, tmp_path / 'female', female)
    assert len(requests) == 1
    assert '/custom-voice/' in str(requests[0].url)
    assert providers._speech_config(male) == providers._speech_config(female)


def test_default_speech_config_preserves_existing_male_cache_key(settings):
    assert providers._speech_config(settings) == {
        'version': 1, 'voice_id': MALE_VOICE, 'model_id': 'eleven_multilingual_v2',
        'output_format': 'mp3_44100_128',
        'voice_settings': {'stability': 0.4, 'similarity_boost': 0.75, 'style': 0.35, 'use_speaker_boost': True, 'speed': 1.0},
        'apply_text_normalization': 'off',
    }


@pytest.mark.parametrize('updates', [{'presenter': 'other'}, {'elevenlabs_voice_id': '../voice'}])
def test_invalid_copied_configuration_cannot_dispatch_speech(settings, script, tmp_path, monkeypatch, updates):
    requests = []
    mock_speech(monkeypatch, requests)
    with pytest.raises(ValueError):
        providers.synthesize(script, tmp_path / 'invalid', settings.model_copy(update=updates))
    assert requests == []


@pytest.mark.parametrize('presenter', ['male', 'female'])
def test_llm_receives_presenter_and_russian_grammar_guidance(settings, script, monkeypatch, presenter):
    configured = settings.model_copy(update={'presenter': presenter, 'language': 'ru'})
    generated = dict(script, language='ru')
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        context = json.loads(payload['contents'][0]['parts'][0]['text'])
        assert context['presenter'] == presenter
        assert context['language'] == 'ru'
        assert 'selected_opponent_threats' in context
        prompt = payload['systemInstruction']['parts'][0]['text']
        assert 'Russian' in prompt
        assert 'feminine' in prompt
        assert 'masculine' in prompt
        return httpx.Response(200, json={
            'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(generated)}]}}],
        })

    monkeypatch.setattr(providers, '_http_client', lambda current: httpx.Client(transport=httpx.MockTransport(handler)))
    result = providers.generate_script(load_game('1. e4 *'), configured)
    assert result['language'] == 'ru'
    assert len(requests) == 1
    assert providers._llm_config(configured)['presenter'] == presenter


def test_presenter_changes_separate_script_cache(settings, script, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(script)}]}}],
        })

    monkeypatch.setattr(providers, '_http_client', lambda current: httpx.Client(transport=httpx.MockTransport(handler)))
    game = load_game('1. e4 *')
    providers.generate_script(game, settings)
    female = settings.model_copy(update={'presenter': 'female'})
    providers.generate_script(game, female)
    providers.generate_script(game, female)
    assert len(requests) == 2
