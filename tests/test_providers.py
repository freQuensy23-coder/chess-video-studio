import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from chess_video import providers
from chess_video.game import load_game
from chess_video.settings import Settings


@pytest.fixture
def game():
    return load_game('[Result "*"]\n\n1. e4 e5 *')


@pytest.fixture
def script():
    return {
        'title': 'A central start',
        'language': 'en',
        'perspective': 'white',
        'beats': [
            {'ply': 0, 'text': 'I want the center.', 'emotion': 'thinking', 'move_anchor': None, 'cues': []},
            {'ply': 1, 'text': 'I push my pawn to e4.', 'emotion': 'neutral', 'move_anchor': 'push', 'cues': [
                {'kind': 'highlight', 'anchor': 'e4', 'square': 'e4'}
            ]},
            {'ply': 2, 'text': 'Now I need to watch e5.', 'emotion': 'worried', 'move_anchor': 'Now', 'cues': [
                {'kind': 'arrow', 'anchor': 'watch e5', 'from_square': 'e4', 'to_square': 'e5'}
            ]},
        ],
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        llm_provider='gemini',
        gemini_api_key='test-gemini-secret',
        elevenlabs_api_key='test-eleven-secret',
        cache_dir=tmp_path / 'cache',
        retry_backoff_seconds=0,
    )


def transport(monkeypatch, handler):
    client_class = httpx.Client
    monkeypatch.setattr(providers, '_http_client', lambda settings: client_class(transport=httpx.MockTransport(handler)))


def gemini_response(script):
    return httpx.Response(200, json={
        'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(script)}]}}],
        'usageMetadata': {'totalTokenCount': 123},
    })


def speech_response(text):
    return {
        'audio_base64': base64.b64encode(b'test-mp3-provider-bytes').decode(),
        'alignment': {
            'characters': list(text),
            'character_start_times_seconds': [index * 0.01 for index in range(len(text))],
            'character_end_times_seconds': [(index + 1) * 0.01 for index in range(len(text))],
        },
        'normalized_alignment': None,
    }


def probe(monkeypatch, duration=3.125):
    def run(command, **kwargs):
        assert command[0] == 'ffprobe'
        assert Path(command[-1]).read_bytes() == b'test-mp3-provider-bytes'
        assert kwargs['timeout'] <= 60
        return SimpleNamespace(returncode=0, stdout=json.dumps({'format': {'duration': str(duration)}}), stderr='')
    monkeypatch.setattr(providers.subprocess, 'run', run)


def test_settings_secrets_hidden_and_env_file_optional(settings, tmp_path):
    assert 'test-gemini-secret' not in repr(settings)
    assert 'test-eleven-secret' not in repr(settings)
    env = tmp_path / '.env'
    env.write_text('LANGUAGE=ru\nELEVENLABS_SPEED=1.1\n', encoding='utf-8')
    configured = Settings(_env_file=env)
    assert configured.language == 'ru'
    assert configured.elevenlabs_speed == 1.1


@pytest.mark.parametrize('field,value', [
    ('elevenlabs_speed', 1.21), ('elevenlabs_stability', -0.1),
    ('elevenlabs_style', 1.1), ('elevenlabs_similarity_boost', float('nan')),
    ('llm_max_repairs', 4), ('max_speech_characters', 0),
    ('llm_base_url', 'http://remote.example/v1'), ('llm_base_url', 'https://user:password@example.org/v1'),
])
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_gemini_structured_request_validated_response_and_cache(monkeypatch, settings, game, script):
    requests = []
    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.headers['x-goog-api-key'] == 'test-gemini-secret'
        assert 'key=' not in str(request.url)
        assert payload['generationConfig']['responseMimeType'] == 'application/json'
        assert payload['generationConfig']['maxOutputTokens'] == settings.llm_max_output_tokens
        assert 'first-person' in payload['systemInstruction']['parts'][0]['text']
        assert game['initial_fen'] in payload['contents'][0]['parts'][0]['text']
        return gemini_response(script)
    transport(monkeypatch, handler)
    result = providers.generate_script(game, settings)
    assert result['beats'][1]['text'] == script['beats'][1]['text']
    assert providers.generate_script(game, settings) == result
    assert len(requests) == 1
    saved = '\n'.join(path.read_text() for path in settings.cache_dir.rglob('*.json'))
    assert 'usageMetadata' in saved
    assert 'test-gemini-secret' not in saved


def test_script_cache_separates_narrative_configuration(monkeypatch, settings, game, script):
    calls = []
    def handler(request):
        calls.append(request)
        current = copy.deepcopy(script)
        current['perspective'] = 'black' if len(calls) == 2 else 'white'
        return gemini_response(current)
    transport(monkeypatch, handler)
    providers.generate_script(game, settings)
    providers.generate_script(game, settings.model_copy(update={'perspective': 'black'}))
    assert len(calls) == 2


def test_openai_compatible_request(monkeypatch, settings, game, script):
    configured = settings.model_copy(update={
        'llm_provider': 'openai', 'llm_base_url': 'https://llm.example/v1',
        'llm_model': 'explicit-model', 'llm_api_key': SecretStr('test-openai-secret'),
    })
    def handler(request):
        assert str(request.url) == 'https://llm.example/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-openai-secret'
        payload = json.loads(request.content)
        assert payload['model'] == 'explicit-model'
        assert payload['response_format']['type'] == 'json_schema'
        assert payload['response_format']['json_schema']['strict'] is True
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(script)}}]})
    transport(monkeypatch, handler)
    assert providers.generate_script(game, configured)['title'] == script['title']


def test_openai_json_object_mode(monkeypatch, settings, game, script):
    configured = settings.model_copy(update={
        'llm_provider': 'openai', 'llm_base_url': 'https://llm.example/v1',
        'llm_model': 'explicit-model', 'llm_api_key': SecretStr('test-openai-secret'),
        'llm_response_format': 'json_object',
    })
    def handler(request):
        assert json.loads(request.content)['response_format'] == {'type': 'json_object'}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(script)}}]})
    transport(monkeypatch, handler)
    assert providers.generate_script(game, configured)['title'] == script['title']


def test_vertex_uses_adc_without_gemini_key(monkeypatch, settings, game, script):
    configured = settings.model_copy(update={
        'llm_provider': 'vertex', 'llm_model': 'gemini-3.1-pro-preview',
        'vertex_project': 'authorized-project', 'vertex_location': 'global', 'gemini_api_key': None,
    })
    monkeypatch.setattr(providers, '_vertex_token', lambda settings: 'test-adc-token')
    def handler(request):
        assert str(request.url) == 'https://aiplatform.googleapis.com/v1/projects/authorized-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent'
        assert request.headers['authorization'] == 'Bearer test-adc-token'
        assert 'x-goog-api-key' not in request.headers
        return gemini_response(script)
    transport(monkeypatch, handler)
    assert providers.generate_script(game, configured)['title'] == script['title']


def test_bounded_repairs_include_invalid_output_and_validation_feedback(monkeypatch, settings, game, script):
    calls = []
    invalid = copy.deepcopy(script)
    invalid['beats'][1]['move_anchor'] = 'absent-anchor'
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return gemini_response(invalid)
        assert len(payload['contents']) > 1
        assert 'absent-anchor' in json.dumps(payload)
        return gemini_response(script)
    transport(monkeypatch, handler)
    assert providers.generate_script(game, settings)['title'] == script['title']
    assert len(calls) == 2


@pytest.mark.parametrize('case', ['json', 'schema', 'anchor', 'language', 'truncated'])
def test_invalid_llm_response_fails_closed(monkeypatch, settings, game, script, case):
    calls = []
    configured = settings.model_copy(update={'llm_max_repairs': 1})
    def handler(request):
        calls.append(request)
        broken = copy.deepcopy(script)
        if case == 'json':
            return httpx.Response(200, text='not provider JSON')
        if case == 'schema':
            broken['beats'] = []
        if case == 'anchor':
            broken['beats'][0]['cues'] = [{'kind': 'zoom', 'anchor': 'not-in-text', 'square': 'e4', 'scale': 1.2}]
        if case == 'language':
            broken['language'] = 'ru'
        if case == 'truncated':
            return httpx.Response(200, json={'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': json.dumps(broken)}]}}]})
        return gemini_response(broken)
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError):
        providers.generate_script(game, configured)
    assert len(calls) <= 2
    assert not list(settings.cache_dir.rglob('script.json'))


@pytest.mark.parametrize('provider', ['llm', 'speech'])
def test_provider_errors_never_echo_secret_payload(monkeypatch, settings, game, script, tmp_path, caplog, provider):
    secret = 'test-eleven-secret test-gemini-secret authorization-private-provider-message'
    def handler(request):
        return httpx.Response(401, text=secret)
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError) as raised:
        if provider == 'llm':
            providers.generate_script(game, settings)
        else:
            providers.synthesize(script, tmp_path / 'output', settings)
    assert secret not in str(raised.value)
    assert 'test-eleven-secret' not in caplog.text
    assert 'test-gemini-secret' not in caplog.text
    assert 'authorization-private-provider-message' not in str(raised.value)


def test_exact_unicode_speech_payload_alignment_duration_and_cache(monkeypatch, settings, script, tmp_path):
    script['beats'][0]['text'] = 'Я думаю… ♟️'
    text = '\n\n'.join(beat['text'] for beat in script['beats'])
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == f'https://api.elevenlabs.io/v1/text-to-speech/{settings.resolved_voice_id}/with-timestamps?output_format=mp3_44100_128'
        assert request.headers['xi-api-key'] == 'test-eleven-secret'
        body = json.loads(request.content)
        assert body['text'] == text
        assert body['model_id'] == 'eleven_multilingual_v2'
        assert body['voice_settings'] == {
            'stability': 0.4, 'similarity_boost': 0.75, 'style': 0.35,
            'speed': 1.0, 'use_speaker_boost': True,
        }
        assert body['apply_text_normalization'] == 'off'
        assert 'previous_text' not in body and 'next_text' not in body
        return httpx.Response(200, json=speech_response(text))
    transport(monkeypatch, handler)
    probe(monkeypatch, duration=3.125)
    result = providers.synthesize(script, tmp_path / 'output', settings)
    assert result['text'] == text
    assert ''.join(result['characters']) == text
    assert result['duration'] == 3.125
    assert result['duration'] != result['character_end_times_seconds'][-1]
    assert Path(result['audio_path']).read_bytes() == b'test-mp3-provider-bytes'
    again = providers.synthesize(script, tmp_path / 'other-output', settings)
    assert Path(again['audio_path']).parent == tmp_path / 'other-output'
    assert len(calls) == 1
    raw = list((tmp_path / 'output').glob('*provider*.json'))
    assert raw
    assert 'audio_base64' not in raw[0].read_text()


@pytest.mark.parametrize('case', ['base64', 'empty_audio', 'missing_alignment', 'normalized_only', 'mismatch', 'length', 'nan', 'backwards', 'negative', 'beyond_audio', 'json'])
def test_malformed_speech_rejected_without_retry(monkeypatch, settings, script, tmp_path, case):
    text = providers.narration_text(script)
    response = speech_response(text)
    if case == 'base64':
        response['audio_base64'] = 'not valid ***'
    if case == 'empty_audio':
        response['audio_base64'] = ''
    if case in {'missing_alignment', 'normalized_only'}:
        response['normalized_alignment'] = response.pop('alignment')
    if case == 'mismatch':
        response['alignment']['characters'][0] = 'Z'
    if case == 'length':
        response['alignment']['character_start_times_seconds'].pop()
    if case == 'nan':
        response['alignment']['character_start_times_seconds'][0] = float('nan')
    if case == 'backwards':
        response['alignment']['character_start_times_seconds'][2] = 0
    if case == 'negative':
        response['alignment']['character_start_times_seconds'][0] = -1
    if case == 'beyond_audio':
        response['alignment']['character_end_times_seconds'][-1] = 30
    calls = []
    def handler(request):
        calls.append(request)
        if case == 'json':
            return httpx.Response(200, text='not JSON')
        return httpx.Response(200, content=json.dumps(response), headers={'Content-Type': 'application/json'})
    transport(monkeypatch, handler)
    probe(monkeypatch)
    with pytest.raises(providers.ProviderError):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1


def test_invalid_audio_decode_fails_closed(monkeypatch, settings, script, tmp_path):
    transport(monkeypatch, lambda request: httpx.Response(200, json=speech_response(providers.narration_text(script))))
    monkeypatch.setattr(providers.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='private diagnostic'))
    with pytest.raises(providers.ProviderError, match='audio'):
        providers.synthesize(script, tmp_path / 'output', settings)


def test_speech_configuration_changes_do_not_reuse_audio(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=speech_response(providers.narration_text(script)))
    transport(monkeypatch, handler)
    probe(monkeypatch)
    providers.synthesize(script, tmp_path / 'one', settings)
    providers.synthesize(script, tmp_path / 'two', settings.model_copy(update={'elevenlabs_speed': 1.1}))
    assert len(calls) == 2


def test_speech_timeout_not_retried_or_rebilled_on_restart(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('private test-eleven-secret', request=request)
    transport(monkeypatch, handler)
    for _ in range(2):
        with pytest.raises(providers.ProviderError):
            providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1


def test_only_rejected_429_retried_with_bound(monkeypatch, settings, script, tmp_path):
    calls = []
    sleeps = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after': '0'}, json={'detail': {'code': 'rate_limit_exceeded'}})
    transport(monkeypatch, handler)
    monkeypatch.setattr(providers.time, 'sleep', sleeps.append)
    with pytest.raises(providers.ProviderError, match='429'):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == settings.provider_429_retries + 1
    assert all(0 <= delay <= settings.max_retry_after_seconds for delay in sleeps)


def test_429_long_retry_after_does_not_retry_early(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after': '86400'}, json={'detail': {'code': 'rate_limit_exceeded'}})
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1


@pytest.mark.parametrize('detail', [{'code': 'insufficient_credits'}, {'status': 'quota_exceeded'}, {}])
def test_nontransient_or_unknown_429_does_not_retry(monkeypatch, settings, script, tmp_path, detail):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={'detail': detail})
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError, match='429'):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1


def test_speech_500_not_retried(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(500, text='private provider detail')
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1


def test_character_budget_prevents_paid_call(monkeypatch, settings, script, tmp_path):
    def forbidden(request):
        pytest.fail('over-budget speech must not call provider')
    transport(monkeypatch, forbidden)
    configured = settings.model_copy(update={'max_speech_characters': 2})
    with pytest.raises(providers.ProviderError, match='character'):
        providers.synthesize(script, tmp_path / 'output', configured)


def test_missing_credentials_do_not_call_provider(monkeypatch, settings, script, game, tmp_path):
    def forbidden(request):
        pytest.fail('missing credentials must not call provider')
    transport(monkeypatch, forbidden)
    with pytest.raises(providers.ProviderError):
        providers.generate_script(game, settings.model_copy(update={'gemini_api_key': None}))
    with pytest.raises(providers.ProviderError):
        providers.synthesize(script, tmp_path / 'output', settings.model_copy(update={'elevenlabs_api_key': None}))


def test_ffprobe_failure_receipt_can_be_recovered_without_rebilling(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=speech_response(providers.narration_text(script)))
    transport(monkeypatch, handler)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('ffprobe')
    monkeypatch.setattr(providers.subprocess, 'run', unavailable)
    with pytest.raises(providers.ProviderError, match='ffprobe'):
        providers.synthesize(script, tmp_path / 'output', settings)
    probe(monkeypatch)
    assert providers.synthesize(script, tmp_path / 'output', settings)['duration'] > 0
    assert len(calls) == 1


def test_list_voices_is_read_only(monkeypatch, settings):
    def handler(request):
        assert request.method == 'GET'
        assert str(request.url) == 'https://api.elevenlabs.io/v1/voices'
        return httpx.Response(200, json={'voices': [{'voice_id': 'public-voice', 'name': 'Example', 'category': 'premade'}]})
    transport(monkeypatch, handler)
    assert providers.list_voices(settings)[0]['voice_id'] == 'public-voice'


def test_gemini_fallback_environment_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_FALLBACK_API_KEY', 'test-fallback-secret')
    assert Settings(_env_file=None).gemini_api_key.get_secret_value() == 'test-fallback-secret'


@pytest.mark.parametrize('claim', ['This is a brilliant move.', 'I have forced mate in three.'])
def test_unsupported_objective_claims_rejected(monkeypatch, settings, game, script, claim):
    configured = settings.model_copy(update={'llm_max_repairs': 0})
    broken = copy.deepcopy(script)
    broken['beats'][0]['text'] = claim
    transport(monkeypatch, lambda request: gemini_response(broken))
    with pytest.raises(providers.ProviderError):
        providers.generate_script(game, configured)


def test_conversational_excitement_is_not_a_brilliance_claim(monkeypatch, settings, game, script):
    script['beats'][0]['text'] = 'Yes!! I want the center.'
    transport(monkeypatch, lambda request: gemini_response(script))
    assert providers.generate_script(game, settings)['beats'][0]['text'] == script['beats'][0]['text']


def test_v3_character_limit_prevents_paid_request(monkeypatch, settings, script, tmp_path):
    configured = settings.model_copy(update={'elevenlabs_model_id': 'eleven_v3', 'elevenlabs_stability': 0.5})
    script['beats'][0]['text'] = 'x' * 5001
    def forbidden(request):
        pytest.fail('v3 model limit must be enforced before billing')
    transport(monkeypatch, forbidden)
    with pytest.raises(providers.ProviderError, match='character'):
        providers.synthesize(script, tmp_path / 'output', configured)


def test_invalid_game_does_not_call_llm(monkeypatch, settings, game):
    game['positions'][1]['uci'] = 'e2e5'
    def forbidden(request):
        pytest.fail('invalid game must not incur provider charges')
    transport(monkeypatch, forbidden)
    with pytest.raises(providers.ProviderError, match='Game'):
        providers.generate_script(game, settings)


def test_failed_validation_does_not_automatically_rebill_on_restart(monkeypatch, settings, game, script):
    configured = settings.model_copy(update={'llm_max_repairs': 0})
    calls = []
    broken = copy.deepcopy(script)
    broken['beats'] = []
    def handler(request):
        calls.append(request)
        return gemini_response(broken)
    transport(monkeypatch, handler)
    for _ in range(2):
        with pytest.raises(providers.ProviderError):
            providers.generate_script(game, configured)
    assert len(calls) == 1


def test_openai_model_namespace_is_preserved():
    configured = Settings(_env_file=None, llm_provider='openai', llm_model='vendor/model-name:latest')
    assert configured.llm_model == 'vendor/model-name:latest'


def test_speech_rejects_invalid_cached_duration(monkeypatch, settings, script, tmp_path):
    transport(monkeypatch, lambda request: httpx.Response(200, json=speech_response(providers.narration_text(script))))
    probe(monkeypatch)
    providers.synthesize(script, tmp_path / 'output', settings)
    cached = next(settings.cache_dir.rglob('speech.json'))
    payload = json.loads(cached.read_text())
    payload['duration'] = 0
    cached.write_text(json.dumps(payload))
    with pytest.raises(providers.ProviderError):
        providers.synthesize(script, tmp_path / 'output', settings)


def test_oversized_provider_response_stops_without_retry(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b'x' * 2048)
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError, match='byte limit'):
        providers.synthesize(script, tmp_path / 'output', settings.model_copy(update={'max_provider_response_bytes': 1024}))
    assert len(calls) == 1


def test_redirect_does_not_forward_authentication(monkeypatch, settings, script, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={'location': 'https://untrusted.example/steal'})
    transport(monkeypatch, handler)
    with pytest.raises(providers.ProviderError, match='307'):
        providers.synthesize(script, tmp_path / 'output', settings)
    assert len(calls) == 1
