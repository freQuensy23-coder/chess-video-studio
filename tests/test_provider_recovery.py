import httpx
import pytest

from chess_video.providers import ProviderError, _http_client, _request
from chess_video.settings import Settings


def test_http_client_respects_configured_ca_bundle(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("chess_video.providers.httpx.Client", Client)
    _http_client(Settings(_env_file=None))
    assert captured["trust_env"] is True


def test_failed_connection_does_not_claim_a_paid_request_was_dispatched():
    def unavailable(request):
        raise httpx.ConnectError("TLS handshake failed", request=request)

    settings = Settings(_env_file=None)
    with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(ProviderError) as caught:
            _request(client, "POST", "https://provider.example/generate", settings, {}, {})
    assert caught.value.outcome_unknown is False
    assert "ConnectError" in str(caught.value)


def test_uncertain_generation_cannot_retry_without_explicit_opt_in(tmp_path):
    from chess_video.providers import _review_pending

    pending = tmp_path / "pending.json"
    pending.write_text('{"status":"request_in_progress"}')
    with pytest.raises(ProviderError, match="unresolved"):
        _review_pending(pending, Settings(_env_file=None), "LLM")
    assert pending.exists()
    _review_pending(pending, Settings(_env_file=None, retry_uncertain_requests=True), "LLM")
    assert not pending.exists()
    assert len(list(tmp_path.glob("operator-retry-*.json"))) == 1
