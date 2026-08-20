import json

import pytest

from src.agent.providers import ChatCompletionProvider, ProviderError


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_provider_reads_groq_configuration(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = ChatCompletionProvider.from_env()
    assert provider.endpoint == "https://api.groq.com/openai/v1/chat/completions"
    assert provider.api_key == "test-key"


def test_provider_posts_openai_compatible_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.data)
        return Response({"choices": [{"message": {"content": "result"}}]})

    monkeypatch.setattr("src.agent.providers.urlopen", fake_urlopen)
    provider = ChatCompletionProvider("key", "https://example.test/chat", "model")
    assert provider("prompt") == "result"
    assert captured["body"] == {"model": "model", "messages": [{"role": "user", "content": "prompt"}], "temperature": 0}
    assert captured["headers"]["Authorization"] == "Bearer key"
    assert captured["headers"]["User-agent"] == "churn-data-analyst/1.0"
    assert captured["headers"]["Accept"] == "application/json"


def test_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY or OPENROUTER_API_KEY"):
        ChatCompletionProvider.from_env()


def test_provider_retries_429_with_backoff_then_succeeds(monkeypatch):
    """A rate-limit response is a transport hiccup, not a bad plan or a real
    failure — it should be retried here (bounded, with backoff) rather than
    propagating straight to the user on the very first 429."""
    from urllib.error import HTTPError

    calls = {"count": 0}
    sleeps = []
    monkeypatch.setattr("src.agent.providers.time.sleep", lambda seconds: sleeps.append(seconds))

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)
        return Response({"choices": [{"message": {"content": "result"}}]})

    monkeypatch.setattr("src.agent.providers.urlopen", fake_urlopen)
    provider = ChatCompletionProvider("key", "https://example.test/chat", "model")
    assert provider("prompt") == "result"
    assert calls["count"] == 3
    assert len(sleeps) == 2


def test_provider_gives_up_after_max_429_retries(monkeypatch):
    from urllib.error import HTTPError

    calls = {"count": 0}
    monkeypatch.setattr("src.agent.providers.time.sleep", lambda seconds: None)

    def always_429(request, timeout):
        calls["count"] += 1
        raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)

    monkeypatch.setattr("src.agent.providers.urlopen", always_429)
    provider = ChatCompletionProvider("key", "https://example.test/chat", "model")
    with pytest.raises(ProviderError, match="HTTP 429"):
        provider("prompt")
    assert calls["count"] == 3  # initial attempt + 2 retries, then give up


def test_provider_does_not_retry_non_429_errors(monkeypatch):
    from urllib.error import HTTPError

    calls = {"count": 0}

    def fake_500(request, timeout):
        calls["count"] += 1
        raise HTTPError(request.full_url, 500, "Server Error", hdrs=None, fp=None)

    monkeypatch.setattr("src.agent.providers.urlopen", fake_500)
    provider = ChatCompletionProvider("key", "https://example.test/chat", "model")
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider("prompt")
    assert calls["count"] == 1  # no retry budget spent on a non-rate-limit error
