"""Minimal OpenAI-compatible chat-completions providers for the agent."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """An API configuration, transport, or response-format error."""


_PROVIDERS = {
    "groq": {
        "key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}


class ChatCompletionProvider:
    """Callable adapter matching the existing ``Callable[[str], str]`` API."""

    def __init__(self, api_key: str, endpoint: str, model: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ProviderError("An API key is required")
        self.api_key, self.endpoint, self.model, self.timeout = api_key, endpoint, model, timeout

    @classmethod
    def from_env(cls) -> "ChatCompletionProvider":
        requested = os.getenv("LLM_PROVIDER", "").strip().lower()
        if requested and requested not in _PROVIDERS:
            raise ProviderError("LLM_PROVIDER must be 'groq' or 'openrouter'")
        names = [requested] if requested else ["groq", "openrouter"]
        for name in names:
            settings = _PROVIDERS[name]
            api_key = os.getenv(settings["key"])
            if api_key:
                model = os.getenv("LLM_MODEL", settings["model"])
                return cls(api_key, settings["url"], model)
        expected = _PROVIDERS[names[0]]["key"] if requested else "GROQ_API_KEY or OPENROUTER_API_KEY"
        raise ProviderError(f"Set {expected} before starting the chat app")

    def __call__(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        request = Request(self.endpoint, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Groq's Cloudflare edge blocks urllib's default client signature.
            "User-Agent": "churn-data-analyst/1.0",
        })
        body: dict[str, Any] = self._send_with_rate_limit_backoff(request)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLM provider returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("LLM provider returned an empty response")
        return content

    def _send_with_rate_limit_backoff(self, request: Request, max_retries: int = 2) -> dict[str, Any]:
        """A 429 is a transport hiccup, not a bad plan — retry it here, once
        or twice with a short backoff, rather than surfacing it straight to
        the user (or worse, burning the agent's plan-retry budget on it,
        which would just hammer an already-rate-limited endpoint harder).
        Any other HTTP status is a real failure and is not retried.
        """
        for attempt in range(max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:  # nosec B310 - fixed HTTPS provider URL
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code != 429 or attempt >= max_retries:
                    raise ProviderError(f"LLM provider returned HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else 2.0 ** attempt
                except ValueError:
                    delay = 2.0 ** attempt
                time.sleep(min(delay, 10.0))
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise ProviderError("LLM provider request failed") from exc
        raise ProviderError("LLM provider returned HTTP 429")  # pragma: no cover - loop always returns/raises above
