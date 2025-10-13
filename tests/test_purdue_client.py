from __future__ import annotations

import importlib
from typing import Any, Dict

import pytest

from src.clients.purdue_client import PurdueClient
from src.net.rate_limiter import RateLimiter


class DummyLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(max_calls=1, period_seconds=1.0)
        self.calls = 0

    def acquire(self) -> None:  # type: ignore[override]
        self.calls += 1


class DummyResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, Dict[str, str], Dict[str, Any]]] = []

    def post(
        self,
        url: str,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: int = 30,
    ) -> DummyResponse:
        self.calls.append((url, headers, json))
        return self.response


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("src.utils.env")
    monkeypatch.setattr(module, "_ENV_LOADED", False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "test-token")


def test_generate_completion_success() -> None:
    response = DummyResponse(200, {"id": "demo"})
    session = DummySession(response)
    limiter = DummyLimiter()

    client = PurdueClient(rate_limiter=limiter, session=session)
    result = client.generate_completion(
        "hello",
        model="demo-model",

    )

    assert result == {"id": "demo"}
    assert limiter.calls == 1
    url, headers, payload = session.calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer test-token"
    assert payload["model"] == "demo-model"
    assert payload["messages"][0]["content"] == "hello"
    assert payload["temperature"] == 0.0
    assert payload["stream"] is False


def test_generate_completion_failure() -> None:
    session = DummySession(DummyResponse(500, {"error": "oops"}))
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.generate_completion("hello")


def test_generate_completion_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    module = importlib.import_module("src.utils.env")
    monkeypatch.setattr(module, "_ENV_LOADED", False)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "")

    with pytest.raises(RuntimeError):
        PurdueClient(
            rate_limiter=DummyLimiter(),
            session=DummySession(DummyResponse(200, {})),
        )


def test_llm_returns_text() -> None:
    response = DummyResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    }
                }
            ]
        },
    )
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    content = client.llm("Hi there")

    assert content == "Hello!"
    assert session.calls[0][2]["messages"][0]["content"] == "Hi there"


def test_llm_raises_on_missing_content() -> None:
    response = DummyResponse(200, {"choices": [{}]})
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.llm("Hi")


def test_llm_handles_delta_content() -> None:
    response = DummyResponse(
        200,
        {
            "choices": [
                {
                    "delta": {
                        "content": "Streaming chunk",
                    }
                }
            ]
        },
    )
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    content = client.llm("Hi", stream=True)
    assert content == "Streaming chunk"
