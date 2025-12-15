"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test purdue client module.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

from src.clients.purdue_client import PurdueClient
from src.net.rate_limiter import RateLimiter


class DummyLimiter(RateLimiter):
    """
    DummyLimiter: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        super().__init__(max_calls=1, period_seconds=1.0)
        self.calls = 0

    def acquire(self) -> None:  # type: ignore[override]
        """
        acquire: Function description.
        :param:
        :returns:
        """

        self.calls += 1


class DummyResponse:
    """
    DummyResponse: Class description.
    """

    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        """
        __init__: Function description.
        :param status_code:
        :param payload:
        :returns:
        """

        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Dict[str, Any]:
        """
        json: Function description.
        :param:
        :returns:
        """

        return self._payload


class DummySession:
    """
    DummySession: Class description.
    """

    def __init__(self, response: DummyResponse) -> None:
        """
        __init__: Function description.
        :param response:
        :returns:
        """

        self.response = response
        self.calls: list[tuple[str, Dict[str, str], Dict[str, Any]]] = []

    def post(
        self,
        url: str,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: int = 30,
    ) -> DummyResponse:
        """
        post: Function description.
        :param url:
        :param headers:
        :param json:
        :param timeout:
        :returns:
        """

        self.calls.append((url, headers, json))
        return self.response


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    reset_env: Function description.
    :param monkeypatch:
    :returns:
    """

    module = importlib.import_module("src.utils.env")
    monkeypatch.setattr(module, "_ENV_LOADED", False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "test-token")


def test_generate_completion_success() -> None:
    """
    test_generate_completion_success: Function description.
    :param:
    :returns:
    """

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


def test_generate_completion_accepts_messages() -> None:
    """
    test_generate_completion_accepts_messages: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(200, {"id": "demo"})
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    result = client.generate_completion(
        messages=[
            {"role": "system", "content": "guide"},
            {"role": "user", "content": "question"},
        ]
    )

    assert result == {"id": "demo"}
    payload = session.calls[0][2]
    roles: List[str] = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "user"]


def test_generate_completion_failure() -> None:
    """
    test_generate_completion_failure: Function description.
    :param:
    :returns:
    """

    session = DummySession(DummyResponse(500, {"error": "oops"}))
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.generate_completion("hello")


def test_generate_completion_rejects_mixed_arguments() -> None:
    """
    test_generate_completion_rejects_mixed_arguments: Function description.
    :param:
    :returns:
    """

    client = PurdueClient(
        rate_limiter=DummyLimiter(),
        session=DummySession(DummyResponse(200, {})),
    )

    with pytest.raises(ValueError):
        client.generate_completion(
            "hello",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_generate_completion_requires_prompt_or_messages() -> None:
    """
    test_generate_completion_requires_prompt_or_messages: Function description.
    :param:
    :returns:
    """

    client = PurdueClient(
        rate_limiter=DummyLimiter(),
        session=DummySession(DummyResponse(200, {})),
    )

    with pytest.raises(
        ValueError, match="Either prompt or messages must be provided"
    ):
        client.generate_completion()


def test_generate_completion_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_generate_completion_requires_token: Function description.
    :param monkeypatch:
    :returns:
    """

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
    """
    test_llm_returns_text: Function description.
    :param:
    :returns:
    """

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


def test_llm_supports_message_payloads() -> None:
    """
    test_llm_supports_message_payloads: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                    }
                }
            ]
        },
    )
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    result = client.llm(
        messages=[
            {"role": "system", "content": "guide"},
            {"role": "user", "content": "question"},
        ]
    )

    assert result == "OK"
    payload = session.calls[0][2]
    assert payload["messages"][0]["role"] == "system"


def test_llm_raises_on_missing_content() -> None:
    """
    test_llm_raises_on_missing_content: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(200, {"choices": [{}]})
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.llm("Hi")


def test_llm_handles_delta_content() -> None:
    """
    test_llm_handles_delta_content: Function description.
    :param:
    :returns:
    """

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


def test_llm_raises_on_unexpected_response_structure() -> None:
    """
    test_llm_raises_on_unexpected_response_structure: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(200, {"choices": None})
    session = DummySession(response)
    client = PurdueClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(
        RuntimeError, match="Unexpected Purdue API response structure"
    ):
        client.llm("Hi")
