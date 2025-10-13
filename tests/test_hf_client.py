from __future__ import annotations

from typing import Any

import pytest

try:
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - environment guard
    pytest.skip(
        "huggingface_hub not installed; skipping HFClient tests",
        allow_module_level=True,
    )

from src.clients.hf_client import HFClient
from src.net.rate_limiter import RateLimiter


class DummyLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(max_calls=1, period_seconds=1.0)
        self.invocations = 0

    def acquire(self) -> None:  # type: ignore[override]
        self.invocations += 1


class DummyApi:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def model_info(self, repo_id: str) -> dict[str, Any]:
        self.calls.append(repo_id)
        return {"modelId": repo_id}


def test_hf_client_fetches_model_info() -> None:
    api = DummyApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    info = client.get_model_info("https://huggingface.co/org/model")

    assert info == {"modelId": "org/model"}
    assert api.calls == ["org/model"]
    assert limiter.invocations == 1


def test_model_exists_true() -> None:
    api = DummyApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    assert client.model_exists("https://huggingface.co/org/model") is True


def test_model_exists_handles_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        status_code = 404
        headers: dict[str, str] = {}
        text = "Not found"
        request = None

    class FailingApi(DummyApi):
        def model_info(self, repo_id: str) -> dict[str, Any]:
            raise HfHubHTTPError("missing", response=DummyResponse())

    failing_api = FailingApi()
    client = HFClient(api=failing_api, rate_limiter=DummyLimiter())

    assert client.model_exists("https://huggingface.co/org/missing") is False


def test_normalize_repo_id_accepts_plain_id() -> None:
    assert HFClient._normalize_repo_id("user/model") == "user/model"


def test_normalize_repo_id_rejects_invalid_host() -> None:
    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://example.com/user/model")


def test_normalize_repo_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("   ")


def test_normalize_repo_id_handles_prefixed_url() -> None:
    assert (
        HFClient._normalize_repo_id(
            "https://huggingface.co/models/user/model/tree/main"
        )
        == "user/model"
    )


def test_normalize_repo_id_requires_two_segments() -> None:
    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://huggingface.co/user")


def test_normalize_repo_id_rejects_empty_path() -> None:
    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://huggingface.co/")


def test_hf_client_instantiates_default_api(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class StubLimiter(DummyLimiter):
        def __init__(self) -> None:
            super().__init__()
            self.invocations = 0

    class StubApi:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def model_info(self, repo_id: str) -> dict[str, Any]:
            self.calls.append(repo_id)
            return {"modelId": repo_id}

    stub_instance = StubApi()

    def fake_hf_api() -> StubApi:
        return stub_instance

    monkeypatch.setattr("huggingface_hub.HfApi", fake_hf_api)

    limiter = StubLimiter()
    client = HFClient(rate_limiter=limiter)

    info = client.get_model_info("user/model")

    assert info == {"modelId": "user/model"}
    assert stub_instance.calls == ["user/model"]
    assert limiter.invocations == 1
