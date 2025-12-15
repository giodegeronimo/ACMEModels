"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test hf client module.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

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
        self.invocations = 0

    def acquire(self) -> None:  # type: ignore[override]
        """
        acquire: Function description.
        :param:
        :returns:
        """

        self.invocations += 1


class DummyApi:
    """
    DummyApi: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.calls: list[str] = []

    def model_info(self, repo_id: str) -> dict[str, Any]:
        """
        model_info: Function description.
        :param repo_id:
        :returns:
        """

        self.calls.append(repo_id)
        return {"modelId": repo_id}


def test_hf_client_fetches_model_info() -> None:
    """
    test_hf_client_fetches_model_info: Function description.
    :param:
    :returns:
    """

    api = DummyApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    info = client.get_model_info("https://huggingface.co/org/model")

    assert info == {"modelId": "org/model"}
    assert api.calls == ["org/model"]
    assert limiter.invocations == 1


def test_model_exists_true() -> None:
    """
    test_model_exists_true: Function description.
    :param:
    :returns:
    """

    api = DummyApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    assert client.model_exists("https://huggingface.co/org/model") is True


def test_model_exists_handles_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_model_exists_handles_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    class DummyResponse:
        """
        DummyResponse: Class description.
        """

        status_code = 404
        headers: dict[str, str] = {}
        text = "Not found"
        request = None

    class FailingApi(DummyApi):
        """
        FailingApi: Class description.
        """

        def model_info(self, repo_id: str) -> dict[str, Any]:
            """
            model_info: Function description.
            :param repo_id:
            :returns:
            """

            raise HfHubHTTPError(
                "missing", response=DummyResponse()  # type: ignore[arg-type]
            )

    failing_api = FailingApi()
    client = HFClient(api=failing_api, rate_limiter=DummyLimiter())

    assert client.model_exists("https://huggingface.co/org/missing") is False


def test_normalize_repo_id_accepts_plain_id() -> None:
    """
    test_normalize_repo_id_accepts_plain_id: Function description.
    :param:
    :returns:
    """

    assert HFClient._normalize_repo_id("user/model") == "user/model"


def test_normalize_repo_id_rejects_invalid_host() -> None:
    """
    test_normalize_repo_id_rejects_invalid_host: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://example.com/user/model")


def test_normalize_repo_id_rejects_empty() -> None:
    """
    test_normalize_repo_id_rejects_empty: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("   ")


def test_normalize_repo_id_handles_prefixed_url() -> None:
    """
    test_normalize_repo_id_handles_prefixed_url: Function description.
    :param:
    :returns:
    """

    assert (
        HFClient._normalize_repo_id(
            "https://huggingface.co/models/user/model/tree/main"
        )
        == "user/model"
    )


def test_normalize_repo_id_requires_two_segments() -> None:
    # Hugging Face supports legacy single-segment model IDs that live at the
    # root of the namespace (e.g. https:/
    # /huggingface.co/distilbert-base-uncased-distilled-squad).
    """
    test_normalize_repo_id_requires_two_segments: Function description.
    :param:
    :returns:
    """

    assert HFClient._normalize_repo_id("https://huggingface.co/user") == "user"


def test_normalize_repo_id_rejects_empty_path() -> None:
    """
    test_normalize_repo_id_rejects_empty_path: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://huggingface.co/")


def test_hf_client_instantiates_default_api(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_hf_client_instantiates_default_api: Function description.
    :param monkeypatch:
    :returns:
    """

    class StubLimiter(DummyLimiter):
        """
        StubLimiter: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            super().__init__()
            self.invocations = 0

    class StubApi:
        """
        StubApi: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.calls: list[str] = []

        def model_info(self, repo_id: str) -> dict[str, Any]:
            """
            model_info: Function description.
            :param repo_id:
            :returns:
            """

            self.calls.append(repo_id)
            return {"modelId": repo_id}

    stub_instance = StubApi()

    def fake_hf_api() -> StubApi:
        """
        fake_hf_api: Function description.
        :param:
        :returns:
        """

        return stub_instance

    monkeypatch.setattr("huggingface_hub.HfApi", fake_hf_api)

    limiter = StubLimiter()
    client = HFClient(rate_limiter=limiter)

    info = client.get_model_info("user/model")

    assert info == {"modelId": "user/model"}
    assert stub_instance.calls == ["user/model"]
    assert limiter.invocations == 1


def test_get_model_readme() -> None:
    """
    test_get_model_readme: Function description.
    :param:
    :returns:
    """

    class DummyResponse:
        """
        DummyResponse: Class description.
        """

        def __init__(self, text: str, status_code: int = 200) -> None:
            """
            __init__: Function description.
            :param text:
            :param status_code:
            :returns:
            """

            self._text = text
            self.status_code = status_code
            self.encoding: str | None = None

        @property
        def text(self) -> str:  # pragma: no cover - simple accessor
            """
            text: Function description.
            :param:
            :returns:
            """

            return self._text

        def raise_for_status(self) -> None:
            """
            raise_for_status: Function description.
            :param:
            :returns:
            """

            if self.status_code >= 400:
                raise requests.HTTPError(
                    "HTTP error {}".format(self.status_code),
                    request=None,  # type: ignore[arg-type]
                    response=None,  # type: ignore[arg-type]
                )

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

            self._response = response
            self.calls: list[tuple[str, int]] = []

        def get(self, url: str, timeout: int = 30) -> DummyResponse:
            """
            get: Function description.
            :param url:
            :param timeout:
            :returns:
            """

            self.calls.append((url, timeout))
            return self._response

    api = DummyApi()
    session = DummySession(DummyResponse("README contents"))
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter, http_session=session)

    contents = client.get_model_readme("https://huggingface.co/org/model")

    assert contents == "README contents"
    assert session.calls and session.calls[0][0].endswith("README.md")
    assert limiter.invocations == 1


def test_list_model_files_uses_repo_tree() -> None:
    """
    test_list_model_files_uses_repo_tree: Function description.
    :param:
    :returns:
    """

    class RepoItem:
        """
        RepoItem: Class description.
        """

        def __init__(self, path: str, size: int) -> None:
            """
            __init__: Function description.
            :param path:
            :param size:
            :returns:
            """

            self.path = path
            self.size = size

    class TreeApi:
        """
        TreeApi: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.calls: list[tuple[str, bool]] = []

        def list_repo_tree(
            self,
            repo_id: str,
            repo_type: str = "model",
            recursive: bool = True,
        ) -> list[RepoItem]:
            """
            list_repo_tree: Function description.
            :param repo_id:
            :param repo_type:
            :param recursive:
            :returns:
            """

            self.calls.append((repo_id, recursive))
            return [
                RepoItem("a.bin", 100),
                RepoItem("b.safetensors", 200),
            ]

    api = TreeApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    files = client.list_model_files("https://huggingface.co/org/model")

    assert files == [("a.bin", 100), ("b.safetensors", 200)]
    assert limiter.invocations == 1


def test_list_model_files_falls_back_to_siblings() -> None:
    """
    test_list_model_files_falls_back_to_siblings: Function description.
    :param:
    :returns:
    """

    class Sibling:
        """
        Sibling: Class description.
        """

        def __init__(self, rfilename: str, size: int) -> None:
            """
            __init__: Function description.
            :param rfilename:
            :param size:
            :returns:
            """

            self.rfilename = rfilename
            self.size = size

    class Info:
        """
        Info: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.siblings = [Sibling("weights/pytorch_model.bin", 123)]

    class FallbackApi:
        """
        FallbackApi: Class description.
        """

        def list_repo_tree(self, *args: Any, **kwargs: Any) -> list[Any]:
            """
            list_repo_tree: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("tree not available")

        def model_info(self, repo_id: str) -> Info:
            """
            model_info: Function description.
            :param repo_id:
            :returns:
            """

            return Info()

    api = FallbackApi()
    limiter = DummyLimiter()
    client = HFClient(api=api, rate_limiter=limiter)

    files = client.list_model_files("org/model")

    assert files == [("weights/pytorch_model.bin", 123)]
    assert limiter.invocations == 1


def test_list_model_files_ignores_missing_sizes() -> None:
    """
    test_list_model_files_ignores_missing_sizes: Function description.
    :param:
    :returns:
    """

    class RepoItem:
        """
        RepoItem: Class description.
        """

        def __init__(self, path: str, size: int | None) -> None:
            """
            __init__: Function description.
            :param path:
            :param size:
            :returns:
            """

            self.path = path
            self.size = size  # type: ignore[assignment]

    class TreeApi:
        """
        TreeApi: Class description.
        """

        def list_repo_tree(self, *args: Any, **kwargs: Any) -> list[RepoItem]:
            """
            list_repo_tree: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            return [RepoItem("a.bin", 100), RepoItem("b.bin", None)]

    api = TreeApi()
    client = HFClient(api=api, rate_limiter=DummyLimiter())

    files = client.list_model_files("org/model")

    assert files == [("a.bin", 100)]


def test_dataset_exists_true_and_false() -> None:
    """
    test_dataset_exists_true_and_false: Function description.
    :param:
    :returns:
    """

    class DatasetApi:
        """
        DatasetApi: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.calls: list[str] = []

        def dataset_info(self, dataset_id: str) -> dict[str, Any]:
            """
            dataset_info: Function description.
            :param dataset_id:
            :returns:
            """

            self.calls.append(dataset_id)
            return {"id": dataset_id}

    api = DatasetApi()
    client = HFClient(api=api, rate_limiter=DummyLimiter())

    assert client.dataset_exists("https://huggingface.co/datasets/org/ds") is True

    class MissingApi(DatasetApi):
        """
        MissingApi: Class description.
        """

        def dataset_info(self, dataset_id: str) -> dict[str, Any]:
            """
            dataset_info: Function description.
            :param dataset_id:
            :returns:
            """

            raise RuntimeError("missing")

    missing = HFClient(api=MissingApi(), rate_limiter=DummyLimiter())
    assert missing.dataset_exists("org/ds") is False


def test_get_dataset_info_and_count_models() -> None:
    """
    test_get_dataset_info_and_count_models: Function description.
    :param:
    :returns:
    """

    class ListApi:
        """
        ListApi: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.datasets: list[str] = []

        def dataset_info(self, dataset_id: str) -> dict[str, Any]:
            """
            dataset_info: Function description.
            :param dataset_id:
            :returns:
            """

            self.datasets.append(dataset_id)
            return {"id": dataset_id}

        def list_models(self, *, trained_dataset: str, limit: int = 1000):
            """
            list_models: Function description.
            :param trained_dataset:
            :param limit:
            :returns:
            """

            assert limit == 3
            yield {"id": "m1"}
            yield {"id": "m2"}

    api = ListApi()
    client = HFClient(api=api, rate_limiter=DummyLimiter())

    assert client.get_dataset_info("org/ds") == {"id": "org/ds"}
    assert client.count_models_trained_on_dataset("org/ds", limit=3) == 2

    class FailingListApi(ListApi):
        """
        FailingListApi: Class description.
        """

        def list_models(self, *, trained_dataset: str, limit: int = 1000):
            """
            list_models: Function description.
            :param trained_dataset:
            :param limit:
            :returns:
            """

            raise RuntimeError("boom")

    failing = HFClient(api=FailingListApi(), rate_limiter=DummyLimiter())
    assert failing.count_models_trained_on_dataset("org/ds") == 0


def test_get_model_readme_handles_404_and_request_failures() -> None:
    """
    test_get_model_readme_handles_404_and_request_failures: Function description.
    :param:
    :returns:
    """

    class DummyResponse:
        """
        DummyResponse: Class description.
        """

        def __init__(self, status_code: int) -> None:
            """
            __init__: Function description.
            :param status_code:
            :returns:
            """

            self.status_code = status_code
            self.encoding: str | None = None
            self.text = "x"

        def raise_for_status(self) -> None:
            """
            raise_for_status: Function description.
            :param:
            :returns:
            """

            if self.status_code >= 400:
                raise requests.HTTPError("boom")

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

            self._response = response

        def get(self, url: str, timeout: int = 30) -> DummyResponse:
            """
            get: Function description.
            :param url:
            :param timeout:
            :returns:
            """

            return self._response

    api = DummyApi()
    client_404 = HFClient(
        api=api,
        rate_limiter=DummyLimiter(),
        http_session=DummySession(DummyResponse(404)),
    )
    assert client_404.get_model_readme("org/model") == ""

    class FailingSession(DummySession):
        """
        FailingSession: Class description.
        """

        def get(self, url: str, timeout: int = 30) -> DummyResponse:
            """
            get: Function description.
            :param url:
            :param timeout:
            :returns:
            """

            raise requests.RequestException("boom")

    client_fail = HFClient(
        api=api,
        rate_limiter=DummyLimiter(),
        http_session=FailingSession(DummyResponse(200)),
    )
    assert client_fail.get_model_readme("org/model") == ""


def test_normalize_repo_and_dataset_ids_edge_cases() -> None:
    """
    test_normalize_repo_and_dataset_ids_edge_cases: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError):
        HFClient._normalize_repo_id("https://huggingface.co/models/")

    assert (
        HFClient._normalize_dataset_id("https://huggingface.co/datasets/org/ds")
        == "org/ds"
    )
    with pytest.raises(ValueError):
        HFClient._normalize_dataset_id("https://example.com/datasets/org/ds")
    with pytest.raises(ValueError):
        HFClient._normalize_dataset_id("https://huggingface.co/datasets/")


def test_normalize_dataset_id_rejects_empty() -> None:
    """
    test_normalize_dataset_id_rejects_empty: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="Dataset identifier cannot be empty"):
        HFClient._normalize_dataset_id("   ")
