from __future__ import annotations

import importlib.util
import io
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

import src.runner as runner_mod
from src.clients.base_client import BaseClient
from src.clients.git_client import GitClient
from src.clients.hf_client import HFClient
from src.clients.purdue_client import ENV_TOKEN_KEY, PurdueClient
from src.logging_config import configure_logging
from src.metrics.base import Metric
from src.metrics.metric_result import MetricResult
from src.metrics.net_score import NetScoreCalculator
from src.metrics.ratings import RatingComputationError, compute_model_rating
from src.metrics.registry import MetricDispatcher
from src.net.rate_limiter import RateLimiter
from src.runner import (_cleanup_coverage_artifacts, _collect_line_coverage,
                        _summarize_pytest_output, install_dependencies,
                        run_parser)
from src.webapp import WebApp, _header_lookup, create_app


def _limiter() -> RateLimiter:
    clock = {"t": 0.0}

    def time_fn() -> float:
        clock["t"] += 1.0
        return clock["t"]

    return RateLimiter(
        max_calls=10_000,
        period_seconds=1.0,
        time_fn=time_fn,
        sleep_fn=lambda _seconds: None,
    )


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        encoding: str | None = "utf-8",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.encoding = encoding

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeGitSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
        self.calls.append(url)
        if url.endswith("/repos/octo/test"):
            return _FakeResponse(payload={"default_branch": "main"})
        if "/git/trees/main" in url:
            return _FakeResponse(
                payload={"tree": [{"type": "blob", "path": "README.md"}]}
            )
        if "/contributors" in url:
            return _FakeResponse(payload=[{"login": "octo", "contributions": 1}])
        if "/commits/" in url and url.endswith("/pulls"):
            return _FakeResponse(payload=[{"number": 1, "title": "PR"}])
        return _FakeResponse(status_code=404, payload={})

    def post(
        self, url: str, json: Any, timeout: int, headers: Any = None
    ) -> _FakeResponse:
        self.calls.append(url)
        if url.endswith("/graphql"):
            return _FakeResponse(
                payload={"data": {"repository": {"ref": {"blame": {"ranges": []}}}}}
            )
        return _FakeResponse(status_code=500, payload={})


def test_base_client_executes_operation_and_calls_rate_limiter() -> None:
    logger = logging.getLogger("test-client")
    logger.setLevel(logging.DEBUG)
    client = BaseClient[int](_limiter(), logger=logger)
    assert client._execute_with_rate_limit(lambda: 5, name="op") == 5


def test_git_client_smoke_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    client = GitClient(rate_limiter=_limiter(), session=_FakeGitSession())

    assert client.get_repo_metadata("https://github.com/octo/test")["default_branch"] == "main"
    assert client.list_repo_files("https://github.com/octo/test") == ["README.md"]
    assert client.list_repo_contributors("https://github.com/octo/test") == [
        {"login": "octo", "contributions": 1}
    ]
    assert client.get_file_blame("https://github.com/octo/test", "README.md") == []
    assert client.get_commit_associated_pr("https://github.com/octo/test", "abc123") == {
        "number": 1,
        "title": "PR",
    }

    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.get_repo_metadata("https://example.com/repo")


def test_git_client_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class ErrorSession:
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(status_code=500, payload={})
            if "/contributors" in url:
                return _FakeResponse(payload={"not": "a-list"})
            if "/git/trees/" in url:
                return _FakeResponse(status_code=500, payload={})
            return _FakeResponse(status_code=404, payload={})

        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(status_code=500, payload={})

    client = GitClient(rate_limiter=_limiter(), session=ErrorSession())

    with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
        client.get_repo_metadata("https://github.com/octo")

    with pytest.raises(RuntimeError, match="Failed to retrieve repo metadata"):
        client.get_repo_metadata("https://github.com/octo/test")

    assert client.list_repo_contributors("https://github.com/octo/test") == []
    assert client.list_repo_contributors("https://github.com/octo/test", per_page=5000) == []

    with pytest.raises(RuntimeError, match="Failed to retrieve repo tree"):
        client.list_repo_files("https://github.com/octo/test", branch="main")


def test_git_client_blame_graphql_error_handling() -> None:
    class Session:
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(payload={"default_branch": "main"})
            return _FakeResponse(payload={})

        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(
                payload={"errors": [{"message": "Path does not exist"}]}
            )

    client = GitClient(rate_limiter=_limiter(), session=Session())
    assert client.get_file_blame("https://github.com/octo/test", "missing.txt") == []

    class Session2(Session):
        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(payload={"errors": [{"message": "boom"}]})

    client2 = GitClient(rate_limiter=_limiter(), session=Session2())
    with pytest.raises(RuntimeError, match="GraphQL error"):
        client2.get_file_blame("https://github.com/octo/test", "README.md")


def test_git_client_commit_pr_paths() -> None:
    class Session:
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(payload={"default_branch": "main"})
            if url.endswith("/pulls"):
                return _FakeResponse(status_code=404, payload={})
            return _FakeResponse(payload={})

        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(payload={})

    client = GitClient(rate_limiter=_limiter(), session=Session())
    assert (
        client.get_commit_associated_pr("https://github.com/octo/test", "abc")
        is None
    )

    class Session2(Session):
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(payload={"default_branch": "main"})
            if url.endswith("/pulls"):
                return _FakeResponse(status_code=200, payload=[])
            return _FakeResponse(payload={})

    client2 = GitClient(rate_limiter=_limiter(), session=Session2())
    assert (
        client2.get_commit_associated_pr("https://github.com/octo/test", "abc")
        is None
    )

    class Session3(Session):
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(payload={"default_branch": "main"})
            if url.endswith("/pulls"):
                return _FakeResponse(status_code=500, payload={})
            return _FakeResponse(payload={})

    client3 = GitClient(rate_limiter=_limiter(), session=Session3())
    with pytest.raises(RuntimeError, match="Failed to get PR"):
        client3.get_commit_associated_pr("https://github.com/octo/test", "abc")


def test_git_client_additional_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    class Session:
        def get(self, url: str, timeout: int, headers: Any = None) -> _FakeResponse:
            if url.endswith("/repos/octo/test"):
                return _FakeResponse(payload={"default_branch": "main"})
            if "/contributors" in url:
                return _FakeResponse(status_code=500, payload={})
            return _FakeResponse(payload={})

        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            if url.endswith("/graphql"):
                return _FakeResponse(status_code=500, payload={})
            return _FakeResponse(payload={})

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = GitClient(rate_limiter=_limiter(), session=Session())

    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.list_repo_files("https://example.com/repo")
    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.list_repo_contributors("https://example.com/repo")

    with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
        client.list_repo_files("https://github.com/octo")
    with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
        client.list_repo_contributors("https://github.com/octo")

    with pytest.raises(RuntimeError, match="Failed to retrieve repo contributors"):
        client.list_repo_contributors("https://github.com/octo/test")

    with pytest.raises(RuntimeError, match="Failed to retrieve blame"):
        client.get_file_blame("https://github.com/octo/test", "README.md")

    class Session2(Session):
        def post(
            self, url: str, json: Any, timeout: int, headers: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(
                payload={
                    "data": {
                        "repository": {"ref": {"blame": {"ranges": "oops"}}}
                    }
                }
            )

    client2 = GitClient(rate_limiter=_limiter(), session=Session2())
    assert client2.get_file_blame("https://github.com/octo/test", "README.md") == []


def test_hf_client_normalization_and_readme_download(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def model_info(self, repo_id: str) -> Any:
            return {"repo": repo_id}

        def dataset_info(self, dataset_id: str) -> Any:
            return {"ds": dataset_id}

        def list_models(self, trained_dataset: str, limit: int = 1000):  # type: ignore[no-untyped-def]
            return iter([object(), object()])

        def list_repo_tree(self, repo_id: str, repo_type: str, recursive: bool = True):  # type: ignore[no-untyped-def]
            return [type("F", (), {"path": "a.txt", "size": 1})()]

    class FakeSession:
        def get(self, url: str, timeout: int = 30) -> _FakeResponse:
            return _FakeResponse(status_code=404)

    monkeypatch.setattr("src.clients.hf_client.HfHubHTTPError", type("E", (Exception,), {}))
    client = HFClient(
        api=FakeApi(),
        rate_limiter=_limiter(),
        http_session=FakeSession(),
    )

    info: Any = client.get_model_info("https://huggingface.co/acme/model")
    assert info["repo"] == "acme/model"
    assert client.dataset_exists("https://huggingface.co/datasets/acme/ds") is True
    assert client.get_model_readme("acme/model") == ""
    assert client.count_models_trained_on_dataset("acme/ds") == 2
    assert client.list_model_files("acme/model") == [("a.txt", 1)]

    with pytest.raises(ValueError, match="cannot be empty"):
        client._normalize_repo_id("   ")


def test_hf_client_error_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.clients.hf_client as hf_mod

    class FakeHubError(Exception):
        pass

    monkeypatch.setattr(hf_mod, "HfHubHTTPError", FakeHubError)

    class Sibling:
        def __init__(self, rfilename: str, size: int) -> None:
            self.rfilename = rfilename
            self.size = size

    class FakeApi:
        def model_info(self, repo_id: str) -> Any:
            if repo_id == "missing/repo":
                raise FakeHubError("nope")
            return type("Info", (), {"siblings": [Sibling("b.txt", 2)]})()

        def dataset_info(self, dataset_id: str) -> Any:
            raise FakeHubError("no dataset")

        def list_models(self, trained_dataset: str, limit: int = 1000):  # type: ignore[no-untyped-def]
            return iter([])

        def list_repo_tree(self, repo_id: str, repo_type: str, recursive: bool = True):  # type: ignore[no-untyped-def]
            raise RuntimeError("tree unavailable")

    class ExplodingSession:
        def get(self, url: str, timeout: int = 30) -> Any:
            raise requests.RequestException("network")

    client = HFClient(
        api=FakeApi(),
        rate_limiter=_limiter(),
        http_session=ExplodingSession(),
    )

    assert client.model_exists("missing/repo") is False
    assert client.dataset_exists("acme/ds") is False
    assert client.get_model_readme("acme/model") == ""
    assert client.list_model_files("acme/model") == [("b.txt", 2)]
    with pytest.raises(ValueError, match="Unsupported Hugging Face host"):
        client._normalize_dataset_id("https://example.com/datasets/acme/ds")


def test_hf_client_additional_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.clients.hf_client as hf_mod

    class FakeHubError(Exception):
        pass

    monkeypatch.setattr(hf_mod, "HfHubHTTPError", FakeHubError)

    class RepoFile:
        def __init__(self, path: str, size: int | None) -> None:
            self.path = path
            self.size = size

    class Sibling:
        def __init__(self, rfilename: str, size: int | None) -> None:
            self.rfilename = rfilename
            self.size = size

    class FakeApi:
        def model_info(self, repo_id: str) -> Any:
            return type("Info", (), {"siblings": [Sibling("c.txt", 3), Sibling("bad", None)]})()

        def dataset_info(self, dataset_id: str) -> Any:
            return {"ds": dataset_id}

        def list_models(self, trained_dataset: str, limit: int = 1000):  # type: ignore[no-untyped-def]
            raise FakeHubError("no listing")

        def list_repo_tree(self, repo_id: str, repo_type: str, recursive: bool = True):  # type: ignore[no-untyped-def]
            return [RepoFile("a.txt", 1), RepoFile("b.txt", None)]

    class FakeSession:
        def get(self, url: str, timeout: int = 30) -> _FakeResponse:
            return _FakeResponse(status_code=200, text="hello", encoding=None)

    client = HFClient(
        api=FakeApi(),
        rate_limiter=_limiter(),
        http_session=FakeSession(),
    )

    assert client.model_exists("https://huggingface.co/acme/model") is True
    assert client.get_dataset_info("https://huggingface.co/datasets/acme/ds")["ds"] == "acme/ds"
    assert client.get_model_readme("https://huggingface.co/models/acme/model") == "hello"
    assert client.count_models_trained_on_dataset("acme/ds") == 0
    assert client.list_model_files("https://huggingface.co/acme/model") == [("a.txt", 1)]

    assert client._normalize_repo_id("https://huggingface.co/distilbert") == "distilbert"

    with pytest.raises(ValueError, match="Unable to extract dataset id"):
        client._normalize_dataset_id("https://huggingface.co/datasets/")


def test_purdue_client_completion_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TOKEN_KEY, "abc")

    class FakeSession:
        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: int = 30,
        ) -> _FakeResponse:
            return _FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})

    client = PurdueClient(
        rate_limiter=_limiter(),
        session=FakeSession(),
        base_url="https://x",
    )
    assert client.llm("hi") == "ok"

    with pytest.raises(ValueError, match="not both"):
        client.generate_completion(prompt="a", messages=[{"role": "user", "content": "b"}])

    with pytest.raises(ValueError, match="must be provided"):
        client.generate_completion()


def test_purdue_client_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_TOKEN_KEY, raising=False)
    with pytest.raises(RuntimeError, match=ENV_TOKEN_KEY):
        PurdueClient(rate_limiter=_limiter(), session=object())  # type: ignore[arg-type]


def test_purdue_client_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TOKEN_KEY, "abc")

    class ExplodingSession:
        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: int = 30,
        ) -> _FakeResponse:
            return _FakeResponse(status_code=500, text="nope", payload={})

    client = PurdueClient(
        rate_limiter=_limiter(),
        session=ExplodingSession(),
        base_url="https://x",
    )
    with pytest.raises(RuntimeError, match="Purdue API returned"):
        client.generate_completion(prompt="hi")


def test_configure_logging_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.logging_config as logging_config

    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setenv("LOG_LEVEL", "2")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    configure_logging()
    configure_logging()

    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setenv("LOG_LEVEL", "1")
    monkeypatch.delenv("LOG_FILE", raising=False)
    configure_logging()

    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setenv("LOG_LEVEL", "0")
    configure_logging()

    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setenv("LOG_LEVEL", "invalid")
    configure_logging()


def test_net_score_calculator_handles_empty_and_numeric_values() -> None:
    calc = NetScoreCalculator()
    results = calc.with_net_score(
        [MetricResult(metric="m", key="k", value=None, latency_ms=1)]
    )
    assert results[0].key == "net_score"

    results = calc.with_net_score(
        [
            MetricResult(metric="m", key="k", value=1.0, latency_ms=1),
            MetricResult(metric="m2", key="k2", value=0.0, latency_ms=1),
        ]
    )
    assert results[0].value == 0.5


def test_compute_model_rating_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.metrics.ratings as ratings_mod

    class FakeDispatcher:
        def compute(self, records: list[dict[str, str]]):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    monkeypatch.setattr(ratings_mod, "MetricDispatcher", FakeDispatcher)
    with pytest.raises(RatingComputationError, match="Failed to compute rating"):
        compute_model_rating("https://huggingface.co/acme/model")


def test_compute_model_rating_success_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.metrics.ratings as ratings_mod

    class FakeDispatcher:
        def compute(self, records: list[dict[str, str]]):  # type: ignore[no-untyped-def]
            return [[MetricResult(metric="m", key="k", value=1.0, latency_ms=1)]]

    class FakeFormatter:
        def format_records(self, records: Any, results: Any):  # type: ignore[no-untyped-def]
            return [{"ok": True}]

    monkeypatch.setattr(ratings_mod, "MetricDispatcher", FakeDispatcher)
    monkeypatch.setattr(ratings_mod, "ResultsFormatter", FakeFormatter)
    assert compute_model_rating("https://huggingface.co/acme/model") == {"ok": True}

    class EmptyFormatter(FakeFormatter):
        def format_records(self, records: Any, results: Any):  # type: ignore[no-untyped-def]
            return []

    monkeypatch.setattr(ratings_mod, "ResultsFormatter", EmptyFormatter)
    with pytest.raises(RatingComputationError, match="No rating results produced"):
        compute_model_rating("https://huggingface.co/acme/model")

    class BadFormatter(FakeFormatter):
        def format_records(self, records: Any, results: Any):  # type: ignore[no-untyped-def]
            return ["not-a-mapping"]

    monkeypatch.setattr(ratings_mod, "ResultsFormatter", BadFormatter)
    with pytest.raises(RatingComputationError, match="Rating payload"):
        compute_model_rating("https://huggingface.co/acme/model")


def test_runner_helpers_and_cleanup(tmp_path: Path) -> None:
    passed, total = _summarize_pytest_output("3 passed, 1 skipped", "")
    assert passed == 3
    assert total >= 3

    artifact = tmp_path / "artifact.tmp"
    artifact.write_text("x", encoding="utf-8")
    _cleanup_coverage_artifacts([artifact])
    assert not artifact.exists()

    assert install_dependencies(tmp_path / "missing.txt") == 1
    assert run_parser(tmp_path / "missing_urls.txt") == 1
    assert _collect_line_coverage(tmp_path / "missing_coverage.dat") == 0.0


def test_runner_run_tests_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    coverage_path = tmp_path / "cov.dat"

    def fake_mkstemp(prefix: str, suffix: str):  # type: ignore[no-untyped-def]
        fd = os.open(str(coverage_path), os.O_CREAT | os.O_RDWR)
        return fd, str(coverage_path)

    def fake_run_pytest_subprocess(
        args: Any,
        env: Any,
    ) -> subprocess.CompletedProcess[str]:
        assert "--cov=src" in args
        assert env.get("COVERAGE_FILE") == str(coverage_path)
        assert env.get("ACME_IGNORE_FAIL") == "1"
        return subprocess.CompletedProcess(
            args=["pytest"],
            returncode=0,
            stdout="3 passed",
            stderr="",
        )

    cleanup_calls: list[tuple[Path, ...]] = []

    monkeypatch.setattr(runner_mod.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(
        runner_mod,
        "_run_pytest_subprocess",
        fake_run_pytest_subprocess,
    )
    monkeypatch.setattr(runner_mod, "_collect_line_coverage", lambda _p: 92.6)
    monkeypatch.setattr(
        runner_mod,
        "_cleanup_coverage_artifacts",
        lambda artifacts=None: cleanup_calls.append(
            tuple(Path(a) for a in (artifacts or []))
        ),
    )

    assert runner_mod.run_tests() == 0
    out = capsys.readouterr().out
    assert "3/3 test cases passed" in out
    assert "93% line coverage achieved" in out
    assert cleanup_calls


def test_runner_run_tests_failure_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    coverage_path = tmp_path / "cov.dat"

    def fake_mkstemp(prefix: str, suffix: str):  # type: ignore[no-untyped-def]
        fd = os.open(str(coverage_path), os.O_CREAT | os.O_RDWR)
        return fd, str(coverage_path)

    def fake_run_pytest_subprocess(
        args: Any,
        env: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["pytest"],
            returncode=2,
            stdout="OUT",
            stderr="ERR",
        )

    monkeypatch.setattr(runner_mod.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(
        runner_mod,
        "_run_pytest_subprocess",
        fake_run_pytest_subprocess,
    )

    assert runner_mod.run_tests() == 2
    captured = capsys.readouterr()
    assert "OUT" in captured.out
    assert "ERR" in captured.err


def test_runner_dispatch_web_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("ACME_WEB_PORT", "not-an-int")
    assert runner_mod.dispatch(["./run", "web"]) == 1
    assert "Invalid ACME_WEB_PORT value" in capsys.readouterr().err


def test_runner_more_coverage_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    # Exercise the "__package__ in (None,'')" bootstrap block by importing the
    # file as a standalone module (different module name, same file path).
    runner_path = Path(runner_mod.__file__)
    saved_sys_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            "runner_standalone_test",
            runner_path,
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        module.__package__ = ""
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert module._PKG_PREFIX == ""
    finally:
        sys.path[:] = saved_sys_path

    # _get_cli_main caching.
    monkeypatch.setattr(runner_mod, "_CLI_MAIN", None)

    class FakeCli:
        @staticmethod
        def main(argv: Any = None) -> int:
            return 0

    monkeypatch.setattr(runner_mod, "_import_from_src", lambda _m: FakeCli)
    first = runner_mod._get_cli_main()
    second = runner_mod._get_cli_main()
    assert first is second

    # install_dependencies happy path and CalledProcessError path.
    requirements = tmp_path / "req.txt"
    requirements.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        runner_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0),
    )
    assert runner_mod.install_dependencies(requirements) == 0

    def raising_run(*_a: Any, **_k: Any) -> Any:
        raise subprocess.CalledProcessError(7, ["pip3"])

    monkeypatch.setattr(runner_mod.subprocess, "run", raising_run)
    assert runner_mod.install_dependencies(requirements) == 7

    # run_pytest branches.
    captured: dict[str, Any] = {}

    def fake_pytest_run(cmd: list[str], env: dict[str, str]):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_pytest_run)
    assert runner_mod.run_pytest() == 0
    assert captured["env"]["ACME_IGNORE_FAIL"] == "1"
    assert runner_mod.run_pytest(["-k", "x"]) == 0
    assert "-k" in captured["cmd"]

    # run_parser success path.
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://huggingface.co/acme/model\n", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "_get_cli_main", lambda: (lambda _argv: 0))
    assert runner_mod.run_parser(url_file) == 0

    # dispatch branches.
    monkeypatch.setattr(runner_mod, "configure_logging", lambda: None)
    monkeypatch.setattr(runner_mod, "install_dependencies", lambda _p: 0)
    assert runner_mod.dispatch(["./run", "install"]) == 0

    monkeypatch.setattr(runner_mod, "run_tests", lambda: 0)
    assert runner_mod.dispatch(["./run", "test"]) == 0

    def fake_run_pytest(args: Any = None) -> int:
        assert args == ["-k", "x"]
        return 0

    monkeypatch.setattr(runner_mod, "run_pytest", fake_run_pytest)
    assert runner_mod.dispatch(["./run", "pytest", "-k", "x"]) == 0

    class FakeApp:
        def run(self, host: str, port: int, debug: bool = False) -> None:
            print(f"running {host}:{port} debug={debug}")

    class FakeWebModule:
        @staticmethod
        def create_app() -> FakeApp:
            return FakeApp()

    monkeypatch.setattr(runner_mod, "_import_from_src", lambda _m: FakeWebModule)
    monkeypatch.setenv("ACME_WEB_PORT", "5001")
    monkeypatch.setenv("ACME_WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("FLASK_DEBUG", "1")
    assert runner_mod.dispatch(["./run", "web"]) == 0
    assert "running 127.0.0.1:5001 debug=True" in capsys.readouterr().out

    monkeypatch.setattr(runner_mod, "validate_runtime_environment", lambda: None)
    monkeypatch.setattr(runner_mod, "run_parser", lambda _p: 0)
    assert runner_mod.dispatch(["./run", "URL_FILE"]) == 0

    assert runner_mod.dispatch(["./run"]) == 1

    monkeypatch.setattr(runner_mod, "dispatch", lambda _argv: 123)
    monkeypatch.setattr(runner_mod.sys, "argv", ["./run", "test"])
    assert runner_mod.main(None) == 123


def test_cli_app_run_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    import src.CLIApp as cli_mod

    urls = tmp_path / "urls.txt"
    urls.write_text("https://huggingface.co/acme/model\n", encoding="utf-8")

    class FakeParser:
        def __init__(self, url_file: Path) -> None:
            self._url_file = url_file

        def parse(self) -> list[dict[str, str]]:
            return [{"hf_url": "https://huggingface.co/acme/model"}]

    class FakeDispatcher:
        def compute(self, url_records: Any):  # type: ignore[no-untyped-def]
            return [[MetricResult(metric="m", key="k", value=1.0, latency_ms=1)]]

    class FakeFormatter:
        def format_records(self, records: Any, results: Any):  # type: ignore[no-untyped-def]
            return [{"x": 1}]

    monkeypatch.setattr(cli_mod, "Parser", FakeParser)
    monkeypatch.setattr(cli_mod, "MetricDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli_mod, "NetScoreCalculator", NetScoreCalculator)
    monkeypatch.setattr(cli_mod, "ResultsFormatter", FakeFormatter)
    monkeypatch.setattr(cli_mod, "to_ndjson_line", lambda record: '{"x": 1}')

    app = cli_mod.CLIApp(urls)
    assert app.run() == 0
    assert '{"x": 1}' in capsys.readouterr().out


def test_registry_dispatcher_compute_and_metric_repr() -> None:
    class DummyMetric(Metric):
        name = "dummy"
        key = "dummy"

        def compute(self, url_record: dict[str, str]) -> float:
            return 1.0

    dispatcher = MetricDispatcher(metrics=[DummyMetric(name="n", key="k")])
    assert len(dispatcher.metrics) == 1
    assert dispatcher.compute([{"hf_url": "x"}])[0][0].key == "k"
    assert "DummyMetric" in repr(DummyMetric(name="n", key="k"))


def test_registry_default_metrics_instantiation_smoke() -> None:
    dispatcher = MetricDispatcher()
    assert len(dispatcher.metrics) >= 1


def test_cli_app_build_arg_parser_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.CLIApp as cli_mod

    urls = tmp_path / "urls.txt"
    urls.write_text("https://huggingface.co/acme/model\n", encoding="utf-8")

    monkeypatch.setattr(cli_mod.CLIApp, "run", lambda self: 0)
    parser = cli_mod.build_arg_parser()
    parsed = parser.parse_args([str(urls)])
    assert parsed.url_file == urls
    assert cli_mod.main([str(urls)]) == 0


def test_webapp_routes_and_wsgi_call() -> None:
    app = create_app({})
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/models").status_code == 200
    assert client.get("/models/acme/solar-safeguard").status_code == 200
    assert client.get("/models/missing").status_code == 404

    assert client.get("/api/models?limit=1").get_json()["total"] == 2
    assert client.get("/api/models?q=[bad").status_code == 400

    assert client.get("/api/models/acme/solar-safeguard/lineage").status_code == 200
    assert client.get("/api/models/missing/lineage").status_code == 404
    assert client.get("/api/models/lineage").status_code == 404

    assert (
        client.post(
            "/api/models/ingest",
            json={"metrics": {"ramp_up_time": 0.1}},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/models/ingest",
            json={"metrics": {"ramp_up_time": 0.9}},
        ).status_code
        == 201
    )

    assert client.post("/api/license-check", json={}).status_code == 400
    assert client.post("/api/license-check", json={"artifact_id": "a", "github_url": "b"}).status_code == 200
    assert client.get("/nope").status_code == 404

    # Exercise the WSGI __call__ path.
    app_wsgi = WebApp()
    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/api/models",
        "QUERY_STRING": "limit=1",
        "CONTENT_LENGTH": "0",
        "wsgi.input": io.BytesIO(b""),
    }
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app_wsgi(environ, start_response))
    assert "200" in str(captured["status"])
    assert b"items" in body

    response = client.get("/api/models?limit=1")
    assert response.get_data(as_text=True)
    assert _header_lookup([("X-Test", "yes")], "x-test") == "yes"
