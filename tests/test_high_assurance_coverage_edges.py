from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.metrics.base import Metric
from src.metrics.registry import MetricDispatcher
from src.models.artifacts import ArtifactMetadata, ArtifactQuery, validate_url
from src.models.audit import _ensure_utc
from src.utils import env as env_mod


class _DummyMetric(Metric):
    name = "dummy"
    key = "dummy"

    def compute(self, url_record: dict[str, str]) -> float:
        return 1.0


def test_metric_repr_includes_name_and_key() -> None:
    metric = _DummyMetric(name="My Metric", key="metric_key")
    assert "My Metric" in repr(metric)
    assert "metric_key" in repr(metric)


def test_metric_dispatcher_exposes_metrics_tuple() -> None:
    dispatcher = MetricDispatcher(metrics=[_DummyMetric(name="m", key="k")])
    assert isinstance(dispatcher.metrics, tuple)
    assert len(dispatcher.metrics) == 1


def test_ensure_utc_converts_non_utc_timezones() -> None:
    dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    converted = _ensure_utc(dt)
    assert converted.tzinfo == timezone.utc


def test_enable_readme_fallback_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACME_ENABLE_README_FALLBACK", raising=False)
    assert env_mod.enable_readme_fallback() is True


def test_validate_runtime_environment_fails_when_log_file_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_12345678")
    monkeypatch.delenv("LOG_FILE", raising=False)
    with pytest.raises(SystemExit):
        env_mod.validate_runtime_environment()


def test_validate_url_rejects_non_absolute_urls() -> None:
    with pytest.raises(ValueError, match="not a valid absolute URI"):
        validate_url("not-a-url")


def test_artifact_metadata_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        ArtifactMetadata(name="artifact", id="abc-123", type="invalid")  # type: ignore[arg-type]


def test_artifact_query_rejects_invalid_type_filters() -> None:
    with pytest.raises(ValueError, match="in query filter is invalid"):
        ArtifactQuery(name="*", types=["invalid"])  # type: ignore[list-item]
