"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for the model rating computation helper.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from src.metrics import ratings


def test_compute_model_rating_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_compute_model_rating_happy_path: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Dispatcher:
        """
        _Dispatcher: Class description.
        """

        def compute(self, records: List[dict[str, str]]) -> list[dict[str, Any]]:
            """
            compute: Function description.
            :param records:
            :returns:
            """

            assert records == [{"hf_url": "https://huggingface.co/org/model"}]
            return [{"bus_factor": 1.0}]

    class _NetScoreCalculator:
        """
        _NetScoreCalculator: Class description.
        """

        def with_net_score(self, results: dict[str, Any]) -> dict[str, Any]:
            """
            with_net_score: Function description.
            :param results:
            :returns:
            """

            return {**results, "net_score": 1.0}

    class _Formatter:
        """
        _Formatter: Class description.
        """

        def format_records(self, records: Any, results: Any) -> list[Any]:
            """
            format_records: Function description.
            :param records:
            :param results:
            :returns:
            """

            return [{"name": "demo", "net_score": 1.0}]

    monkeypatch.setattr(ratings, "MetricDispatcher", _Dispatcher)
    monkeypatch.setattr(ratings, "NetScoreCalculator", _NetScoreCalculator)
    monkeypatch.setattr(ratings, "ResultsFormatter", _Formatter)

    rating = ratings.compute_model_rating("https://huggingface.co/org/model")

    assert rating["net_score"] == 1.0


def test_compute_model_rating_wraps_dispatch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_compute_model_rating_wraps_dispatch_errors: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Dispatcher:
        """
        _Dispatcher: Class description.
        """

        def compute(self, records: Any) -> Any:
            """
            compute: Function description.
            :param records:
            :returns:
            """

            raise RuntimeError("boom")

    monkeypatch.setattr(ratings, "MetricDispatcher", _Dispatcher)

    with pytest.raises(ratings.RatingComputationError):
        ratings.compute_model_rating("https://huggingface.co/org/model")


def test_compute_model_rating_requires_non_empty_formatted_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_compute_model_rating_requires_non_empty_formatted_results: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Dispatcher:
        """
        _Dispatcher: Class description.
        """

        def compute(self, records: Any) -> Any:
            """
            compute: Function description.
            :param records:
            :returns:
            """

            return [{"bus_factor": 1.0}]

    class _NetScoreCalculator:
        """
        _NetScoreCalculator: Class description.
        """

        def with_net_score(self, results: Any) -> Any:
            """
            with_net_score: Function description.
            :param results:
            :returns:
            """

            return results

    class _Formatter:
        """
        _Formatter: Class description.
        """

        def format_records(self, records: Any, results: Any) -> list[Any]:
            """
            format_records: Function description.
            :param records:
            :param results:
            :returns:
            """

            return []

    monkeypatch.setattr(ratings, "MetricDispatcher", _Dispatcher)
    monkeypatch.setattr(ratings, "NetScoreCalculator", _NetScoreCalculator)
    monkeypatch.setattr(ratings, "ResultsFormatter", _Formatter)

    with pytest.raises(ratings.RatingComputationError):
        ratings.compute_model_rating("https://huggingface.co/org/model")


def test_compute_model_rating_requires_mapping_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_compute_model_rating_requires_mapping_payload: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Dispatcher:
        """
        _Dispatcher: Class description.
        """

        def compute(self, records: Any) -> Any:
            """
            compute: Function description.
            :param records:
            :returns:
            """

            return [{"bus_factor": 1.0}]

    class _NetScoreCalculator:
        """
        _NetScoreCalculator: Class description.
        """

        def with_net_score(self, results: Any) -> Any:
            """
            with_net_score: Function description.
            :param results:
            :returns:
            """

            return results

    class _Formatter:
        """
        _Formatter: Class description.
        """

        def format_records(self, records: Any, results: Any) -> list[Any]:
            """
            format_records: Function description.
            :param records:
            :param results:
            :returns:
            """

            return [123]

    monkeypatch.setattr(ratings, "MetricDispatcher", _Dispatcher)
    monkeypatch.setattr(ratings, "NetScoreCalculator", _NetScoreCalculator)
    monkeypatch.setattr(ratings, "ResultsFormatter", _Formatter)

    with pytest.raises(ratings.RatingComputationError):
        ratings.compute_model_rating("https://huggingface.co/org/model")
