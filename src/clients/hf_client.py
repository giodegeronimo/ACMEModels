"""Hugging Face Hub client that wraps HfApi with rate limiting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast
from urllib.parse import urlparse

import requests
from huggingface_hub import hf_hub_url

if TYPE_CHECKING:
    from huggingface_hub import HfApi, ModelInfo  # type: ignore[import]
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore[import]
else:
    HfApi = Any  # type: ignore[assignment]
    ModelInfo = Any  # type: ignore[assignment]
    HfHubHTTPError = Exception  # type: ignore[assignment]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter

DEFAULT_MAX_CALLS = 5
DEFAULT_PERIOD_SECONDS = 1.0


class _SessionWithGet(Protocol):
    def get(self, url: str, timeout: int = 30) -> Any: ...


class HFClient(BaseClient[Any]):
    """Thin wrapper around ``huggingface_hub.HfApi`` with rate limiting."""

    def __init__(
        self,
        *,
        api: Optional[Any] = None,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        http_session: Optional[_SessionWithGet] = None,
    ) -> None:
        limiter = rate_limiter or RateLimiter(
            max_calls=DEFAULT_MAX_CALLS,
            period_seconds=DEFAULT_PERIOD_SECONDS,
        )
        super().__init__(limiter, logger=logger)
        self._http_session = cast(
            _SessionWithGet, http_session or requests.Session()
        )
        if api is not None:
            self._api = api
        else:
            try:
                from huggingface_hub import \
                    HfApi as RuntimeHfApi  # type: ignore[import]
            except ModuleNotFoundError as error:  # pragma: no cover
                message = (
                    "huggingface_hub is not installed. "
                    "Run './run install' first."
                )
                raise RuntimeError(message) from error

            self._api = cast(HfApi, RuntimeHfApi())

    def get_model_info(self, repo_id: str) -> ModelInfo:
        """Fetch model metadata from the Hugging Face Hub."""

        normalized_repo = self._normalize_repo_id(repo_id)

        def _operation() -> ModelInfo:
            self._logger.debug("Requesting model info for %s", normalized_repo)
            return self._api.model_info(normalized_repo)

        return self._execute_with_rate_limit(
            _operation,
            name=f"hf.model_info({normalized_repo})",
        )

    def model_exists(self, repo_id: str) -> bool:
        """Return ``True`` when the model exists on the Hub."""
        try:
            self.get_model_info(repo_id)
        except HfHubHTTPError:
            return False
        return True

    def dataset_exists(self, dataset_id: str) -> bool:
        """Return ``True`` when the dataset slug resolves on the Hub."""

        normalized_dataset = self._normalize_dataset_id(dataset_id)

        def _operation() -> bool:
            self._logger.debug(
                "Checking dataset existence for %s", normalized_dataset
            )
            self._api.dataset_info(normalized_dataset)
            return True

        try:
            return self._execute_with_rate_limit(
                _operation,
                name=f"hf.dataset_exists({normalized_dataset})",
            )
        except HfHubHTTPError:
            self._logger.debug(
                "Dataset %s does not exist or is inaccessible",
                normalized_dataset,
            )
            return False

    def get_dataset_info(self, dataset_id: str) -> Any:
        """Fetch dataset metadata from the Hugging Face Hub."""

        normalized_dataset = self._normalize_dataset_id(dataset_id)

        def _operation() -> Any:
            self._logger.debug(
                "Requesting dataset info for %s", normalized_dataset
            )
            return self._api.dataset_info(normalized_dataset)

        return self._execute_with_rate_limit(
            _operation,
            name=f"hf.dataset_info({normalized_dataset})",
        )

    def count_models_trained_on_dataset(
        self,
        dataset_id: str,
        limit: int = 1000,
    ) -> int:
        """Count models that reference the dataset via trained_dataset."""

        normalized_dataset = self._normalize_dataset_id(dataset_id)

        def _operation() -> int:
            self._logger.debug(
                "Listing models trained on dataset %s (limit=%d)",
                normalized_dataset,
                limit,
            )
            count = 0
            iterator = self._api.list_models(
                trained_dataset=normalized_dataset,
                limit=limit,
            )
            for _ in iterator:
                count += 1
            return count

        try:
            return self._execute_with_rate_limit(
                _operation,
                name=f"hf.models_for_dataset({normalized_dataset})",
            )
        except HfHubHTTPError:
            self._logger.debug(
                "Unable to list models for dataset %s",
                normalized_dataset,
            )
            return 0

    def get_model_readme(self, repo_id: str) -> str:
        """Fetch the model card README as UTF-8 text."""

        normalized_repo = self._normalize_repo_id(repo_id)

        def _operation() -> str:
            self._logger.debug("Downloading README.md for %s", normalized_repo)
            url = hf_hub_url(
                repo_id=normalized_repo,
                filename="README.md",
            )
            response = self._http_session.get(url, timeout=30)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text

        try:
            return self._execute_with_rate_limit(
                _operation,
                name=f"hf.model_readme({normalized_repo})",
            )
        except (requests.RequestException, HfHubHTTPError) as exc:
            self._logger.info(
                "README unavailable for %s: %s",
                normalized_repo,
                exc,
            )
            return ""

    def list_model_files(
        self,
        repo_id: str,
        *,
        recursive: bool = True,
    ) -> list[tuple[str, int]]:
        """Return a list of (path, size_bytes) for files in the model repo.

        The implementation prefers ``list_repo_tree`` (which provides sizes),
        falls back to ``model_info().siblings`` if necessary, and finally
        attempts HEAD requests to obtain ``Content-Length`` for entries with
        unknown sizes.
        """

        normalized_repo = self._normalize_repo_id(repo_id)

        def _list_via_tree() -> list[tuple[str, int]]:
            try:
                items = self._api.list_repo_tree(  # type: ignore[attr-defined]
                    normalized_repo,
                    repo_type="model",
                    recursive=recursive,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.debug(
                    "list_repo_tree failed for %s: %s", normalized_repo, exc
                )
                return []

            results: list[tuple[str, int]] = []
            for item in items:
                # RepoFile typically has 'path' and 'size' attrs
                path = getattr(item, "path", None) or getattr(
                    item, "rfilename", None
                )
                size = getattr(item, "size", None)
                if isinstance(path, str) and isinstance(size, int):
                    results.append((path, size))
            return results

        def _list_via_siblings() -> list[tuple[str, int]]:
            try:
                info = self._api.model_info(normalized_repo)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.debug(
                    "model_info failed for %s: %s", normalized_repo, exc
                )
                return []

            results: list[tuple[str, int]] = []
            siblings = getattr(info, "siblings", None) or []
            for sibling in siblings:
                path = getattr(sibling, "rfilename", None)
                size = getattr(sibling, "size", None)
                if isinstance(path, str) and isinstance(size, int):
                    results.append((path, size))
            return results

        def _operation() -> list[tuple[str, int]]:
            files = _list_via_tree()
            if files:
                return files
            files = _list_via_siblings()
            # Optionally fill missing sizes via HEAD; avoid over-fetching.
            # Here we keep it simple and return only known sizes.
            return files

        return self._execute_with_rate_limit(
            _operation,
            name=f"hf.list_files({normalized_repo})",
        )

    @staticmethod
    def _normalize_repo_id(repo_identifier: str) -> str:
        trimmed = repo_identifier.strip()
        if not trimmed:
            raise ValueError("Repository identifier cannot be empty.")

        if "://" not in trimmed:
            return trimmed

        parsed = urlparse(trimmed)
        if parsed.netloc != "huggingface.co":
            raise ValueError(
                f"Unsupported Hugging Face host: {parsed.netloc}"
            )

        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise ValueError(f"Unable to extract repo id from URL: {trimmed}")

        if segments[0] in {"datasets", "spaces", "models"}:
            segments = segments[1:]

        if len(segments) < 2:
            raise ValueError(f"Unable to extract repo id from URL: {trimmed}")

        return "/".join(segments[:2])

    @staticmethod
    def _normalize_dataset_id(dataset_identifier: str) -> str:
        trimmed = dataset_identifier.strip()
        if not trimmed:
            raise ValueError("Dataset identifier cannot be empty.")

        if "://" not in trimmed:
            if "/" not in trimmed:
                raise ValueError("Dataset identifier must include owner/name")
            return trimmed

        parsed = urlparse(trimmed)
        if parsed.netloc != "huggingface.co":
            raise ValueError(
                f"Unsupported Hugging Face host in dataset id: {parsed.netloc}"
            )

        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise ValueError(
                f"Unable to extract dataset id from URL: {dataset_identifier}"
            )

        if segments[0] == "datasets":
            segments = segments[1:]

        if len(segments) < 2:
            raise ValueError(
                f"Unable to extract dataset id from URL: {dataset_identifier}"
            )

        return "/".join(segments[:2])
