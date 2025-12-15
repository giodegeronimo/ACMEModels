"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for reviewedness metric module.
"""

# mypy: disable-error-code="arg-type,assignment,var-annotated"

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.metrics.reviewedness import ReviewednessMetric, _is_code_file


class _FakeGitClient:
    """
    _FakeGitClient: Class description.
    """

    def __init__(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        blame_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        commit_pr_map: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        pr_reviews_map: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        fail_metadata: bool = False,
        fail_files: bool = False,
    ) -> None:
        """
        __init__: Function description.
        :param metadata:
        :param files:
        :param blame_data:
        :param commit_pr_map:
        :param pr_reviews_map:
        :param fail_metadata:
        :param fail_files:
        :returns:
        """

        self._metadata = metadata
        self._files = files or []
        self._blame_data = blame_data or {}
        self._commit_pr_map = commit_pr_map or {}
        self._pr_reviews_map = pr_reviews_map or {}
        self._fail_metadata = fail_metadata
        self._fail_files = fail_files

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        """
        get_repo_metadata: Function description.
        :param repo_url:
        :returns:
        """

        if self._fail_metadata or self._metadata is None:
            raise RuntimeError("metadata unavailable")
        return dict(self._metadata)

    def list_repo_files(
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> List[str]:
        """
        list_repo_files: Function description.
        :param repo_url:
        :param branch:
        :returns:
        """

        if self._fail_files:
            raise RuntimeError("files unavailable")
        return list(self._files)

    def get_file_blame(
        self,
        repo_url: str,
        file_path: str,
        *,
        branch: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        get_file_blame: Function description.
        :param repo_url:
        :param file_path:
        :param branch:
        :returns:
        """

        return self._blame_data.get(file_path, [])

    def get_commit_associated_pr(
        self,
        repo_url: str,
        commit_sha: str,
    ) -> Optional[Dict[str, Any]]:
        """
        get_commit_associated_pr: Function description.
        :param repo_url:
        :param commit_sha:
        :returns:
        """

        return self._commit_pr_map.get(commit_sha)


def _iter_github_pr_reviews_mock(
    git_client: _FakeGitClient,
    repo_url: str,
    pr_number: int,
) -> List[Dict[str, Any]]:
    """Mock for _iter_github_pr_reviews function."""
    return git_client._pr_reviews_map.get(pr_number, [])


def test_reviewedness_returns_negative_for_non_github() -> None:
    """
    test_reviewedness_returns_negative_for_non_github: Function description.
    :param:
    :returns:
    """

    metric = ReviewednessMetric(
        git_client=_FakeGitClient()  # type: ignore[arg-type]
    )

    score = metric.compute({"git_url": "https://gitlab.com/org/repo"})

    assert score == -1.0


def test_reviewedness_returns_negative_for_missing_url() -> None:
    """
    test_reviewedness_returns_negative_for_missing_url: Function description.
    :param:
    :returns:
    """

    metric = ReviewednessMetric(git_client=_FakeGitClient())

    score = metric.compute({"git_url": ""})

    assert score == -1.0


def test_reviewedness_returns_zero_for_metadata_failure() -> None:
    """
    test_reviewedness_returns_zero_for_metadata_failure: Function description.
    :param:
    :returns:
    """

    git_client = _FakeGitClient(fail_metadata=True)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_returns_zero_for_files_failure() -> None:
    """
    test_reviewedness_returns_zero_for_files_failure: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    git_client = _FakeGitClient(metadata=metadata, fail_files=True)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_returns_zero_for_no_code_files() -> None:
    """
    test_reviewedness_returns_zero_for_no_code_files: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    files = ["README.md", "model.pth", "data.parquet", "image.png"]
    git_client = _FakeGitClient(metadata=metadata, files=files)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_high_score_with_all_reviewed_commits() -> None:
    """
    test_reviewedness_high_score_with_all_reviewed_commits: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    files = ["src/main.py", "src/utils.py", "tests/test_main.py"]

    # All lines come from reviewed commits
    blame_data = {
        "src/main.py": [
            {"startingLine": 1, "endingLine": 50, "commit": {"sha": "abc123"}},
        ],
        "src/utils.py": [
            {"startingLine": 1, "endingLine": 30, "commit": {"sha": "def456"}},
        ],
        "tests/test_main.py": [
            {"startingLine": 1, "endingLine": 20, "commit": {"sha": "abc123"}},
        ],
    }

    # Both commits are from reviewed PRs
    commit_pr_map = {
        "abc123": {"number": 1, "state": "closed", "merged_at": "2024-01-15"},
        "def456": {"number": 2, "state": "closed", "merged_at": "2024-01-20"},
    }

    pr_reviews_map = {
        1: [
            {
                "state": "APPROVED",
                "user": {"login": "reviewer1", "type": "User"},
                "submitted_at": "2024-01-14T12:00:00Z",
            }
        ],
        2: [
            {
                "state": "APPROVED",
                "user": {"login": "reviewer2", "type": "User"},
                "submitted_at": "2024-01-19T12:00:00Z",
            }
        ],
    }

    git_client = _FakeGitClient(
        metadata=metadata,
        files=files,
        blame_data=blame_data,
        commit_pr_map=commit_pr_map,
        pr_reviews_map=pr_reviews_map,
    )

    # Monkey-patch the _iter_github_pr_reviews function
    import src.metrics.reviewedness as rev_module
    original_iter_reviews = rev_module._iter_github_pr_reviews
    rev_module._iter_github_pr_reviews = (
        lambda git, url, pr_num: _iter_github_pr_reviews_mock(
            git, url, pr_num
        )
    )

    try:
        metric = ReviewednessMetric(git_client=git_client)
        score = metric.compute({"git_url": "https://github.com/org/repo"})

        assert isinstance(score, float)
        # With constant seed and sampling, score should be high (close to 1.0)
        assert score >= 0.8
    finally:
        rev_module._iter_github_pr_reviews = original_iter_reviews


def test_reviewedness_low_score_with_no_reviewed_commits() -> None:
    """
    test_reviewedness_low_score_with_no_reviewed_commits: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    files = ["src/main.py", "src/utils.py"]

    # All lines come from unreviewed commits
    blame_data = {
        "src/main.py": [
            {"startingLine": 1, "endingLine": 50, "commit": {"sha": "abc123"}},
        ],
        "src/utils.py": [
            {"startingLine": 1, "endingLine": 30, "commit": {"sha": "def456"}},
        ],
    }

    # Commits are NOT from reviewed PRs (no PR association)
    commit_pr_map = {
        "abc123": None,
        "def456": None,
    }

    git_client = _FakeGitClient(
        metadata=metadata,
        files=files,
        blame_data=blame_data,
        commit_pr_map=commit_pr_map,
    )

    metric = ReviewednessMetric(git_client=git_client)
    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert isinstance(score, float)
    # All commits are unreviewed, so score should be 0.0
    assert score == 0.0


def test_reviewedness_filters_non_code_files() -> None:
    """
    test_reviewedness_filters_non_code_files: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    # Mix of code and non-code files
    files = [
        "src/main.py",  # code
        "model.pth",  # weight
        "data.parquet",  # data
        "image.png",  # binary
        "README.md",  # code (documentation)
        "config.yml",  # code
    ]

    blame_data = {
        "src/main.py": [
            {"startingLine": 1, "endingLine": 50, "commit": {"sha": "abc123"}},
        ],
        "README.md": [
            {"startingLine": 1, "endingLine": 10, "commit": {"sha": "def456"}},
        ],
        "config.yml": [
            {"startingLine": 1, "endingLine": 20, "commit": {"sha": "ghi789"}},
        ],
    }

    commit_pr_map = {
        "abc123": {"number": 1, "merged_at": "2024-01-15"},
        "def456": None,
        "ghi789": None,
    }

    pr_reviews_map = {
        1: [
            {
                "state": "APPROVED",
                "user": {"login": "reviewer1", "type": "User"},
                "submitted_at": "2024-01-14T12:00:00Z",
            }
        ],
    }

    git_client = _FakeGitClient(
        metadata=metadata,
        files=files,
        blame_data=blame_data,
        commit_pr_map=commit_pr_map,
        pr_reviews_map=pr_reviews_map,
    )

    import src.metrics.reviewedness as rev_module
    original_iter_reviews = rev_module._iter_github_pr_reviews
    rev_module._iter_github_pr_reviews = (
        lambda git, url, pr_num: _iter_github_pr_reviews_mock(
            git, url, pr_num
        )
    )

    try:
        metric = ReviewednessMetric(git_client=git_client)
        score = metric.compute({"git_url": "https://github.com/org/repo"})

        # Should only sample from code files
        # (src/main.py, README.md, config.yml)
        # Not from model.pth, data.parquet, or image.png
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    finally:
        rev_module._iter_github_pr_reviews = original_iter_reviews


def test_is_code_file_returns_true_for_python() -> None:
    """
    test_is_code_file_returns_true_for_python: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("main.py") is True


def test_is_code_file_returns_true_for_javascript() -> None:
    """
    test_is_code_file_returns_true_for_javascript: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("app.js") is True


def test_is_code_file_returns_true_for_typescript() -> None:
    """
    test_is_code_file_returns_true_for_typescript: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("component.tsx") is True


def test_is_code_file_returns_true_for_markdown() -> None:
    """
    test_is_code_file_returns_true_for_markdown: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("README.md") is True


def test_is_code_file_returns_true_for_yaml() -> None:
    """
    test_is_code_file_returns_true_for_yaml: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("config.yml") is True


def test_is_code_file_returns_true_for_makefile() -> None:
    """
    test_is_code_file_returns_true_for_makefile: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("Makefile") is True


def test_is_code_file_returns_false_for_weights() -> None:
    """
    test_is_code_file_returns_false_for_weights: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("model.pth") is False
    assert _is_code_file("weights.bin") is False
    assert _is_code_file("model.safetensors") is False


def test_is_code_file_returns_false_for_data_files() -> None:
    """
    test_is_code_file_returns_false_for_data_files: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("dataset.parquet") is False
    assert _is_code_file("data.arrow") is False
    assert _is_code_file("samples.csv") is False


def test_is_code_file_returns_false_for_binaries() -> None:
    """
    test_is_code_file_returns_false_for_binaries: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("image.png") is False
    assert _is_code_file("video.mp4") is False
    assert _is_code_file("archive.zip") is False


def test_is_code_file_returns_false_for_empty_string() -> None:
    """
    test_is_code_file_returns_false_for_empty_string: Function description.
    :param:
    :returns:
    """

    assert _is_code_file("") is False


def test_reviewedness_handles_empty_blame_ranges() -> None:
    """
    test_reviewedness_handles_empty_blame_ranges: Function description.
    :param:
    :returns:
    """

    metadata = {"default_branch": "main"}
    files = ["src/main.py"]

    # File exists but has no blame data (empty file or error)
    blame_data = {
        "src/main.py": [],
    }

    git_client = _FakeGitClient(
        metadata=metadata,
        files=files,
        blame_data=blame_data,
    )

    metric = ReviewednessMetric(git_client=git_client)
    score = metric.compute({"git_url": "https://github.com/org/repo"})

    # Should handle gracefully and return 0.0
    assert score == 0.0


def test_reviewedness_uses_constant_seed() -> None:
    """Test that the metric uses a constant seed for reproducibility."""
    metadata = {"default_branch": "main"}
    files = ["src/main.py", "src/utils.py", "src/helper.py"]

    blame_data = {
        "src/main.py": [
            {"startingLine": 1, "endingLine": 50, "commit": {"sha": "abc123"}},
        ],
        "src/utils.py": [
            {"startingLine": 1, "endingLine": 30, "commit": {"sha": "def456"}},
        ],
        "src/helper.py": [
            {"startingLine": 1, "endingLine": 20, "commit": {"sha": "ghi789"}},
        ],
    }

    commit_pr_map = {
        "abc123": {"number": 1, "merged_at": "2024-01-15"},
        "def456": None,
        "ghi789": None,
    }

    pr_reviews_map = {
        1: [
            {
                "state": "APPROVED",
                "user": {"login": "reviewer1", "type": "User"},
                "submitted_at": "2024-01-14T12:00:00Z",
            }
        ],
    }

    git_client = _FakeGitClient(
        metadata=metadata,
        files=files,
        blame_data=blame_data,
        commit_pr_map=commit_pr_map,
        pr_reviews_map=pr_reviews_map,
    )

    import src.metrics.reviewedness as rev_module
    original_iter_reviews = rev_module._iter_github_pr_reviews
    rev_module._iter_github_pr_reviews = (
        lambda git, url, pr_num: _iter_github_pr_reviews_mock(
            git, url, pr_num
        )
    )

    try:
        metric1 = ReviewednessMetric(git_client=git_client)
        score1 = metric1.compute({"git_url": "https://github.com/org/repo"})

        metric2 = ReviewednessMetric(git_client=git_client)
        score2 = metric2.compute({"git_url": "https://github.com/org/repo"})

        # Both runs should produce identical results due to constant seed
        assert score1 == score2
    finally:
        rev_module._iter_github_pr_reviews = original_iter_reviews


def test_parse_owner_repo_strips_git_suffix() -> None:
    """
    test_parse_owner_repo_strips_git_suffix: Function description.
    :param:
    :returns:
    """

    import src.metrics.reviewedness as rev_module

    assert rev_module._parse_owner_repo("https://github.com/org/repo.git") == (
        "org",
        "repo",
    )
    with pytest.raises(ValueError):
        rev_module._parse_owner_repo("https://gitlab.com/org/repo")


def test_iter_github_pr_reviews_paginates() -> None:
    """
    test_iter_github_pr_reviews_paginates: Function description.
    :param:
    :returns:
    """

    import src.metrics.reviewedness as rev_module

    class _Response:
        """
        _Response: Class description.
        """

        def __init__(self, status_code: int, payload: list[dict[str, Any]]):
            """
            __init__: Function description.
            :param status_code:
            :param payload:
            :returns:
            """

            self.status_code = status_code
            self._payload = payload

        def json(self) -> list[dict[str, Any]]:
            """
            json: Function description.
            :param:
            :returns:
            """

            return list(self._payload)

    class _Session:
        """
        _Session: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.calls: list[str] = []

        def get(self, url: str, timeout: int = 10) -> _Response:
            """
            get: Function description.
            :param url:
            :param timeout:
            :returns:
            """

            self.calls.append(url)
            if "&page=1" in url:
                return _Response(200, [{"id": 1}])
            return _Response(200, [])

    class _Git:
        """
        _Git: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self._session = _Session()

        def _execute_with_rate_limit(self, op, *, name=None):  # type: ignore[no-untyped-def]
            """
            _execute_with_rate_limit: Function description.
            :param op:
            :param name:
            :returns:
            """

            return op()

    git = _Git()
    reviews = list(
        rev_module._iter_github_pr_reviews(  # type: ignore[arg-type]
            git,
            "https://github.com/org/repo",
            5,
        )
    )
    assert reviews == [{"id": 1}]
    assert git._session.calls


def test_iter_github_pr_reviews_raises_on_non_200() -> None:
    """
    test_iter_github_pr_reviews_raises_on_non_200: Function description.
    :param:
    :returns:
    """

    import src.metrics.reviewedness as rev_module

    class _Response:
        """
        _Response: Class description.
        """

        status_code = 500

        def json(self):  # type: ignore[no-untyped-def]
            """
            json: Function description.
            :param:
            :returns:
            """

            return []

    class _Session:
        """
        _Session: Class description.
        """

        def get(self, url: str, timeout: int = 10) -> _Response:
            """
            get: Function description.
            :param url:
            :param timeout:
            :returns:
            """

            return _Response()

    class _Git:
        """
        _Git: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self._session = _Session()

        def _execute_with_rate_limit(self, op, *, name=None):  # type: ignore[no-untyped-def]
            """
            _execute_with_rate_limit: Function description.
            :param op:
            :param name:
            :returns:
            """

            return op()

    git = _Git()
    with pytest.raises(RuntimeError):
        list(
            rev_module._iter_github_pr_reviews(  # type: ignore[arg-type]
                git,
                "https://github.com/org/repo",
                7,
            )
        )


def test_reviewedness_helpers_cover_edge_cases() -> None:
    """
    test_reviewedness_helpers_cover_edge_cases: Function description.
    :param:
    :returns:
    """

    import src.metrics.reviewedness as rev_module

    assert rev_module._login_of(None) is None
    assert rev_module._login_of({"login": "alice"}) == "alice"

    assert rev_module._is_human({"login": "bot[bot]", "type": "User"}) is False
    assert rev_module._is_human({"login": "alice", "type": "Bot"}) is False
    assert rev_module._is_human({"login": "alice", "type": "User"}) is True

    assert rev_module._parse_ts(None) is None
    assert rev_module._parse_ts("invalid") is None
    assert rev_module._parse_ts("2024-01-01T00:00:00Z") is not None

    merged_at = rev_module._parse_ts("2024-01-10T00:00:00Z")
    assert merged_at is not None
    reviews = [
        {
            "state": "APPROVED",
            "user": {"login": "reviewer", "type": "User"},
            "submitted_at": "2024-01-09T00:00:00Z",
        },
        {
            "state": "DISMISSED",
            "user": {"login": "reviewer", "type": "User"},
            "submitted_at": "2024-01-09T01:00:00Z",
        },
        {
            "state": "APPROVED",
            "user": {"login": "author", "type": "User"},
            "submitted_at": "2024-01-09T00:00:00Z",
        },
        {
            "state": "APPROVED",
            "user": {"login": "bot[bot]", "type": "User"},
            "submitted_at": "2024-01-09T00:00:00Z",
        },
        {
            "state": "APPROVED",
            "user": {"login": "late", "type": "User"},
            "submitted_at": "2024-02-09T00:00:00Z",
        },
    ]
    assert rev_module._has_valid_approval(reviews, "author", merged_at) is False
    assert rev_module._has_valid_approval([reviews[0]], "author", merged_at) is True

    metric = ReviewednessMetric(git_client=_FakeGitClient())
    assert metric._wilson_score_interval(0.0, 0.0) == (0.0, 0.0)
    low, high = metric._wilson_score_interval(1.0, 2.0, confidence=0.99)
    assert 0.0 <= low <= high <= 1.0
