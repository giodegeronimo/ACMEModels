"""Tests for reviewedness metric module."""

# mypy: disable-error-code="arg-type,assignment,var-annotated"

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.metrics.reviewedness import ReviewednessMetric, _is_code_file


class _FakeGitClient:
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
        self._metadata = metadata
        self._files = files or []
        self._blame_data = blame_data or {}
        self._commit_pr_map = commit_pr_map or {}
        self._pr_reviews_map = pr_reviews_map or {}
        self._fail_metadata = fail_metadata
        self._fail_files = fail_files

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        if self._fail_metadata or self._metadata is None:
            raise RuntimeError("metadata unavailable")
        return dict(self._metadata)

    def list_repo_files(
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> List[str]:
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
        return self._blame_data.get(file_path, [])

    def get_commit_associated_pr(
        self,
        repo_url: str,
        commit_sha: str,
    ) -> Optional[Dict[str, Any]]:
        return self._commit_pr_map.get(commit_sha)


def _iter_github_pr_reviews_mock(
    git_client: _FakeGitClient,
    repo_url: str,
    pr_number: int,
) -> List[Dict[str, Any]]:
    """Mock for _iter_github_pr_reviews function."""
    return git_client._pr_reviews_map.get(pr_number, [])


def test_reviewedness_returns_negative_for_non_github() -> None:
    metric = ReviewednessMetric(
        git_client=_FakeGitClient()  # type: ignore[arg-type]
    )

    score = metric.compute({"git_url": "https://gitlab.com/org/repo"})

    assert score == -1.0


def test_reviewedness_returns_negative_for_missing_url() -> None:
    metric = ReviewednessMetric(git_client=_FakeGitClient())

    score = metric.compute({"git_url": ""})

    assert score == -1.0


def test_reviewedness_returns_zero_for_metadata_failure() -> None:
    git_client = _FakeGitClient(fail_metadata=True)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_returns_zero_for_files_failure() -> None:
    metadata = {"default_branch": "main"}
    git_client = _FakeGitClient(metadata=metadata, fail_files=True)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_returns_zero_for_no_code_files() -> None:
    metadata = {"default_branch": "main"}
    files = ["README.md", "model.pth", "data.parquet", "image.png"]
    git_client = _FakeGitClient(metadata=metadata, files=files)
    metric = ReviewednessMetric(git_client=git_client)

    score = metric.compute({"git_url": "https://github.com/org/repo"})

    assert score == 0.0


def test_reviewedness_high_score_with_all_reviewed_commits() -> None:
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
    assert _is_code_file("main.py") is True


def test_is_code_file_returns_true_for_javascript() -> None:
    assert _is_code_file("app.js") is True


def test_is_code_file_returns_true_for_typescript() -> None:
    assert _is_code_file("component.tsx") is True


def test_is_code_file_returns_true_for_markdown() -> None:
    assert _is_code_file("README.md") is True


def test_is_code_file_returns_true_for_yaml() -> None:
    assert _is_code_file("config.yml") is True


def test_is_code_file_returns_true_for_makefile() -> None:
    assert _is_code_file("Makefile") is True


def test_is_code_file_returns_false_for_weights() -> None:
    assert _is_code_file("model.pth") is False
    assert _is_code_file("weights.bin") is False
    assert _is_code_file("model.safetensors") is False


def test_is_code_file_returns_false_for_data_files() -> None:
    assert _is_code_file("dataset.parquet") is False
    assert _is_code_file("data.arrow") is False
    assert _is_code_file("samples.csv") is False


def test_is_code_file_returns_false_for_binaries() -> None:
    assert _is_code_file("image.png") is False
    assert _is_code_file("video.mp4") is False
    assert _is_code_file("archive.zip") is False


def test_is_code_file_returns_false_for_empty_string() -> None:
    assert _is_code_file("") is False


def test_reviewedness_handles_empty_blame_ranges() -> None:
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
