from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from src.clients.git_client import GitClient
from src.metrics.base import Metric, MetricOutput

_LOGGER = logging.getLogger(__name__)

# Reproducible random seed for consistent sampling across runs
_SAMPLING_SEED = 42

# File extensions for model weights and data files to exclude
_WEIGHT_EXTENSIONS = {
    ".bin", ".pth", ".pt", ".h5", ".pb", ".onnx", ".tflite",
    ".safetensors", ".ckpt", ".weights", ".model", ".pkl", ".pickle",
    ".msgpack", ".npz", ".npy",
}

# Large data file extensions to exclude
_DATA_EXTENSIONS = {
    ".parquet", ".arrow", ".feather", ".hdf5", ".h5",
}

# Binary and media files to exclude
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".mp3", ".wav", ".flac", ".ogg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".class", ".pyc", ".o",
    ".woff", ".woff2", ".ttf", ".eot",
}

# Code file extensions to include
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".m", ".mm",
    ".sh", ".bash", ".zsh", ".fish", ".pl", ".r", ".lua", ".vim",
    ".html", ".css", ".scss", ".sass", ".less",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg",
    ".json", ".jsonl", ".md", ".rst", ".txt", ".sql",
    ".makefile", ".gradle", ".cmake", ".dockerfile",
}


def _is_code_file(filename: str) -> bool:
    """Determine if a file should be counted as code (not weights/data).

    Returns True for source code files, False for weights, data, and binaries.
    """
    if not filename:
        return False

    filename_lower = filename.lower()

    # Check for known weight/data/binary extensions
    for ext in _WEIGHT_EXTENSIONS | _DATA_EXTENSIONS | _BINARY_EXTENSIONS:
        if filename_lower.endswith(ext):
            return False

    # Check if it has a code extension
    for ext in _CODE_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True

    # Special cases: files without extensions that are typically code
    basename = filename.split("/")[-1] if "/" in filename else filename
    code_files = {"Makefile", "Dockerfile", "Rakefile", "Gemfile", "Pipfile"}
    if basename in code_files:
        return True

    # Default: exclude files with no recognized extension
    return False


def _is_github_url(url: str) -> bool:
    """Check if URL is a GitHub repository."""
    return url.strip().startswith("https://github.com/")


class ReviewednessMetric(Metric):
    """
    Line-based reviewedness metric using inverse-probability weighted sampling.

    Uses pure line sampling: randomly picks files, randomly picks lines
    within them, and applies inverse-probability weighting
    (Horvitz-Thompson estimator) to correct for file-size bias.
    This approach:

    - Only fetches blame for sampled files (not all files)
    - Weights samples by file size (lines) to remain unbiased
    - Uses Wilson score interval for early stopping when estimate converges
    - Minimizes API calls dramatically

    Returns:
        -1.0: No GitHub URL or unsupported host
         0.0: GitHub repo but can't analyze (API errors or no code files)
         [0.0-1.0]: Estimated fraction of code lines from reviewed PRs
    """

    def __init__(self, git_client: Optional[GitClient] = None) -> None:
        super().__init__(name="Reviewedness", key="reviewedness")
        self._git: GitClient = git_client or GitClient()
        # Cache for commit -> PR lookups to minimize API calls
        self._commit_pr_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        # Cache for PR -> review status
        self._pr_review_cache: Dict[int, bool] = {}
        # Cache for file blame data by (file_path, branch)
        self._blame_cache: Dict[tuple[str, str], list[Dict[str, Any]]] = {}

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        """Compute reviewedness using inverse-probability weighted
        line sampling."""
        # 1) Validate git_url presence and host
        git_url = url_record.get("git_url")
        if not git_url or not _is_github_url(git_url):
            _LOGGER.info(f"Skipping non-GitHub URL: {git_url}")
            return -1.0

        _LOGGER.info(f"Computing reviewedness for {git_url}")

        # Set constant random seed for reproducible sampling across all repos
        random.seed(_SAMPLING_SEED)
        _LOGGER.debug(
            f"Using constant seed for reproducibility: {_SAMPLING_SEED}"
        )

        # 2) Fetch repository metadata
        try:
            meta = self._git.get_repo_metadata(git_url)
        except Exception as e:
            _LOGGER.warning(f"Failed to fetch metadata for {git_url}: {e}")
            return 0.0

        # 3) Extract default branch
        default_branch = meta.get("default_branch") or "main"
        _LOGGER.info(f"Default branch: {default_branch}")

        # 4) Get all files in repository
        try:
            all_files = self._git.list_repo_files(
                git_url, branch=default_branch
            )
        except Exception as e:
            _LOGGER.warning(
                f"Failed to list repo files for {git_url}: {e}"
            )
            return 0.0

        # 5) Filter to code files only
        code_files = [f for f in all_files if _is_code_file(f)]
        _LOGGER.info(
            f"Found {len(code_files)} code files out of "
            f"{len(all_files)} total files"
        )

        if not code_files:
            _LOGGER.info("No code files found")
            return 0.0

        # 6) Pure line sampling with inverse-probability weighting
        MAX_SAMPLES = 300
        MIN_SAMPLES_FOR_EARLY_STOP = 50
        CONFIDENCE_LEVEL = 0.95
        TARGET_INTERVAL_WIDTH = 0.10  # Stop when 95% CI width < 10%
        # Set to False to always take MAX_SAMPLES
        ENABLE_EARLY_STOPPING = True

        sum_weights = 0.0
        sum_weighted_x = 0.0
        samples_taken = 0

        _LOGGER.info(
            f"Starting weighted line sampling "
            f"(max {MAX_SAMPLES} samples)..."
        )

        for sample_idx in range(MAX_SAMPLES):
            # Pick a random code file uniformly
            file_path = random.choice(code_files)

            # Fetch blame for this file (with caching)
            cache_key = (file_path, default_branch)
            if cache_key in self._blame_cache:
                blame_ranges = self._blame_cache[cache_key]
            else:
                try:
                    blame_ranges = self._git.get_file_blame(
                        git_url, file_path, branch=default_branch
                    )
                    self._blame_cache[cache_key] = blame_ranges
                except Exception as e:
                    _LOGGER.debug(f"Failed to get blame for {file_path}: {e}")
                    continue

            if not blame_ranges:
                _LOGGER.debug(f"No blame data for {file_path}")
                continue

            # Get file line count: n = max(endingLine)
            n = max((r.get("endingLine", 0) for r in blame_ranges), default=0)
            if n == 0:
                continue

            # Pick a random line ℓ ∈ [1, n]
            sampled_line = random.randint(1, n)

            # Map line to commit_sha via blame ranges
            commit_sha = None
            for blame_range in blame_ranges:
                start = blame_range.get("startingLine", 0)
                end = blame_range.get("endingLine", 0)
                if start <= sampled_line <= end:
                    commit_sha = blame_range.get("commit", {}).get("sha")
                    break

            if not commit_sha:
                continue

            # Check if commit came from reviewed PR
            reviewed = self._is_commit_from_reviewed_pr(git_url, commit_sha)

            # Accumulate weighted sample
            w = float(n)  # weight = file size (line count)
            x = 1.0 if reviewed else 0.0

            sum_weights += w
            sum_weighted_x += w * x
            samples_taken += 1

            # Log progress periodically
            if samples_taken % 25 == 0:
                p_hat = (
                    sum_weighted_x / sum_weights
                    if sum_weights > 0
                    else 0.0
                )
                _LOGGER.info(
                    f"Sample {samples_taken}/{MAX_SAMPLES}: "
                    f"p_hat = {p_hat:.4f}"
                )

            # Early stopping: check Wilson score interval width
            if (
                ENABLE_EARLY_STOPPING
                and samples_taken >= MIN_SAMPLES_FOR_EARLY_STOP
            ):
                p_hat = (
                    sum_weighted_x / sum_weights
                    if sum_weights > 0
                    else 0.0
                )
                ci_lower, ci_upper = self._wilson_score_interval(
                    sum_weighted_x, sum_weights, CONFIDENCE_LEVEL
                )
                interval_width = ci_upper - ci_lower

                if interval_width < TARGET_INTERVAL_WIDTH:
                    _LOGGER.info(
                        f"Early stopping at {samples_taken} samples: "
                        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], "
                        f"width = {interval_width:.4f} < "
                        f"{TARGET_INTERVAL_WIDTH}"
                    )
                    break

        # Calculate final estimate
        if sum_weights == 0:
            _LOGGER.info("No valid samples collected")
            return 0.0

        p_hat = sum_weighted_x / sum_weights
        ci_lower, ci_upper = self._wilson_score_interval(
            sum_weighted_x, sum_weights, CONFIDENCE_LEVEL
        )

        _LOGGER.info(
            f"Sampling complete: {samples_taken} samples taken"
        )
        _LOGGER.info(
            f"Weighted estimate: p_hat = {p_hat:.4f}"
        )
        _LOGGER.info(
            f"95% Wilson CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
        )

        final_score = max(0.0, min(float(p_hat), 1.0))
        _LOGGER.info(f"Final reviewedness score: {final_score:.4f}")

        return final_score

    def _wilson_score_interval(
        self, successes: float, total_weight: float, confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Compute Wilson score confidence interval for weighted proportion.

        For weighted samples, we approximate using the ratio estimator
        variance. This is a simplified version - for true weighted
        Wilson interval, you'd need the full sample variance, but this
        gives a reasonable approximation.

        Args:
            successes: Sum of weighted successes (Σ w_i * x_i)
            total_weight: Sum of all weights (Σ w_i)
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if total_weight == 0:
            return (0.0, 0.0)

        p = successes / total_weight

        # For weighted sampling, approximate effective sample size
        # This is a simplification - assumes weights are roughly uniform
        # A more sophisticated approach would use design effect
        n_eff = total_weight / (total_weight / max(1, total_weight))
        n_eff = min(n_eff, 1000)  # Cap for numerical stability

        # Z-score for confidence level
        if confidence == 0.95:
            z = 1.96
        elif confidence == 0.99:
            z = 2.576
        else:
            # Approximate using normal distribution
            from math import sqrt
            z = sqrt(2) * 1.96  # Fallback

        # Wilson score interval formula
        denominator = 1 + (z * z) / n_eff
        center = (p + (z * z) / (2 * n_eff)) / denominator
        margin = (z / denominator) * math.sqrt(
            (p * (1 - p) / n_eff)
            + (z * z) / (4 * n_eff * n_eff)
        )

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    def _is_commit_from_reviewed_pr(
        self, git_url: str, commit_sha: str
    ) -> bool:
        """Check if a commit came from a reviewed PR (with caching)."""
        # Check commit -> PR cache
        if commit_sha in self._commit_pr_cache:
            pr_data = self._commit_pr_cache[commit_sha]
        else:
            # Fetch PR associated with this commit
            try:
                pr_data = self._git.get_commit_associated_pr(
                    git_url, commit_sha
                )
                self._commit_pr_cache[commit_sha] = pr_data
            except Exception as e:
                _LOGGER.debug(f"Failed to get PR for commit {commit_sha}: {e}")
                self._commit_pr_cache[commit_sha] = None
                return False

        # If no PR, this was a direct commit
        if not pr_data:
            return False

        pr_number = pr_data.get("number")
        if not pr_number:
            return False

        # Check PR review cache
        if pr_number in self._pr_review_cache:
            return self._pr_review_cache[pr_number]

        # Check if PR was reviewed
        pr_author = _login_of(pr_data.get("user"))
        merged_at = _parse_ts(pr_data.get("merged_at"))

        if merged_at is None:
            self._pr_review_cache[pr_number] = False
            return False

        # Fetch reviews for this PR
        try:
            reviews = list(
                _iter_github_pr_reviews(self._git, git_url, pr_number)
            )
        except Exception as e:
            _LOGGER.debug(f"Failed to fetch reviews for PR #{pr_number}: {e}")
            reviews = []

        # Check for valid approval
        has_approval = _has_valid_approval(reviews, pr_author, merged_at)
        self._pr_review_cache[pr_number] = has_approval

        return has_approval


def _parse_owner_repo(repo_url: str) -> tuple[str, str]:
    """Parse owner and repo name from GitHub URL."""
    normalized = repo_url.strip().rstrip("/")
    if not normalized.startswith("https://github.com/"):
        raise ValueError(f"Unsupported git repository host: {repo_url}")
    tail = normalized.removeprefix("https://github.com/")
    parts = tail.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _iter_github_pr_reviews(
    git: GitClient, repo_url: str, pr_number: int
) -> Iterable[Dict[str, Any]]:
    """Iterate over all reviews for a pull request with pagination."""
    owner, repo = _parse_owner_repo(repo_url)

    page = 1
    per_page = 100
    while True:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/pulls/"
            f"{pr_number}/reviews"
            f"?per_page={per_page}&page={page}"
        )
        resp = git._execute_with_rate_limit(  # type: ignore[attr-defined]
            lambda: git._session.get(
                url,
                timeout=10,
            ),  # type: ignore[attr-defined]
            name=(
                f"github.reviews({owner}/{repo}#{pr_number})#p{page}"
            ),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to list PR reviews (#{pr_number}): {resp.status_code}"
            )
        batch = resp.json()
        if not batch:
            break
        yield from batch
        page += 1


def _login_of(user_obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(user_obj, dict):
        return None
    login = user_obj.get("login")
    if isinstance(login, str):
        return login
    return None


def _is_human(user_obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(user_obj, dict):
        return False
    utype = user_obj.get("type")
    login = user_obj.get("login")
    if not isinstance(utype, str) or not isinstance(login, str):
        return False
    if utype != "User":
        return False
    if login.endswith("[bot]"):
        return False
    return True


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _has_valid_approval(
    reviews: Iterable[Dict[str, Any]],
    pr_author: Optional[str],
    merged_at: datetime,
) -> bool:
    approved_by: set[str] = set()
    dismissed_by: set[str] = set()

    for rv in reviews:
        state = rv.get("state")
        user = rv.get("user")
        submitted_at = _parse_ts(rv.get("submitted_at"))

        # Only consider reviews submitted at or before merge time
        if submitted_at is None or submitted_at > merged_at:
            continue

        reviewer_login = _login_of(user)

        if state == "DISMISSED" and reviewer_login:
            dismissed_by.add(reviewer_login)

        if state == "APPROVED":
            if not _is_human(user):
                continue
            if reviewer_login is None:
                continue
            if pr_author and reviewer_login == pr_author:
                continue
            approved_by.add(reviewer_login)

    # Exclude approvals from reviewers whose approvals were dismissed
    effective = [r for r in approved_by if r not in dismissed_by]
    return len(effective) > 0
