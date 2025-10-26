from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_readme_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "1")
