
import os, types
import LLM_Analyzer as A

def test_fallback_rules_detects_signals(monkeypatch):
    # force fallback path (no endpoint configured)
    monkeypatch.delenv("GENAI_ENDPOINT", raising=False)
    text = (
        "# Project\n\n"
        "```python\n"
        'print("hi")\n'
        "```\n\n"
        "## Results\n"
        "| metric | value |\n"
        "| --- | --- |\n"
        "| acc | 99 |\n\n"
        "See paper: https://arxiv.org/abs/1234.5678\n"
        "License: MIT\n\n"
        "Dataset: https://huggingface.co/datasets/EdinburghNLP/xsum\n"
        "Code: https://github.com/huggingface/transformers\n"
    )
    out = A.analyze_readme_and_metadata(text, {})
    assert out["has_examples"] is True
    assert out["has_benchmarks"] in ("third_party", "self_reported")
    assert out["license_name"].lower().startswith("mit")
    assert out["has_dataset_links"] is True
    assert out["has_code_links"] is True

def test_remote_success_and_value_coercion(monkeypatch):
    # force remote path
    monkeypatch.setenv("GENAI_ENDPOINT", "https://fake-endpoint")
    monkeypatch.setenv("GENAI_API_KEY", "abc123")

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            # mixed types/casing to exercise coercions
            return {
                "has_examples": False,
                "has_benchmarks": "THIRD_PARTY",
                "license_name": "Apache-2.0",
                "has_dataset_links": 1,
                "has_code_links": "yes",
            }

    monkeypatch.setattr(A, "requests", types.SimpleNamespace(post=lambda *a, **k: Resp()))
    out = A.analyze_readme_and_metadata("x", {})
    assert out["has_examples"] is False
    assert out["has_benchmarks"] == "third_party"
    assert out["license_name"] == "Apache-2.0"
    assert out["has_dataset_links"] is True
    assert out["has_code_links"] is True

def test_remote_failure_falls_back(monkeypatch):
    # capture baseline fallback on same input
    monkeypatch.delenv("GENAI_ENDPOINT", raising=False)
    baseline = A.analyze_readme_and_metadata("Usage: run\n", {})
    # now force remote but make it fail -> equals baseline fallback
    monkeypatch.setenv("GENAI_ENDPOINT", "https://fake-endpoint")
    def boom_post(*a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(A, "requests", types.SimpleNamespace(post=boom_post))
    out = A.analyze_readme_and_metadata("Usage: run\n", {})
    assert out == baseline

