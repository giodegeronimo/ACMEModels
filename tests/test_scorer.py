import types, os
import Scorer as S
from URL_Fetcher import hasLicenseSection as real_hasLicenseSection

def mk(readme=None, md=None):
    return S.Inputs(resource=None, metadata=md or {}, readme=readme)

def test_metrics_rampup_busfactor_claims_license(monkeypatch):
    i1 = mk("# Readme\n" + "x"*300, {"likes":15,"fileCount":9,"lastModified":"2024-06-01"})
    assert 0.5 <= S.metric_ramp_up_time(i1) <= 1.0
    i2 = mk("", {"fileCount":20,"lastModified":"2024-07-01"})
    assert 0.4 <= S.metric_bus_factor(i2) <= 1.0
    txt = "Benchmark results.\n\n| metric | value |\n|---|---|\n| acc | 90 |"
    assert S.metric_performance_claims(mk(txt)) > 0.7
    monkeypatch.setattr(S, "hasLicenseSection", lambda t: True)
    assert S.metric_license(mk("whatever")) == 1.0
    monkeypatch.setattr(S, "hasLicenseSection", lambda t: False)
    assert S.metric_license(mk("# r")) == 0.0
    monkeypatch.setattr(S, "hasLicenseSection", real_hasLicenseSection)

def test_dataset_code_quality_and_links_and_edges():
    t = "See https://huggingface.co/datasets/xsum and https://github.com/huggingface/transformers"
    assert S.metric_dataset_and_code_score(mk(t)) == 1.0
    assert 0.49 <= S.metric_dataset_and_code_score(mk("https://huggingface.co/datasets/xsum")) <= 0.51
    assert S.metric_dataset_quality(mk("", {"downloads":5})) == 0.0
    assert 0.3 <= S.metric_code_quality(mk("readme "*60, {"fileCount":15,"lastModified":"2024-04-01"})) <= 1.0

def test_size_score_buckets_and_helpers(monkeypatch):
    small = S.metric_size_score(mk("", {"fileCount":3}))
    mid   = S.metric_size_score(mk("", {"fileCount":12}))
    large = S.metric_size_score(mk("", {"fileCount":30}))
    huge  = S.metric_size_score(mk("", {"fileCount":50}))
    assert small["raspberry_pi"] >= mid["raspberry_pi"] >= large["raspberry_pi"]
    assert S._size_scalar({}) == 0.0
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert 2 <= S._cpu_workers() <= 8  # capped

def test_latency_wrapper_and_score_resource_paths(monkeypatch):
    v, ms = S._latency_wrapper(lambda: 0.7)
    assert v == 0.7 and ms >= 0
    def bad(): raise RuntimeError("x")
    bad.__name__ = "metric_size_score"
    assert S._latency_wrapper(bad)[0] == {}

    class Cat: 
        def __init__(self, name): self.name = name
    class Ref: 
        def __init__(self, name, cat): self.name=name; self.category=Cat(cat)
    class Res:
        def __init__(self): self.ref = Ref("n","CODE")
        def fetchMetadata(self): return {"fileCount":10,"lastModified":"2024-01-01"}
        def fetchReadme(self):   return "# Readme\n- a\n- b\n- c\n"
    out = S.score_resource(Res())
    assert out["category"] == "CODE" and 0.0 <= out["net_score"] <= 1.0
    # latencies present
    for k in ("net_score_latency","ramp_up_time_latency","bus_factor_latency",
              "performance_claims_latency","license_latency","size_score_latency",
              "dataset_and_code_score_latency","dataset_quality_latency","code_quality_latency"):
        assert isinstance(out[k], int) and out[k] >= 0

    class Bad:
        def __init__(self): self.ref = Ref("n","CODE")
        def fetchMetadata(self): raise RuntimeError("m")
        def fetchReadme(self):   raise RuntimeError("r")
    out2 = S.score_resource(Bad())
    assert out2["category"] == "CODE" and "size_score" in out2

def test_net_weights_sum_to_one():
    assert abs(sum(S.NET_WEIGHTS.values()) - 1.0) < 1e-6


def test_latency_wrapper_non_size_error_returns_zero():
    def bad(): 
        raise RuntimeError("boom")
    bad.__name__ = "metric_other"
    v, ms = S._latency_wrapper(bad)
    assert v == 0.0 and ms >= 0

def test_size_score_threshold_edges():
    def score(fc): return S.metric_size_score(mk("", {"fileCount": fc}))
    # Check stepdowns at boundaries (5/6, 10/11, 15/16, 40/41)
    rp_5 = score(5)["raspberry_pi"];   rp_6 = score(6)["raspberry_pi"]
    rp_10 = score(10)["jetson_nano"];  rp_11 = score(11)["jetson_nano"]
    dp_15 = score(15)["desktop_pc"];   dp_16 = score(16)["desktop_pc"]
    aws_40 = score(40)["aws_server"];  aws_41 = score(41)["aws_server"]
    assert rp_5 >= rp_6
    assert rp_10 >= rp_11
    assert dp_15 >= dp_16
    assert aws_40 >= aws_41

def test_score_resource_licenses_section_true_false(monkeypatch):
    # Toggle hasLicenseSection to cover both branches during assembly
    class Cat:  # shim
        def __init__(self, name): self.name = name
    class Ref:
        def __init__(self): self.name = "n"; self.category = Cat("CODE")
    class Res:
        def __init__(self): self.ref = Ref()
        def fetchMetadata(self): return {"fileCount": 1}
        def fetchReadme(self):   return "# License\nMIT"
    # First, force no license
    monkeypatch.setattr(S, "hasLicenseSection", lambda _t: False)
    out0 = S.score_resource(Res())
    # Then, has license
    monkeypatch.setattr(S, "hasLicenseSection", lambda _t: True)
    out1 = S.score_resource(Res())
    assert out0["license"] in (0.0, 1.0) and out1["license"] == 1.0
