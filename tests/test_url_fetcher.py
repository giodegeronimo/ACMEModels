import json, types
import URL_Fetcher as UF

def test_classify_and_determine_resource_types_variants():
    r1 = UF.classifyUrl("https://huggingface.co/google/flan-t5-base/tree/main")
    r2 = UF.classifyUrl("https://huggingface.co/datasets/EdinburghNLP/xsum")
    r3 = UF.classifyUrl("https://github.com/huggingface/transformers")
    r4 = UF.classifyUrl("https://example.com/whatever")
    assert r1.category == UF.UrlCategory.MODEL and r1.repoId == "google/flan-t5-base"
    assert r2.category == UF.UrlCategory.DATASET and r2.repoId == "EdinburghNLP/xsum"
    assert r3.host == UF.Host.GITHUB and r3.category == UF.UrlCategory.CODE
    assert r4.category == UF.UrlCategory.UNKNOWN
    assert UF.determineResource(r3.url).ref.category == UF.UrlCategory.CODE

def test_parse_url_file_and_license(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("# cmt\nhttps://github.com/a/b\n\nhttps://huggingface.co/datasets/x/y\n", encoding="ascii")
    refs = UF.parseUrlFile(str(p))
    assert len(refs) == 2 and refs[0].repoId == "a/b"
    assert UF.hasLicenseSection("# License\nMIT")
    assert not UF.hasLicenseSection("# Not license\nfoo")

# ---- HTTP helper coverage: cache, retry, json decode ----
class _Resp:
    def __init__(self, status=200, text="", json_obj=None):
        self.status_code = status; self.text = text; self._json = json_obj
    def raise_for_status(self):
        if self.status_code >= 400: raise types.SimpleNamespace()
    def json(self):
        if isinstance(self._json, Exception): raise self._json
        return self._json

def test_http_get_json_retries_429_then_ok(monkeypatch):
    calls = {"n": 0}
    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(status=429, json_obj={"x":"y"}) if calls["n"] == 1 else _Resp(status=200, json_obj={"ok":True})
    monkeypatch.setattr(UF, "_session", types.SimpleNamespace(get=fake_get))
    UF.clearCache()
    out = UF._http_get_json("https://api/x", headers={})
    assert out == {"ok": True} and calls["n"] >= 2

def test_http_get_json_decode_error(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _Resp(status=200, json_obj=json.JSONDecodeError("bad","x",0))
    monkeypatch.setattr(UF, "_session", types.SimpleNamespace(get=fake_get))
    UF.clearCache()
    assert UF._http_get_json("https://api/x", headers={}) is None

def test_http_get_text_cache_and_retry(monkeypatch):
    calls = {"n": 0}
    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1: return _Resp(status=429, text="")
        return _Resp(status=200, text="# README")
    monkeypatch.setattr(UF, "_session", types.SimpleNamespace(get=fake_get))
    UF.clearCache()
    t1 = UF._http_get_text("https://raw/readme.md", headers={})
    t2 = UF._http_get_text("https://raw/readme.md", headers={})
    assert t1.startswith("# README") and t2.startswith("# README")
    assert calls["n"] >= 2  # retried once then cached

def test_model_dataset_code_metadata_and_readme(monkeypatch):
    # route based on URL substrings used in fetchers
    def fake_json(url, headers):
        if "/api/models/" in url:
            return {"downloads": 10, "likes": 5, "lastModified":"2024-06-01","sha":"abc","siblings":[1,2,3]}
        if "/api/datasets/" in url:
            return {"downloads": 20, "likes": 7, "lastModified":"2024-05-01","sha":"def","siblings":[1]}
        if "api.github.com/repos/" in url:
            return {"stargazers_count": 100, "forks_count": 5, "open_issues_count": 1,
                    "license": {"spdx_id": "MIT"}, "updated_at":"2024-01-01",
                    "default_branch":"main", "archived": False}
        return None
    def fake_text(url, headers):
        if "/raw/main/README.md" in url: return "# Readme"
        if "/raw/master/README.md" in url: return "# Readme master"
        if "raw.githubusercontent.com" in url: return "# GH Readme"
        return None
    monkeypatch.setattr(UF, "_http_get_json", fake_json)
    monkeypatch.setattr(UF, "_http_get_text", fake_text)

    mref = UF.classifyUrl("https://huggingface.co/google/flan-t5-base")
    dref = UF.classifyUrl("https://huggingface.co/datasets/EdinburghNLP/xsum")
    cref = UF.classifyUrl("https://github.com/huggingface/transformers")
    m = UF.ModelResource(mref); d = UF.DatasetResource(dref); c = UF.CodeResource(cref)

    mm = m.fetchMetadata(); dm = d.fetchMetadata(); cm = c.fetchMetadata()
    assert mm["fileCount"] == 3 and dm["fileCount"] == 1 and cm["licenseSpdx"] == "MIT"
    assert m.fetchReadme().startswith("# Readme")
    assert d.fetchReadme().startswith("# Readme")
    assert c.fetchReadme().startswith("# GH Readme")

def test_unknown_host_noop_resource():
    ref = UF.classifyUrl("https://example.com/x")
    res = UF.determineResource(ref.url)
    assert res.fetchMetadata() == {} and res.fetchReadme() is None


def test_parse_url_file_skips_bad_line(monkeypatch, tmp_path):
    # Force classifyUrl to raise on one line; parseUrlFile should skip it (no crash)
    p = tmp_path / "u_bad.txt"
    p.write_text("https://good\nhttps://bad\n", encoding="ascii")
    real = UF.classifyUrl
    def fake(url):
        if "bad" in url:
            raise ValueError("boom")
        return real(url)
    monkeypatch.setattr(UF, "classifyUrl", fake)
    refs = UF.parseUrlFile(str(p))
    assert any("good" in r.normalizedUrl for r in refs)
    # and only the good one made it through
    assert len(refs) == 1

def test_http_get_json_429_backs_off_with_sleep(monkeypatch):
    # Verify 429 triggers time.sleep backoff path
    class _Resp:
        def __init__(self, status=200, js=None): self.status_code=status; self._js=js
        def raise_for_status(self): 
            if self.status_code >= 400: raise RuntimeError("HTTP")
        def json(self): return self._js
    calls = {"n": 0, "sleep": 0}
    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(status=429, js={"x":"y"}) if calls["n"] == 1 else _Resp(status=200, js={"ok": True})
    def fake_sleep(_secs): calls["sleep"] += 1
    monkeypatch.setattr(UF, "_session", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(UF.time, "sleep", fake_sleep)
    UF.clearCache()
    out = UF._http_get_json("https://api/with429", headers={})
    assert out == {"ok": True} and calls["sleep"] >= 1
