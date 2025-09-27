import json, tempfile, types
import CLI as CLI_mod

def test_iter_urls_merge_dedup():
    tmp = tempfile.NamedTemporaryFile("w+", delete=False)
    try:
        tmp.write("https://a\n# comment\nhttps://b\n")
        tmp.flush()
        got = list(CLI_mod.iter_urls(["https://a","https://c","https://a"], tmp.name))
        assert got == ["https://a","https://c","https://b"]
    finally:
        tmp.close()

def test_do_score_stdout_success(monkeypatch, capsys):
    monkeypatch.setattr(CLI_mod, "determineResource", lambda url: types.SimpleNamespace(url=url))
    monkeypatch.setattr(CLI_mod, "score_resource",
                        lambda res: {"name":"n","category":"CODE","net_score":0.5,"net_score_latency":2})
    rc = CLI_mod.do_score(["https://ok"], None, "-", append=False)
    assert rc == 0
    o = json.loads(capsys.readouterr().out.strip())
    assert o["category"] == "CODE" and 0.0 <= o["net_score"] <= 1.0

def test_do_score_file_and_append(monkeypatch, tmp_path):
    monkeypatch.setattr(CLI_mod, "determineResource", lambda url: types.SimpleNamespace(url=url))
    monkeypatch.setattr(CLI_mod, "score_resource",
                        lambda res: {"name":"n","category":"CODE","net_score":1.0,"net_score_latency":1})
    out = tmp_path / "out.ndjson"
    assert CLI_mod.do_score(["u1"], None, str(out), append=False) == 0
    assert CLI_mod.do_score(["u2"], None, str(out), append=True) == 0
    assert sum(1 for _ in open(out, "r", encoding="utf-8")) == 2

def test_do_score_handles_exception(monkeypatch, capsys):
    def boom(url): raise RuntimeError("bad")
    monkeypatch.setattr(CLI_mod, "determineResource", boom)
    rc = CLI_mod.do_score(["https://bad"], None, "-", append=False)
    assert rc == 1
    o = json.loads(capsys.readouterr().out.strip())
    assert o["category"] == "UNKNOWN" and "error" in o

def test_do_score_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(CLI_mod, "determineResource", lambda url: types.SimpleNamespace(url=url))
    calls = {"n": 0}
    def fake_score(res):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"name":"n","category":"CODE","net_score":0.1,"net_score_latency":1}
        raise KeyboardInterrupt()
    monkeypatch.setattr(CLI_mod, "score_resource", fake_score)
    rc = CLI_mod.do_score(["u1","u2"], None, "-", append=False)
    assert rc == 130  # interrupted
    # exactly one line printed
    assert len(capsys.readouterr().out.strip().splitlines()) == 1
