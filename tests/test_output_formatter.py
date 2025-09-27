import io, json, pytest
from Output_Formatter import OutputFormatter, clamp01, as_nonneg_int

def test_clamp01_and_nonneg_int_rounding():
    assert clamp01(-1) == 0.0
    assert clamp01(0.33333) == 0.3333
    assert clamp01(4) == 1.0
    assert as_nonneg_int(-3) == 0 and as_nonneg_int("7") == 7

def test_coerce_and_write_line():
    buf = io.StringIO()
    fmt = OutputFormatter(fh=buf, score_keys={"s"}, latency_keys={"l"})
    fmt.write_line({"s": 1.7, "l": -9, "x": 2})
    o = json.loads(buf.getvalue())
    assert o["s"] == 1.0 and o["l"] == 0 and o["x"] == 2

def test_to_path_context_and_multiple_lines(tmp_path):
    p = tmp_path / "o.ndjson"
    with OutputFormatter.to_path(str(p)) as fmt:
        fmt.write_line({"a": 1})
        fmt.write_line({"b": 2})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["a"] == 1

def test_schema_validation_raises_if_jsonschema_present(tmp_path):
    try:
        import jsonschema  # noqa: F401
    except Exception:
        pytest.skip("jsonschema not installed")
    p = tmp_path / "o2.ndjson"
    fmt = OutputFormatter.to_path(str(p), schema={
        "type":"object","properties":{"a":{"type":"string"}}, "required":["a"]
    })
    with pytest.raises(ValueError):
        fmt.write_line({"a": 123})
    fmt.close()

def test_close_idempotent(tmp_path):
    p = tmp_path / "out.ndjson"
    fmt = OutputFormatter.to_path(str(p))
    fmt.write_line({"x": 1})
    fmt.close()
    fmt.close()  # should not raise
    assert p.read_text(encoding="utf-8").strip() == '{"x": 1}'


def test_coerce_fields_does_not_mutate_input():
    from copy import deepcopy
    rec = {"s": 2.0, "l": -3, "k": "v"}
    original = deepcopy(rec)
    fmt = OutputFormatter(fh=io.StringIO(), score_keys={"s"}, latency_keys={"l"})
    out = fmt.coerce_fields(rec)
    # Original unchanged
    assert rec == original
    # Coerced copy
    assert out["s"] == 1.0 and out["l"] == 0 and out["k"] == "v"
