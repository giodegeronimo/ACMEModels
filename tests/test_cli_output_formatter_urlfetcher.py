import io, json, os, tempfile
from CLI import iter_urls
from Output_Formatter import OutputFormatter
from URL_Fetcher import hasLicenseSection
def test_iter_urls_dedup_and_file_merge(tmp_path):
    f = tmp_path/"urls.txt"
    f.write_text("# c\nhttps://a\n\nhttps://b\nhttps://a\n")
    got = list(iter_urls(["https://b","https://c"], str(f)))
    assert got == ["https://b","https://c","https://a"]
def test_output_formatter_coercions():
    buf = io.StringIO()
    fmt = OutputFormatter(fh=buf, score_keys={"s1","s2"}, latency_keys={"l1","l2"})
    fmt.write_line({"s1":1.234,"s2":-0.5,"l1":-3,"l2":"7","name":"x"})
    buf.seek(0)
    obj = json.loads(buf.read())
    assert obj["s1"]==1.0 and obj["s2"]==0.0
    assert obj["l1"]==0 and obj["l2"]==7
def test_has_license_section():
    t = "# Title\n\n## LICENSE\nMIT"
    assert hasLicenseSection(t) is True
    assert hasLicenseSection("# readme\nnope") is False
