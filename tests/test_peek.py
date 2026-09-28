import json

import pytest

import peek


def test_jsonl_summary_and_rows(tmp_path, capsys):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in [
        {"id": 1, "prompt": "short", "attachments": ["a.csv"]},
        {"id": 2, "prompt": "a much longer prompt " * 20, "attachments": []},
        {"id": 2, "prompt": "x"}]))
    peek.main([str(p), "--n", "2", "--chars", "30"])
    out = capsys.readouterr().out
    assert "rows: 3" in out and "prompt (3)" in out and "attachments (2)" in out
    assert "longest prompt: 420 chars (row 1)" in out and "attachments (attachments): 1 across 1 rows" in out
    assert "id: 2 distinct, 1 DUPLICATE" in out and out.count("--- row") == 2
    assert "prompt [420 chars]: ...er prompt a much longer prompt" in out  # long field: the tail only


def test_csv_and_json_object(tmp_path, capsys):
    c = tmp_path / "t.csv"
    c.write_text('Task ID,Prompt,File Attachments\n7,hello,"a.csv\nb.pdf"\n8,bye,\n')
    peek.main([str(c)])
    out = capsys.readouterr().out
    assert "rows: 2" in out and "attachments (File Attachments): 2 across 1 rows, max 2" in out
    j = tmp_path / "t.json"
    j.write_text(json.dumps({"t1": {"prompt": "p"}, "t2": {"prompt": "q"}}))
    rows = peek.load(j)
    assert rows[0] == {"_key": "t1", "prompt": "p"}


def test_unknown_suffix_dies(tmp_path):
    (tmp_path / "x.parquet").write_bytes(b"0")
    with pytest.raises(SystemExit, match="not supported"):
        peek.load(tmp_path / "x.parquet")
