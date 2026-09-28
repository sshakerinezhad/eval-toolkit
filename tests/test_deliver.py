import json

import pytest

import deliver


def _src(tmp_path):
    src = tmp_path / "judge"
    src.mkdir()
    (src / "scores.jsonl").write_text('{"task": 1, "variant": "a", "sample": 0, "score": 1.0}\n')
    (src / "metrics.json").write_text(json.dumps({
        "variants": {"a": {"mean": {"value": 0.61, "lo": 0.52, "hi": 0.70, "n": 30}},
                     "b": {"mean": {"value": 0.68, "lo": 0.59, "hi": 0.77, "n": 30}}},
        "diff": {"a": "a", "b": "b", "metrics": {"mean": {"value": 0.07, "lo": -0.02, "hi": 0.16, "n": 30, "up": 17, "down": 9, "same": 4}}},
        "judge_errors": {"timeout": 1, "api_error": 0, "unparsed": 2}}))
    (src / "plot_means_mean.png").write_bytes(b"png")
    (src / "plot_means_mean.json").write_text("{}")
    (src / "report.md").write_text("# Model b beats a by 7 points on 30 tasks\n\n- x\n")
    return src


def test_collect_zip_and_email(tmp_path, capsys, monkeypatch):
    src = _src(tmp_path)
    slides = tmp_path / "slides.pdf"
    slides.write_bytes(b"pdf")
    monkeypatch.setattr(deliver, "repo_link", lambda: "https://github.com/x/y")
    assert deliver.main([str(src), "--also", str(slides), "--out", str(tmp_path / "deliverables")]) == 0
    out = tmp_path / "deliverables"
    assert sorted(p.name for p in out.iterdir()) == ["metrics.json", "plot_means_mean.json", "plot_means_mean.png", "report.md", "results.jsonl", "slides.pdf"]
    assert (tmp_path / "deliverables.zip").exists()
    text = capsys.readouterr().out
    assert "Subject: LLM Evaluation and Analysis Submission - Shayan Shakeri" in text
    assert "Model b beats a by 7 points on 30 tasks" in text
    assert "b: mean 0.680 (95% CI 0.590 to 0.770, n = 30 tasks)" in text
    assert "b minus a: +0.070 (95% CI -0.020 to +0.160), up 17 / down 9 / same 4 of 30 tasks" in text
    assert "judge errors: 1 timeouts, 0 api, 2 unparsed" in text and "https://github.com/x/y" in text
    assert "Attached: deliverables.zip, results.jsonl" in text


def test_headline_falls_back_to_bracket_when_report_has_blanks(tmp_path):
    out = tmp_path / "d"
    out.mkdir()
    (out / "report.md").write_text("# [headline: fill me]\n")
    assert deliver.headline(out).startswith("[FINDING")


def test_missing_scores_dies(tmp_path):
    with pytest.raises(SystemExit, match="scores.jsonl not found"):
        deliver.collect(tmp_path, tmp_path / "d", [])
