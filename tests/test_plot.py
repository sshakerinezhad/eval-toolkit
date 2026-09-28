import json

import pytest

import plot


def _metrics(tmp_path):
    doc = {"n_tasks": 4, "variants": {"a": {"mean": {"value": 0.5, "lo": 0.3, "hi": 0.7, "n": 4}},
                                      "b": {"mean": {"value": 0.7, "lo": 0.5, "hi": 0.9, "n": 4}}},
           "diff": {"a": "a", "b": "b", "metrics": {"mean": {"value": 0.2, "lo": 0.05, "hi": 0.35, "n": 4, "up": 3, "down": 1, "same": 0}}}}
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(doc))
    return p


def test_means_writes_png(tmp_path):
    out = plot.means(_metrics(tmp_path))
    assert out.name == "plot_means_mean.png" and out.stat().st_size > 1000
    d = json.loads(out.with_suffix(".json").read_text())
    assert d["kind"] == "means" and d["series"][1] == {"model": "b", "value": 0.7, "lo": 0.5, "hi": 0.9}


def test_means_unknown_metric_dies(tmp_path):
    with pytest.raises(SystemExit, match="not in metrics.json"):
        plot.means(_metrics(tmp_path), metric="pass@3")


def test_paired_uses_ci_band_when_metrics_beside(tmp_path):
    _metrics(tmp_path)
    s = tmp_path / "scores.jsonl"
    s.write_text("".join(json.dumps(r) + "\n" for r in [
        {"task": 1, "variant": "a", "sample": 0, "score": 0.5}, {"task": 1, "variant": "b", "sample": 0, "score": 1.0},
        {"task": 2, "variant": "a", "sample": 0, "score": 1.0}, {"task": 2, "variant": "b", "sample": 0, "score": 0.5}]))
    out = plot.paired(s)
    assert out.name == "plot_paired.png" and out.stat().st_size > 1000
    d = json.loads(out.with_suffix(".json").read_text())
    assert d["up"] == 1 and d["down"] == 1 and d["ci"] == [0.05, 0.35] and d["per_task"][0]["diff"] == -0.5


def test_paired_needs_two_variants(tmp_path):
    s = tmp_path / "scores.jsonl"
    s.write_text(json.dumps({"task": 1, "variant": "a", "sample": 0, "score": 1}) + "\n")
    with pytest.raises(SystemExit, match="exactly two variants"):
        plot.paired(s)


def test_taxonomy_from_judge_scores(tmp_path):
    doc = {"categories": ["missing", "wrong_value", "unparsed"],
           "models": {"openai:x": {"n": 5, "variants": {"m": {"n": 5, "counts": {"missing": 3, "wrong_value": 2, "unparsed": 0},
                                                              "share": {}}}, "agreement": None}}}
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(doc))
    out = plot.taxonomy(p)
    assert out.name == "plot_taxonomy.png" and out.stat().st_size > 1000
    assert json.loads(out.with_suffix(".json").read_text())["series"][0]["counts"] == [3, 2, 0]


def test_cli(tmp_path, capsys):
    assert plot.main(["--kind", "means", str(_metrics(tmp_path))]) == 0
    assert "wrote" in capsys.readouterr().out
