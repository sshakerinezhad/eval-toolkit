import os
from pathlib import Path

import demo


def test_demo_runs_the_whole_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert demo.main() == 0
    d = tmp_path / "results/demo"
    for f in ("full/raw.jsonl", "judge/raw.jsonl", "judge/scores.jsonl", "judge/metrics.json", "judge/judge_labels.csv",
              "judge/failures.md", "judge/failure_labels.csv", "classify/scores.json", "judge/plot_paired.png",
              "deliverables/results.jsonl", "deliverables.zip"):
        assert (d / f).exists(), f
    out = capsys.readouterr().out
    assert "gold:" in out and "judge errors: timeouts 1" in out and "judge vs you:" in out and "demo ok" in out
    assert not (tmp_path / ".cache").exists()  # nothing cached outside results/demo
