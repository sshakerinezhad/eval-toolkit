import json

import pytest

import metrics


# ---------- helpers ----------

# Toy set: two tasks, two variants, two attempts each. Untrained rows come first, so it is the baseline.
TOY = [
    ("A", "untrained", 1), ("A", "untrained", 0),
    ("B", "untrained", 0), ("B", "untrained", 0),
    ("A", "trained", 1), ("A", "trained", 1),
    ("B", "trained", 0), ("B", "trained", 1),
]


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "scores.jsonl"
    p.write_text("\n".join(json.dumps({"task": t, "variant": v, "score": s}) for t, v, s in rows) + "\n",
                 encoding="utf-8")
    return p


def _write_csv(tmp_path, rows):
    p = tmp_path / "scores.csv"
    p.write_text("task,variant,score\n" + "".join(f"{t},{v},{s}\n" for t, v, s in rows), encoding="utf-8")
    return p


def _run(path, *extra):
    assert metrics.main([str(path), "--boot", "2000", *extra]) == 0
    return json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))


def _check_toy(doc):
    tr, un, d = doc["variants"]["trained"], doc["variants"]["untrained"], doc["diff"]
    assert (d["a"], d["b"]) == ("untrained", "trained")
    assert tr["mean"]["value"] == pytest.approx(0.75)
    assert un["mean"]["value"] == pytest.approx(0.25)
    assert d["metrics"]["mean"]["value"] == pytest.approx(0.50)
    assert tr["pass@2"]["value"] == pytest.approx(1.0)
    assert un["pass@2"]["value"] == pytest.approx(0.5)
    assert tr["pass^2"]["value"] == pytest.approx(0.5)
    assert un["pass^2"]["value"] == pytest.approx(0.0)
    m = d["metrics"]["mean"]
    assert (m["up"], m["down"], m["same"]) == (2, 0, 0)
    assert tr["mean"]["n"] == 2


def _every_interval(doc):
    for v in doc["variants"].values():
        yield from v.values()
    yield from doc["diff"]["metrics"].values()


# ---------- estimators (hand-computed) ----------

@pytest.mark.parametrize("n,c,k,want", [
    (5, 2, 2, 0.7),   # 1 - C(3,2)/C(5,2) = 1 - 3/10
    (4, 0, 2, 0.0),   # no passes -> never
    (3, 3, 1, 1.0),   # all pass
    (5, 4, 2, 1.0),   # only 1 failure, can't pick 2 failures
    (4, 1, 1, 0.25),  # k=1 is just c/n
])
def test_pass_at_k(n, c, k, want):
    assert metrics.pass_at_k(n, c, k) == pytest.approx(want)


@pytest.mark.parametrize("n,c,k,want", [
    (5, 2, 2, 0.1),   # C(2,2)/C(5,2) = 1/10
    (4, 4, 3, 1.0),   # all pass
    (5, 1, 2, 0.0),   # only 1 pass, can't pick 2 passes
    (4, 2, 1, 0.5),   # k=1 is just c/n
])
def test_pass_pow_k(n, c, k, want):
    assert metrics.pass_pow_k(n, c, k) == pytest.approx(want)


# ---------- toy set end to end ----------

def test_toy_jsonl(tmp_path):
    doc = _run(_write_jsonl(tmp_path, TOY))
    _check_toy(doc)
    assert doc["ks"] == [1, 2]  # default: 1 and attempts per task


def test_toy_csv_same_numbers(tmp_path):
    _check_toy(_run(_write_csv(tmp_path, TOY)))


def test_every_interval_contains_point(tmp_path):
    doc = _run(_write_jsonl(tmp_path, TOY))
    for r in _every_interval(doc):
        assert r["lo"] <= r["value"] <= r["hi"]


def test_baseline_flag_overrides_file_order(tmp_path):
    trained_first = TOY[4:] + TOY[:4]
    doc = _run(_write_jsonl(tmp_path, trained_first))
    assert doc["diff"]["metrics"]["mean"]["value"] == pytest.approx(-0.50)  # file order: untrained - trained
    doc = _run(_write_jsonl(tmp_path, trained_first), "--baseline", "untrained")
    _check_toy(doc)


def test_same_seed_same_intervals(tmp_path):
    p = _write_jsonl(tmp_path, TOY)
    assert _run(p) == _run(p)


# ---------- load / by_task ----------

def test_load_rows_and_by_task(tmp_path):
    rows = metrics.load_rows(_write_jsonl(tmp_path, [(1, "x", 1), ("1", "x", 0.5)]))
    assert rows == [{"task": "1", "variant": "x", "score": 1.0}, {"task": "1", "variant": "x", "score": 0.5}]
    assert metrics.by_task(rows) == {"1": {"x": [1.0, 0.5]}}  # int 1 and "1" are the same task


def test_bad_score_dies(tmp_path):
    with pytest.raises(SystemExit, match="not a number"):
        metrics.load_rows(_write_jsonl(tmp_path, [("A", "x", "high")]))


# ---------- errors ----------

def test_missing_variant_dies(tmp_path):
    rows = TOY + [("C", "trained", 1)]  # task C has no untrained rows
    with pytest.raises(SystemExit, match="no rows"):
        _run(_write_jsonl(tmp_path, rows))


def test_k_above_attempts_dies(tmp_path):
    with pytest.raises(SystemExit, match="every k must be"):
        _run(_write_jsonl(tmp_path, TOY), "--k", "3")


def test_unknown_baseline_dies(tmp_path):
    with pytest.raises(SystemExit, match="--baseline"):
        _run(_write_jsonl(tmp_path, TOY), "--baseline", "nope")


# ---------- non-binary ----------

def test_threshold_on_percentages(tmp_path):
    rows = [("A", "x", 80), ("A", "x", 40), ("B", "x", 50), ("B", "x", 10)]
    doc = _run(_write_jsonl(tmp_path, rows), "--threshold", "50")
    x = doc["variants"]["x"]
    assert x["mean"]["value"] == pytest.approx(45.0)      # raw scores: (60 + 30) / 2
    assert x["pass@1"]["value"] == pytest.approx(0.5)     # each task 1 of 2 passes
    assert x["pass@2"]["value"] == pytest.approx(1.0)
    assert x["pass^2"]["value"] == pytest.approx(0.0)
    assert doc["diff"] is None                            # one variant -> no diff
