import csv
import json

import pytest

import grade


@pytest.mark.parametrize("text,expected", [
    ('{"rationale": "r", "is_criteria_true": true}', (True, "r")),
    ('```json\n{"rationale": "r", "is_criteria_true": false}\n```', (False, "r")),
    ("the criterion is met", (None, "the criterion is met")),
    ('{"rationale": "r", "is_criteria_true": "yes"}', (None, '{"rationale": "r", "is_criteria_true": "yes"}')),
    (None, (None, "(no reply)")),
    ('{"rationale": "line1\nline2", "is_criteria_true": true}', (True, "line1\nline2")),
])
def test_parse_verdict_plain_fenced_bad(text, expected):
    assert grade.parse_verdict(text) == expected


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path, n_samples):
    """Two tasks, one model, n samples each; task 1 has 2 criteria, task 2 has 1."""
    raw = tmp_path / "raw.jsonl"
    _write(raw, [{"task_id": t, "model": "m", "sample_index": i, "response": f"answer {t}-{i}",
                  "metadata": {"prompt_raw": f"question {t}"}}
                 for t in (1, 2) for i in range(n_samples)])
    data = tmp_path / "train.csv"
    data.write_text('Task ID,Rubric JSON\n1,"{""c1"": {""description"": ""says A""}, ""c2"": {""description"": ""says B""}}"\n'
                    '2,"{""c1"": {""description"": ""says C""}}"\n', encoding="utf-8")
    prompt = tmp_path / "judge.md"
    prompt.write_text("<!-- test -->\nJudge it.", encoding="utf-8")
    return raw, data, prompt


def test_build_grades_every_sample(tmp_path):
    raw, data, prompt = _fixture(tmp_path, n_samples=3)
    out = tmp_path / "tasks.jsonl"
    assert grade.build(raw, data, prompt, out) == 3 * (2 + 1)  # 3 samples x 3 criteria
    ids = [json.loads(l)["id"] for l in out.read_text().splitlines()]
    assert "1|m|0|c1" in ids and "1|m|2|c2" in ids and "2|m|1|c1" in ids
    row = json.loads(out.read_text().splitlines()[0])
    assert row["sample_index"] == 0 and "answer 1-0" in row["prompt"]


def test_build_dies_when_a_sample_is_missing(tmp_path):
    raw, data, prompt = _fixture(tmp_path, n_samples=2)
    lines = raw.read_text().splitlines()
    raw.write_text("\n".join(lines[:-1]) + "\n")  # drop task 2 sample 1
    with pytest.raises(SystemExit, match="have no answer"):
        grade.build(raw, data, prompt, tmp_path / "tasks.jsonl")


def test_score_one_row_per_sample_and_spread(tmp_path, capsys):
    raw, data, prompt = _fixture(tmp_path, n_samples=2)
    tasks = tmp_path / "tasks.jsonl"
    grade.build(raw, data, prompt, tasks)
    # judge says: task 1 sample 0 meets both, sample 1 meets one; task 2 both samples meet it
    verdict = {"1|m|0|c1": True, "1|m|0|c2": True, "1|m|1|c1": True, "1|m|1|c2": False,
               "2|m|0|c1": True, "2|m|1|c1": True}
    judged = tmp_path / "judge_raw.jsonl"
    _write(judged, [{"task_id": t["id"], "metadata": {k: t[k] for k in ("task_id", "model", "sample_index", "criterion_id")},
                     "response": json.dumps({"rationale": "r", "is_criteria_true": verdict[t["id"]]})}
                    for t in map(json.loads, tasks.read_text().splitlines())])
    grade.score(judged, tasks, baseline="m")
    scores = [json.loads(l) for l in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert [(s["task"], s["sample"], s["score"]) for s in scores] == [(1, 0, 1.0), (1, 1, 0.5), (2, 0, 1.0), (2, 1, 1.0)]
    text = capsys.readouterr().out
    assert "samples/task 2" in text and "mean score 0.875" in text and "mean spread over samples 0.250" in text


def _judged(tmp_path, verdict):
    """Build judge tasks on the 2-sample fixture and fake the judge's answers from `verdict` {id: bool|str}."""
    raw, data, prompt = _fixture(tmp_path, n_samples=2)
    tasks = tmp_path / "tasks.jsonl"
    grade.build(raw, data, prompt, tasks)
    judged = tmp_path / "judge_raw.jsonl"
    _write(judged, [{"task_id": t["id"], "prompt": t["prompt"],
                     "metadata": {k: t[k] for k in ("task_id", "model", "sample_index", "criterion_id")},
                     "response": (verdict[t["id"]] if isinstance(verdict[t["id"]], str)
                                  else json.dumps({"rationale": "r", "is_criteria_true": verdict[t["id"]]}))}
                    for t in map(json.loads, tasks.read_text().splitlines())])
    return judged, tasks


ALL_MET = {"1|m|0|c1": True, "1|m|0|c2": True, "1|m|1|c1": True, "1|m|1|c2": True, "2|m|0|c1": True, "2|m|1|c1": True}


@pytest.mark.parametrize("source", ["tasks", "judged"])
def test_label_writes_csv_and_q_keeps_what_was_done(tmp_path, source):
    judged, tasks = _judged(tmp_path, ALL_MET)
    judged = tasks if source == "tasks" else judged  # labelling works from build's output, before any judge ran
    out = tmp_path / "labels.csv"
    answers = iter(["maybe", "y", "n", "q"])  # a bad key is asked again
    shown = []
    n = grade.label(judged, 40, out, ask=lambda _: next(answers), say=shown.append)
    assert n == 2
    rows = list(csv.DictReader(out.open()))
    assert [r["label"] for r in rows] == ["y", "n"] and set(rows[0]) >= {"id", "task_id", "criterion_id"}
    text = "\n".join(shown)
    assert "CRITERION: says" in text and "RESPONSE" in text and "is_criteria_true" not in text  # verdict hidden
    # rerun skips the two already labelled
    n2 = grade.label(judged, 40, out, ask=lambda _: "q", say=shown.append)
    assert n2 == 0 and len(list(csv.DictReader(out.open()))) == 2


def test_agreement_by_hand():
    truth = dict(zip("abcdef", [True, True, False, False, True, False]))
    other = dict(zip("abcdef", [True, False, False, False, True, True]))
    a = grade.agreement(truth, other)
    assert a["n"] == 6 and a["accuracy"] == pytest.approx(4 / 6)
    assert a["kappa"] == pytest.approx((4 / 6 - 0.5) / 0.5)  # chance = .5*.5 + .5*.5
    assert a["precision_met"] == pytest.approx(2 / 3) and a["recall_met"] == pytest.approx(2 / 3)
    assert a["base_rate_met"] == 0.5 and [d["id"] for d in a["disagreements"]] == ["b", "f"]


def test_agreement_always_met_judge_has_kappa_zero():
    truth = {str(i): i < 9 for i in range(10)}          # 90% Met
    other = {str(i): True for i in range(10)}           # judge says Met on everything
    a = grade.agreement(truth, other)
    assert a["accuracy"] == pytest.approx(0.9) and a["kappa"] == pytest.approx(0.0)


def test_score_with_labels_prints_agreement(tmp_path, capsys):
    judged, tasks = _judged(tmp_path, ALL_MET)
    (tmp_path / "labels.csv").write_text("id,task_id,model,sample_index,criterion_id,label\n"
                                         "1|m|0|c1,1,m,0,c1,y\n1|m|0|c2,1,m,0,c2,n\n2|m|1|c1,2,m,1,c1,y\n")
    grade.score(judged, tasks, baseline="m", labels=tmp_path / "labels.csv")
    text = capsys.readouterr().out
    assert "judge vs you: n 3  accuracy 0.667" in text and "1|m|0|c2: you not, judge Met" in text
    assert (tmp_path / "judge_agreement.json").exists()


def test_agree_two_judge_runs(tmp_path, capsys):
    a, _ = _judged(tmp_path, ALL_MET)
    b_dir = tmp_path / "b"; b_dir.mkdir()
    b, _ = _judged(b_dir, {**ALL_MET, "2|m|0|c1": False, "1|m|1|c2": "not json at all"})
    grade.agree(a, b)
    text = capsys.readouterr().out
    assert "unparseable left out: A 0, B 1" in text and "n 5" in text and "disagreements: 1" in text
    assert (tmp_path / "agreement.json").exists()


def test_score_mid_run_reports_progress_and_partial_labels(tmp_path, capsys):
    judged, tasks = _judged(tmp_path, ALL_MET)
    lines = judged.read_text().splitlines()
    judged.write_text("\n".join(lines[:4]) + "\n")  # judge has finished 4 of 6 calls
    (tmp_path / "labels.csv").write_text("id,task_id,model,sample_index,criterion_id,label\n"
                                         "1|m|0|c1,1,m,0,c1,y\n2|m|1|c1,2,m,1,c1,n\n")  # second one not judged yet
    assert grade.score(judged, tasks, baseline="m", labels=tmp_path / "labels.csv") == 1
    text = capsys.readouterr().out
    assert "judge not finished: 4 of 6" in text and "without a judge verdict yet: 1 of 2" in text
    assert "judge vs you: n 1" in text and "metrics wait for the full run" in text
    assert not (tmp_path / "scores.jsonl").exists()


def test_error_kind():
    assert grade.error_kind({"response": "not json"}) == "unparsed"
    assert grade.error_kind({"response": None, "errors": [{"type": "APITimeoutError"}]}) == "timeout"
    assert grade.error_kind({"response": None, "errors": [{"type": "RateLimitError"}]}) == "api_error"


def test_score_errors_in_their_own_bucket(tmp_path, capsys):
    # task 1 sample 0: c1 met, c2 timed out -> score 0.5 as fail, 1.0 dropped; everything else met
    judged, tasks = _judged(tmp_path, {**ALL_MET, "1|m|0|c2": "garbage"})
    rows = [json.loads(l) for l in judged.read_text().splitlines()]
    for r in rows:
        if r["task_id"] == "1|m|0|c2":
            r["response"] = None
            r["errors"] = [{"type": "APITimeoutError"}]
    _write(judged, rows)
    grade.score(judged, tasks, baseline="m")
    scores = {(s["task"], s["sample"]): s for s in map(json.loads, (tmp_path / "scores.jsonl").read_text().splitlines())}
    assert scores[(1, 0)]["score"] == 0.5 and scores[(1, 0)]["score_errors_dropped"] == 1.0 and scores[(1, 0)]["errors"] == 1
    assert scores[(2, 0)]["errors"] == 0
    text = capsys.readouterr().out
    assert "judge errors: timeouts 1  api errors 0  unparsed 0" in text
    assert "mean score 0.875 (errors as fail)  1.000 (errors dropped)" in text
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["judge_errors"]["timeout"] == 1 and m["judge_errors"]["per_model"]["m"]["mean_errors_dropped"] == 1.0


def test_gold_mode_graded_apart(tmp_path, capsys):
    raw, data, prompt = _fixture(tmp_path, n_samples=1)
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"1": "reference one", "2": "reference two"}))
    tasks = tmp_path / "tasks.jsonl"
    assert grade.build(raw, data, prompt, tasks, gold=gold) == 3 + 3  # model rows + gold rows
    rows = [json.loads(l) for l in tasks.read_text().splitlines()]
    g = [r for r in rows if r["model"] == "gold"]
    assert len(g) == 3 and "reference one" in g[0]["prompt"] and g[0]["sample_index"] == 0
    # judge: gold fails one criterion, the model meets everything
    verdict = {r["id"]: not (r["model"] == "gold" and r["criterion_id"] == "c2") for r in rows}
    judged = tmp_path / "judge_raw.jsonl"
    _write(judged, [{"task_id": r["id"], "prompt": r["prompt"],
                     "metadata": {k: r[k] for k in ("task_id", "model", "sample_index", "criterion_id")},
                     "response": json.dumps({"rationale": "r", "is_criteria_true": verdict[r["id"]]})} for r in rows])
    grade.score(judged, tasks, baseline="m")
    text = capsys.readouterr().out
    assert "gold: tasks 2  criteria 3  mean 0.667  not met 1" in text and "1|gold|0|c2" in text
    scores = [json.loads(l) for l in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert {s["variant"] for s in scores} == {"m"}  # gold never enters the comparison
