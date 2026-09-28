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
