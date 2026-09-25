import json

import pytest

import judge


# ---------- helpers ----------

PROMPT = "Classify the failure.\n\nCategories:\n- env_fail: the sandbox broke\n- bug: the code had a bug\n"
CATS = ["env_fail", "bug"]


def _trajectory(tid, task_name, score, statuses):
    passed = sum(v == "pass" for v in statuses.values())
    return {"trajectory": {
        "trajectory_id": tid, "task_name": task_name, "trajectory_status": "completed",
        "initial_messages": [{"role": "system", "content": "sys"},
                             {"role": "user", "content": f"TASK TEXT for {task_name}"}],
        "trajectory_messages": [
            {"role": "user", "content": "HARNESS PREAMBLE"},
            {"role": "assistant", "content": "step one"},
            {"role": "tool", "content": None},
            {"role": "assistant", "content": "I am done"},
        ],
        "trajectory_output": {"score": score, "outcome": "completed", "test_statuses": statuses,
                              "tests_passed": passed, "tests_total": len(statuses), "error_message": None},
    }}


def _layout(tmp_path, variant="modelA_traces"):
    """Real tbench shape: index + tasks/<task_name>__<task_id>/<trajectory_id>.json, one pass one fail."""
    d = tmp_path / variant
    entries = [
        ("traj_pass", "tb_alpha", "task_1", 1.0, {"test_x": "pass"}),
        ("traj_fail", "tb_beta", "task_2", 0.0, {"test_x": "pass", "test_y": "fail"}),
    ]
    index = []
    for tid, name, task_id, score, statuses in entries:
        index.append({"trajectory_id": tid, "task_name": name, "task_id": task_id, "final_score": score})
        f = d / "tasks" / f"{name}__{task_id}" / f"{tid}.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(_trajectory(tid, name, score, statuses)), encoding="utf-8")
    (d / "trajectories_index.json").write_text(json.dumps(index), encoding="utf-8")
    (d / "README.md").write_text("not for the loader", encoding="utf-8")
    return d


def _prompt(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text(PROMPT, encoding="utf-8")
    return p


def _build(tmp_path, *extra):
    d, out = _layout(tmp_path), tmp_path / "judge_tasks.jsonl"
    assert judge.main(["build", str(d), "--prompt", str(_prompt(tmp_path)), "--out", str(out), *extra]) == 0
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


# ---------- build ----------

def test_build_fails_default(tmp_path):
    rows = _build(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert (r["id"], r["task"], r["variant"], r["trajectory_id"]) == ("traj_fail", "tb_beta", "modelA_traces", "traj_fail")
    assert r["categories"] == CATS
    assert r["prompt"].startswith(PROMPT.strip())
    p = r["prompt"]
    assert p.index("TASK TEXT for tb_beta") < p.index("I am done") < p.index("FAIL test_y")  # task, turns, tests
    assert "HARNESS PREAMBLE" not in p
    assert "tests: 1/2 passed" in p


@pytest.mark.parametrize("select,want", [("all", ["traj_pass", "traj_fail"]), ("passes", ["traj_pass"])])
def test_build_select(tmp_path, select, want):
    assert [r["id"] for r in _build(tmp_path, "--select", select)] == want


def test_build_rows_load_in_run_py(tmp_path):
    """run.py must accept the file as-is; extra fields become metadata."""
    import run
    _build(tmp_path)
    tasks = run.load_tasks(tmp_path / "judge_tasks.jsonl")
    assert tasks[0]["metadata"] == {"task": "tb_beta", "variant": "modelA_traces",
                                    "trajectory_id": "traj_fail", "categories": CATS}


def test_duplicate_folder_dies(tmp_path):
    d = _layout(tmp_path)
    with pytest.raises(SystemExit, match="duplicate trajectory"):
        judge.build([d, d], _prompt(tmp_path), tmp_path / "out.jsonl")


def test_prompt_without_categories_dies(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("no categories here", encoding="utf-8")
    with pytest.raises(SystemExit, match="no category lines"):
        judge.build([_layout(tmp_path)], p, tmp_path / "out.jsonl")


def test_crash_without_score_counts_as_fail(tmp_path):
    d = _layout(tmp_path)
    f = d / "tasks" / "tb_alpha__task_1" / "traj_pass.json"
    t = json.loads(f.read_text(encoding="utf-8"))
    t["trajectory"]["trajectory_output"]["score"] = None
    f.write_text(json.dumps(t), encoding="utf-8")
    index = json.loads((d / "trajectories_index.json").read_text(encoding="utf-8"))
    index[0]["final_score"] = None
    (d / "trajectories_index.json").write_text(json.dumps(index), encoding="utf-8")
    judge.build([d], _prompt(tmp_path), tmp_path / "out.jsonl")
    ids = [json.loads(line)["id"] for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ids == ["traj_pass", "traj_fail"]


# ---------- excerpt ----------

def _record(n_turns=10, size=100):
    return {"instruction": "THE TASK", "test_output": "FAIL test_b",
            "turns": [{"role": "assistant", "content": f"t{i + 1}:" + "x" * size} for i in range(n_turns)]}


def test_excerpt_last_n_turns():
    e = judge.excerpt(_record(), 3, 60000)
    assert "### [8]" in e and "### [10]" in e and "### [7]" not in e
    assert "Last 3 of 10 turns" in e and "cut" not in e


def test_excerpt_cap_keeps_task_tests_and_newest():
    e = judge.excerpt(_record(), 10, 500)
    assert len(e) <= 500
    assert "THE TASK" in e and "FAIL test_b" in e and "### [10]" in e and "### [1]" not in e
    assert "earlier turn(s) cut" in e


def test_excerpt_huge_newest_turn_keeps_its_tail():
    rec = _record(n_turns=1, size=5000)
    rec["turns"][0]["content"] += "THE_END"
    e = judge.excerpt(rec, 30, 1000)
    assert len(e) <= 1000 and "THE_END" in e and "FAIL test_b" in e


@pytest.mark.parametrize("cap", [1, 20, 60])
def test_excerpt_tiny_cap_never_exceeded(cap):
    assert len(judge.excerpt(_record(), 30, cap)) <= cap


# ---------- parse_response ----------

@pytest.mark.parametrize("text,want", [
    ('{"category": "bug", "why": "off by one"}', ("bug", "off by one")),
    ('```json\n{"category": "env_fail", "why": "no pip"}\n```', ("env_fail", "no pip")),
    ('{"category": "cosmic_rays", "why": "?"}', ("unparsed", '{"category": "cosmic_rays", "why": "?"}')),
    ("I think it is a bug", ("unparsed", "I think it is a bug")),
    ('["bug"]', ("unparsed", '["bug"]')),
    (None, ("unparsed", "(no reply)")),
])
def test_parse_response(text, want):
    assert judge.parse_response(text, CATS) == want


# ---------- score ----------

def _raw_row(tid, variant, response):
    return {"task_id": tid, "provider": "anthropic", "model": "judge-1", "response": response,
            "metadata": {"task": "t", "variant": variant, "trajectory_id": tid, "categories": CATS}}


def _score_fixture(tmp_path):
    rows = [
        _raw_row("t1", "A", '{"category": "bug", "why": "w1"}'),
        _raw_row("t2", "A", '{"category": "env_fail", "why": "w2"}'),
        _raw_row("t3", "B", '{"category": "bug", "why": "w3"}'),
        _raw_row("t4", "B", "sorry, no json"),
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("trajectory_id,category\nt1,bug\nt2,env_fail\nt3,env_fail\n", encoding="utf-8")
    return raw, labels


def test_score_counts_and_agreement(tmp_path):
    raw, labels = _score_fixture(tmp_path)
    assert judge.main(["score", str(raw), "--labels", str(labels)]) == 0
    doc = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    m = doc["models"]["anthropic:judge-1"]
    a, b = m["variants"]["A"], m["variants"]["B"]
    assert a["counts"] == {"env_fail": 1, "bug": 1, "unparsed": 0}
    assert b["counts"] == {"env_fail": 0, "bug": 1, "unparsed": 1}
    assert a["share"]["bug"] == pytest.approx(0.5) and b["share"]["unparsed"] == pytest.approx(0.5)
    ag = m["agreement"]
    assert (ag["matches"], ag["labelled"]) == (2, 3)
    assert ag["value"] == pytest.approx(2 / 3)
    assert ag["disagreements"] == [{"trajectory_id": "t3", "mine": "env_fail", "judge": "bug", "why": "w3"}]


def test_score_without_labels(tmp_path):
    raw, _ = _score_fixture(tmp_path)
    doc = judge.score(raw)
    assert doc["models"]["anthropic:judge-1"]["agreement"] is None


def test_score_unknown_label_dies(tmp_path):
    raw, labels = _score_fixture(tmp_path)
    labels.write_text("trajectory_id,category\nt1,bugg\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown category"):
        judge.score(raw, labels)


def test_score_duplicate_reply_dies(tmp_path):
    raw, _ = _score_fixture(tmp_path)
    raw.write_text(raw.read_text(encoding="utf-8") + json.dumps(_raw_row("t1", "A", "{}")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        judge.score(raw)

def test_load_prompt_strips_provenance_header(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("<!-- source: mine, 2026-09-25 -->\nThe prompt.", encoding="utf-8")
    assert judge.load_prompt(p) == "The prompt."
    p.write_text("<!-- source: mine -->\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        judge.load_prompt(p)