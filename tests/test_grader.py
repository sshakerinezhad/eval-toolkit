import asyncio
import csv
import json
from types import SimpleNamespace as NS

import pytest

import grader
import llm
import traces
from tests.helpers import A, T, msg, write_run

GOOD = {
    "category": "IMPL_BUG", "secondary": None, "evidence": "T3: off-by-one", "verification": 3,
    "false_completion": True, "efficiency": 2, "recovery": 4, "wasted_turns_pct": 40,
    "notable": ["polled sleep"], "summary": "Right idea, wrong loop bound.",
}


def _ok(payload=GOOD, tin=100, tout=50):
    return llm.CallResult(json.dumps(payload), tin, tout, "stop")


def _fail(status, retryable=False, fatal=False):
    r = llm.CallResult(None, None, None, None, fatal=fatal)
    r.errors.append(llm.Attempt(1, status, "Err", None, f"status {status}", retryable))
    return r


@pytest.fixture
def scripted(monkeypatch):
    state = {"script": [], "calls": []}

    async def fake_call(client, provider, model, system_prompt, prompt, temperature, max_tokens, max_retries,
                        extra_body=None):
        state["calls"].append({"prompt": prompt, "extra_body": extra_body})
        item = state["script"].pop(0) if state["script"] else _ok()
        return item() if callable(item) else item

    monkeypatch.setattr(grader.llm, "call", fake_call)
    monkeypatch.setattr(grader.llm, "make_client", lambda provider, timeout: _FakeClient())

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(grader.llm, "smoke", fake_smoke)
    monkeypatch.setattr(grader, "tqdm", _NoBar)
    return state


class _FakeClient:
    async def close(self): pass


class _NoBar:
    def __init__(self, *a, **k): pass
    def update(self, n=1): pass
    def set_postfix_str(self, s): pass
    def close(self): pass


@pytest.fixture
def two_variants(tmp_path):
    """foo: trained passes 2/3, untrained 1/3. bar: both fail 3/3. Returns (trained_dir, untrained_dir)."""
    t = tmp_path / "M_trained_traces"
    u = tmp_path / "M_untrained_traces"
    for d, tag, foo_scores in ((t, "t", [1.0, 1.0, 0.0]), (u, "u", [1.0, 0.0, 0.0])):
        d.mkdir()  # tid carries the variant tag so trained/untrained transcripts differ (prompt is blind)
        for i, s in enumerate(foo_scores):
            write_run(d, task="terminal_bench_foo", tid=f"{tag}f{i}", score=s)
        for i in range(3):
            write_run(d, task="terminal_bench_bar", tid=f"{tag}b{i}", score=0.0, tail="E AssertionError")
    return t, u


def _cfg(tmp_path, **over):
    kw = dict(model="anthropic:claude-opus-5-5", workers=2, max_retries=0, timeout=5, effort="high")
    kw.update(over)
    return grader.GradeConfig(**kw)


def _items(dirs, limit=None):
    items = grader.build_items([traces.load_variant(d) for d in dirs])
    return items[:limit] if limit else items


# ---------- parse_judgment ----------

def test_parse_judgment_strips_fences_and_validates():
    txt = "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nDone."
    assert grader.parse_judgment(txt)["category"] == "IMPL_BUG"


@pytest.mark.parametrize("bad", [
    {**GOOD, "category": "MADE_UP"},
    {**GOOD, "verification": 6},
    {**GOOD, "efficiency": 0},
    {k: v for k, v in GOOD.items() if k != "summary"},
    {**GOOD, "false_completion": "yes"},
])
def test_parse_judgment_rejects_invalid(bad):
    with pytest.raises(ValueError):
        grader.parse_judgment(json.dumps(bad))


def test_parse_judgment_rejects_non_json():
    with pytest.raises(ValueError):
        grader.parse_judgment("no json here")


def test_parse_judgment_wasted_pct_optional_and_secondary_validated():
    ok = {k: v for k, v in GOOD.items() if k != "wasted_turns_pct"}
    assert grader.parse_judgment(json.dumps(ok))["wasted_turns_pct"] is None
    with pytest.raises(ValueError):
        grader.parse_judgment(json.dumps({**GOOD, "secondary": "NOPE"}))


# ---------- build_prompt ----------

def test_build_prompt_has_metadata_task_prelabel_and_transcript(tmp_path):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d, score=0.0, duration=10800, msgs=[msg(A, traces.DEAD), msg(T, "p")] * 3, task_text="Build a widget.")
    item = _items([d])[0]
    p = grader.build_prompt(item)
    assert "Build a widget." in p and "pre_label: CONTEXT_OVERFLOW_LOOP" in p
    assert "termination: timeout_in_context_overflow_loop" in p and "# Transcript" in p
    assert "duration_s: 10800" in p


def test_build_items_pre_label_uses_per_task_modal_tests(tmp_path):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d, tid="a", tests_total=9)
    write_run(d, tid="b", tests_total=9)
    write_run(d, tid="c", tests_total=1)
    labels = {it.run.traj_id: it.pre_label for it in _items([d])}
    assert labels == {"traj_a": None, "traj_b": None, "traj_c": "INFRA_ERROR"}


# ---------- judge_run ----------

def _judge(item, cfg, scripted):
    client = NS()
    killed = asyncio.Event()
    return asyncio.run(grader.judge_run(client, asyncio.Semaphore(1), item, cfg, killed))


def test_judge_run_calls_with_effort_and_caches(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d)
    item = _items([d])[0]
    cfg = _cfg(tmp_path)
    rec = _judge(item, cfg, scripted)
    assert rec["judge"]["category"] == "IMPL_BUG" and rec["cached"] is False
    assert scripted["calls"][0]["extra_body"] == {"reasoning_effort": "high"}
    rec2 = _judge(item, cfg, scripted)
    assert rec2["cached"] is True and len(scripted["calls"]) == 1


def test_judge_run_retries_once_on_bad_json_with_repair_suffix(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d)
    item = _items([d])[0]
    scripted["script"] = [llm.CallResult("not json", 1, 1, "stop"), _ok()]
    rec = _judge(item, _cfg(tmp_path), scripted)
    assert rec["judge"]["category"] == "IMPL_BUG" and rec["parse_retries"] == 1
    assert scripted["calls"][1]["prompt"].endswith(grader.REPAIR_SUFFIX)


def test_judge_run_gives_up_after_second_bad_json(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d)
    item = _items([d])[0]
    scripted["script"] = [llm.CallResult("x", 1, 1, "stop"), llm.CallResult("y", 1, 1, "stop")]
    rec = _judge(item, _cfg(tmp_path), scripted)
    assert rec["judge"] is None and "parse" in rec["error"]


def test_judge_run_records_api_failure(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d)
    item = _items([d])[0]
    scripted["script"] = [_fail(500, retryable=True)]
    rec = _judge(item, _cfg(tmp_path), scripted)
    assert rec["judge"] is None and "500" in rec["error"]


# ---------- execute ----------

def test_execute_kills_on_fatal_and_skips_rest(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    for i in range(6):
        write_run(d, tid=f"t{i}")
    items = _items([d])
    scripted["script"] = [_fail(401, fatal=True)]
    out = tmp_path / "j.jsonl"
    s = asyncio.run(grader.execute(items, _cfg(tmp_path, workers=1), out))
    assert s.killed and "401" in s.kill_reason
    assert s.ok + s.failed < len(items)


def test_execute_kills_after_consecutive_non_retryable(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    for i in range(8):
        write_run(d, tid=f"t{i}")
    items = _items([d])
    scripted["script"] = [_fail(400)] * 8
    s = asyncio.run(grader.execute(items, _cfg(tmp_path, workers=1), tmp_path / "j.jsonl"))
    assert s.killed and s.failed == grader.KILL_AFTER_CONSECUTIVE


def test_execute_appends_one_line_per_item(tmp_path, scripted):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    for i in range(3):
        write_run(d, tid=f"t{i}")
    out = tmp_path / "j.jsonl"
    asyncio.run(grader.execute(_items([d]), _cfg(tmp_path), out))
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and all(x["judge"]["category"] == "IMPL_BUG" for x in lines)


# ---------- reports ----------

def _records(two_variants, cat_by_variant=None):
    recs = []
    for it in _items(two_variants):
        j = None if it.pre_label in ("PASS",) and False else dict(GOOD)
        if it.pre_label == "PASS":
            j = {**GOOD, "category": "PASS", "false_completion": False, "verification": 5}
        elif cat_by_variant:
            j = {**GOOD, "category": cat_by_variant[it.run.variant]}
        recs.append(grader.make_record(it, judge=j, error=None, cached=True, tokens_in=1, tokens_out=1,
                                       parse_retries=0))
    return recs


def test_write_reports_summary_has_taxonomy_and_paired_table(tmp_path, two_variants):
    recs = _records(two_variants, {"M_trained": "IMPL_BUG", "M_untrained": "UNVERIFIED_CLAIM"})
    out = tmp_path / "out"
    grader.write_reports(recs, out, model="m")
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| IMPL_BUG |" in summary and "| UNVERIFIED_CLAIM |" in summary
    assert "M_trained" in summary and "M_untrained" in summary
    # paired table: foo trained [110] vs untrained [100] -> delta +0.33
    assert "[110]" in summary and "[100]" in summary and "+0.33" in summary
    assert "[000]" in summary


def test_write_reports_task_md_stacks_both_variants(tmp_path, two_variants):
    recs = _records(two_variants)
    out = tmp_path / "out"
    grader.write_reports(recs, out, model="m")
    md = (out / "tasks" / "foo.md").read_text(encoding="utf-8")
    assert md.index("## M_trained") < md.index("## M_untrained")
    assert "Right idea, wrong loop bound." in md and "polled sleep" in md
    assert (out / "tasks" / "bar.md").exists()


def test_write_reports_scores_csv_one_row_per_run(tmp_path, two_variants):
    recs = _records(two_variants)
    out = tmp_path / "out"
    grader.write_reports(recs, out, model="m")
    rows = list(csv.DictReader(open(out / "scores.csv", encoding="utf-8")))
    assert len(rows) == 12
    assert {"variant", "task", "traj_id", "pre_label", "judge_category", "judge_verification", "termination"} <= set(rows[0])


def test_write_reports_tolerates_unjudged_records(tmp_path, two_variants):
    recs = _records(two_variants)
    recs[0]["judge"], recs[0]["error"] = None, "boom"
    out = tmp_path / "out"
    grader.write_reports(recs, out, model="m")
    assert "UNJUDGED" in (out / "summary.md").read_text(encoding="utf-8")


# ---------- main ----------

def test_main_dry_run_makes_no_calls_and_writes_prompts(tmp_path, two_variants, scripted, monkeypatch):
    monkeypatch.setattr(grader, "RESULTS_DIR", tmp_path / "results")
    rc = grader.main([str(two_variants[0]), "--name", "x", "--dry-run", "--limit", "2"])
    assert rc == 0 and scripted["calls"] == []
    prompts = list((tmp_path / "results" / "grade_x" / "prompts").glob("*.txt"))
    assert len(prompts) == 2


def test_main_end_to_end_then_report_only(tmp_path, two_variants, scripted, monkeypatch):
    monkeypatch.setattr(grader, "RESULTS_DIR", tmp_path / "results")
    t, u = two_variants
    rc = grader.main([str(t), str(u), "--name", "x", "--yes"])
    assert rc == 0 and len(scripted["calls"]) == 12
    out = tmp_path / "results" / "grade_x"
    assert (out / "summary.md").exists() and (out / "tasks" / "foo.md").exists()
    n_calls = len(scripted["calls"])
    rc = grader.main([str(t), str(u), "--name", "x", "--report-only"])
    assert rc == 0 and len(scripted["calls"]) == n_calls


def test_main_rerun_same_name_uses_cache_and_dedupes(tmp_path, two_variants, scripted, monkeypatch):
    monkeypatch.setattr(grader, "RESULTS_DIR", tmp_path / "results")
    t, u = two_variants
    grader.main([str(t), str(u), "--name", "x", "--yes"])
    grader.main([str(t), str(u), "--name", "x", "--yes"])
    assert len(scripted["calls"]) == 12
    out = tmp_path / "results" / "grade_x"
    assert len((out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()) == 24
    rows = list(csv.DictReader(open(out / "scores.csv", encoding="utf-8")))
    assert len(rows) == 12


def test_main_task_filter(tmp_path, two_variants, scripted, monkeypatch):
    monkeypatch.setattr(grader, "RESULTS_DIR", tmp_path / "results")
    rc = grader.main([str(two_variants[0]), "--name", "x", "--yes", "--task", "bar"])
    assert rc == 0 and len(scripted["calls"]) == 3


def test_build_prompt_is_blind_to_variant(tmp_path):
    d = tmp_path / "X_trained_traces"
    d.mkdir()
    write_run(d)
    assert "X_trained" not in grader.build_prompt(_items([d])[0])


# ---------- selection flags ----------

def test_select_items_fails_only_and_size_cutoff(tmp_path, two_variants):
    items = _items(two_variants)
    todo, skipped = grader.select_items(items, fails_only=True, max_prompt_tokens=None)
    assert all(it.pre_label is None for it in todo) and len(todo) + len(skipped) == 12
    assert all(r["judge"] is None and r["error_kind"] == "skipped" for r in skipped)
    todo2, skipped2 = grader.select_items(items, fails_only=False, max_prompt_tokens=1)
    assert todo2 == [] and len(skipped2) == 12 and "max-prompt-tokens" in skipped2[0]["error"]


def test_main_fails_only_skips_passes_but_reports_them(tmp_path, two_variants, scripted, monkeypatch):
    monkeypatch.setattr(grader, "RESULTS_DIR", tmp_path / "results")
    t, u = two_variants
    rc = grader.main([str(t), str(u), "--name", "x", "--yes", "--fails-only"])
    assert rc == 0 and len(scripted["calls"]) == 9  # 12 runs, 3 passes skipped
    out = tmp_path / "results" / "grade_x"
    rows = list(csv.DictReader(open(out / "scores.csv", encoding="utf-8")))
    assert len(rows) == 12
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "[110]" in summary  # pass rate context kept even though passes were not judged
