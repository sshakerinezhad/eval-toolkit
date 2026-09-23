import json

import pytest

import report
import traces
from tests.helpers import A, T, msg, write_run
from traces import CONFIRM, DEAD

CONFIRMED = [msg(A, "a"), msg(T, "out"), msg(A, "done"), msg(T, CONFIRM + "?"), msg(A, "yes")]
STUCK = "Analysis: The terminal is stuck. Plan: send Ctrl+C"


def one(tmp_path, **kw):
    root = tmp_path / "M_trained_traces"
    root.mkdir(exist_ok=True)
    write_run(root, **kw)
    r = traces.load_variant(root)[0]
    return r, traces.features(r)


# ---------- tail classifier ----------

@pytest.mark.parametrize("tail,label", [
    (None, "done_no_test_output"),
    ("   ", "done_no_test_output"),
    ("some log without a summary", "done_unclassified"),
    ("FAILED ../tests/test_outputs.py::test_gcov_enabled - FileNotFoundError: [Errn...", "done_missing_output"),
    ("FAILED ../tests/test_outputs.py::test_hello_html_exists - AssertionError: Did...", "done_missing_output"),
    ("FAILED ../tests/t.py::test_gpt2 - subprocess.Timeou...", "done_too_slow"),
    ("FAILED ../tests/t.py::test_compare_golden_vs_solution_runtime - as...", "done_too_slow"),
    ("FAILED ../tests/t.py::test_ccomplexity - AttributeError: module 'n...", "done_crash"),
    ("FAILED ../tests/t.py::test_count - Ass...", "done_wrong_value"),
    ("FAILED ../tests/t.py::test_x\nFAILED ../tests/t.py::test_y - AssertionError", "done_wrong_value"),
    # AssertionError must not read as a crash; upstream failure wins across several tests
    ("FAILED ../t.py::test_a - AssertionError: x\nFAILED ../t.py::test_b - RuntimeError: y", "done_crash"),
])
def test_classify_tail(tail, label):
    assert report.classify_tail(tail) == label


# ---------- ending precedence ----------

def test_pass_wins_even_with_lockup(tmp_path):
    r, f = one(tmp_path, score=1.0, msgs=[msg(A, STUCK), msg(T, "x")] * 25)
    assert report.ending(r, f, None) == "pass"


def test_infra_error_beats_stall_and_cap(tmp_path):
    r, f = one(tmp_path, exception="ConflictError", duration=20000, n_calls=3)
    assert report.ending(r, f, None) == "infra_error"


def test_verifier_short_when_below_task_mode(tmp_path):
    r, f = one(tmp_path, tests_total=1, msgs=CONFIRMED)
    assert report.ending(r, f, 6) == "verifier_short"
    assert report.ending(r, f, 1) != "verifier_short"


def test_stall(tmp_path):
    r, f = one(tmp_path, duration=11000, n_calls=5)
    assert report.ending(r, f, None) == "stall_suspected"


def test_lockup_beats_overflow_loop(tmp_path):
    # >= STALL_CALLS calls, else the stall rule (checked first) claims it; real lockups had 64+ calls
    msgs = [msg(A, STUCK), msg(T, "x")] * 65 + [msg(A, DEAD + ". Please continue."), msg(T, "err")] * 3
    r, f = one(tmp_path, duration=11000, msgs=msgs)
    assert f["termination"] == "timeout_in_context_overflow_loop"
    assert report.ending(r, f, None) == "lockup"


def test_overflow_loop_without_lockup(tmp_path):
    msgs = [msg(A, "a"), msg(T, "x")] * 70 + [msg(A, DEAD + "."), msg(T, "err")] * 3
    r, f = one(tmp_path, duration=11000, msgs=msgs)
    assert report.ending(r, f, None) == "overflow_loop"


def test_confirmed_failure_uses_test_tail(tmp_path):
    r, f = one(tmp_path, msgs=CONFIRMED, tail="FAILED ../t.py::test_v - assert 1 == 2")
    assert report.ending(r, f, None) == "done_wrong_value"


def test_cap_working_and_cut_other(tmp_path):
    r, f = one(tmp_path, duration=11000, msgs=[msg(A, "a"), msg(T, "x")] * 70)
    assert report.ending(r, f, None) == "cap_working"
    other = tmp_path / "other"
    other.mkdir()
    r2, f2 = one(other, duration=500)
    assert report.ending(r2, f2, None) == "cut_other"


# ---------- traces features added for the report ----------

def test_stuck_streak_is_consecutive_and_skips_dead_calls(tmp_path):
    msgs = [msg(A, STUCK), msg(T, "x"), msg(A, STUCK), msg(T, "x"), msg(A, DEAD + "."), msg(T, "e"),
            msg(A, STUCK), msg(T, "x"), msg(A, "Analysis: fine"), msg(T, "x"), msg(A, STUCK), msg(T, "x")]
    _, f = one(tmp_path, msgs=msgs)
    assert f["max_stuck_streak"] == 3
    assert f["live_calls"] == 5 and f["n_dead_calls"] == 1


def test_think_tokens_default_zero(tmp_path):
    r, _ = one(tmp_path)
    assert r.think_tokens == 0


# ---------- stats ----------

def test_sign_test_p():
    assert report.sign_test_p(0, 0) == 1.0
    assert report.sign_test_p(5, 0) == pytest.approx(0.0625)
    assert report.sign_test_p(16, 7) == report.sign_test_p(7, 16)
    assert report.sign_test_p(16, 7) == pytest.approx(0.0931, abs=1e-4)  # matches scipy binomtest


def _row(task, variant, passed):
    return {"task": task, "variant": variant, "passed": passed}


def test_paired_passk_drops_tasks_short_of_k():
    rows = [_row("a", v, p) for v, p in [("trained", True)] * 3 + [("untrained", False)] * 3]
    rows += [_row("b", "trained", True), _row("b", "untrained", True)]  # only 1 run each
    out = report.paired_passk(rows, 3, B=50, seed=0)
    assert out["n_tasks"] == 1 and out["n_tasks_dropped"] == 1
    assert out["diff"]["mean"] == 1.0


def test_geo_mean_ratio_symmetric():
    assert report.geo_mean_ratio([(2, 1), (1, 2)]) == 1.0
    assert report.geo_mean_ratio([(0, 1)]) is None


# ---------- end to end ----------

@pytest.fixture
def traces_dir(tmp_path):
    for v, scores in (("trained", (1.0, 1.0, 0.0)), ("untrained", (1.0, 0.0, 0.0))):
        root = tmp_path / f"M_{v}_traces"
        root.mkdir()
        for task in ("terminal_bench_alpha", "terminal_bench_beta"):
            for i, s in enumerate(scores):
                write_run(root, task=task, tid=f"{v[0]}{task[-1]}{i}", score=s, msgs=CONFIRMED,
                          tail=None if s else "FAILED ../t.py::test_v - AssertionError: 1 != 2")
    return tmp_path


def test_main_writes_three_outputs(traces_dir, tmp_path):
    out = tmp_path / "out"
    assert report.main(["--model", "M", "--traces", str(traces_dir), "--out", str(out), "--boot", "50"]) == 0
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert {r["ending"] for r in rows} == {"pass", "done_wrong_value"}
    assert sorted({r["trial"] for r in rows}) == [0, 1, 2]
    assert not any(r["path"].startswith(("/", "C:")) for r in rows)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert {"meta", "q1_improvement", "q2_behavior", "q3_failures", "q4_trust"} <= set(m)
    assert m["q1_improvement"]["pass_at_1"]["diff"]["mean"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["q2_behavior"]["per_variant"]["untrained"]["false_done_runs"] == 4
    assert m["q3_failures"]["diff_trained_minus_untrained"]["done_wrong_value"] == -2
    assert all(not e["found"] for e in m["q2_behavior"]["examples"])  # curated runs absent from fixture
    assert (out / "endings.png").stat().st_size > 10_000
