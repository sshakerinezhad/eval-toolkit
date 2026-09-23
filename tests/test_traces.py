import json

import pytest

import traces
from tests.helpers import A, T, msg, write_run
from traces import CONFIRM, DEAD

@pytest.fixture
def root(tmp_path):
    r = tmp_path / "Qwen_trained_traces"
    r.mkdir()
    return r


def one(root, **kw):
    write_run(root, **kw)
    return traces.load_variant(root)[0]


# ---------- load_variant ----------

def test_load_variant_strips_prefixes_and_reads_task_text(root):
    r = one(root, score=1.0)
    assert r.variant == "Qwen_trained"
    assert r.task == "foo" and r.category == "Sci"
    assert r.task_text == "Do the thing."
    assert r.passed is True and r.n_calls == 1 and r.tests_total == 2


def test_load_variant_sorted_by_task_then_traj(root):
    write_run(root, task="terminal_bench_zeta", tid="b")
    write_run(root, task="terminal_bench_alpha", tid="c")
    write_run(root, task="terminal_bench_alpha", tid="a")
    assert [(r.task, r.traj_id) for r in traces.load_variant(root)] == \
        [("alpha", "traj_a"), ("alpha", "traj_c"), ("zeta", "traj_b")]


def test_load_variant_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        traces.load_variant(tmp_path / "nope")


def test_load_variant_null_assistant_content_tolerated(root):
    r = one(root, msgs=[msg(A, None), msg(T, "x")])
    f = traces.features(r)
    assert f["n_dead_calls"] == 0 and f["termination"] == "other_cut"


# ---------- features / termination ----------

def test_termination_error_wins_over_everything(root):
    r = one(root, exception="VerifierTimeoutError", duration=20000)
    assert traces.features(r)["termination"] == "error:VerifierTimeoutError"


def test_termination_completed_confirmed(root):
    r = one(root, msgs=[msg(A, "a"), msg(T, "out"), msg(A, "done"), msg(T, CONFIRM + "?"), msg(A, "yes")])
    f = traces.features(r)
    assert f["termination"] == "completed_confirmed"
    assert f["n_confirm_prompts"] == 1 and f["backed_off"] == 0


def test_termination_confirmed_then_trailing_tool_still_confirmed(root):
    r = one(root, msgs=[msg(A, "done"), msg(T, CONFIRM), msg(A, "yes"), msg(T, "final out")])
    assert traces.features(r)["termination"] == "completed_confirmed"


def test_backed_off_counts_confirm_prompts_not_followed_by_end(root):
    r = one(root, msgs=[msg(A, "done"), msg(T, CONFIRM), msg(A, "wait no"), msg(T, "out"),
                        msg(A, "fix"), msg(T, "out2")])
    f = traces.features(r)
    assert f["n_confirm_prompts"] == 1 and f["backed_off"] == 1 and f["termination"] == "other_cut"


def test_termination_cut_at_confirm_prompt(root):
    r = one(root, msgs=[msg(A, "done"), msg(T, CONFIRM)])
    assert traces.features(r)["termination"] == "cut_at_confirm_prompt"


def test_termination_overflow_loop_needs_cap_and_recent_dead_call(root):
    dead = [msg(A, DEAD + ". Please continue with the task."), msg(T, "Previous response had parsing errors")]
    r = one(root, msgs=[msg(A, "a"), msg(T, "o")] + dead * 5, duration=10800)
    f = traces.features(r)
    assert f["termination"] == "timeout_in_context_overflow_loop"
    assert f["n_dead_calls"] == 5 and f["max_dead_streak"] == 5 and f["n_parse_err"] == 5


def test_termination_timeout_3h_when_still_working(root):
    r = one(root, msgs=[msg(A, "a"), msg(T, "o")] * 3, duration=10900)
    assert traces.features(r)["termination"] == "timeout_3h"


def test_dead_streak_is_longest_consecutive_not_total(root):
    d = msg(A, DEAD)
    r = one(root, msgs=[d, msg(T, "x"), d, msg(T, "x"), msg(A, "ok"), msg(T, "x"), d, msg(T, "x")])
    f = traces.features(r)
    assert f["n_dead_calls"] == 3 and f["max_dead_streak"] == 2


def test_command_features(root):
    ch = "pytest tests/\nsleep 30\nsleep 30\nsleep 30\nls\n"
    r = one(root, cmd_history=ch, msgs=[msg(A, "I am unable to complete this"), msg(T, "x")])
    f = traces.features(r)
    assert f["ran_tests"] == 1 and f["n_sleep_cmds"] == 3 and f["sleep_sum"] == 90.0
    assert f["repeated_cmd_lines"] == 2 and f["giveup_final"] == 1


def test_stalled_flag(root):
    r = one(root, duration=10500, n_calls=10)
    assert traces.features(r)["stalled"] is True
    r2 = one(root, duration=10500, n_calls=100)
    assert traces.features(r2)["stalled"] is False


# ---------- pre_label ----------

def test_pre_label_precedence(root):
    f = lambda r: traces.pre_label(r, traces.features(r), modal_tests=2)  # noqa: E731
    assert f(one(root, exception="X", score=1.0)) == "INFRA_ERROR"
    assert f(one(root, tests_total=1)) == "INFRA_ERROR"
    assert f(one(root, score=1.0)) == "PASS"
    dead = [msg(A, DEAD), msg(T, "p")]
    assert f(one(root, msgs=dead * 4, duration=10800)) == "CONTEXT_OVERFLOW_LOOP"
    assert f(one(root, duration=10500, n_calls=5)) == "SERVING_STALL"
    assert f(one(root, duration=500)) is None


def test_modal_tests_total_ignores_none_and_picks_mode(root):
    write_run(root, tid="a", tests_total=9)
    write_run(root, tid="b", tests_total=9)
    write_run(root, tid="c", tests_total=1)
    write_run(root, task="terminal_bench_bar", tid="d", tests_total=None)
    m = traces.modal_tests_total(traces.load_variant(root))
    assert m == {"foo": 9}


# ---------- render_transcript ----------

def test_render_transcript_numbers_turns_and_appends_tests(root):
    r = one(root, msgs=[msg(A, "Analysis: a1"), msg(T, "out1"), msg(A, "Analysis: a2"), msg(T, "out2")],
            tail="E  AssertionError: nope")
    txt = traces.render_transcript(r)
    assert "[T1 assistant]\nAnalysis: a1" in txt and "[T1 terminal]\nout1" in txt
    assert "[T2 assistant]" in txt and "[T2 terminal]" in txt
    assert "harness prompt" not in txt  # first user message is dropped; task text comes from initial_messages
    assert "FAIL test_outputs.py::t_b" in txt and "PASS test_outputs.py::t_a" in txt
    assert "AssertionError: nope" in txt


def test_render_transcript_collapses_dead_streaks(root):
    dead = [msg(A, DEAD + ". Please continue with the task."), msg(T, "Previous response had parsing errors")]
    r = one(root, msgs=[msg(A, "a"), msg(T, "o")] + dead * 40 + [msg(A, "back"), msg(T, "z")])
    txt = traces.render_transcript(r)
    assert txt.count("Technical difficulties") == 0
    assert "[T2-T41: 40 consecutive empty responses (context overflow), no commands ran]" in txt
    assert "[T42 assistant]\nback" in txt


def test_render_transcript_mid_run_user_message_is_labelled_harness(root):
    r = one(root, msgs=[msg(A, "a"), msg(T, "o"), msg("user", "injected note"), msg(A, "b"), msg(T, "p")])
    txt = traces.render_transcript(r)
    assert "[harness]\ninjected note" in txt
