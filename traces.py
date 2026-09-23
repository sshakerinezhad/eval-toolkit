"""ONE VARIANT FOLDER. Load Terminal-Bench trajectory traces and compute deterministic features.

No API calls here. Everything the grader (or any other analysis) needs to know about a run
*without* asking a model: metadata, termination type, dead-call loops, command patterns, and a
judge-ready rendering of the transcript.

Folder layout (verified 2026-09-23):
  <root>/trajectories_index.json         list of rows; world_name carries the category
  <root>/tasks/<task>__task_<id>/traj_*.json   {"trajectory": {...}}

Harness = Terminus. Assistant content keeps only the "Analysis:/Plan:" text; the JSON commands
and task_complete flag are NOT stored. Commands live in trajectory_output.command_history.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CONFIRM = "Are you sure you want to mark the task as complete"
DEAD = "Technical difficulties"  # harness text injected when the model returned 0 tokens
PARSE_ERR = "Previous response had parsing errors"
WARN = "Previous response had warnings"

CAP_S = 10_800        # 3h wall-clock cap
STALL_S = 10_000      # >= this with few calls = serving stall (all 5 seen were trained runs)
STALL_CALLS = 60

TASK_PREFIX = "terminal_bench_"
WORLD_PREFIX = "Terminal-Bench 2.1 / "

VERIFY_CMD = re.compile(r"\b(pytest|unittest|test_\w+\.py|tests?/|run_tests|make test|cargo test|go test|"
                        r"npm test|diff\b|cmp\b|sha256sum|md5sum)", re.I)
GIVEUP = re.compile(r"(unable to (complete|solve|find|fix)|cannot (complete|solve|be (done|completed|solved))|"
                    r"not possible|give up|giving up|impossible|best (i|we) can do|out of (time|ideas)|"
                    r"partial solution|limitation|could not (complete|solve|find))", re.I)
SLEEP = re.compile(r"\bsleep\s+(\d+(?:\.\d+)?)")
# The model's own description of a terminal it has lost control of. Only commands are hidden from the
# log, so this is the only lockup signal; it is noisy per turn (tasks *about* interrupts match), hence
# LOCKUP_STREAK: 397B runs split cleanly into <=10 or >=20 consecutive matching turns.
STUCK = re.compile(
    r"(stuck|frozen|freez|unresponsive|not respond|hang(s|ing)?\b|heredoc|"
    r"ctrl\+?-?c|\bc-c\b|ctrl\+?-?d|\bc-d\b|\beof\b|interrupt|"
    r"still (running|waiting)|terminal (is|seems|appears)|escape (the|from)|"
    r"garbled|corrupt|echo(ed|ing) (back|literally))", re.I)
LOCKUP_STREAK = 20

# Eval-side problems found by reading 397B traces (2026-09-23). Flagged, never silently dropped.
SUSPECT_TASKS = {
    "mteb-leaderboard": "answer key tracks a live leaderboard; all 6 runs found a newer #1",
    "mteb-retrieve": "all 6 runs return the identical 'wrong' line: method/env mismatch likely",
    "install-windows-3.11": "test_windows_keys_with_visual_feedback fails 6/6; harness cannot drive VNC keys",
    "sanitize-git-repo": "all 6 runs fail the same 2 tests after editing the same 2 files",
    "model-extraction-relu-logits": "local forward.py exposes the true weights (leak/cheat path)",
    "build-pov-ray": "working builds fail only on a missing file_id.diz",
    "torch-pipeline-parallelism": "tests run in a newer uv env than the agent container (no torch)",
}


@dataclass
class Run:
    variant: str
    task: str
    task_id: str
    traj_id: str
    category: str
    status: str
    score: float
    tests_passed: int | None
    tests_total: int | None
    test_statuses: dict
    duration: float
    n_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    max_prompt: int | None
    think_tokens: int          # sum of call_log reasoning_summary_tokens (index reasoning_tokens is always 0)
    exception_type: str | None
    error_message: str | None
    test_stdout_tail: str | None
    task_text: str
    messages: list[dict]
    command_history: str
    path: str

    @property
    def passed(self) -> bool:
        return self.score >= 1.0

    def assistant_texts(self) -> list[str]:
        return [(m.get("content") or "") for m in self.messages if m.get("role") == "assistant"]

    def tool_texts(self) -> list[str]:
        return [(m.get("content") or "") for m in self.messages if m.get("role") == "tool"]


# ---------- loading ----------

def variant_label(root: Path) -> str:
    name = root.name
    return name[:-len("_traces")] if name.endswith("_traces") else name


def _load_one(path: Path, variant: str, idx: dict) -> Run:
    t = json.loads(path.read_text(encoding="utf-8"))["trajectory"]
    o = t.get("trajectory_output") or {}
    u = o.get("usage_metrics") or {}
    calls = u.get("call_log") or []
    row = idx.get(t["trajectory_id"], {})
    init = t.get("initial_messages") or []
    task_text = next((m.get("content") or "" for m in init if m.get("role") == "user"), "")
    return Run(
        variant=variant,
        task=t["task_name"].removeprefix(TASK_PREFIX),
        task_id=t.get("task_id", ""),
        traj_id=t["trajectory_id"],
        category=(row.get("world_name") or "").removeprefix(WORLD_PREFIX),
        status=row.get("trajectory_status") or t.get("trajectory_status") or "",
        score=float(o.get("score") or 0.0),
        tests_passed=o.get("tests_passed"),
        tests_total=o.get("tests_total"),
        test_statuses=o.get("test_statuses") or {},
        duration=float(o.get("duration_seconds") or row.get("trajectory_time_elapsed") or 0.0),
        n_calls=len(calls),
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
        max_prompt=u.get("max_prompt_tokens"),
        think_tokens=sum(c.get("reasoning_summary_tokens") or 0 for c in calls),
        exception_type=o.get("exception_type"),
        error_message=o.get("error_message"),
        test_stdout_tail=(o.get("test_summary_metadata") or {}).get("test_stdout_tail"),
        task_text=task_text,
        messages=t.get("trajectory_messages") or [],
        command_history=o.get("command_history") or "",
        path=str(path),
    )


def load_variant(root: str | Path, variant: str | None = None) -> list[Run]:
    """All runs under <root>, sorted by (task, traj_id). variant defaults to the folder name sans _traces."""
    root = Path(root)
    idx_path = root / "trajectories_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"no trajectories_index.json under {root}")
    idx = {r["trajectory_id"]: r for r in json.loads(idx_path.read_text(encoding="utf-8"))}
    variant = variant or variant_label(root)
    runs = [_load_one(p, variant, idx) for p in sorted((root / "tasks").glob("*/traj_*.json"))]
    runs.sort(key=lambda r: (r.task, r.traj_id))
    return runs


def modal_tests_total(runs: list[Run]) -> dict[str, int]:
    """task -> most common tests_total. A run below its task's mode = verifier did not run fully."""
    per: dict[str, Counter] = {}
    for r in runs:
        if r.tests_total is not None:
            per.setdefault(r.task, Counter())[r.tests_total] += 1
    return {task: c.most_common(1)[0][0] for task, c in per.items()}


# ---------- deterministic features ----------

def _is_dead(text: str | None) -> bool:
    return (text or "").startswith(DEAD)


def features(run: Run) -> dict:
    m = run.messages
    acont = run.assistant_texts()
    tcont = run.tool_texts()

    conf_idx = [i for i, x in enumerate(m) if x.get("role") == "tool" and CONFIRM in (x.get("content") or "")]
    tail_roles = [x["role"] for x in m[conf_idx[-1] + 1:]] if conf_idx else []
    # Terminus: task_complete -> confirm prompt; task_complete again -> stop. Commands in that final
    # reply still run, so a confirmed run ends on [assistant] or [assistant, tool].
    confirmed = bool(conf_idx) and tail_roles in (["assistant"], ["assistant", "tool"]) and run.duration < CAP_S
    last = m[-1] if m else {}

    if run.exception_type:
        term = f"error:{run.exception_type}"
    elif confirmed:
        term = "completed_confirmed"
    elif last.get("role") == "tool" and CONFIRM in (last.get("content") or ""):
        term = "cut_at_confirm_prompt"
    elif run.duration >= CAP_S and any(_is_dead(a) for a in acont[-3:]):
        term = "timeout_in_context_overflow_loop"
    elif run.duration >= CAP_S:
        term = "timeout_3h"
    else:
        term = "other_cut"

    streak = best = 0
    for a in acont:
        streak = streak + 1 if _is_dead(a) else 0
        best = max(best, streak)

    # Stuck streak runs over live turns only: dead calls are harness filler, not the model's view.
    s = stuck_best = 0
    for a in (a for a in acont if not _is_dead(a)):
        s = s + 1 if STUCK.search(a) else 0
        stuck_best = max(stuck_best, s)

    lines = [ln for ln in run.command_history.split("\n")]
    repeated = sum(1 for a, b in zip(lines, lines[1:]) if a.strip() and a == b)
    sleeps = [float(x) for x in SLEEP.findall(run.command_history)]

    return {
        "termination": term,
        "n_confirm_prompts": len(conf_idx),
        "completed_confirmed": int(confirmed),
        "backed_off": len(conf_idx) - int(confirmed),
        "n_dead_calls": sum(_is_dead(a) for a in acont),
        "max_dead_streak": best,
        "max_stuck_streak": stuck_best,
        "live_calls": sum(not _is_dead(a) for a in acont),
        "n_parse_err": sum(x.startswith(PARSE_ERR) for x in tcont),
        "n_warn": sum(x.startswith(WARN) for x in tcont),
        "n_assistant": len(acont),
        "ran_tests": int(bool(VERIFY_CMD.search(run.command_history))),
        "giveup_final": int(bool(GIVEUP.search(" ".join(acont[-3:])))),
        "sleep_sum": sum(sleeps),
        "n_sleep_cmds": len(sleeps),
        "repeated_cmd_lines": repeated,
        "stalled": run.duration >= STALL_S and run.n_calls < STALL_CALLS,
    }


def pre_label(run: Run, feats: dict, modal_tests: int | None) -> str | None:
    """Mechanically certain categories. None = the judge decides."""
    if run.exception_type:
        return "INFRA_ERROR"
    if modal_tests is not None and run.tests_total is not None and run.tests_total < modal_tests:
        return "INFRA_ERROR"
    if run.passed:
        return "PASS"
    if feats["termination"] == "timeout_in_context_overflow_loop":
        return "CONTEXT_OVERFLOW_LOOP"
    if feats["stalled"]:
        return "SERVING_STALL"
    return None


# ---------- judge-ready rendering ----------

def render_transcript(run: Run) -> str:
    """Numbered turns. The harness user prompt (messages[0]) is dropped: task text is passed separately.
    Consecutive dead calls (and the harness replies to them) collapse to one marker line so a 1000-message
    overflow loop costs a few tokens instead of 100k. Nothing else is truncated."""
    out: list[str] = []
    turn = 0
    i = 0
    m = run.messages
    if m and m[0].get("role") == "user":
        i = 1
    while i < len(m):
        msg = m[i]
        role, content = msg.get("role"), msg.get("content") or ""
        if role == "assistant" and _is_dead(content):
            start = turn + 1
            n = 0
            while i < len(m) and m[i].get("role") == "assistant" and _is_dead(m[i].get("content") or ""):
                n += 1
                turn += 1
                i += 1
                if i < len(m) and m[i].get("role") == "tool":  # the harness "please continue" reply
                    i += 1
            out.append(f"[T{start}-T{turn}: {n} consecutive empty responses (context overflow), no commands ran]")
            continue
        if role == "assistant":
            turn += 1
            out.append(f"[T{turn} assistant]\n{content}")
        elif role == "tool":
            out.append(f"[T{turn} terminal]\n{content}")
        else:
            out.append(f"[harness]\n{content}")
        i += 1

    tests = "\n".join(f"- {'PASS' if v == 'pass' else 'FAIL'} {k}" for k, v in sorted(run.test_statuses.items()))
    footer = [f"# Test results\nscore: {run.score} ({run.tests_passed}/{run.tests_total} tests passed)", tests]
    if run.test_stdout_tail:
        footer.append(f"## test_stdout_tail (hidden tests, tail of output)\n{run.test_stdout_tail}")
    return "# Transcript\n" + "\n\n".join(out) + "\n\n" + "\n".join(footer)
