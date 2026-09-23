"""ONE REPORT. Free, deterministic trained-vs-untrained comparison over tbench traces. No API calls.

    python report.py [--model Qwen3p5-397B] [--boot 10000] [--seed 0]

Reads  tbench-traces/<model>_{trained,untrained}_traces  (via traces.py; stats via passk.py)
Writes results/report_<model>/results.jsonl   one row per run: facts + an `ending` label
       results/report_<model>/metrics.json    four sections, one per question:
                                             q1_improvement  did it get better (paired, with CIs)
                                             q2_behavior     how behaviour changed (+ verified examples)
                                             q3_failures     how each variant failed (taxonomy)
                                             q4_trust        can the score be trusted
       results/report_<model>/endings.png     how runs ended, share of runs vs share of tokens

Why no judge: grader.py (LLM judge) says *why* a run failed but costs ~$0.20/run. Everything here comes
from fields already in the traces: harness exceptions, wall time, dead calls, the model's own
"terminal is stuck" text, and the hidden tests' stdout. That says *what* went wrong, not why.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import passk
import traces

ROOT = Path(__file__).parent
VARIANTS = ("trained", "untrained")
TAIL_CHARS = 500

# ---------- ending taxonomy ----------
# One label per run. Precedence matters: an infra error explains a failure better than the stall or
# overflow it caused, and a lockup explains the overflow loop it drove the run into.
HARNESS = ("infra_error", "verifier_short", "stall_suspected", "overflow_loop")
UNFINISHED = ("lockup", "cap_working", "cut_other")
FALSE_DONE = ("done_missing_output", "done_too_slow", "done_crash", "done_wrong_value", "done_no_test_output",
              "done_unclassified")
ENDINGS = ("pass", *FALSE_DONE, *UNFINISHED, *HARNESS)
DEFINITIONS = {
    "pass": "hidden tests passed (score 1)",
    "done_missing_output": "model confirmed done; a failing test raised FileNotFoundError or is a test_*exist* check",
    "done_too_slow": "model confirmed done; a failing test timed out or is a runtime/performance check",
    "done_crash": "model confirmed done; a failing test raised a non-assertion exception (import/attr/runtime/subprocess)",
    "done_wrong_value": "model confirmed done; failing tests are plain assertions on the output's content",
    "done_no_test_output": "model confirmed done; no test stdout recorded to classify",
    "done_unclassified": "model confirmed done; test stdout has no pytest FAILED/ERROR summary line",
    "lockup": f">= {traces.LOCKUP_STREAK} consecutive live turns where the model says the terminal is stuck",
    "cap_working": "hit the 3h wall cap while still issuing live calls (polling, grinding)",
    "cut_other": "ended without a confirmed completion and below the cap",
    "infra_error": "harness/verifier exception (sandbox shutdown, verifier timeout)",
    "verifier_short": "fewer tests ran than this task's usual count (e.g. verifier download 504)",
    "stall_suspected": f">= {traces.STALL_S}s wall time with < {traces.STALL_CALLS} calls; serving stall inferred, not proven",
    "overflow_loop": "prompt passed ~180k (262k ctx - 81,920 max_tokens); every later call empty until the cap",
}
# The tail classifier reads pytest's short summary: "FAILED path::test_name - ExcType: msg". pytest
# truncates ExcType ("Ass...", "FileNo...", "subprocess.Timeou..."), so match on prefixes. Free-text
# regex over the whole tail was tried first and failed: "Error" matches AssertionError, "not found"
# and "Timeout" appear in unrelated log lines. Assertions are refined by test name.
SUMMARY_LINE = re.compile(r"^(?:FAILED|ERROR) \S*?::(\S+)(?: - (.*))?$", re.M)
# Most upstream wins when a run fails several tests: no output explains a crash, a crash a wrong value.
TAIL_PRIORITY = ("done_missing_output", "done_crash", "done_too_slow", "done_wrong_value")


def _classify_failed_test(test: str, message: str) -> str:
    exc = message.split(":")[0].strip().lower()
    name = test.lower()
    if exc.startswith(("fileno", "filenotfound")):
        return "done_missing_output"
    if "timeou" in exc:
        return "done_too_slow"
    if exc and not exc.startswith("as"):  # anything not an assertion: import/attr/runtime/subprocess error
        return "done_crash"
    if "exist" in name:
        return "done_missing_output"
    if any(k in name for k in ("runtime", "performance", "speed", "fast")):
        return "done_too_slow"
    return "done_wrong_value"


# Plot folds 13 labels into 7 groups (categorical palette max 8, validated light mode, fixed order).
PLOT_GROUPS = (
    ("Passed", ("pass",), "#2a78d6"),
    ("Said done: wrong value", ("done_wrong_value",), "#eb6834"),
    ("Said done: missing / crash / slow", ("done_missing_output", "done_too_slow", "done_crash", "done_no_test_output",
                                           "done_unclassified"), "#1baf7a"),
    ("Terminal lockup", ("lockup",), "#eda100"),
    ("Out of time / cut", ("cap_working", "cut_other"), "#e87ba4"),
    ("Context overflow loop", ("overflow_loop",), "#008300"),
    ("Infra / verifier / stall", ("infra_error", "verifier_short", "stall_suspected"), "#4a3aa7"),
)

# Runs read by hand on 2026-09-23 (subagent reports + spot checks). Matched by substring of traj_id.
EXAMPLES = (
    ("verification", "qemu-alpine-ssh", "trained", "77f3d144",
     "Configured sshd in the VM, backgrounded qemu, then logged in with ssh -p 2222 from the host before declaring done."),
    ("false_done", "qemu-alpine-ssh", "untrained", "d927e3c9",
     "Saw sshd listening inside the VM and declared done in 10 calls without testing login from the host."),
    ("lockup", "qemu-alpine-ssh", "untrained", "34f177ad",
     "Lost the qemu console: 516 consecutive stuck turns resending the same tmux keys; 48.6M tokens."),
    ("lockup", "password-recovery", "trained", "b6b10e75",
     "Opened a python heredoc that never closed, then ~350 turns of 'send EOF'; 64M tokens, 1/2 tests."),
    ("lockup", "large-scale-text-editing", "trained", "db641280",
     "Stuck in vim ex-mode for 488 consecutive turns sending Ctrl+D; the other two trials passed in ~10 calls."),
    ("verification", "fix-ocaml-gc", "trained", "3164f339",
     "Found the one-line GC sweep bug, checked sibling loops, changed one line, ran the testsuite twice."),
    ("spec_misread", "count-dataset-tokens", "untrained", "1277d157",
     "Concatenated two text fields before tokenizing: 79566 vs expected 79586 on an exact-match test."),
    ("spec_misread", "sparql-university", "trained", "65c7c8c2",
     "Applied the EU filter to the returned country list, dropping US; the one untrained pass returned all countries."),
    ("false_done", "query-optimize", "untrained", None,
     "All 5 failing runs claimed ~380x speedup measured against the original query, not the golden one."),
    ("harness", "polyglot-rust-c", "trained", None,
     "3 trained runs sat ~11,000s with 3-51 calls; untrained solved it in 8 calls. Serving stall inferred."),
    ("harness", "rstan-to-pystan", "trained", "bb315822",
     "Verifier download returned HTTP 504 (uvx: command not found); the agent's work was never graded."),
    ("trust", "break-filter-js-from-html", "trained", None,
     "Copied its filter into /tests/filter.py so the local test passed; the real grader uses its own copy."),
)


def classify_tail(tail: str | None) -> str:
    tail = tail or ""
    if not tail.strip():
        return "done_no_test_output"
    labels = {_classify_failed_test(t, m or "") for t, m in SUMMARY_LINE.findall(tail)}
    return next((p for p in TAIL_PRIORITY if p in labels), "done_unclassified")


def ending(run: traces.Run, feats: dict, modal_tests: int | None) -> str:
    if run.passed:
        return "pass"
    if run.exception_type:
        return "infra_error"
    if modal_tests is not None and run.tests_total is not None and run.tests_total < modal_tests:
        return "verifier_short"
    if feats["stalled"]:
        return "stall_suspected"
    if feats["max_stuck_streak"] >= traces.LOCKUP_STREAK:
        return "lockup"
    if feats["termination"] == "timeout_in_context_overflow_loop":
        return "overflow_loop"
    if feats["termination"] == "completed_confirmed":
        return classify_tail(run.test_stdout_tail)
    if feats["termination"] == "timeout_3h":
        return "cap_working"
    return "cut_other"


def group_of(label: str) -> str:
    return ("pass" if label == "pass" else "false_done" if label in FALSE_DONE
            else "unfinished" if label in UNFINISHED else "harness")


# ---------- rows ----------

def build_rows(model: str, traces_dir: Path) -> list[dict]:
    runs = {v: traces.load_variant(traces_dir / f"{model}_{v}_traces", variant=v) for v in VARIANTS}
    modal = traces.modal_tests_total([r for rs in runs.values() for r in rs])  # across both variants
    rows = []
    for v in VARIANTS:
        trial = Counter()
        for r in runs[v]:
            f = traces.features(r)
            label = ending(r, f, modal.get(r.task))
            rows.append({
                "variant": v, "model": model, "task": r.task, "category": r.category,
                "trial": trial[r.task], "traj_id": r.traj_id,
                "path": Path(r.path).relative_to(traces_dir).as_posix(),
                "score": r.score, "passed": r.passed,
                "tests_passed": r.tests_passed, "tests_total": r.tests_total,
                "failed_tests": sorted(k for k, s in r.test_statuses.items() if s != "pass"),
                "test_tail": (r.test_stdout_tail or "")[-TAIL_CHARS:],
                "ending": label, "ending_group": group_of(label),
                "partial": bool(r.tests_total and 0 < (r.tests_passed or 0) < r.tests_total),
                "suspect_task": r.task in traces.SUSPECT_TASKS,
                "termination": f["termination"],
                "claimed_done": f["n_confirm_prompts"] > 0,
                "completed_confirmed": bool(f["completed_confirmed"]),
                "backed_off": f["backed_off"],
                "max_stuck_streak": f["max_stuck_streak"],
                "live_calls": f["live_calls"], "dead_calls": f["n_dead_calls"],
                "duration_s": round(r.duration, 1),
                "prompt_tokens": r.prompt_tokens or 0, "completion_tokens": r.completion_tokens or 0,
                "think_tokens": r.think_tokens, "max_prompt": r.max_prompt,
                "repeated_cmd_lines": f["repeated_cmd_lines"],
            })
            trial[r.task] += 1
    return rows


# ---------- stats helpers ----------

def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test (ties dropped). Stdlib only."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def by_task(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {v: [] for v in VARIANTS})
    for r in rows:
        out[r["task"]][r["variant"]].append(r)
    return out


def paired_passk(rows: list[dict], k: int, B: int, seed: int) -> dict:
    """Task-level paired bootstrap, same draws for both variants (as passk.py). Tasks where either variant
    has fewer than k runs left after filtering are dropped and counted."""
    tasks = by_task(rows)
    names = sorted(t for t, v in tasks.items() if all(len(v[x]) >= k for x in VARIANTS))
    vals = {v: [passk.pass_at_k([1.0 if r["passed"] else 0.0 for r in tasks[t][v]], k) for t in names]
            for v in VARIANTS}
    draws = passk.make_draws(len(names), B, seed)
    out: dict = {"n_tasks": len(names), "n_tasks_dropped": len(tasks) - len(names)}
    for v in VARIANTS:
        lo, hi = passk.bootstrap_ci(draws, vals[v])
        out[v] = {"mean": round(st.mean(vals[v]), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    diff = [a - b for a, b in zip(vals["trained"], vals["untrained"])]
    lo, hi = passk.bootstrap_ci(draws, diff)
    out["diff"] = {"mean": round(st.mean(diff), 4), "ci95": [round(lo, 4), round(hi, 4)],
                   "p_le_0": round(sum(1 for d in draws if sum(diff[i] for i in d) <= 0) / B, 4)}
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def geo_mean_ratio(pairs: list[tuple[float, float]]) -> float | None:
    logs = [math.log(a / b) for a, b in pairs if a and b]
    return round(math.exp(st.mean(logs)), 3) if logs else None


def pattern(rs: list[dict]) -> str:
    return "[" + "".join("1" if r["passed"] else "0" for r in rs) + "]"


# ---------- metrics ----------

def q1_improvement(rows, B, seed):
    tasks = by_task(rows)
    wins = losses = 0
    for v in tasks.values():
        d = sum(r["passed"] for r in v["trained"]) - sum(r["passed"] for r in v["untrained"])
        wins, losses = wins + (d > 0), losses + (d < 0)
    solved = {t: {x: any(r["passed"] for r in v[x]) for x in VARIANTS} for t, v in tasks.items()}
    infra = {"infra_error", "verifier_short", "stall_suspected"}
    subsets = {
        "all_runs": rows,
        "excl_infra": [r for r in rows if r["ending"] not in infra],
        "excl_infra_and_overflow": [r for r in rows if r["ending"] not in infra | {"overflow_loop"}],
        "excl_suspect_tasks": [r for r in rows if not r["suspect_task"]],
        "excl_all_of_the_above": [r for r in rows if r["ending"] not in infra | {"overflow_loop"} and not r["suspect_task"]],
    }
    cats = defaultdict(lambda: {v: [] for v in VARIANTS})
    cat_tasks = defaultdict(set)
    for r in rows:
        cats[r["category"]][r["variant"]].append(r["passed"])
        cat_tasks[r["category"]].add(r["task"])
    return {
        "pass_at_1": paired_passk(rows, 1, B, seed),
        "pass_at_3": paired_passk(rows, 3, B, seed),
        "task_flips_at_3": {
            "trained_only": sum(s["trained"] and not s["untrained"] for s in solved.values()),
            "untrained_only": sum(s["untrained"] and not s["trained"] for s in solved.values()),
            "both": sum(s["trained"] and s["untrained"] for s in solved.values()),
            "neither": sum(not s["trained"] and not s["untrained"] for s in solved.values()),
        },
        "per_task_pass_count": {"trained_better": wins, "untrained_better": losses,
                                "tie": len(tasks) - wins - losses, "sign_test_p": round(sign_test_p(wins, losses), 4)},
        "sensitivity_pass_at_1": {
            "note": "Failures that are not the model's fault removed from both variants; per-task rate "
                    "over the remaining trials. Does the gain survive?",
            **{name: paired_passk(sub, 1, B, seed) for name, sub in subsets.items()},
        },
        "per_category_pass_rate": {
            c: {"n_tasks": len(cat_tasks[c]), **{x: round(st.mean(v[x]), 3) if v[x] else None for x in VARIANTS}}
            for c, v in sorted(cats.items())},
    }


def q2_behavior(rows):
    out: dict = {"per_variant": {}}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        conf = [r for r in rs if r["completed_confirmed"]]
        lock = [r for r in rs if r["max_stuck_streak"] >= traces.LOCKUP_STREAK]
        out["per_variant"][v] = {
            "runs": len(rs),
            "confirmed_done": len(conf),
            "false_done_runs": sum(not r["passed"] for r in conf),
            "false_done_rate": round(sum(not r["passed"] for r in conf) / len(conf), 3) if conf else None,
            "runs_backed_off_after_are_you_sure": sum(r["backed_off"] > 0 for r in rs),
            "lockup_runs": len(lock),
            "lockup_runs_that_still_passed": sum(r["passed"] for r in lock),
            "overflow_loop_runs": sum(r["dead_calls"] > 0 and r["termination"] == "timeout_in_context_overflow_loop" for r in rs),
            "cap_runs": sum(r["duration_s"] >= traces.CAP_S for r in rs),
            "median_passed": {k: med(r[k] for r in rs if r["passed"])
                              for k in ("duration_s", "live_calls", "completion_tokens", "think_tokens")},
            "median_failed": {k: med(r[k] for r in rs if not r["passed"])
                              for k in ("duration_s", "live_calls", "completion_tokens", "think_tokens")},
            "total_hours": round(sum(r["duration_s"] for r in rs) / 3600, 1),
            "total_prompt_tokens_M": round(sum(r["prompt_tokens"] for r in rs) / 1e6, 1),
        }
    # Efficiency, apples to apples: tasks both variants pass 3/3. Per-task means, then geo-mean of T/U.
    tasks = by_task(rows)
    both = {t: v for t, v in tasks.items() if all(v[x] and all(r["passed"] for r in v[x]) for x in VARIANTS)}
    eff: dict = {"n_tasks": len(both)}
    for k in ("duration_s", "live_calls", "completion_tokens", "think_tokens", "prompt_tokens"):
        pairs = [(st.mean(r[k] for r in v["trained"]), st.mean(r[k] for r in v["untrained"])) for v in both.values()]
        lo, hi = sum(a < b for a, b in pairs), sum(a > b for a, b in pairs)
        eff[k] = {"geo_mean_ratio_trained_over_untrained": geo_mean_ratio(pairs),
                  "trained_lower_on": lo, "trained_higher_on": hi, "sign_test_p": round(sign_test_p(lo, hi), 4)}
    out["efficiency_on_tasks_both_pass_3of3"] = eff
    out["examples"] = resolve_examples(rows)
    return out


def resolve_examples(rows):
    out = []
    for theme, task, variant, frag, note in EXAMPLES:
        hits = [r for r in rows if r["task"] == task and r["variant"] == variant and (frag is None or frag in r["traj_id"])]
        ex: dict = {"theme": theme, "task": task, "variant": variant, "note": note}
        if frag is not None:
            ex["traj_id"] = hits[0]["traj_id"] if len(hits) == 1 else None
            ex["ending"] = hits[0]["ending"] if len(hits) == 1 else None
        else:
            ex["task_pattern"] = pattern(hits)
        ex["found"] = bool(hits) and (frag is None or len(hits) == 1)
        out.append(ex)
    return out


def q3_failures(rows):
    out: dict = {"per_variant": {}}
    for v in VARIANTS:
        fails = [r for r in rows if r["variant"] == v and not r["passed"]]
        c = Counter(r["ending"] for r in fails)
        g = Counter(r["ending_group"] for r in fails)
        out["per_variant"][v] = {
            "failed_runs": len(fails),
            "by_group": {k: g.get(k, 0) for k in ("false_done", "unfinished", "harness")},
            "by_ending": {k: c.get(k, 0) for k in ENDINGS if k != "pass"},
            "share_by_ending": {k: round(c.get(k, 0) / len(fails), 3) for k in ENDINGS if k != "pass"} if fails else {},
            "false_done_partial_tests_passed": sum(r["partial"] for r in fails if r["ending_group"] == "false_done"),
        }
    t, u = out["per_variant"]["trained"]["by_ending"], out["per_variant"]["untrained"]["by_ending"]
    out["diff_trained_minus_untrained"] = {k: t[k] - u[k] for k in t}
    flips = []
    for task, v in sorted(by_task(rows).items()):
        pt, pu = sum(r["passed"] for r in v["trained"]), sum(r["passed"] for r in v["untrained"])
        if pt != pu:
            flips.append({"task": task, "trained": pattern(v["trained"]), "untrained": pattern(v["untrained"]),
                          "diff": pt - pu, "suspect": task in traces.SUSPECT_TASKS,
                          "trained_fail_endings": [r["ending"] for r in v["trained"] if not r["passed"]],
                          "untrained_fail_endings": [r["ending"] for r in v["untrained"] if not r["passed"]]})
    out["tasks_where_variants_differ"] = sorted(flips, key=lambda f: (-f["diff"], f["task"]))
    return out


def q4_trust(rows, q1):
    tasks = by_task(rows)
    out: dict = {"harness_caused_failures": {}, "token_share_by_group": {}}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        h = [r for r in rs if not r["passed"] and r["ending_group"] == "harness"]
        out["harness_caused_failures"][v] = {
            "count": len(h),
            "by_ending": dict(Counter(r["ending"] for r in h)),
            "non_overflow_runs": [{"task": r["task"], "traj_id": r["traj_id"], "ending": r["ending"]}
                                  for r in h if r["ending"] != "overflow_loop"],
        }
        tot = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) or 1
        out["token_share_by_group"][v] = {
            g: round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs if r["ending_group"] == g) / tot, 3)
            for g in ("pass", "false_done", "unfinished", "harness")}
    sus = {t: {"reason": traces.SUSPECT_TASKS[t], **{x: pattern(tasks[t][x]) for x in VARIANTS}}
           for t in sorted(traces.SUSPECT_TASKS) if t in tasks}
    out["suspect_tasks"] = sus
    out["suspect_tasks_net_pass_diff"] = sum(sum(r["passed"] for r in tasks[t]["trained"])
                                             - sum(r["passed"] for r in tasks[t]["untrained"]) for t in sus)
    mixed = {v: sum(0 < sum(r["passed"] for r in tv[v]) < len(tv[v]) for tv in tasks.values()) for v in VARIANTS}
    diffs = [sum(r["passed"] for r in v["trained"]) - sum(r["passed"] for r in v["untrained"]) for v in tasks.values()]
    out["trial_noise"] = {
        "tasks_with_mixed_trials": mixed,
        "tasks_differing_by_exactly_1_of_3": sum(abs(d) == 1 for d in diffs),
        "tasks_differing_by_2_or_3": sum(abs(d) >= 2 for d in diffs),
        "note": "A 1-of-3 difference is within temperature-0.6 trial noise; only the aggregate is testable.",
    }
    s = q1["sensitivity_pass_at_1"]
    out["gain_after_exclusions"] = {k: s[k]["diff"] for k in s if k != "note"}
    out["known_unreliable_fields"] = [
        "index reasoning_tokens / tool_calls / compaction_count: always 0",
        "trajectory_output.empty_response_loop: False even inside overflow loops",
        "trajectory_output.outcome: 'completed' for 3h-cap runs",
        "command_history: lossy (heredocs dropped); commands and task_complete not stored in messages",
        "final_answer: harness placeholder in dead-call runs; model claims success on most failures",
        "last message role == tool: normal after a confirmed completion, not a cut-off signal",
    ]
    return out


# ---------- plot ----------

def plot(rows, path: Path, model: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    ink, muted, surface = "#0b0b0b", "#898781", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True, facecolor=surface)
    panels = (("Share of runs", lambda r: 1), ("Share of tokens (prompt + completion)",
                                                lambda r: r["prompt_tokens"] + r["completion_tokens"]))
    for ax, (title, weight) in zip(axes, panels):
        ax.set_facecolor(surface)
        for yi, v in enumerate(VARIANTS):
            rs = [r for r in rows if r["variant"] == v]
            total = sum(weight(r) for r in rs) or 1
            left = 0.0
            for name, labels, color in PLOT_GROUPS:
                sel = [r for r in rs if r["ending"] in labels]
                w = sum(weight(r) for r in sel) / total
                if w:
                    ax.barh(yi, w, left=left, color=color, edgecolor=surface, linewidth=2, height=0.6)
                    if w >= 0.045:  # direct labels: palette has sub-3:1 slots, so text carries identity too
                        txt = f"{len(sel)}" if title.startswith("Share of runs") else f"{w:.0%}"
                        ax.text(left + w / 2, yi, txt, ha="center", va="center", fontsize=9, color="white",
                                fontweight="bold")
                left += w
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(colors=muted, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(VARIANTS)), [f"{v}\n(n={sum(r['variant'] == v for r in rows)})" for v in VARIANTS],
                       color=ink, fontsize=10)
    axes[0].invert_yaxis()
    handles = [Patch(color=c, label=n) for n, _, c in PLOT_GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle(f"{model} on Terminal-Bench 2.1: how runs ended (left: run counts; right: where the tokens went)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ---------- main ----------

def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_metrics(rows, model, B, seed):
    q1 = q1_improvement(rows, B, seed)
    return {
        "meta": {
            "model": model, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": git_rev(), "n_runs": len(rows),
            "n_tasks": len({r["task"] for r in rows}),
            "bootstrap": {"unit": "task", "B": B, "seed": seed, "ci": "95% percentile"},
            "thresholds": {"lockup_streak": traces.LOCKUP_STREAK, "cap_s": traces.CAP_S,
                           "stall_s": traces.STALL_S, "stall_calls": traces.STALL_CALLS},
            "ending_definitions": DEFINITIONS,
            "ending_groups": {"false_done": FALSE_DONE, "unfinished": UNFINISHED, "harness": HARNESS},
        },
        "q1_improvement": q1,
        "q2_behavior": q2_behavior(rows),
        "q3_failures": q3_failures(rows),
        "q4_trust": q4_trust(rows, q1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3p5-397B")
    ap.add_argument("--traces", type=Path, default=ROOT / "tbench-traces")
    ap.add_argument("--out", type=Path, default=None, help="default results/report_<model>")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    out = a.out or ROOT / "results" / f"report_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(a.model, a.traces)
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = build_metrics(rows, a.model, a.boot, a.seed)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if not a.no_plot:
        plot(rows, out / "endings.png", a.model)

    q1, q3 = metrics["q1_improvement"], metrics["q3_failures"]["per_variant"]
    p1, p3 = q1["pass_at_1"], q1["pass_at_3"]
    print(f"{a.model}: {len(rows)} runs -> {out}")
    print(f"  pass@1 T {p1['trained']['mean']:.3f} U {p1['untrained']['mean']:.3f} diff {p1['diff']['mean']:+.3f} {p1['diff']['ci95']}")
    print(f"  pass@3 T {p3['trained']['mean']:.3f} U {p3['untrained']['mean']:.3f} diff {p3['diff']['mean']:+.3f} {p3['diff']['ci95']}")
    for v in VARIANTS:
        print(f"  {v:<9} failures by group {q3[v]['by_group']}")
    missing = [e for e in metrics["q2_behavior"]["examples"] if not e["found"]]
    if missing:
        print(f"  WARNING: {len(missing)} curated examples not found: {[e['task'] for e in missing]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
