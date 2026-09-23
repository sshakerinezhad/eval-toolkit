"""ONE GRADE. LLM-judge every run in one or more trace folders; aggregate into hand-readable reports.

    python grader.py DIR [DIR ...] --name NAME [--model anthropic:claude-opus-5-5] [--workers 8]
                     [--limit N] [--task SUBSTR] [--effort high] [--dry-run] [--yes] [--report-only]

Flow: load variants (traces.py) -> deterministic features + pre-label -> one judge call per run
(llm.call + cache, asyncio, semaphore, kill-switch) -> append results/grade_<name>/judgments.jsonl
-> scores.csv + summary.md + tasks/<task>.md.

Why a judge at all: score is binary and ~45% of runs end at a wall cap; the *reason* a run fails and
how it behaved on the way (verification, honesty, efficiency) is the signal that separates trained
from untrained. Why a deterministic layer first: infra errors, overflow loops, and stalls are
mechanically certain; the judge only confirms them and spends its effort on the ambiguous 1/3.
Exit codes mirror run.py: 0 ok, 1 pre-run failure / declined, 2 killed mid-run, 130 Ctrl-C.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

import llm
import traces
from run import git_rev, load_pricing
from traces import Run

RESULTS_DIR = Path("results")
KILL_AFTER_CONSECUTIVE = 4
CHARS_PER_TOKEN = 2.2  # measured on pilot: terminal dumps tokenize dense (1.9-2.4 chars/token), not 4
ASSUMED_OUT_TOKENS = 2_500   # judge JSON + thinking; used only for the pre-run estimate
DEFAULT_MODEL = "anthropic:claude-opus-5-5"
TASK_TEXT_PREVIEW = 600

# ---------- taxonomy ----------

PRE_CATEGORIES = {
    "INFRA_ERROR": "Harness/verifier exception, HTTP 504, or the verifier did not run the full test set.",
    "CONTEXT_OVERFLOW_LOOP": "Prompt exceeded the context budget; every later call returned nothing until the 3h cap.",
    "SERVING_STALL": "Very long wall time with very few model calls: the serving side stalled, not the model.",
}
JUDGE_CATEGORIES = {
    "MISREAD_TASK": "Solved the wrong problem or missed an explicit requirement stated in the task.",
    "WRONG_APPROACH": "Understood the task but chose a method that could not work.",
    "IMPL_BUG": "Right approach; the delivered solution has defects the hidden tests catch.",
    "INCOMPLETE": "Was still making real progress when it ran out of time/turns (e.g. 3h cap while working).",
    "ENV_BLOCKER": "Missing dependency, no network, permissions, or similar; could not work around it.",
    "TOOL_INTERACTION": "Stuck on terminal mechanics: tmux, vim, qemu, heredocs, pagers, interactive prompts.",
    "UNVERIFIED_CLAIM": "Declared complete without running any check that would have caught the failure.",
    "VERIFIED_BUT_WRONG": "Ran its own checks, they passed, hidden tests disagree (spec ambiguity or mismatch).",
    "GAVE_UP": "Explicitly abandoned the task.",
    "HARNESS_ARTIFACT": "Fixture pollution or eval-environment mismatch, not a model failure.",
}
ALL_CATEGORIES = ["PASS", *PRE_CATEGORIES, *JUDGE_CATEGORIES]
SUSPECT_TASKS = tuple(traces.SUSPECT_TASKS)  # single source of truth, with reasons, in traces.py


def _taxonomy_block() -> str:
    rows = [f"- {k}: {v}" for k, v in {**PRE_CATEGORIES, **JUDGE_CATEGORIES}.items()]
    return "\n".join(rows)


SYSTEM_PROMPT = f"""You are grading one run of an autonomous coding agent on a Terminal-Bench task.
The agent (Terminus harness) works in a Linux container by emitting shell commands; only its
"Analysis/Plan" text and the terminal output are recorded, not the raw commands. Hidden tests decide
pass/fail. You see: task metadata, deterministic features computed from the log, the task text, the
full numbered transcript, and the hidden-test results.

Your job: say WHY the run ended the way it did and HOW the agent behaved. Be specific and cite turn
numbers (T12). Prefer evidence over inference; when the log does not support a claim, say so.

## Failure categories (pick exactly one `category`; `secondary` optional)
- PASS: the run passed (score 1.0). Use only when the metadata says passed.
{_taxonomy_block()}

If `pre_label` is given it was set by deterministic rules and is almost always right. Keep it unless
the transcript clearly contradicts it, and if you override, quote the evidence. For PASS runs the
category is PASS; still score behaviour honestly.

Known suspect tasks (eval-side issues seen before; use HARNESS_ARTIFACT only with evidence):
{", ".join(SUSPECT_TASKS)}.

## Scores (integers 1-5)
- verification: 1 = never checked anything; 3 = ran the obvious command once; 5 = tested the actual
  requirement (ran tests / compared outputs / reproduced the spec) before declaring done.
- efficiency: 1 = most turns wasted (polling sleeps, repeating the same command, re-reading files);
  3 = some waste; 5 = tight, each turn advanced the task.
- recovery: 1 = repeated the same failing action or ignored errors; 3 = eventually adjusted;
  5 = diagnosed errors quickly and changed approach appropriately. Use 3 if no errors occurred.
- false_completion (bool): the agent declared the task complete while a cheap check it did not run
  would have shown it was not. false for PASS runs and for runs that never declared completion.
- wasted_turns_pct (int 0-100, optional): your estimate of turns that did not advance the task.

## Output
Return ONLY a JSON object, no prose, no code fence:
{{"category": "...", "secondary": null, "evidence": "T..: ...", "verification": 1-5,
  "false_completion": true/false, "efficiency": 1-5, "recovery": 1-5, "wasted_turns_pct": 0-100,
  "notable": ["short observation", "..."], "summary": "Two sentences: what happened and why."}}
"""

REPAIR_SUFFIX = "\n\nYour previous reply was not a valid JSON object. Return ONLY the JSON object."


# ---------- config / items ----------

@dataclass
class GradeConfig:
    model: str = DEFAULT_MODEL
    workers: int = 8
    max_retries: int = 4
    timeout: float = 600.0      # long prompts + thinking; per-call
    effort: str | None = "high"  # reasoning_effort via extra_body; None = provider default
    max_tokens: int = 16_000
    temperature: float | None = None

    @property
    def extra_body(self) -> dict | None:
        return {"reasoning_effort": self.effort} if self.effort else None


@dataclass
class Item:
    run: Run
    feats: dict
    pre_label: str | None


def build_items(variants: list[list[Run]]) -> list[Item]:
    """Modal tests_total is computed per task across ALL variants so a verifier short-run is caught
    even when it happened in every trial of one variant."""
    all_runs = [r for v in variants for r in v]
    modal = traces.modal_tests_total(all_runs)
    items = []
    for r in all_runs:
        f = traces.features(r)
        items.append(Item(r, f, traces.pre_label(r, f, modal.get(r.task))))
    return items


def build_prompt(item: Item) -> str:
    r, f = item.run, item.feats
    # Variant name deliberately omitted: the judge must not know whether it is grading trained or untrained.
    meta = [
        f"task: {r.task}", f"task_category: {r.category}",
        f"passed: {r.passed}", f"score: {r.score}", f"tests: {r.tests_passed}/{r.tests_total}",
        f"duration_s: {int(r.duration)}", f"model_calls: {r.n_calls}",
        f"termination: {f['termination']}", f"pre_label: {item.pre_label or 'none (you decide)'}",
        f"exception: {r.exception_type or 'none'}",
        f"deterministic: confirm_prompts={f['n_confirm_prompts']} backed_off={f['backed_off']} "
        f"dead_calls={f['n_dead_calls']} parse_errors={f['n_parse_err']} ran_test_like_cmd={f['ran_tests']} "
        f"sleep_cmds={f['n_sleep_cmds']} sleep_total_s={int(f['sleep_sum'])} "
        f"repeated_cmd_lines={f['repeated_cmd_lines']} giveup_phrases_at_end={f['giveup_final']}",
    ]
    return ("# Run metadata\n" + "\n".join(meta) + "\n\n# Task\n" + r.task_text.strip()
            + "\n\n" + traces.render_transcript(r))


# ---------- judgment parsing ----------

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_judgment(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON object in reply")
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON: {e}") from e
    if not isinstance(j, dict):
        raise ValueError("JSON is not an object")
    for k in ("category", "evidence", "verification", "false_completion", "efficiency", "recovery", "summary"):
        if k not in j:
            raise ValueError(f"missing field {k}")
    if j["category"] not in ALL_CATEGORIES:
        raise ValueError(f"unknown category {j['category']!r}")
    sec = j.get("secondary")
    if sec is not None and sec not in ALL_CATEGORIES:
        raise ValueError(f"unknown secondary {sec!r}")
    for k in ("verification", "efficiency", "recovery"):
        v = j[k]
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
            raise ValueError(f"{k} must be int 1-5, got {v!r}")
    if not isinstance(j["false_completion"], bool):
        raise ValueError("false_completion must be bool")
    w = j.get("wasted_turns_pct")
    if w is not None and (not isinstance(w, int) or isinstance(w, bool) or not 0 <= w <= 100):
        raise ValueError(f"wasted_turns_pct must be int 0-100, got {w!r}")
    notable = j.get("notable") or []
    if not isinstance(notable, list):
        raise ValueError("notable must be a list")
    return {
        "category": j["category"], "secondary": sec, "evidence": str(j["evidence"]),
        "verification": j["verification"], "false_completion": j["false_completion"],
        "efficiency": j["efficiency"], "recovery": j["recovery"], "wasted_turns_pct": w,
        "notable": [str(x) for x in notable], "summary": str(j["summary"]),
    }


# ---------- one judge call ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_record(item: Item, *, judge: dict | None, error: str | None, cached: bool, tokens_in, tokens_out,
                parse_retries: int, error_kind: str | None = None, prompt_chars: int = 0) -> dict:
    r = item.run
    return {
        "variant": r.variant, "task": r.task, "category": r.category, "task_id": r.task_id,
        "traj_id": r.traj_id, "path": r.path, "passed": r.passed, "score": r.score,
        "tests_passed": r.tests_passed, "tests_total": r.tests_total, "duration": round(r.duration, 1),
        "n_calls": r.n_calls, "total_tokens": r.total_tokens, "max_prompt": r.max_prompt,
        "features": item.feats, "pre_label": item.pre_label,
        "judge": judge, "error": error, "error_kind": error_kind,
        "cached": cached, "tokens_in": tokens_in, "tokens_out": tokens_out,
        "parse_retries": parse_retries, "prompt_chars": prompt_chars, "graded_at": now_iso(),
    }


async def judge_run(client, sem: asyncio.Semaphore, item: Item, cfg: GradeConfig,
                    killed: asyncio.Event) -> dict | None:
    """Returns a record, or None if skipped because the run was killed. Never raises."""
    provider, model = llm.parse_model(cfg.model)
    prompt = build_prompt(item)
    retries = 0
    tokens_in = tokens_out = 0
    cached_all = True
    while True:
        key = llm.cache_key(provider, model, SYSTEM_PROMPT, prompt, 0, cfg.temperature, cfg.max_tokens,
                            cfg.extra_body)
        rec = llm.cache_get(key)
        if rec is None:
            cached_all = False
            async with sem:
                if killed.is_set():
                    return None
                res = await llm.call(client, provider, model, SYSTEM_PROMPT, prompt, cfg.temperature,
                                     cfg.max_tokens, cfg.max_retries, extra_body=cfg.extra_body)
            if res.response is None:
                last = res.errors[-1]
                kind = "fatal" if res.fatal else ("retryable" if last.retryable else "non_retryable")
                return make_record(item, judge=None, error=f"{last.type} status={last.status}: {last.message[:300]}",
                                   cached=False, tokens_in=tokens_in, tokens_out=tokens_out,
                                   parse_retries=retries, error_kind=kind, prompt_chars=len(prompt))
            rec = {"response": res.response, "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                   "finish_reason": res.finish_reason}
            llm.cache_put(key, rec)
        tokens_in += rec.get("tokens_in") or 0
        tokens_out += rec.get("tokens_out") or 0
        try:
            judge = parse_judgment(rec["response"])
        except ValueError as e:
            if retries == 0:
                retries = 1
                prompt = prompt + REPAIR_SUFFIX
                continue
            return make_record(item, judge=None, error=f"parse failed twice: {e}", cached=cached_all,
                               tokens_in=tokens_in, tokens_out=tokens_out, parse_retries=retries,
                               error_kind="parse", prompt_chars=len(prompt))
        return make_record(item, judge=judge, error=None, cached=cached_all, tokens_in=tokens_in,
                           tokens_out=tokens_out, parse_retries=retries, prompt_chars=len(prompt))


# ---------- execute ----------

@dataclass
class Summary:
    total: int = 0
    ok: int = 0
    cached: int = 0
    failed: int = 0
    skipped: int = 0
    killed: bool = False
    kill_reason: str = ""
    error_counts: Counter = field(default_factory=Counter)
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0


async def execute(items: list[Item], cfg: GradeConfig, out_path: Path) -> Summary:
    summary = Summary(total=len(items))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(out_path, "a", encoding="utf-8")  # append-only; reports dedupe by (variant, traj_id)
    provider, _ = llm.parse_model(cfg.model)
    client = llm.make_client(provider, cfg.timeout)
    sem = asyncio.Semaphore(cfg.workers)
    killed = asyncio.Event()
    consecutive = 0
    bar = tqdm(total=len(items), unit="run", dynamic_ncols=True)
    t0 = time.perf_counter()

    def kill(reason: str) -> None:
        if not killed.is_set():
            summary.killed, summary.kill_reason = True, reason
            killed.set()

    async def one(item: Item) -> None:
        nonlocal consecutive
        rec = await judge_run(client, sem, item, cfg, killed)
        if rec is None:
            summary.skipped += 1
            return
        if rec["judge"] is not None:
            summary.ok += 1
            summary.cached += rec["cached"]
            consecutive = 0
        else:
            summary.failed += 1
            summary.error_counts[rec["error_kind"] or "?"] += 1
            if rec["error_kind"] == "fatal":
                kill(f"fatal error: {rec['error']}")
            elif rec["error_kind"] == "non_retryable":
                consecutive += 1
                if consecutive >= KILL_AFTER_CONSECUTIVE:
                    kill(f"{KILL_AFTER_CONSECUTIVE} consecutive non-retryable errors; last: {rec['error']}")
        if not rec["cached"]:
            summary.tokens_in += rec["tokens_in"] or 0
            summary.tokens_out += rec["tokens_out"] or 0
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        bar.update(1)
        bar.set_postfix_str(f"ok={summary.ok} cached={summary.cached} err={summary.failed}")

    try:
        await asyncio.gather(*(one(it) for it in items))
    finally:
        summary.wall_s = time.perf_counter() - t0
        bar.close()
        out.close()
        await client.close()  # else Windows proactor logs 'Event loop is closed' at interpreter exit
    return summary


# ---------- selection ----------

def prompt_tokens(item: Item) -> int:
    return int((len(SYSTEM_PROMPT) + len(build_prompt(item))) / CHARS_PER_TOKEN)


def select_items(items: list[Item], fails_only: bool, max_prompt_tokens: int | None) -> tuple[list[Item], list[dict]]:
    """Split into (to judge, skipped records). Skipped runs still get a record so pass rates and
    deterministic pre-labels show up in the reports; only the judge fields are missing."""
    todo, skipped = [], []
    for it in items:
        reason = None
        if fails_only and it.pre_label is not None:
            reason = f"not judged: --fails-only (pre_label {it.pre_label})"
        elif max_prompt_tokens is not None and prompt_tokens(it) > max_prompt_tokens:
            reason = f"not judged: over --max-prompt-tokens {max_prompt_tokens:,} (~{prompt_tokens(it):,} tokens)"
        if reason:
            skipped.append(make_record(it, judge=None, error=reason, cached=False, tokens_in=0, tokens_out=0,
                                       parse_retries=0, error_kind="skipped"))
        else:
            todo.append(it)
    return todo, skipped


# ---------- estimate ----------

def estimate(items: list[Item], cfg: GradeConfig, pricing: dict) -> dict:
    provider, model = llm.parse_model(cfg.model)
    todo_chars = 0
    todo = 0
    for it in items:
        p = build_prompt(it)
        key = llm.cache_key(provider, model, SYSTEM_PROMPT, p, 0, cfg.temperature, cfg.max_tokens, cfg.extra_body)
        if llm.cache_get(key) is None:
            todo += 1
            todo_chars += len(SYSTEM_PROMPT) + len(p)
    in_tokens = todo_chars / CHARS_PER_TOKEN
    out_tokens = todo * ASSUMED_OUT_TOKENS
    price = pricing.get(cfg.model)
    cost = None
    if price:
        cost = in_tokens / 1e6 * price["input_per_m"] + out_tokens / 1e6 * price["output_per_m"]
    return {"runs": len(items), "cached": len(items) - todo, "to_run": todo,
            "est_in_tokens": int(in_tokens), "est_out_tokens": out_tokens, "cost_est": cost}


# ---------- reports ----------

def load_judgments(path: Path) -> list[dict]:
    """Last record per (variant, traj_id) wins, so re-runs into the same name supersede."""
    if not path.exists():
        return []
    by_key: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_key[(rec["variant"], rec["traj_id"])] = rec
    return sorted(by_key.values(), key=lambda r: (r["task"], r["variant"], r["traj_id"]))


def _cat(rec: dict) -> str:
    if rec["judge"]:
        return rec["judge"]["category"]
    return rec["pre_label"] or "UNJUDGED"


def _score_str(recs: list[dict]) -> str:
    return "[" + "".join("1" if r["passed"] else ("p" if r["score"] > 0 else "0") for r in recs) + "]"


def _mean(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.2f}" if xs else "-"


def _pct(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"


def _fail_cats(recs: list[dict]) -> str:
    c = Counter(_cat(r) for r in recs if not r["passed"])
    return ", ".join(f"{k}×{v}" if v > 1 else k for k, v in c.most_common()) or "-"


def _summary_md(records: list[dict], model: str) -> str:
    variants = sorted({r["variant"] for r in records})
    by_v = {v: [r for r in records if r["variant"] == v] for v in variants}
    L = [f"# Grade summary", f"judge: `{model}` · generated {now_iso()}", ""]
    L += ["## Runs", "| variant | runs | judged | unjudged | passed | pass rate |", "|---|---|---|---|---|---|"]
    for v in variants:
        rs = by_v[v]
        judged = sum(r["judge"] is not None for r in rs)
        passed = sum(r["passed"] for r in rs)
        L.append(f"| {v} | {len(rs)} | {judged} | {len(rs) - judged} | {passed} | {100 * passed / len(rs):.1f}% |")

    L += ["", "## Failure taxonomy (failed runs; count and share of that variant's failures)",
          "| category | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    fails = {v: [r for r in by_v[v] if not r["passed"]] for v in variants}
    cats = [c for c in ALL_CATEGORIES if c != "PASS"] + ["UNJUDGED"]
    seen = {_cat(r) for v in variants for r in fails[v]}
    for c in cats:
        if c not in seen:
            continue
        cells = []
        for v in variants:
            n = sum(_cat(r) == c for r in fails[v])
            cells.append(f"{n} ({100 * n / len(fails[v]):.0f}%)" if fails[v] else "0")
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    L += ["", "## Behaviour (mean of 1-5 scores; false_completion = share of runs)",
          "| metric | " + " | ".join(f"{v} all | {v} pass | {v} fail" for v in variants) + " |",
          "|---|" + "---|---|---|" * len(variants)]
    metrics = [("verification", _mean), ("efficiency", _mean), ("recovery", _mean),
               ("wasted_turns_pct", _mean), ("false_completion", _pct)]
    for m, fn in metrics:
        cells = []
        for v in variants:
            js = [(r["judge"], r["passed"]) for r in by_v[v] if r["judge"]]
            cells += [fn([j[m] for j, _ in js]), fn([j[m] for j, p in js if p]), fn([j[m] for j, p in js if not p])]
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    tasks = sorted({r["task"] for r in records})
    by_tv: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_tv[r["task"]][r["variant"]].append(r)
    paired = len(variants) == 2
    hdr = "| task | category | " + " | ".join(f"{v} | {v} fail cats" for v in variants)
    hdr += f" | Δ {variants[0]}−{variants[1]} |" if paired else " |"
    L += ["", "## Per task (trial scores in order; 1 pass, p partial, 0 fail)", hdr,
          "|---|---|" + "---|---|" * len(variants) + ("---|" if paired else "")]
    rows = []
    for t in tasks:
        cells = []
        means = []
        for v in variants:
            rs = by_tv[t].get(v, [])
            cells += [_score_str(rs) if rs else "-", _fail_cats(rs)]
            means.append(st.mean(r["score"] for r in rs) if rs else None)
        cat = next((r["category"] for v in variants for r in by_tv[t].get(v, [])), "")
        delta = None
        if paired and None not in means:
            delta = means[0] - means[1]
        rows.append((delta if delta is not None else 0.0, t, cat, cells, delta))
    rows.sort(key=lambda x: (-x[0], x[1]))
    for _, t, cat, cells, delta in rows:
        line = f"| {t} | {cat} | " + " | ".join(cells)
        line += f" | {delta:+.2f} |" if delta is not None else " |"
        L.append(line)

    if paired:
        a, b = variants
        for better, worse in ((a, b), (b, a)):
            L += ["", f"## {better} better than {worse}"]
            any_ = False
            for _, t, cat, cells, delta in rows:
                d = delta if better == a else (-delta if delta is not None else None)
                if d is not None and d > 0:
                    any_ = True
                    ra, rb = by_tv[t].get(better, []), by_tv[t].get(worse, [])
                    L.append(f"- **{t}** ({cat}): {_score_str(ra)} vs {_score_str(rb)} — {worse} fails: {_fail_cats(rb)}"
                             + (f"; {better} fails: {_fail_cats(ra)}" if any(not r['passed'] for r in ra) else ""))
            if not any_:
                L.append("- none")
    return "\n".join(L) + "\n"


def _task_md(task: str, recs: list[dict], task_text: str) -> str:
    variants = sorted({r["variant"] for r in recs})
    cat = recs[0]["category"]
    L = [f"# {task} — {cat}", "", "> " + task_text.strip().replace("\n", "\n> ")[:TASK_TEXT_PREVIEW], ""]
    for v in variants:
        rs = sorted((r for r in recs if r["variant"] == v), key=lambda r: r["traj_id"])
        L += [f"## {v}  {_score_str(rs)}", "",
              "| trial | score | tests | dur_s | calls | termination | pre_label | category | ver | eff | rec | false_compl |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rs, 1):
            j = r["judge"] or {}
            L.append(f"| {i} | {r['score']:.2f} | {r['tests_passed']}/{r['tests_total']} | {int(r['duration'])} | "
                     f"{r['n_calls']} | {r['features']['termination']} | {r['pre_label'] or '-'} | {_cat(r)} | "
                     f"{j.get('verification', '-')} | {j.get('efficiency', '-')} | {j.get('recovery', '-')} | "
                     f"{j.get('false_completion', '-')} |")
        L.append("")
        for i, r in enumerate(rs, 1):
            j = r["judge"]
            L.append(f"### {v} · trial {i} · {r['traj_id']} · {_cat(r)}")
            if not j:
                L += [f"- **UNJUDGED**: {r['error']}", ""]
                continue
            if j.get("secondary"):
                L.append(f"- **secondary**: {j['secondary']}")
            L += [f"- **evidence**: {j['evidence']}", f"- **summary**: {j['summary']}"]
            if j.get("notable"):
                L.append("- **notable**: " + "; ".join(j["notable"]))
            if j.get("wasted_turns_pct") is not None:
                L.append(f"- **wasted turns**: ~{j['wasted_turns_pct']}%")
            L.append("")
    return "\n".join(L)


CSV_FEATS = ["termination", "n_confirm_prompts", "backed_off", "n_dead_calls", "max_dead_streak", "n_parse_err",
             "ran_tests", "giveup_final", "sleep_sum", "n_sleep_cmds", "repeated_cmd_lines", "stalled"]
CSV_JUDGE = ["category", "secondary", "verification", "false_completion", "efficiency", "recovery",
             "wasted_turns_pct", "evidence", "summary"]


def write_reports(records: list[dict], out_dir: Path, model: str, task_texts: dict[str, str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks").mkdir(exist_ok=True)
    task_texts = task_texts or {}

    cols = ["variant", "task", "category", "traj_id", "passed", "score", "tests_passed", "tests_total", "duration",
            "n_calls", "total_tokens", "max_prompt", *CSV_FEATS, "pre_label",
            *[f"judge_{k}" for k in CSV_JUDGE], "error", "path"]
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in cols if k in r}
            row.update({k: r["features"].get(k) for k in CSV_FEATS})
            j = r["judge"] or {}
            row.update({f"judge_{k}": j.get(k) for k in CSV_JUDGE})
            w.writerow(row)

    (out_dir / "summary.md").write_text(_summary_md(records, model), encoding="utf-8")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    for task, recs in by_task.items():
        (out_dir / "tasks" / f"{task}.md").write_text(_task_md(task, recs, task_texts.get(task, "")),
                                                       encoding="utf-8")


# ---------- CLI ----------

def _fmt_time(s: float) -> str:
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def print_dashboard(cfg: GradeConfig, name: str, variants: list[str], est: dict, latency: float) -> None:
    cost = "unknown (no pricing entry)" if est["cost_est"] is None else f"${est['cost_est']:.2f}"
    print("-" * 78)
    print(f"grade: {name}   judge: {cfg.model}   effort: {cfg.effort or 'default'}   workers: {cfg.workers}")
    print(f"variants: {', '.join(variants)}")
    print(f"runs: {est['runs']} total | cached {est['cached']} | to judge {est['to_run']}")
    print(f"est input tokens: {est['est_in_tokens']:,}   est output tokens: {est['est_out_tokens']:,}")
    print(f"est cost: {cost}   est time: {_fmt_time(latency * est['to_run'] / cfg.workers)} (smoke latency x calls / workers)")
    print("-" * 78)


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\ninterrupted; completed judgments are cached and appended. Rerun the same command to resume.")
        return 130


def _main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="variant folders, each with trajectories_index.json + tasks/")
    ap.add_argument("--name", required=True, help="-> results/grade_<name>/")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--effort", default="high", help="reasoning_effort; 'none' to omit")
    ap.add_argument("--max-tokens", type=int, default=16_000)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N runs (after --task filter)")
    ap.add_argument("--task", default=None, help="only tasks whose name contains this substring")
    ap.add_argument("--fails-only", action="store_true",
                    help="judge only runs with no deterministic pre-label (ambiguous failures)")
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help="skip runs whose prompt exceeds this many (estimated) tokens")
    ap.add_argument("--dry-run", action="store_true", help="render prompts + estimate; no calls")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--report-only", action="store_true", help="rebuild reports from judgments.jsonl; no calls")
    args = ap.parse_args(argv)

    cfg = GradeConfig(model=args.model, workers=args.workers, max_retries=args.max_retries, timeout=args.timeout,
                      effort=None if args.effort == "none" else args.effort, max_tokens=args.max_tokens)
    out_dir = RESULTS_DIR / f"grade_{args.name}"
    jpath = out_dir / "judgments.jsonl"

    variants = [traces.load_variant(d) for d in args.dirs]
    task_texts = {r.task: r.task_text for v in variants for r in v}
    names = [v[0].variant for v in variants if v]

    if args.report_only:
        records = load_judgments(jpath)
        if not records:
            print(f"no judgments at {jpath}")
            return 1
        write_reports(records, out_dir, cfg.model, task_texts)
        print(f"reports rebuilt from {len(records)} judgments -> {out_dir}")
        return 0

    items = build_items(variants)
    if args.task:
        items = [it for it in items if args.task in it.run.task]
    if args.limit:
        items = items[:args.limit]
    items, skipped = select_items(items, args.fails_only, args.max_prompt_tokens)
    if skipped:
        print(f"skipping {len(skipped)} run(s) (recorded as unjudged): "
              + ", ".join(f"{k}={v}" for k, v in Counter(r["error"].split(" (")[0] for r in skipped).most_common()))
    if not items:
        print("no runs selected")
        return 1
    pricing = load_pricing()
    est = estimate(items, cfg, pricing)

    if args.dry_run:
        pdir = out_dir / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        for it in items:
            (pdir / f"{it.run.variant}__{it.run.task}__{it.run.traj_id}.txt").write_text(
                "### SYSTEM\n" + SYSTEM_PROMPT + "\n\n### USER\n" + build_prompt(it), encoding="utf-8")
        print_dashboard(cfg, args.name, names, est, 0.0)
        print(f"dry run: {len(items)} prompt(s) written to {pdir}")
        return 0

    print(f"smoke test: {cfg.model} ...")
    smoke = asyncio.run(llm.smoke([cfg.model], cfg.timeout))
    print("  " + smoke[0].line())
    if not smoke[0].ok:
        print("smoke test failed; nothing judged.")
        return 1
    print_dashboard(cfg, args.name, names, est, smoke[0].latency_s)
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grade.json").write_text(json.dumps({
        "started_at": now_iso(), "config": asdict(cfg), "cli_args": vars(args), "git_rev": git_rev(),
        "n_items": len(items), "estimate": est}, indent=2), encoding="utf-8")
    with open(jpath, "a", encoding="utf-8") as fh:
        for rec in skipped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = asyncio.run(execute(items, cfg, jpath))

    print("-" * 78)
    if summary.killed:
        print(f"RUN KILLED: {summary.kill_reason}")
        print("Fix the cause and rerun the same command; judged runs are cached.")
    print(f"runs: {summary.total} | ok {summary.ok} (cached {summary.cached}) | failed {summary.failed} | "
          f"skipped {summary.skipped}   tokens in {summary.tokens_in:,} / out {summary.tokens_out:,}   "
          f"wall {_fmt_time(summary.wall_s)}")
    for k, v in summary.error_counts.most_common():
        print(f"  errors {k}: {v}")
    records = load_judgments(jpath)
    write_reports(records, out_dir, cfg.model, task_texts)
    print(f"reports: {out_dir / 'summary.md'}  ({len(records)} runs)")
    return 2 if summary.killed else 0


if __name__ == "__main__":
    sys.exit(main())
