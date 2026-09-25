"""GRADE. Score APEX-v1 runner answers against each task's rubric criteria with an LLM judge.

    python grade.py build [--runs RAW] [--data CSV] [--prompt FILE] [--out FILE]
    python grade.py score RAW [--tasks FILE] [--baseline MODEL]

build: runner raw.jsonl (last line per task+model wins) + train.csv rubrics -> one run.py task per
(task, model, criterion). User message = judge prompt (Mercor's APEX grader prompt, scattergun guard
included; run.py configs can't load a system prompt from a file) + <TASK_PROMPT> (raw prompt, no
attachments: criteria state their expected values) + <RESPONSE> + <CRITERION>. No LLM calls here.

score: judge raw.jsonl from run.py -> parse {"rationale", "is_criteria_true"} per criterion
(unparseable = fail, counted as a judge error) -> score = passed/total per (task, model), unweighted
like Mercor's scoring -> scores.jsonl (task, variant, score) next to RAW -> metrics.py: mean with
bootstrap CI per model, paired diff (other model minus --baseline) with up/down/same.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

import metrics
from judge import FENCE, load_prompt

NO_RESPONSE = "(the model returned no response)"


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


# ---------- load ----------

def load_answers(path: str | Path) -> dict[tuple[int, str], dict]:
    """Runner raw.jsonl -> {(task_id, model): line}. Later lines win (appended reruns)."""
    path = Path(path)
    if not path.exists():
        die(f"runs not found: {path}")
    answers = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            die(f"{path}:{n}: invalid JSON: {e}")
        answers[(row["task_id"], row["model"])] = row
    if not answers:
        die(f"{path}: no rows")
    models = sorted({m for _, m in answers})
    tasks = sorted({t for t, _ in answers})
    missing = [(t, m) for t in tasks for m in models if (t, m) not in answers]
    if missing:
        die(f"{len(missing)} (task, model) pair(s) have no answer, e.g. {missing[:5]}")
    return answers


def load_rubrics(path: str | Path) -> dict[int, list[dict]]:
    """train.csv -> {task_id: [{"id", "description"}, ...]} in rubric order."""
    path = Path(path)
    if not path.exists():
        die(f"data not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["Task ID"]): [{"id": k, "description": v["description"]}
                                    for k, v in json.loads(r["Rubric JSON"]).items()]
                for r in csv.DictReader(f)}


# ---------- build ----------

def judge_message(judge_prompt: str, task_prompt: str, response: str | None, criterion: str) -> str:
    return (f"{judge_prompt}\n\n<TASK_PROMPT>\n{task_prompt}\n</TASK_PROMPT>\n\n"
            f"<RESPONSE>\n{NO_RESPONSE if response is None else response}\n</RESPONSE>\n\n"
            f"<CRITERION>\n{criterion}\n</CRITERION>")


def build(runs: str | Path, data: str | Path, prompt: str | Path, out: str | Path) -> int:
    answers = load_answers(runs)
    rubrics = load_rubrics(data)
    judge_prompt = load_prompt(prompt)
    rows = []
    for (task_id, model), a in sorted(answers.items()):
        if task_id not in rubrics:
            die(f"task {task_id} has no rubric in {data}")
        for c in rubrics[task_id]:
            rows.append({"id": f"{task_id}|{model}|{c['id']}",
                         "prompt": judge_message(judge_prompt, a["metadata"]["prompt_raw"], a["response"],
                                                 c["description"]),
                         "task_id": task_id, "model": model, "criterion_id": c["id"]})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_tasks, n_models = len({t for t, _ in answers}), len({m for _, m in answers})
    print(f"wrote {len(rows)} judge task(s) ({n_tasks} tasks x {n_models} models x criteria) -> {out}")
    return len(rows)


# ---------- score ----------

def parse_verdict(text: str | None) -> tuple[bool | None, str]:
    """Judge reply -> (is_criteria_true, rationale). None = unparseable (caller counts it as a fail)."""
    if not isinstance(text, str):
        return None, "(no reply)"
    s = text.strip()
    m = FENCE.match(s)  # a ```json fence is a formatting habit, not a wrong verdict
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s, strict=False)
    except json.JSONDecodeError:
        m = re.search(r'"is_criteria_true"\s*:\s*(true|false)', s)
        if m:
            return m.group(1) == "true", ""
        return None, text.strip()[:300]
    if not isinstance(obj, dict) or not isinstance(obj.get("is_criteria_true"), bool):
        return None, text.strip()[:300]
    why = obj.get("rationale")
    return obj["is_criteria_true"], why if isinstance(why, str) else ""


def score(raw: str | Path, tasks: str | Path, baseline: str) -> int:
    raw, tasks = Path(raw), Path(tasks)
    for p in (raw, tasks):
        if not p.exists():
            die(f"not found: {p}")
    expected = {json.loads(l)["id"] for l in tasks.read_text(encoding="utf-8").splitlines() if l.strip()}
    passed: dict[str, bool] = {}    # judge task id -> criterion met
    cell: dict[str, tuple] = {}     # judge task id -> (task_id, model)
    errors = []
    for n, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        if any(k not in meta for k in ("task_id", "model", "criterion_id")):
            die(f"{raw}:{n}: metadata lacks task_id/model/criterion_id; tasks not made by grade.py build?")
        verdict, why = parse_verdict(row.get("response"))
        if verdict is None:
            errors.append((row["task_id"], why))
        passed[row["task_id"]] = bool(verdict)
        cell[row["task_id"]] = (meta["task_id"], meta["model"])
    missing = expected - set(passed)
    if missing:
        die(f"{len(missing)} judge task(s) in {tasks} have no row in {raw}, e.g. {sorted(missing)[:3]}; "
            f"rerun run.py with a new --name (finished calls are cached)")

    per: dict[tuple, list[bool]] = {}
    for jid, ok in passed.items():
        per.setdefault(cell[jid], []).append(ok)
    out = raw.parent / "scores.jsonl"
    out.write_text("".join(json.dumps({"task": t, "variant": m, "score": sum(v) / len(v)}) + "\n"
                           for (t, m), v in sorted(per.items())), encoding="utf-8")

    models = sorted({m for _, m in per})
    for m in models:
        mine = [sum(v) / len(v) for (_, mm), v in per.items() if mm == m]
        print(f"{m:24s} tasks {len(mine)}  mean score {sum(mine) / len(mine):.3f}  "
              f"all criteria met {sum(s == 1.0 for s in mine)}")
    print(f"judge_errors (unparseable, counted as fail): {len(errors)}")
    for jid, why in errors[:10]:
        print(f"  {jid}: {' '.join(why.split())[:120]}")
    print(f"wrote {out}\n")
    if baseline not in models:
        die(f"--baseline {baseline!r} not in models {models}")
    return metrics.main([str(out), "--baseline", baseline])


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="runner answers + rubrics -> judge task JSONL for run.py")
    b.add_argument("--runs", default="results/apex30_v1/raw.jsonl")
    b.add_argument("--data", default="apex-v1/data/train.csv")
    b.add_argument("--prompt", default="prompts/apex_judge.md")
    b.add_argument("--out", default="results/grade_v1/tasks.jsonl")
    s = sub.add_parser("score", help="judge raw.jsonl -> scores.jsonl + metrics.json")
    s.add_argument("raw")
    s.add_argument("--tasks", default="results/grade_v1/tasks.jsonl", help="build output, checks completeness")
    s.add_argument("--baseline", default="gpt-5.6-luna", help="diff = other model minus this one")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.runs, args.data, args.prompt, args.out)
        return 0
    return score(args.raw, args.tasks, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
