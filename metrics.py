"""METRICS. Per-variant pass@1 / pass@k / pass^k with task-level bootstrap intervals, plus a paired diff.

    python metrics.py FILE [--k K ...] [--threshold T] [--boot 10000] [--seed 0] [--baseline NAME]

FILE is .jsonl or .csv, one row per attempt, columns: task, variant, score.
Flow: load rows -> group {task: {variant: [scores]}} -> check every task has every variant
-> per-task value of each metric -> one shared set of bootstrap draws over tasks
-> interval per variant (+ paired diff if exactly two variants) -> print table + write metrics.json
next to FILE.

Metrics: mean = average raw score (pass@1 on raw scores). pass@k / pass^k count an attempt as a
pass when score >= --threshold. Diff = other variant minus baseline (first in file, or --baseline).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import NoReturn


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


# ---------- load ----------

def load_rows(path: str | Path) -> list[dict]:
    """Each row -> {"task": str, "variant": str, "score": float}. Extra columns are ignored."""
    path = Path(path)
    if not path.exists():
        die(f"input not found: {path}")
    raw = []  # (line number, row dict)
    if path.suffix == ".jsonl":
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw.append((n, json.loads(line)))
            except json.JSONDecodeError as e:
                die(f"{path}:{n}: invalid JSON: {e}")
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:  # -sig: Excel CSVs start with a BOM
            raw = list(enumerate(csv.DictReader(f), 2))  # line 1 is the header
    else:
        die(f"{path}: expected a .jsonl or .csv file")

    rows = []
    for n, r in raw:
        if not isinstance(r, dict) or any(r.get(k) in (None, "") for k in ("task", "variant", "score")):
            die(f"{path}:{n}: row needs task, variant, score")
        try:
            score = float(r["score"])
        except (TypeError, ValueError):
            die(f"{path}:{n}: score {r['score']!r} is not a number")
        if not math.isfinite(score):
            die(f"{path}:{n}: score {r['score']!r} is not finite")
        # str(): CSV ids are strings, JSONL ids may be ints; 1 and "1" must be the same task
        rows.append({"task": str(r["task"]), "variant": str(r["variant"]), "score": score})
    if not rows:
        die(f"{path}: no rows")
    return rows


def by_task(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """{task: {variant: [score of each attempt]}}."""
    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        groups.setdefault(r["task"], {}).setdefault(r["variant"], []).append(r["score"])
    return groups


# ---------- estimators ----------
# n = attempts on a task, c = how many passed, k = attempts we imagine drawing from those n.

def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k attempts passes."""
    return 1 - math.comb(n - c, k) / math.comb(n, k)  # comb(n-c, k) = ways to pick k that all fail


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Chance that all k attempts pass."""
    return math.comb(c, k) / math.comb(n, k)  # comb(c, k) = ways to pick k that all pass


# ---------- bootstrap ----------

def bootstrap_ci(values: list[float], draws: list[list[int]], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of the mean. Each draw is a list of task indices picked with replacement."""
    means = sorted(sum(values[i] for i in d) / len(d) for d in draws)
    last = len(means) - 1
    return means[math.floor(alpha / 2 * last)], means[math.ceil((1 - alpha / 2) * last)]


def paired(a: list[float], b: list[float], draws: list[list[int]]) -> tuple[float, tuple[float, float], tuple[int, int, int]]:
    """b minus a, task by task. Resampling a task keeps its a and b together; that is the pairing."""
    diffs = [y - x for x, y in zip(a, b)]
    up = sum(d > 0 for d in diffs)
    down = sum(d < 0 for d in diffs)
    return sum(diffs) / len(diffs), bootstrap_ci(diffs, draws), (up, down, len(diffs) - up - down)


# ---------- CLI ----------

def _fmt(r: dict, sign: str = "") -> str:
    """{'value': .75, 'lo': .5, 'hi': 1} -> '0.750 [0.500, 1.000]'. sign='+' always shows the sign."""
    return f"{r['value']:{sign}.3f} [{r['lo']:{sign}.3f}, {r['hi']:{sign}.3f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--k", type=int, action="append", help="repeatable; default: 1 and attempts per task")
    ap.add_argument("--threshold", type=float, default=1.0, help="score >= this counts as a pass")
    ap.add_argument("--boot", type=int, default=10000, help="number of bootstrap draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", help="variant the other is compared against (default: first in file)")
    args = ap.parse_args(argv)
    if args.boot < 1:
        die("--boot must be >= 1")

    rows = load_rows(args.file)
    groups = by_task(rows)
    tasks = sorted(groups)  # one fixed task order, so per-task lists line up across variants
    variants = list(dict.fromkeys(r["variant"] for r in rows))  # file order, no repeats
    if args.baseline is not None:
        if args.baseline not in variants:
            die(f"--baseline {args.baseline!r} not in variants {variants}")
        variants.remove(args.baseline)
        variants.insert(0, args.baseline)

    missing = [(t, v) for t in tasks for v in variants if v not in groups[t]]
    if missing:
        die(f"{len(missing)} (task, variant) pair(s) have no rows, e.g. {missing[:5]}")
    n_min = min(len(scores) for t in tasks for scores in groups[t].values())
    ks = sorted(set(args.k or [1, n_min]))
    if ks[0] < 1 or ks[-1] > n_min:
        die(f"every k must be 1..{n_min} (fewest attempts in any task/variant); got {ks}")

    # per_task[variant][metric] = [value on each task, in `tasks` order]
    per_task: dict[str, dict[str, list[float]]] = {v: {} for v in variants}
    for v in variants:
        for t in tasks:
            scores = groups[t][v]
            n, c = len(scores), sum(s >= args.threshold for s in scores)
            cell = {"mean": sum(scores) / n}
            cell |= {f"pass@{k}": pass_at_k(n, c, k) for k in ks}
            cell |= {f"pass^{k}": pass_pow_k(n, c, k) for k in ks}
            for name, value in cell.items():
                per_task[v].setdefault(name, []).append(value)
    names = list(per_task[variants[0]])

    # One set of draws for everything: every interval resamples the same tasks.
    rng = random.Random(args.seed)
    T = len(tasks)
    draws = [rng.choices(range(T), k=T) for _ in range(args.boot)]

    result: dict[str, dict[str, dict]] = {v: {} for v in variants}
    for v in variants:
        for name in names:
            vals = per_task[v][name]
            lo, hi = bootstrap_ci(vals, draws)
            result[v][name] = {"value": sum(vals) / T, "lo": lo, "hi": hi, "n": T}

    diff = None
    if len(variants) == 2:
        a, b = variants
        diff = {"a": a, "b": b, "metrics": {}}
        for name in names:
            d, (lo, hi), (up, down, same) = paired(per_task[a][name], per_task[b][name], draws)
            diff["metrics"][name] = {"value": d, "lo": lo, "hi": hi, "n": T, "up": up, "down": down, "same": same}

    # ---- table ----
    all_scores = [r["score"] for r in rows]
    kind = "binary" if set(all_scores) <= {0.0, 1.0} else "non-binary"
    print(f"metrics: {args.file}   tasks (n): {T}   boot: {args.boot}   seed: {args.seed}")
    print(f"scores: {kind}, range [{min(all_scores):g}, {max(all_scores):g}], pass = score >= {args.threshold:g}")
    w = max(22, *(len(v) for v in variants))
    label = f"diff = {variants[-1]} - {variants[0]}"
    wd = max(w + 1, len(label))  # diff column: wide enough for its values and its label
    head = f"{'metric':8s}  " + "  ".join(f"{v:{w}s}" for v in variants)
    if diff:
        head += f"  {label:{wd}s}  up/down/same"
    print(head)
    print("-" * len(head))
    for name in names:
        line = f"{name:8s}  " + "  ".join(f"{_fmt(result[v][name]):{w}s}" for v in variants)
        if diff:
            m = diff["metrics"][name]
            line += f"  {_fmt(m, '+'):{wd}s}  {m['up']}/{m['down']}/{m['same']}"
        print(line)

    out = Path(args.file).parent / "metrics.json"
    doc = {"input": str(args.file), "threshold": args.threshold, "boot": args.boot, "seed": args.seed,
           "ks": ks, "n_tasks": T, "baseline": variants[0], "variants": result, "diff": diff}
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
