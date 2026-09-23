"""pass@k over tbench trajectory traces, with task-level bootstrap CIs.

Usage:
    python passk.py                        # Qwen3p5-397B, k=3, 10000 bootstrap draws
    python passk.py --model Qwen3p6-35B    # same for 35B dirs
    python passk.py --k 1                  # pass@1

Score source: each traj_*.json -> trajectory.trajectory_output.score (binary 0/1).
Trajectories that errored before grading carry score 0.0 and count as failures;
their count is reported so it is visible.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

TRACES = Path(__file__).parent / "tbench-traces"
VARIANTS = ("trained", "untrained")


def load_variant(root: Path):
    """Return ({task_dir_name: [score, ...]}, n_errored). Scores ordered by file name."""
    tasks, errored = {}, 0
    for f in sorted(root.glob("tasks/*/traj_*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))["trajectory"]
        score = float((t.get("trajectory_output") or {}).get("score") or 0.0)
        if t.get("trajectory_status") != "completed":
            errored += 1
        tasks.setdefault(f.parent.name, []).append(score)
    if not tasks:
        sys.exit(f"no trajectories under {root}")
    return tasks, errored


def pass_at_k(scores, k):
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    n, c = len(scores), sum(1 for s in scores if s == 1.0)
    if k > n:
        raise ValueError(f"k={k} > n={n} samples")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_ci(draws, values, alpha=0.05):
    """Percentile CI of the mean of `values` under resampled index sets `draws`."""
    means = sorted(sum(values[i] for i in d) / len(d) for d in draws)
    lo, hi = int(alpha / 2 * len(means)), int((1 - alpha / 2) * len(means)) - 1
    return means[lo], means[hi]


def make_draws(n, B, seed):
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(B)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3p5-397B")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap draws")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_task = {}   # variant -> {task: pass@k}
    errored = {}
    for v in VARIANTS:
        tasks, errored[v] = load_variant(TRACES / f"{args.model}_{v}_traces")
        per_task[v] = {t: pass_at_k(s, args.k) for t, s in tasks.items()}
        print(f"\n== {v}: {len(tasks)} tasks, {sum(len(s) for s in tasks.values())} trajectories")
        for t, s in tasks.items():
            print(f"  {per_task[v][t]:.3f}  {s}  {t}")

    names = sorted(per_task["trained"])
    if names != sorted(per_task["untrained"]):
        sys.exit("trained/untrained task sets differ; paired diff undefined")
    vals = {v: [per_task[v][t] for t in names] for v in VARIANTS}
    draws = make_draws(len(names), args.boot, args.seed)

    print(f"\n== {args.model}  pass@{args.k}  (95% CI, task-level bootstrap B={args.boot}, seed={args.seed})")
    for v in VARIANTS:
        m = sum(vals[v]) / len(names)
        lo, hi = bootstrap_ci(draws, vals[v])
        print(f"  {v:<10} {m:.3f}  [{lo:.3f}, {hi:.3f}]  n_tasks={len(names)} errored_trajs={errored[v]}")
    diff = [a - b for a, b in zip(vals["trained"], vals["untrained"])]
    lo, hi = bootstrap_ci(draws, diff)
    print(f"  {'diff':<10} {sum(diff) / len(names):+.3f}  [{lo:+.3f}, {hi:+.3f}]  (trained - untrained, paired)")

    out = Path(__file__).parent / "results" / f"passk_{args.model}_k{args.k}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "trained", "untrained", "diff"])
        for t, a, b, d in zip(names, vals["trained"], vals["untrained"], diff):
            w.writerow([t, a, b, d])
    print(f"\nper-task CSV -> {out}")


if __name__ == "__main__":
    main()
