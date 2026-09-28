"""PLOT. One figure from the toolkit's outputs, for the deck. PNG next to the input unless --out.

    python plot.py --kind means    results/judge/metrics.json   [--metric mean]
    python plot.py --kind paired   results/judge/scores.jsonl   [--metrics results/judge/metrics.json]
    python plot.py --kind taxonomy results/classify/scores.json

means:    one bar per model, the metric's value with its bootstrap interval; n in the title.
paired:   two models: per-task difference (B minus A, mean over samples), sorted, zero line, the mean
          difference with its interval as a band when metrics.json is beside it; up/down/same in the title.
taxonomy: judge.py's scores.json: counts per category, one bar group per variant.
Plain style: no grid, no colour beyond one per series, every number in the title or on the bar.
Beside every PNG, the same name with .json: the exact series drawn (labels, values, intervals, counts),
so the deck can draw the chart natively (Claude Design) from the numbers; the PNG is the fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import matplotlib
matplotlib.use("Agg")  # files only, never a window
import matplotlib.pyplot as plt


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


def _load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        die(f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _data(out: Path, doc: dict) -> None:
    """The numbers a PNG was drawn from, beside it as JSON, for a native chart in the deck."""
    out.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def means(metrics_path: str | Path, metric: str = "mean", out: Path | None = None) -> Path:
    m = _load(metrics_path)
    variants = m["variants"]
    if not variants:
        die("metrics.json has no variants")
    names = list(variants)
    if metric not in variants[names[0]]:
        die(f"metric {metric!r} not in metrics.json; have {list(variants[names[0]])}")
    vals = [variants[v][metric]["value"] for v in names]
    lo = [vals[i] - variants[v][metric]["lo"] for i, v in enumerate(names)]
    hi = [variants[v][metric]["hi"] - vals[i] for i, v in enumerate(names)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, vals, yerr=[lo, hi], capsize=6, color="#4a6fa5")
    for i, v in enumerate(vals):  # label above the whisker top, never on it
        ax.text(i, v + hi[i] + 0.02, f"{v:.3f}  [{variants[names[i]][metric]['lo']:.3f}, {variants[names[i]][metric]['hi']:.3f}]",
                ha="center", fontsize=9)
    ax.set_ylim(0, min(1.0, max(v + h for v, h in zip(vals, hi)) + 0.12) if max(vals) <= 1 else None)
    _style(ax, f"{metric} per model, 95% task-bootstrap interval, n = {m['n_tasks']} tasks")
    out = out or Path(metrics_path).parent / f"plot_means_{metric.replace('@', '_at_').replace('^', '_pow_')}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    _data(out, {"kind": "means", "metric": metric, "n_tasks": m["n_tasks"], "interval": "95% task bootstrap",
                "series": [{"model": v, "value": variants[v][metric]["value"], "lo": variants[v][metric]["lo"],
                            "hi": variants[v][metric]["hi"]} for v in names]})
    return out


def paired(scores_path: str | Path, metrics_path: str | Path | None = None, out: Path | None = None) -> Path:
    path = Path(scores_path)
    if not path.exists():
        die(f"not found: {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    per: dict[str, dict[str, list[float]]] = {}  # task -> variant -> sample scores
    for r in rows:
        per.setdefault(str(r["task"]), {}).setdefault(r["variant"], []).append(float(r["score"]))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    if len(variants) != 2:
        die(f"paired needs exactly two variants, got {variants}")
    a, b = variants
    diffs = sorted((sum(v[b]) / len(v[b]) - sum(v[a]) / len(v[a]), t) for t, v in per.items() if a in v and b in v)
    if not diffs:
        die("no task has both variants")
    d = [x for x, _ in diffs]
    up, down, same = sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)
    mean = sum(d) / len(d)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(d)), d, color=["#4a6fa5" if x > 0 else "#b55a4a" if x < 0 else "#bbbbbb" for x in d], width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    band = ""
    mp = Path(metrics_path) if metrics_path else path.parent / "metrics.json"
    if mp.exists():
        dm = (_load(mp).get("diff") or {}).get("metrics", {}).get("mean")
        if dm:
            ax.axhspan(dm["lo"], dm["hi"], color="#4a6fa5", alpha=0.15)
            band = f", 95% CI [{dm['lo']:+.3f}, {dm['hi']:+.3f}]"
    ax.axhline(mean, color="#4a6fa5", linewidth=1.2, linestyle="--")
    ax.set_xticks([]); ax.set_xlabel(f"tasks, sorted by difference (n = {len(d)})")
    ax.set_ylabel(f"{b} minus {a}")
    _style(ax, f"per-task difference, mean {mean:+.3f}{band}\nup {up} / down {down} / same {same} of {len(d)} tasks")
    out = out or path.parent / "plot_paired.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    _data(out, {"kind": "paired", "a": a, "b": b, "n_tasks": len(d), "mean_diff": mean,
                "ci": [dm["lo"], dm["hi"]] if band else None, "up": up, "down": down, "same": same,
                "per_task": [{"task": t, "diff": x} for x, t in diffs]})
    return out


def taxonomy(scores_path: str | Path, out: Path | None = None) -> Path:
    doc = _load(scores_path)
    models = doc.get("models") or {}
    if not models:
        die("scores.json has no models (made by judge.py score?)")
    model, block = next(iter(models.items()))
    cats = doc["categories"]
    variants = list(block["variants"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.8 / len(variants)
    for j, v in enumerate(variants):
        counts = [block["variants"][v]["counts"][c] for c in cats]
        xs = [i + j * width for i in range(len(cats))]
        ax.bar(xs, counts, width=width, label=f"{v} (n = {block['variants'][v]['n']})")
        for x, c in zip(xs, counts):
            if c:
                ax.text(x, c + 0.1, str(c), ha="center", fontsize=8)
    ax.set_xticks([i + width * (len(variants) - 1) / 2 for i in range(len(cats))])
    ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, f"failure categories, classifier {model}")
    out = out or Path(scores_path).parent / "plot_taxonomy.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    _data(out, {"kind": "taxonomy", "classifier": model, "categories": cats,
                "series": [{"variant": v, "n": block["variants"][v]["n"],
                            "counts": [block["variants"][v]["counts"][c] for c in cats]} for v in variants]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["means", "paired", "taxonomy"])
    ap.add_argument("file", help="metrics.json (means), scores.jsonl (paired) or judge.py scores.json (taxonomy)")
    ap.add_argument("--metric", default="mean", help="means: which metric (mean, pass@k, pass^k)")
    ap.add_argument("--metrics", help="paired: metrics.json for the CI band (default: beside the scores file)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if a.kind == "means":
        p = means(a.file, a.metric, out)
    elif a.kind == "paired":
        p = paired(a.file, a.metrics, out)
    else:
        p = taxonomy(a.file, out)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
