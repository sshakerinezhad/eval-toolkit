"""DELIVER. Package a run for the email: python deliver.py results/<judge> [--also FILE ...] [--out deliverables]

Copies what a grader needs from the judge's results folder into deliverables/: scores.jsonl as results.jsonl,
metrics.json, every plot_*.png and plot_*.json, report.md, failures.md and judge_agreement.json when present,
plus any --also file. Zips the folder. Prints the email body: subject, the headline (report.md's first line
if it has one, else a bracket to fill), the number with its interval and n from metrics.json, the paired
difference when two models, the repo link (git remote), the attachment list. Nothing is sent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

WANT = ["metrics.json", "report.md", "failures.md", "judge_agreement.json"]
SUBJECT = "LLM Evaluation and Analysis Submission - Shayan Shakeri"


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


def collect(src: Path, out: Path, also: list[str]) -> list[Path]:
    """Copy the deliverable files from src into out (emptied first). Returns the copied paths."""
    if not (src / "scores.jsonl").exists():
        die(f"{src}/scores.jsonl not found; run grade.py score first")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied = [shutil.copy(src / "scores.jsonl", out / "results.jsonl")]
    for name in WANT:
        if (src / name).exists():
            copied.append(shutil.copy(src / name, out / name))
    for p in sorted(src.glob("plot_*.png")) + sorted(src.glob("plot_*.json")):
        copied.append(shutil.copy(p, out / p.name))
    for extra in also:
        p = Path(extra)
        if not p.exists():
            die(f"--also file not found: {p}")
        copied.append(shutil.copy(p, out / p.name))
    return [Path(c) for c in copied]


def headline(out: Path) -> str:
    r = out / "report.md"
    if r.exists():
        first = next((l.strip() for l in r.read_text(encoding="utf-8").splitlines() if l.strip()), "")
        if first.startswith("#") and "[" not in first:
            return first.lstrip("# ").strip()
    return "[FINDING: one sentence, from facts.md]"


def numbers(out: Path) -> list[str]:
    """Lines for the email from metrics.json: each model's mean with CI and n, the paired diff if present."""
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8")) if (out / "metrics.json").exists() else None
    if not m:
        return ["[NUMBER with 95% CI and n: metrics.json missing]"]
    lines = []
    for v, block in m["variants"].items():
        x = block["mean"]
        lines.append(f"{v}: mean {x['value']:.3f} (95% CI {x['lo']:.3f} to {x['hi']:.3f}, n = {x['n']} tasks)")
    d = (m.get("diff") or {}).get("metrics", {}).get("mean")
    if d:
        lines.append(f"{m['diff']['b']} minus {m['diff']['a']}: {d['value']:+.3f} (95% CI {d['lo']:+.3f} to {d['hi']:+.3f}), "
                     f"up {d['up']} / down {d['down']} / same {d['same']} of {d['n']} tasks")
    je = m.get("judge_errors")
    if je:
        lines.append(f"judge errors: {je.get('timeout', 0)} timeouts, {je.get('api_error', 0)} api, {je.get('unparsed', 0)} unparsed (counted as fail)")
    return lines


def repo_link() -> str:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout.strip()
        return url.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "[REPO LINK]"


def email(out: Path, zip_path: Path, copied: list[Path]) -> str:
    body = [f"Subject: {SUBJECT}", "", headline(out), "", *numbers(out), "",
            f"Code and results: {repo_link()}", "",
            "Attached: " + ", ".join([zip_path.name] + [p.name for p in copied]), "",
            "Shayan"]
    return "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the judge's results folder (has scores.jsonl and metrics.json)")
    ap.add_argument("--also", action="append", default=[], help="extra file to include (repeatable), e.g. the slides PDF")
    ap.add_argument("--out", default="deliverables")
    a = ap.parse_args(argv)
    out = Path(a.out)
    copied = collect(Path(a.src), out, a.also)
    zip_path = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    print(f"copied {len(copied)} file(s) -> {out}/ ; zipped -> {zip_path} ({zip_path.stat().st_size:,} bytes)\n")
    print(email(out, zip_path, copied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
