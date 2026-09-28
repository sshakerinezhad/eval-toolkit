"""PEEK. Look at a data file before touching it: python peek.py FILE [--n 5] [--seed 0] [--chars 200]

jsonl (one object per line), json (a list, or an object of rows), or csv. Prints: row count, the fields with
how many rows have each, the longest prompt-like field (prompt, trajectory, response, context...) with a warning
when rows exceed run.py's 200,000-char attachment cap, attachment counts when a field looks like a file list,
then n random rows, every field with its length and its text; a long field shows only its last --chars
characters, since the end is where an answer usually is. Read only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import NoReturn

PROMPT_KEYS = ("prompt", "question", "input", "task", "instruction", "trajectory", "response", "context")
CAP = 200_000  # run.py's attachment cap (attachments.MAX_CHARS); a field over it means a decision, not a default
ATTACH_KEYS = ("attachments", "file attachments", "files", "documents")


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


def load(path: str | Path) -> list[dict]:
    """File -> list of row dicts. Unknown suffix -> die."""
    path = Path(path)
    if not path.exists():
        die(f"not found: {path}")
    ext = path.suffix.lower()
    if ext == ".jsonl":
        rows = []
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    die(f"{path}:{n}: invalid JSON: {e}")
    elif ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = [{"_key": k, **v} if isinstance(v, dict) else {"_key": k, "value": v} for k, v in obj.items()]
        else:
            die(f"{path}: JSON is neither a list nor an object")
    elif ext in (".csv", ".tsv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
    else:
        die(f"{path}: {ext or 'no extension'} not supported (jsonl, json, csv, tsv)")
    if not rows:
        die(f"{path}: no rows")
    rows = [r if isinstance(r, dict) else {"value": r} for r in rows]
    return rows


def _text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _attachments(v) -> int:
    if isinstance(v, list):
        return len(v)
    if isinstance(v, str):
        return len([p for p in v.replace(",", "\n").splitlines() if p.strip()])
    return 0


def summary(rows: list[dict]) -> list[str]:
    fields: dict[str, int] = {}
    for r in rows:
        for k in r:
            fields[k] = fields.get(k, 0) + 1
    out = [f"rows: {len(rows)}", "fields (rows having it): " + ", ".join(f"{k} ({c})" for k, c in fields.items())]
    pk = next((k for k in fields if k.lower() in PROMPT_KEYS), None)
    if pk:
        lens = [len(_text(r.get(pk, ""))) for r in rows]
        i = max(range(len(rows)), key=lambda j: lens[j])
        out.append(f"longest {pk}: {lens[i]:,} chars (row {i}), mean {sum(lens) // len(lens):,}, shortest {min(lens):,}")
        over = sum(l > CAP for l in lens)
        if over:
            out.append(f"OVER THE {CAP:,}-CHAR CAP: {over} row(s); run.py stops on files this size. Decide: "
                       f"--max-attachment-chars N, filter, or drop them, and say which in the deliverable.")
    ak = next((k for k in fields if k.lower() in ATTACH_KEYS), None)
    if ak:
        counts = [_attachments(r.get(ak)) for r in rows]
        out.append(f"attachments ({ak}): {sum(counts)} across {sum(c > 0 for c in counts)} rows, max {max(counts)} per row")
    ids = next((k for k in fields if k.lower() in ("id", "task_id", "task id")), None)
    if ids:
        vals = [json.dumps(r.get(ids)) for r in rows]
        dups = len(vals) - len(set(vals))
        out.append(f"{ids}: {len(set(vals))} distinct" + (f", {dups} DUPLICATE" if dups else ""))
    return out


def show(rows: list[dict], n: int, seed: int, chars: int) -> list[str]:
    pick = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    out = []
    for i in pick:
        out.append(f"\n--- row {i} ---")
        for k, v in rows[i].items():
            t = _text(v)
            one = " ".join(t.split())
            if len(one) > chars:  # long field: the end, where the answer usually is
                out.append(f"{k} [{len(t):,} chars]: ...{one[-chars:]}")
            else:
                out.append(f"{k} [{len(t):,} chars]: {one}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", type=int, default=200)
    a = ap.parse_args(argv)
    rows = load(a.file)
    print("\n".join(summary(rows) + show(rows, a.n, a.seed, a.chars)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
