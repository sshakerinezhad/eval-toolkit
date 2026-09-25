"""JUDGE. Turn agent traces into LLM-judge tasks for run.py, then score the judge's replies.

    python judge.py build DIR [DIR ...] --prompt FILE --out FILE [--select all|passes|fails] [--turns 30] [--max-chars 60000]
    python judge.py score RAW [--labels labels.csv]

build: prompt file -> category names -> for each trace folder, iter_trajectories() -> keep the selected
ones -> excerpt (task, last N turns, test output; hard-capped) -> one task row per trajectory:
{id, prompt, task, variant, trajectory_id, categories}. run.py passes the non-prompt fields through as
metadata, so the category list travels with each reply. No LLM calls here.

score: raw.jsonl from run.py -> parse each reply as {"category", "why"} (bad JSON or unknown category
-> "unparsed") -> counts and share per category per variant, one table per judge model
-> optional agreement with hand labels -> print + write scores.json next to RAW.

Pass = score >= 1.0; everything else (including no score, e.g. the sandbox died) is a fail.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator, NoReturn

UNPARSED = "unparsed"
CUT = "\n[... cut to fit --max-chars ...]\n"
CATEGORY_LINE = re.compile(r"^- ([a-z_]+):", re.M)
FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.S)


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


# ---------- prompt ----------

def load_prompt(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        die(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("<!--"):  # provenance header, never sent to the model
        text = text.split("-->", 1)[1].strip()
    if not text:
        die(f"{path}: prompt is empty")
    return text


def categories(prompt: str) -> list[str]:
    """Category names = lines shaped '- name: description'."""
    return CATEGORY_LINE.findall(prompt)


# ---------- trace loader (the only code that knows the trace layout) ----------

def iter_trajectories(dir: str | Path) -> Iterator[dict]:
    """tbench layout: trajectories_index.json + tasks/<task_name>__<task_id>/<trajectory_id>.json.

    Yields {id, task, variant, score, instruction, turns: [{role, content}], test_output: str}.
    task = task name (the key metrics.py groups on), instruction = the task text the agent got,
    variant = folder name.
    """
    dir = Path(dir)
    index_path = dir / "trajectories_index.json"
    if not index_path.exists():
        die(f"{dir}: no trajectories_index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, list):
        die(f"{index_path}: expected a list")
    for e in index:
        path = dir / "tasks" / f"{e['task_name']}__{e['task_id']}" / f"{e['trajectory_id']}.json"
        if not path.exists():
            die(f"trajectory file missing: {path}")
        t = json.loads(path.read_text(encoding="utf-8"))["trajectory"]
        out = t.get("trajectory_output") or {}

        instruction = next((m["content"] for m in t.get("initial_messages", []) if m["role"] == "user"), None)
        if not isinstance(instruction, str):
            die(f"{path}: no user message with the task in initial_messages")
        msgs = t.get("trajectory_messages") or []
        if msgs and msgs[0]["role"] == "user":
            msgs = msgs[1:]  # harness preamble that repeats the task plus format rules; noise for the judge
        turns = [{"role": m["role"], "content": _text(m.get("content"))} for m in msgs]

        # index final_score is None when the run crashed; the file still has the grader's score
        score = out.get("score") if out.get("score") is not None else e.get("final_score")
        yield {"id": e["trajectory_id"], "task": e["task_name"], "variant": dir.name, "score": score,
               "instruction": instruction, "turns": turns, "test_output": _test_output(t, out)}


def _text(content) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _test_output(t: dict, out: dict) -> str:
    """tbench keeps no raw test log, only pass/fail per test name plus any run error."""
    lines = [f"status: {t.get('trajectory_status')}, outcome: {out.get('outcome')}, score: {out.get('score')}"]
    if out.get("tests_total") is not None:
        lines.append(f"tests: {out.get('tests_passed')}/{out.get('tests_total')} passed")
    statuses = out.get("test_statuses") or {}
    for name, status in sorted(statuses.items(), key=lambda kv: (kv[1] == "pass", kv[0])):  # fails first
        lines.append(f"{str(status).upper()} {name}")
    if not statuses:
        lines.append("(no per-test results)")
    for key in ("exception_type", "error_message", "exception_message"):
        if out.get(key):
            lines.append(f"{key}: {out[key]}")
    return "\n".join(lines)


# ---------- excerpt ----------

def _clip(text: str, room: int, keep_end: bool = False) -> str:
    """Shorten text to at most `room` chars, marking the cut. keep_end keeps the tail instead of the head."""
    if len(text) <= room:
        return text
    if room <= len(CUT):
        return "" if room <= 0 else (text[-room:] if keep_end else text[:room])
    n = room - len(CUT)
    return CUT + text[-n:] if keep_end else text[:n] + CUT


def excerpt(record: dict, turns: int, max_chars: int) -> str:
    """Task, then the last `turns` turns, then test output; never longer than max_chars.

    Priority when over the cap: test output, then task, then turns newest first (the end of a
    trajectory is where it fails).
    """
    tests = _clip(f"## Test output\n{record['test_output']}", max_chars)
    room = max_chars - len(tests)
    task = _clip(f"## Task\n{record['instruction']}\n\n", room)
    room -= len(task)

    all_turns = record["turns"]
    start = max(0, len(all_turns) - turns)
    blocks = [(f"### [{i}] {t['role']}\n", f"{t['content']}\n\n")
              for i, t in enumerate(all_turns[start:], start + 1)]
    head = f"## Last {len(blocks)} of {len(all_turns)} turns\n\n"

    if len(head) + sum(len(h) + len(c) for h, c in blocks) <= room:
        kept = [h + c for h, c in blocks]
        marker = ""
    else:
        marker_room = len(f"[... {len(blocks)} earlier turn(s) cut to fit --max-chars ...]\n\n")
        left = room - len(head) - marker_room
        kept = []
        for h, c in reversed(blocks):
            if len(h) + len(c) > left:
                break
            kept.insert(0, h + c)
            left -= len(h) + len(c)
        if not kept and blocks:  # newest turn alone is too big: keep its tail
            h, c = blocks[-1]
            kept = [h + _clip(c, left - len(h), keep_end=True)]
        marker = f"[... {len(blocks) - len(kept)} earlier turn(s) cut to fit --max-chars ...]\n\n"
    body = _clip(head + marker + "".join(kept), room)  # only bites when the cap is tiny
    return task + body + tests


# ---------- build ----------

def build(dirs: list[str | Path], prompt_path: str | Path, out: str | Path, select: str = "fails",
          turns: int = 30, max_chars: int = 60000) -> int:
    """Write one run.py task row per selected trajectory. Returns rows written."""
    if select not in ("all", "passes", "fails"):
        die(f"select must be all, passes or fails, got {select!r}")
    prompt = load_prompt(prompt_path)
    cats = categories(prompt)
    if not cats:
        die(f"{prompt_path}: no category lines ('- name: description') found")

    rows, seen = [], set()
    counts: dict[str, list[int]] = {}  # variant -> [selected, total]
    for d in dirs:
        for r in iter_trajectories(d):
            if r["id"] in seen:
                die(f"duplicate trajectory {r['id']} (same folder given twice?)")
            seen.add(r["id"])
            c = counts.setdefault(r["variant"], [0, 0])
            c[1] += 1
            passed = r["score"] is not None and r["score"] >= 1.0
            if (select == "passes" and not passed) or (select == "fails" and passed):
                continue
            c[0] += 1
            rows.append({"id": r["id"], "prompt": prompt + "\n\n" + excerpt(r, turns, max_chars),
                         "task": r["task"], "variant": r["variant"],
                         "trajectory_id": r["id"], "categories": cats})
    if not rows:
        die(f"no trajectories selected (--select {select})")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for v, (sel, total) in counts.items():
        print(f"  {v:40s} {sel:4d} of {total:4d} selected ({select})")
    print(f"wrote {len(rows)} task(s) -> {out}")
    return len(rows)


# ---------- score ----------

def parse_response(text: str | None, cats: list[str]) -> tuple[str, str]:
    """Reply -> (category, why). Anything but one JSON object with a known category -> ("unparsed", reply)."""
    if not isinstance(text, str):
        return UNPARSED, "(no reply)"  # the call itself failed
    s = text.strip()
    m = FENCE.match(s)  # a ```json fence around the object is a formatting habit, not a wrong answer
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return UNPARSED, text.strip()[:200]
    if not isinstance(obj, dict) or obj.get("category") not in cats:
        return UNPARSED, text.strip()[:200]
    why = obj.get("why")
    return obj["category"], why if isinstance(why, str) else ""


def load_labels(path: str | Path, cats: list[str]) -> dict[str, str]:
    """labels.csv (trajectory_id, category) -> {trajectory_id: category}."""
    path = Path(path)
    if not path.exists():
        die(f"labels not found: {path}")
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as f:  # -sig: Excel CSVs start with a BOM
        for n, r in enumerate(csv.DictReader(f), 2):  # line 1 is the header
            tid, cat = (r.get("trajectory_id") or "").strip(), (r.get("category") or "").strip()
            if not tid or not cat:
                die(f"{path}:{n}: row needs trajectory_id, category")
            if cat not in cats:
                die(f"{path}:{n}: unknown category {cat!r}; known: {cats}")
            if tid in labels:
                die(f"{path}:{n}: duplicate trajectory_id {tid}")
            labels[tid] = cat
    return labels


def score(raw: str | Path, labels: str | Path | None = None) -> dict:
    """Count judge categories per variant (one table per judge model), compare to labels, write scores.json."""
    raw = Path(raw)
    if not raw.exists():
        die(f"input not found: {raw}")
    cats: list[str] | None = None
    judged: dict[str, dict[str, dict]] = {}  # model -> {trajectory_id: {variant, category, why}}
    for n, line in enumerate(raw.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            die(f"{raw}:{n}: invalid JSON: {e}")
        meta = row.get("metadata") or {}
        if any(k not in meta for k in ("trajectory_id", "variant", "categories")):
            die(f"{raw}:{n}: metadata lacks trajectory_id/variant/categories; tasks not made by judge.py build?")
        if cats is None:
            cats = meta["categories"]
        elif meta["categories"] != cats:
            die(f"{raw}:{n}: category list differs from earlier rows; one prompt per run")
        model = f"{row.get('provider')}:{row.get('model')}"
        tid = meta["trajectory_id"]
        if tid in judged.setdefault(model, {}):
            die(f"{raw}:{n}: {model} judged {tid} twice; n_samples > 1 is not supported")
        category, why = parse_response(row.get("response"), cats)
        judged[model][tid] = {"variant": meta["variant"], "category": category, "why": why}
    if cats is None:
        die(f"{raw}: no rows")
    gold = load_labels(labels, cats) if labels else None

    names = cats + [UNPARSED]
    doc = {"input": str(raw), "labels": None if labels is None else str(labels), "categories": names, "models": {}}
    for model, rows in judged.items():
        variants = list(dict.fromkeys(r["variant"] for r in rows.values()))  # file order, no repeats
        per_variant = {}
        for v in variants:
            mine = [r["category"] for r in rows.values() if r["variant"] == v]
            counts = {c: mine.count(c) for c in names}
            per_variant[v] = {"n": len(mine), "counts": counts,
                              "share": {c: counts[c] / len(mine) for c in names}}

        agreement = None
        if gold is not None:
            labelled = [t for t in gold if t in rows]
            disagreements = [{"trajectory_id": t, "mine": gold[t], "judge": rows[t]["category"], "why": rows[t]["why"]}
                             for t in labelled if rows[t]["category"] != gold[t]]
            matches = len(labelled) - len(disagreements)
            agreement = {"matches": matches, "labelled": len(labelled),
                         "value": matches / len(labelled) if labelled else None,
                         "not_judged": len(gold) - len(labelled), "disagreements": disagreements}
        doc["models"][model] = {"n": len(rows), "variants": per_variant, "agreement": agreement}

        # ---- table ----
        print(f"judge: {model}   replies: {len(rows)}   input: {raw}")
        w = max(12, *(len(v) for v in variants))
        head = f"{'category':24s}  " + "  ".join(f"{v:>{w}s}" for v in variants)
        print(head)
        print("-" * len(head))
        for c in names:
            print(f"{c:24s}  " + "  ".join(
                f"{per_variant[v]['counts'][c]:>{w - 9}d} ({per_variant[v]['share'][c]:6.1%})" for v in variants))
        print(f"{'total':24s}  " + "  ".join(f"{per_variant[v]['n']:>{w}d}" for v in variants))
        if agreement is not None:
            a = agreement
            value = "n/a" if a["value"] is None else f"{a['value']:.3f}"
            print(f"agreement with labels: {a['matches']}/{a['labelled']} = {value}"
                  + (f"   ({a['not_judged']} label(s) not in this run, skipped)" if a["not_judged"] else ""))
            for d in a["disagreements"]:
                why = " ".join(d["why"].split())[:120]
                print(f"  {d['trajectory_id']}  mine={d['mine']}  judge={d['judge']}  why={why}")
        print()

    out = raw.parent / "scores.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return doc


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="trace folders -> task JSONL for run.py")
    b.add_argument("dirs", nargs="+", help="trace folders; folder name = variant")
    b.add_argument("--prompt", required=True, help="judge prompt file; lists categories as '- name: ...'")
    b.add_argument("--out", required=True, help="task JSONL to write")
    b.add_argument("--select", choices=["all", "passes", "fails"], default="fails")
    b.add_argument("--turns", type=int, default=30, help="how many of the last turns (messages) the judge sees")
    b.add_argument("--max-chars", type=int, default=60000, help="hard cap on the excerpt")
    s = sub.add_parser("score", help="run.py raw.jsonl -> category counts per variant (+ label agreement)")
    s.add_argument("raw")
    s.add_argument("--labels", help="CSV with columns trajectory_id, category")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        if args.turns < 0:
            die("--turns must be >= 0")
        if args.max_chars < 1:
            die("--max-chars must be >= 1")
        build(args.dirs, args.prompt, args.out, args.select, args.turns, args.max_chars)
    else:
        score(args.raw, args.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
