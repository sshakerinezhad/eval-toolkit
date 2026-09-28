"""GRADE. Score APEX-v1 runner answers against each task's rubric criteria with an LLM judge.

    python grade.py build [--runs RAW] [--data CSV] [--prompt FILE] [--out FILE] [--gold JSON]
    python grade.py score RAW [--tasks FILE] [--baseline MODEL] [--labels judge_labels.csv]
    python grade.py label TASKS [--n 40] [--out CSV] [--seed 0]
    python grade.py agree RAW_A RAW_B
    python grade.py failures RAW [--out DIR] [--prompt FILE] [--excerpt 600] [--label [--n 20] [--seed 0]]

build: runner raw.jsonl (last line per task+model+sample wins) + train.csv rubrics -> one run.py task
per (task, model, sample, criterion). With n_samples 3 every sample is graded, not just the last. User message = judge prompt (Mercor's APEX grader prompt, scattergun guard
included; run.py configs can't load a system prompt from a file) + <TASK_PROMPT> (raw prompt, no
attachments: criteria state their expected values) + <RESPONSE> + <CRITERION>. No LLM calls here.

score: judge raw.jsonl from run.py -> parse {"rationale", "is_criteria_true"} per criterion
(a judge call that timed out, errored or did not parse is a judge error, kept in its own bucket:
timeout / api_error / unparsed) -> score = met/total per (task, model, sample), errors as fail, unweighted
like Mercor's scoring, plus score_errors_dropped = met/(total - errors) -> scores.jsonl (task, variant,
sample, score, score_errors_dropped, errors) next to RAW, one row per sample -> metrics.py: mean over samples and tasks with bootstrap CI per model, pass@k / pass^k when
there are several samples, paired diff (other model minus --baseline) with up/down/same. metrics.json also gets a judge_errors
block: the three counts and, per model, the mean with errors as fail and with errors dropped.

build --gold: also grade the dataset's reference answers ({task_id: text}, the shape the loader prompt
asks for) as a pretend model named "gold", sample 0. score prints gold apart (mean, criteria not met)
and keeps it out of scores.jsonl: it checks the judge and the rubric, it is not a model to compare.
If gold does not score near 1.0, fix the judge or the rubric before trusting any model's number.

label: pick n judge calls at random from TASKS (the build output, so it works before or while the judge
runs; the judge's raw.jsonl is accepted too), show each one (task, model, sample, criterion text, response
excerpt), take y / n from the keyboard (q stops and keeps what was done), append to judge_labels.csv. Rerunning
skips ids already labelled. The judge's verdict is never shown.

score --labels: judge vs your labels on the ids you labelled: accuracy, precision and recall on Met,
Cohen's kappa, n, base rate, the disagreements. Precision on Met = of what the judge called Met, how
many you did. Recall = of what you called Met, how many the judge caught. Kappa = agreement beyond
chance: a judge that says Met on everything in a 90% Met set agrees 90% and has kappa 0.

failures: every criterion the judge marked not met (judge errors stay in their bucket) -> failures.md for
hand reading (task, model, sample, criterion text, the judge's reason, a response excerpt) and, when
--prompt names a classifier prompt with '- name: description' category lines (prompts/classify_criteria.md),
failures.jsonl: one run.py task per failure in classifier.py's row shape, so `python classifier.py score RAW
--labels failure_labels.csv` gives the count table and "agrees with me on x of n" (failure_labels.csv: trajectory_id,
category; trajectory_id is the failure id printed in failures.md). --label opens a terminal loop on a random
n of the failures: criterion, judge reason, response excerpt, the numbered category list; type the number
(or q to stop, keeping what was done); appends to failure_labels.csv next to RAW, rerun skips ids already done.
Under 40 failures: hand-label all, skip the classifier.

agree: two judge runs over the same judge tasks (e.g. two judge models, or the same one twice):
raw agreement, kappa, n, disagreements. Unparseable verdicts are left out and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import NoReturn

import metrics
from classifier import FENCE, categories, load_prompt

NO_RESPONSE = "(the model returned no response)"


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


# ---------- load ----------

def load_answers(path: str | Path) -> dict[tuple[int, str, int], dict]:
    """Runner raw.jsonl -> {(task_id, model, sample_index): line}. Later lines win (appended reruns)."""
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
        answers[(row["task_id"], row["model"], row.get("sample_index", 0))] = row
    if not answers:
        die(f"{path}: no rows")
    models = sorted({m for _, m, _ in answers})
    tasks = sorted({t for t, _, _ in answers})
    samples = sorted({i for _, _, i in answers})
    missing = [(t, m, i) for t in tasks for m in models for i in samples if (t, m, i) not in answers]
    if missing:
        die(f"{len(missing)} (task, model, sample) triple(s) have no answer, e.g. {missing[:5]}")
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


def load_gold(path: str | Path) -> dict[int, str]:
    """{task_id: reference answer text} JSON -> same with int keys where the id is a number."""
    path = Path(path)
    if not path.exists():
        die(f"gold answers not found: {path}")
    gold = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gold, dict) or not gold:
        die(f"{path}: expected a non-empty {{task_id: text}} object")
    return {int(k) if str(k).isdigit() else k: v for k, v in gold.items()}


# ---------- build ----------

def task_prompt_of(a: dict) -> str:
    """The task prompt the judge sees: metadata.prompt_raw (APEX: prompt without attachments) or the prompt sent."""
    meta = a.get("metadata") or {}
    return meta["prompt_raw"] if isinstance(meta.get("prompt_raw"), str) else a["prompt"]


def judge_message(judge_prompt: str, task_prompt: str, response: str | None, criterion: str) -> str:
    return (f"{judge_prompt}\n\n<TASK_PROMPT>\n{task_prompt}\n</TASK_PROMPT>\n\n"
            f"<RESPONSE>\n{NO_RESPONSE if response is None else response}\n</RESPONSE>\n\n"
            f"<CRITERION>\n{criterion}\n</CRITERION>")


GOLD = "gold"


def build(runs: str | Path, data: str | Path, prompt: str | Path, out: str | Path,
          gold: str | Path | None = None) -> int:
    answers = load_answers(runs)
    rubrics = load_rubrics(data)
    judge_prompt = load_prompt(prompt)
    if gold:  # reference answers ride along as model "gold", sample 0, using each task's own prompt
        ref = load_gold(gold)
        prompt_of = {t: task_prompt_of(a) for (t, _, _), a in answers.items()}
        hit = [t for t in prompt_of if t in ref]
        if not hit:
            die(f"none of the {len(prompt_of)} tasks in {runs} has a gold answer in {gold}")
        for t in hit:
            answers[(t, GOLD, 0)] = {"response": ref[t], "prompt": prompt_of[t], "metadata": {"prompt_raw": prompt_of[t]}}
        print(f"gold answers for {len(hit)} of {len(prompt_of)} tasks")
    rows = []
    for (task_id, model, sample), a in sorted(answers.items()):
        if task_id not in rubrics:
            die(f"task {task_id} has no rubric in {data}")
        for c in rubrics[task_id]:
            rows.append({"id": f"{task_id}|{model}|{sample}|{c['id']}",
                         "prompt": judge_message(judge_prompt, task_prompt_of(a), a["response"], c["description"]),
                         "task_id": task_id, "model": model, "sample_index": sample, "criterion_id": c["id"]})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_tasks = len({t for t, _, _ in answers})
    n_models = len({m for _, m, _ in answers})
    n_samples = len({i for _, _, i in answers})
    print(f"wrote {len(rows)} judge task(s) ({n_tasks} tasks x {n_models} models x {n_samples} samples "
          f"x criteria) -> {out}")
    return len(rows)


# ---------- score ----------

def error_kind(row: dict) -> str:
    """Why a judge row has no verdict: timeout (or connection), api_error (any other failed call), unparsed (text came back)."""
    if isinstance(row.get("response"), str):
        return "unparsed"
    types = {e.get("type", "") for e in row.get("errors") or []}
    if any("Timeout" in t or "Connection" in t for t in types):
        return "timeout"
    return "api_error"


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


def score(raw: str | Path, tasks: str | Path, baseline: str, labels: str | Path | None = None) -> int:
    raw, tasks = Path(raw), Path(tasks)
    for p in (raw, tasks):
        if not p.exists():
            die(f"not found: {p}")
    expected = {json.loads(l)["id"] for l in tasks.read_text(encoding="utf-8").splitlines() if l.strip()}
    passed: dict[str, bool] = {}    # judge task id -> criterion met
    cell: dict[str, tuple] = {}     # judge task id -> (task_id, model, sample_index)
    kind: dict[str, str] = {}       # judge task id -> "timeout" / "api_error" / "unparsed" when the verdict is missing
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
            kind[row["task_id"]] = error_kind(row)
            errors.append((row["task_id"], f"{kind[row['task_id']]}: {why}"))
        passed[row["task_id"]] = bool(verdict)
        cell[row["task_id"]] = (meta["task_id"], meta["model"], meta.get("sample_index", 0))
    missing = expected - set(passed)
    if missing:
        print(f"judge not finished: {len(passed)} of {len(expected)} calls in {raw}, {len(missing)} still to come "
              f"(if the run ended, rerun run.py with a new --name; finished calls are cached)")
        if labels:
            _labels_check(labels, passed, raw)
        print("metrics wait for the full run")
        return 1

    verdicts_all = dict(passed)  # every judged call, gold included: what the labels are compared against
    gold_rows = {jid: ok for jid, ok in passed.items() if cell[jid][1] == GOLD}
    if gold_rows:  # the judge on the reference answers: a check of judge and rubric, printed apart
        failed = sorted(jid for jid, ok in gold_rows.items() if not ok)
        n_tasks = len({cell[j][0] for j in gold_rows})
        print(f"gold: tasks {n_tasks}  criteria {len(gold_rows)}  mean {sum(gold_rows.values()) / len(gold_rows):.3f}  "
              f"not met {len(failed)}" + ("  <- judge or rubric wrong on these; check before trusting any model" if failed else ""))
        for jid in failed[:10]:
            print(f"  {jid}")
        passed = {jid: ok for jid, ok in passed.items() if jid not in gold_rows}
        if not passed:
            die("only gold rows in the run; nothing to score")
    per: dict[tuple, list[bool]] = {}      # cell -> met flag per criterion (errors count as not met)
    errs: dict[tuple, int] = {}            # cell -> judge errors in it
    for jid, ok in passed.items():
        per.setdefault(cell[jid], []).append(ok)
        errs[cell[jid]] = errs.get(cell[jid], 0) + (jid in kind)
    def dropped(c: tuple) -> float | None:  # met / (total - errors); None when every criterion errored
        good = len(per[c]) - errs[c]
        return sum(per[c]) / good if good else None
    out = raw.parent / "scores.jsonl"
    out.write_text("".join(json.dumps({"task": t, "variant": m, "sample": i, "score": sum(v) / len(v),
                                       "score_errors_dropped": dropped((t, m, i)), "errors": errs[(t, m, i)]}) + "\n"
                           for (t, m, i), v in sorted(per.items())), encoding="utf-8")

    counts = {k: sum(v == k for v in kind.values()) for k in ("timeout", "api_error", "unparsed")}
    per_model: dict[str, dict] = {}
    models = sorted({m for _, m, _ in per})
    for m in models:
        by_task: dict[int, list[float]] = {}  # task -> score of each sample, errors as fail
        drop: list[float] = []                # cell scores with errors dropped, where defined
        for c, v in per.items():
            if c[1] == m:
                by_task.setdefault(c[0], []).append(sum(v) / len(v))
                if dropped(c) is not None:
                    drop.append(dropped(c))
        task_means = [sum(x) / len(x) for x in by_task.values()]
        spread = [max(x) - min(x) for x in by_task.values()]  # per-task gap between best and worst sample
        n_samples = max(len(x) for x in by_task.values())
        mean_fail = sum(task_means) / len(task_means)
        mean_drop = sum(drop) / len(drop) if drop else None
        per_model[m] = {"mean_errors_as_fail": mean_fail, "mean_errors_dropped": mean_drop,
                        "cells_all_errors": sum(dropped(c) is None for c in per if c[1] == m)}
        print(f"{m:24s} tasks {len(by_task)}  samples/task {n_samples}  mean score {mean_fail:.3f} (errors as fail)  "
              f"{_fmt_opt(mean_drop)} (errors dropped)  all criteria met {sum(s == 1.0 for s in task_means)}  "
              f"mean spread over samples {sum(spread) / len(spread):.3f}")
    print(f"judge errors: timeouts {counts['timeout']}  api errors {counts['api_error']}  unparsed {counts['unparsed']}  "
          f"(counted as fail in `score`, dropped from the denominator in `score_errors_dropped`)")
    for jid, why in errors[:10]:
        print(f"  {jid}: {' '.join(why.split())[:120]}")
    print(f"wrote {out}\n")
    if labels:
        _labels_check(labels, verdicts_all, raw)
    if baseline not in models:
        die(f"--baseline {baseline!r} not in models {models}")
    rc = metrics.main([str(out), "--baseline", baseline])
    mpath = raw.parent / "metrics.json"
    if mpath.exists():  # add the error bucket beside metrics.py's numbers
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["judge_errors"] = {**counts, "per_model": per_model}
        mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return rc


# ---------- labels ----------

def judge_rows(raw: str | Path) -> list[dict]:
    """Judge raw.jsonl (run.py output on grade.py build tasks) -> rows, metadata checked."""
    raw = Path(raw)
    if not raw.exists():
        die(f"not found: {raw}")
    rows = []
    for n, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        if any(k not in meta for k in ("task_id", "model", "criterion_id")):
            die(f"{raw}:{n}: metadata lacks task_id/model/criterion_id; tasks not made by grade.py build?")
        rows.append(row)
    if not rows:
        die(f"{raw}: no rows")
    return rows


def blocks(prompt: str) -> tuple[str, str]:
    """Judge message -> (response, criterion) pulled back out of its tags."""
    r = re.search(r"<RESPONSE>\n(.*?)\n</RESPONSE>", prompt, re.S)
    c = re.search(r"<CRITERION>\n(.*?)\n</CRITERION>", prompt, re.S)
    return (r.group(1) if r else ""), (c.group(1) if c else "")


def read_labels(path: str | Path) -> dict[str, bool]:
    """labels.csv -> {judge task id: human said Met}. Missing file = no labels yet."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["label"] == "y" for r in csv.DictReader(f)}


def call_fields(row: dict) -> tuple[str, dict]:
    """One judge call from tasks.jsonl (build output) or from the judge's raw.jsonl -> (judge id, fields)."""
    if "metadata" in row:  # judge raw.jsonl: run.py put build's fields under metadata
        return row["task_id"], row["metadata"]
    return row["id"], row  # tasks.jsonl: fields at top level


def label(tasks: str | Path, n: int, out: str | Path, seed: int = 0, ask=input, say=print,
          excerpt: int = 1200) -> int:
    """Terminal loop: show n random judge calls, verdict hidden, take y/n, append to OUT (judge_labels.csv). Returns count."""
    tasks, out = Path(tasks), Path(out)
    if not tasks.exists():
        die(f"not found: {tasks}")
    done = read_labels(out)
    rows = [r for r in map(json.loads, filter(str.strip, tasks.read_text(encoding="utf-8").splitlines()))
            if call_fields(r)[0] not in done]
    if not rows:
        say(f"nothing left to label ({len(done)} already in {out})")
        return 0
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    new = out.exists() and out.stat().st_size > 0
    count = 0
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not new:
            w.writerow(["id", "task_id", "model", "sample_index", "criterion_id", "label"])
        for k, row in enumerate(pick, 1):
            jid, m = call_fields(row)
            response, criterion = blocks(row.get("prompt") or "")
            say(f"\n[{k}/{len(pick)}]  task {m['task_id']}  model {m['model']}  sample {m.get('sample_index', 0)}  "
                f"criterion {m['criterion_id']}\nCRITERION: {criterion}\n"
                f"RESPONSE (first {excerpt} chars):\n{response[:excerpt]}\n")
            while True:
                a = ask("met?  y / n / q: ").strip().lower()
                if a in ("y", "n", "q"):
                    break
            if a == "q":
                break
            w.writerow([jid, m["task_id"], m["model"], m.get("sample_index", 0), m["criterion_id"], a])
            f.flush()
            count += 1
    say(f"labelled {count} this time, {len(done) + count} total in {out}")
    return count


def verdicts(raw: str | Path) -> tuple[dict[str, bool], int]:
    """Judge raw.jsonl -> ({judge task id: Met}, number of unparseable verdicts left out)."""
    good, bad = {}, 0
    for row in judge_rows(raw):
        v, _ = parse_verdict(row.get("response"))
        if v is None:
            bad += 1
        else:
            good[row["task_id"]] = v
    return good, bad


def agreement(truth: dict[str, bool], other: dict[str, bool]) -> dict:
    """Compare two yes/no maps on their shared ids; truth is the reference (your labels, or judge A)."""
    ids = sorted(set(truth) & set(other))
    n = len(ids)
    if n == 0:
        die("no shared ids to compare")
    tp = sum(truth[i] and other[i] for i in ids)
    fp = sum(other[i] and not truth[i] for i in ids)
    fn = sum(truth[i] and not other[i] for i in ids)
    agree = sum(truth[i] == other[i] for i in ids)
    p_yes_t, p_yes_o = sum(truth[i] for i in ids) / n, sum(other[i] for i in ids) / n
    po = agree / n
    pe = p_yes_t * p_yes_o + (1 - p_yes_t) * (1 - p_yes_o)  # chance agreement from each side's Met rate
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return {"n": n, "accuracy": po, "kappa": kappa,
            "precision_met": tp / (tp + fp) if tp + fp else None,
            "recall_met": tp / (tp + fn) if tp + fn else None,
            "base_rate_met": p_yes_t,
            "disagreements": [{"id": i, "truth": truth[i], "other": other[i]} for i in ids if truth[i] != other[i]]}


def _labels_check(labels: str | Path, passed: dict[str, bool], raw: Path) -> None:
    """Judge vs your labels on the labelled calls that have a verdict; says how many are still pending."""
    human = read_labels(labels)
    if not human:
        die(f"no labels in {labels}")
    judged = {jid: ok for jid, ok in passed.items() if jid in human}
    pending = len(human) - len(judged)
    if pending:
        print(f"labelled calls without a judge verdict yet: {pending} of {len(human)}")
    a = agreement(human, judged)
    print_agreement(a, "you", "judge")
    (raw.parent / "judge_agreement.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    print(f"wrote {raw.parent / 'judge_agreement.json'}\n")


def _fmt_opt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def print_agreement(a: dict, truth_name: str, other_name: str, say=print) -> None:
    say(f"{other_name} vs {truth_name}: n {a['n']}  accuracy {_fmt_opt(a['accuracy'])}  kappa {_fmt_opt(a['kappa'])}  "
        f"precision on Met {_fmt_opt(a['precision_met'])}  recall on Met {_fmt_opt(a['recall_met'])}  "
        f"base rate Met ({truth_name}) {_fmt_opt(a['base_rate_met'])}")
    say(f"disagreements: {len(a['disagreements'])}")
    for d in a["disagreements"]:
        say(f"  {d['id']}: {truth_name} {'Met' if d['truth'] else 'not'}, {other_name} {'Met' if d['other'] else 'not'}")


def agree(raw_a: str | Path, raw_b: str | Path) -> int:
    """Two judge runs on the same judge tasks -> agreement block; writes agreement.json next to A."""
    va, bad_a = verdicts(raw_a)
    vb, bad_b = verdicts(raw_b)
    a = agreement(va, vb)
    print(f"unparseable left out: A {bad_a}, B {bad_b}")
    print_agreement(a, "A", "B")
    out = Path(raw_a).parent / "agreement.json"
    out.write_text(json.dumps({"a": str(raw_a), "b": str(raw_b), "unparseable": {"a": bad_a, "b": bad_b}, **a},
                              indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


# ---------- failures ----------

def failures(raw: str | Path, out: str | Path | None = None, prompt: str | Path | None = None,
             excerpt: int = 600, label_n: int | None = None, seed: int = 0, ask=input, say=print) -> int:
    """Not-met criteria -> failures.md (+ failures.jsonl classifier tasks when a prompt is given;
    + a hand-labelling loop over label_n random failures when label_n is set). Returns the failure count."""
    raw = Path(raw)
    out = Path(out) if out else raw.parent
    out.mkdir(parents=True, exist_ok=True)
    fails = []
    for row in judge_rows(raw):
        v, why = parse_verdict(row.get("response"))
        if v is False:
            response, criterion = blocks(row.get("prompt") or "")
            m = row["metadata"]
            fails.append({"id": row["task_id"], "task": m["task_id"], "variant": m["model"],
                          "sample": m.get("sample_index", 0), "criterion_id": m["criterion_id"],
                          "criterion": criterion, "reason": why, "response": response})
    fails.sort(key=lambda f: (str(f["task"]), f["variant"], f["sample"], f["criterion_id"]))
    md = [f"# Failures: {len(fails)} criteria not met  (source {raw})\n",
          "Read them. Label about 20 with --label (writes failure_labels.csv: trajectory_id,category), or by hand using the id line of each; "
          "under 40, label them all and skip the classifier.\n"]
    for f in fails:
        md.append(f"## {f['id']}\ntask {f['task']} · model {f['variant']} · sample {f['sample']} · criterion {f['criterion_id']}\n\n"
                  f"**Criterion:** {f['criterion']}\n\n**Judge:** {f['reason'] or '(no reason given)'}\n\n"
                  f"**Response (first {excerpt} chars):**\n\n```\n{f['response'][:excerpt]}\n```\n")
    (out / "failures.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(fails)} failure(s) -> {out / 'failures.md'}" + ("  (under 40: hand-label all, no classifier)" if len(fails) < 40 else ""))
    if prompt:
        text = load_prompt(prompt)
        cats = categories(text)
        if not cats:
            die(f"{prompt}: no category lines ('- name: description') found")
        rows = [{"id": f["id"], "task": f["task"], "variant": f["variant"], "trajectory_id": f["id"], "categories": cats,
                 "prompt": (f"{text}\n\n<CRITERION>\n{f['criterion']}\n</CRITERION>\n\n<JUDGE_REASON>\n{f['reason']}\n</JUDGE_REASON>\n\n"
                            f"<RESPONSE>\n{f['response']}\n</RESPONSE>")} for f in fails]
        (out / "failures.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} classifier task(s) with categories {cats} -> {out / 'failures.jsonl'}; "
              f"run them with run.py, then `python classifier.py score <raw> --labels failure_labels.csv`")
        if label_n:
            label_failures(fails, cats, out / "failure_labels.csv", label_n, seed, excerpt, ask, say)
    elif label_n:
        die("--label needs --prompt: the categories come from the prompt's '- name: description' lines")
    return len(fails)


def label_failures(fails: list[dict], cats: list[str], out: Path, n: int, seed: int, excerpt: int,
                   ask=input, say=print) -> int:
    """Terminal loop: n random failures, type the category number (q stops), append to failure_labels.csv. Returns count."""
    done = set()
    if out.exists() and out.stat().st_size > 0:
        with open(out, encoding="utf-8-sig", newline="") as f:
            done = {r["trajectory_id"] for r in csv.DictReader(f)}
    left = [f for f in fails if f["id"] not in done]
    if not left:
        say(f"nothing left to label ({len(done)} already in {out})")
        return 0
    pick = random.Random(seed).sample(left, min(n, len(left)))
    menu = "\n".join(f"  {i}. {c}" for i, c in enumerate(cats, 1))
    count = 0
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["trajectory_id", "category"])
        for k, fl in enumerate(pick, 1):
            say(f"\n[{k}/{len(pick)}]  {fl['id']}\nCRITERION: {fl['criterion']}\nJUDGE: {fl['reason'] or '(no reason)'}\n"
                f"RESPONSE (first {excerpt} chars):\n{fl['response'][:excerpt]}\n\nCategories:\n{menu}")
            while True:
                a = ask("category number, or q: ").strip().lower()
                if a == "q" or a in cats or (a.isdigit() and 1 <= int(a) <= len(cats)):
                    break
            if a == "q":
                break
            w.writerow([fl["id"], cats[int(a) - 1] if a.isdigit() else a])
            f.flush()
            count += 1
    say(f"labelled {count} this time, {len(done) + count} total in {out}")
    return count


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="runner answers + rubrics -> judge task JSONL for run.py")
    b.add_argument("--runs", default="results/apex30_v1/raw.jsonl")
    b.add_argument("--data", default="apex-v1/data/train.csv")
    b.add_argument("--prompt", default="prompts/apex_judge.md")
    b.add_argument("--out", default="results/grade_v1/tasks.jsonl")
    b.add_argument("--gold", help="{task_id: reference answer} JSON; graded as model 'gold', reported apart")
    s = sub.add_parser("score", help="judge raw.jsonl -> scores.jsonl + metrics.json")
    s.add_argument("raw")
    s.add_argument("--tasks", default="results/grade_v1/tasks.jsonl", help="build output, checks completeness")
    s.add_argument("--baseline", default="gpt-5.6-luna", help="diff = other model minus this one")
    s.add_argument("--labels", help="judge_labels.csv from `label`; prints judge-vs-you agreement")
    l = sub.add_parser("label", help="label n random judge calls by hand from build's tasks.jsonl -> judge_labels.csv")
    l.add_argument("tasks", help="grade.py build output (works before the judge runs); judge raw.jsonl also accepted")
    l.add_argument("--n", type=int, default=40)
    l.add_argument("--out", help="default: judge_labels.csv next to TASKS")
    l.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("failures", help="not-met criteria -> failures.md to read, failures.jsonl for a classifier")
    f.add_argument("raw")
    f.add_argument("--out", help="folder; default: next to RAW")
    f.add_argument("--prompt", help="classifier prompt with category lines, e.g. prompts/classify_criteria.md")
    f.add_argument("--excerpt", type=int, default=600)
    f.add_argument("--label", action="store_true", help="then hand-label --n random failures in the terminal -> failure_labels.csv")
    f.add_argument("--n", type=int, default=20)
    f.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("agree", help="two judge runs on the same tasks -> agreement, kappa, disagreements")
    g.add_argument("raw_a")
    g.add_argument("raw_b")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.runs, args.data, args.prompt, args.out, args.gold)
        return 0
    if args.cmd == "label":
        label(args.tasks, args.n, args.out or Path(args.tasks).parent / "judge_labels.csv", args.seed)
        return 0
    if args.cmd == "agree":
        return agree(args.raw_a, args.raw_b)
    if args.cmd == "failures":
        failures(args.raw, args.out, args.prompt, args.excerpt, args.n if args.label else None, args.seed)
        return 0
    return score(args.raw, args.tasks, args.baseline, args.labels)


if __name__ == "__main__":
    sys.exit(main())
