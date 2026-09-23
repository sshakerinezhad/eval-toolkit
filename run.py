"""ONE RUN. Batch a jsonl of prompts across models, with cache, retries, kill-switch, and provenance.

    python run.py configs/x.yaml --name my_run [--yes] [--max-retries N] [--timeout S]

Flow: load config + tasks -> build jobs (cache lookup) -> smoke each model -> dashboard + estimate
-> confirm -> results/<name>/run.json -> execute (asyncio, per-model semaphore) -> append raw.jsonl
-> summary. Exit codes: 0 ok, 1 pre-run failure / declined, 2 killed mid-run, 130 Ctrl-C.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import yaml
from tqdm import tqdm

import llm

RESULTS_DIR = Path("results")
PRICING_PATH = Path(__file__).with_name("pricing.json")
KILL_AFTER_CONSECUTIVE = 4
CHARS_PER_TOKEN = 4  # rough input-token estimate; good enough for a go/no-go number


def die(msg: str) -> NoReturn:
    raise SystemExit(f"ERROR: {msg}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------- config / tasks / jobs ----------

@dataclass
class RunConfig:
    tasks: str
    models: list[str]
    system_prompt: str
    max_tokens: int
    temperature: float | None = None
    n_samples: int = 1
    workers: int = 4       # per model
    max_retries: int = 4
    timeout: float = 60    # seconds per call


def load_config(path: str | Path, overrides: dict) -> RunConfig:
    path = Path(path)
    if not path.exists():
        die(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        die("config must be a YAML mapping")
    known = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        die(f"unknown config keys {sorted(unknown)}; known: {sorted(known)}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    missing = [k for k in ("tasks", "models", "system_prompt", "max_tokens") if k not in raw]
    if missing:
        die(f"config missing required keys {missing}")
    cfg = RunConfig(**raw)

    if not isinstance(cfg.models, list) or not cfg.models:
        die("models must be a non-empty list of 'provider:model' strings")
    for spec in cfg.models:
        try:
            llm.parse_model(spec)
        except ValueError as e:
            die(str(e))
    if len(set(cfg.models)) != len(cfg.models):
        die("models list has duplicates")
    if not isinstance(cfg.system_prompt, str):
        die("system_prompt must be a string")
    if cfg.temperature is not None and not isinstance(cfg.temperature, (int, float)):
        die("temperature must be a number or null")
    for name in ("max_tokens", "n_samples", "workers"):
        v = getattr(cfg, name)
        if not isinstance(v, int) or v < 1:
            die(f"{name} must be an integer >= 1, got {v!r}")
    if not isinstance(cfg.max_retries, int) or cfg.max_retries < 0:
        die(f"max_retries must be an integer >= 0, got {cfg.max_retries!r}")
    if not isinstance(cfg.timeout, (int, float)) or cfg.timeout <= 0:
        die(f"timeout must be a positive number, got {cfg.timeout!r}")
    if not Path(cfg.tasks).exists():
        die(f"tasks file not found: {cfg.tasks}")
    return cfg


def load_tasks(path: str | Path) -> list[dict]:
    """Each row -> {"id", "prompt", "metadata": {everything else}}."""
    tasks, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            die(f"{path}:{n}: invalid JSON: {e}")
        if not isinstance(row, dict) or "id" not in row:
            die(f"{path}:{n}: row has no 'id'")
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            die(f"{path}:{n}: 'prompt' missing or not a non-empty string")
        key = json.dumps(row["id"])
        if key in seen:
            die(f"{path}:{n}: duplicate id {row['id']!r}")
        seen.add(key)
        meta = {k: v for k, v in row.items() if k not in ("id", "prompt")}
        tasks.append({"id": row["id"], "prompt": row["prompt"], "metadata": meta})
    if not tasks:
        die(f"{path}: no tasks")
    return tasks


@dataclass
class Job:
    task: dict
    provider: str
    model: str
    sample_index: int
    key: str
    cached: dict | None


def build_jobs(cfg: RunConfig, tasks: list[dict]) -> list[Job]:
    jobs = []
    for spec in cfg.models:
        provider, model = llm.parse_model(spec)
        for task in tasks:
            for i in range(cfg.n_samples):
                key = llm.cache_key(provider, model, cfg.system_prompt, task["prompt"], i,
                                    cfg.temperature, cfg.max_tokens)
                jobs.append(Job(task, provider, model, i, key, llm.cache_get(key)))
    return jobs


# ---------- estimate / dashboard ----------

def load_pricing() -> dict:
    if not PRICING_PATH.exists():
        return {}
    return json.loads(PRICING_PATH.read_text(encoding="utf-8"))


def estimate(cfg: RunConfig, jobs: list[Job], smoke_results: list[llm.SmokeResult], pricing: dict) -> dict:
    """Cost = upper bound (max_tokens out on every call). Time = rough floor (smoke latency)."""
    latency = {s.spec: s.latency_s for s in smoke_results}
    per_model: dict[str, dict] = {}
    for spec in cfg.models:
        provider, model = llm.parse_model(spec)
        mine = [j for j in jobs if j.provider == provider and j.model == model]
        todo = [j for j in mine if j.cached is None]
        in_tokens = sum(len(cfg.system_prompt) + len(j.task["prompt"]) for j in todo) / CHARS_PER_TOKEN
        out_tokens = len(todo) * cfg.max_tokens
        price = pricing.get(spec)
        cost = None
        if price:
            cost = in_tokens / 1e6 * price["input_per_m"] + out_tokens / 1e6 * price["output_per_m"]
        per_model[spec] = {
            "calls": len(mine), "cached": len(mine) - len(todo), "to_run": len(todo),
            "est_in_tokens": int(in_tokens), "max_out_tokens": out_tokens, "cost_max": cost,
            "time_s": latency.get(spec, 0.0) * len(todo) / cfg.workers,
        }
    costs = [m["cost_max"] for m in per_model.values()]
    return {
        "per_model": per_model,
        "cost_max": None if any(c is None for c in costs) else sum(costs),
        "time_s": max((m["time_s"] for m in per_model.values()), default=0.0),
    }


def _fmt_cost(c: float | None) -> str:
    return "unknown (no price)" if c is None else f"${c:.4f}"


def _fmt_time(s: float) -> str:
    return f"{s / 60:.1f} min" if s >= 90 else f"{s:.0f} s"


def print_dashboard(cfg: RunConfig, name: str, jobs: list[Job], est: dict) -> None:
    cached = sum(j.cached is not None for j in jobs)
    n_tasks = len({json.dumps(j.task["id"]) for j in jobs})
    w = 78
    print("=" * w)
    print(f"RUN {name}  ->  {(RESULTS_DIR / name / 'raw.jsonl').as_posix()}")
    print("-" * w)
    print(f"tasks file   : {cfg.tasks}  ({n_tasks} tasks)")
    print(f"models       : {', '.join(cfg.models)}")
    print(f"n_samples    : {cfg.n_samples}    workers/model: {cfg.workers}    "
          f"max_retries: {cfg.max_retries}    timeout: {cfg.timeout}s")
    print(f"temperature  : {'not sent (model default)' if cfg.temperature is None else cfg.temperature}"
          f"    max_tokens: {cfg.max_tokens}")
    sp = cfg.system_prompt.replace("\n", " ")
    print(f"system_prompt: {sp[:60]!r}{'...' if len(sp) > 60 else ''}  ({len(cfg.system_prompt)} chars)")
    print(f"jobs         : {len(jobs)} total = {cached} cached + {len(jobs) - cached} to run")
    print("-" * w)
    for spec, m in est["per_model"].items():
        print(f"  {spec:40s} run {m['to_run']:5d}  cached {m['cached']:5d}  "
              f"cost<= {_fmt_cost(m['cost_max']):18s} time~ {_fmt_time(m['time_s'])}")
    print("-" * w)
    print(f"EST cost (upper bound, max_tokens out every call): {_fmt_cost(est['cost_max'])}")
    print(f"EST time (rough floor, from smoke latency)       : {_fmt_time(est['time_s'])}")
    print("=" * w)


# ---------- run dir ----------

def resolve_run_name(name: str, yes: bool) -> str:
    """Pick a free run name BEFORE anything costs money. Creates nothing."""
    while True:
        d = RESULTS_DIR / name
        if not d.exists():
            return name
        if yes:
            die(f"run folder already exists: {d} (pick a new --name)")
        name = input(f"{d} already exists. New run name (blank to abort): ").strip()
        if not name:
            die("aborted")


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - git absent or not a repo; provenance is best-effort
        return None


def write_run_json(run_dir: Path, cfg: RunConfig, cli_args: dict, n_tasks: int, git_rev: str | None) -> None:
    doc = {"started_at": now_iso(), "config": asdict(cfg), "cli_args": cli_args,
           "n_tasks": n_tasks, "git_rev": git_rev}
    (run_dir / "run.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- execute ----------

@dataclass
class Summary:
    total: int = 0
    ok: int = 0
    cached: int = 0
    failed: int = 0
    truncated: int = 0  # ok responses cut off by max_tokens (finish_reason == "length")
    killed: bool = False
    kill_reason: str = ""
    error_counts: Counter = field(default_factory=Counter)
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0


def _line(cfg: RunConfig, job: Job, *, cached: bool, response, finish_reason, tokens_in, tokens_out,
          errors: list, params_sent: dict, started: str, ended: str, total_time_s: float) -> dict:
    return {
        "task_id": job.task["id"], "provider": job.provider, "model": job.model,
        "system_prompt": cfg.system_prompt, "prompt": job.task["prompt"], "sample_index": job.sample_index,
        "params": {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens},
        "params_sent": params_sent, "cached": cached,
        "response": response, "finish_reason": finish_reason,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "errors": [asdict(e) for e in errors],
        "started_at": started, "ended_at": ended, "total_time_s": round(total_time_s, 3),
        "metadata": job.task["metadata"],
    }


async def execute(cfg: RunConfig, jobs: list[Job], run_dir: Path) -> Summary:
    summary = Summary(total=len(jobs))
    out = open(run_dir / "raw.jsonl", "a", encoding="utf-8")  # append-only, never truncate
    sems = {spec: asyncio.Semaphore(cfg.workers) for spec in cfg.models}
    clients = {p: llm.make_client(p, cfg.timeout) for p in {j.provider for j in jobs}}
    consecutive = 0
    killed = asyncio.Event()
    bar = tqdm(total=len(jobs), unit="job", dynamic_ncols=True)
    t0 = time.perf_counter()

    def emit(line: dict) -> None:
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
        out.flush()
        bar.update(1)
        pct = 100 * summary.failed / max(1, summary.ok + summary.cached + summary.failed)
        bar.set_postfix_str(f"ok={summary.ok} cached={summary.cached} err={summary.failed} ({pct:.0f}%)")

    def kill(reason: str) -> None:
        if not killed.is_set():
            summary.killed, summary.kill_reason = True, reason
            killed.set()

    async def one(job: Job) -> None:
        nonlocal consecutive
        if job.cached is not None:
            c, ts = job.cached, now_iso()
            summary.cached += 1
            summary.truncated += c.get("finish_reason") == "length"
            emit(_line(cfg, job, cached=True, response=c["response"], finish_reason=c.get("finish_reason"),
                       tokens_in=c["tokens_in"], tokens_out=c["tokens_out"], errors=[],
                       params_sent=llm.build_params(job.provider, job.model, cfg.temperature, cfg.max_tokens),
                       started=ts, ended=ts, total_time_s=0.0))
            return
        async with sems[f"{job.provider}:{job.model}"]:
            if killed.is_set():
                return
            started, t = now_iso(), time.perf_counter()
            r = await llm.call(clients[job.provider], job.provider, job.model, cfg.system_prompt,
                               job.task["prompt"], cfg.temperature, cfg.max_tokens, cfg.max_retries)
            elapsed, ended = time.perf_counter() - t, now_iso()
        if r.response is not None:
            llm.cache_put(job.key, {"response": r.response, "tokens_in": r.tokens_in,
                                    "tokens_out": r.tokens_out, "finish_reason": r.finish_reason})
            summary.ok += 1
            summary.truncated += r.finish_reason == "length"
            summary.tokens_in += r.tokens_in or 0
            summary.tokens_out += r.tokens_out or 0
            consecutive = 0
        else:
            summary.failed += 1
            last = r.errors[-1]
            if r.fatal:
                kill(f"fatal error {last.status} {last.type}: {last.message}")
            elif not last.retryable:
                consecutive += 1
                if consecutive >= KILL_AFTER_CONSECUTIVE:
                    kill(f"{KILL_AFTER_CONSECUTIVE} consecutive non-retryable errors; "
                         f"last: {last.status} {last.type}: {last.message}")
        for e in r.errors:
            summary.error_counts[f"{e.type}/{e.status}"] += 1
        emit(_line(cfg, job, cached=False, response=r.response, finish_reason=r.finish_reason,
                   tokens_in=r.tokens_in, tokens_out=r.tokens_out, errors=r.errors,
                   params_sent=r.params_sent, started=started, ended=ended, total_time_s=elapsed))

    try:
        await asyncio.gather(*(one(j) for j in jobs))
    finally:
        summary.wall_s = time.perf_counter() - t0
        bar.close()
        out.close()
    return summary


def print_summary(s: Summary, pricing: dict, cfg: RunConfig) -> None:
    print("-" * 78)
    if s.killed:
        print(f"RUN KILLED: {s.kill_reason}")
        print("Fix the cause, then rerun with a new --name; completed calls are cached and cost nothing.")
    skipped = s.total - s.ok - s.cached - s.failed
    print(f"jobs: {s.total} total | ok {s.ok} | cached {s.cached} | failed {s.failed} | not run {skipped}")
    if s.truncated:
        print(f"WARNING: {s.truncated} ok response(s) hit max_tokens (finish_reason=length); "
              f"raise max_tokens if that matters")
    print(f"tokens: in {s.tokens_in} / out {s.tokens_out}    wall: {_fmt_time(s.wall_s)}")
    if s.error_counts:
        print("errors by type/status:")
        for k, v in s.error_counts.most_common():
            print(f"  {k:32s} {v}")


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\ninterrupted; completed calls are cached, rerun with a new --name to resume.")
        return 130


def _main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--name", required=True, help="run name -> results/<name>/")
    ap.add_argument("--yes", action="store_true", help="skip confirmation; abort instead of prompting")
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"max_retries": args.max_retries, "timeout": args.timeout})
    args.name = resolve_run_name(args.name, args.yes)  # before smoke: a name clash must not cost a call
    tasks = load_tasks(cfg.tasks)
    jobs = build_jobs(cfg, tasks)
    pricing = load_pricing()

    print(f"smoke test: {len(cfg.models)} model(s) ...")
    smoke = asyncio.run(llm.smoke(cfg.models, cfg.timeout))
    for s in smoke:
        print("  " + s.line())
    if not all(s.ok for s in smoke):
        print("smoke test failed; run not started. Fix the above and rerun.")
        return 1

    est = estimate(cfg, jobs, smoke, pricing)
    print_dashboard(cfg, args.name, jobs, est)
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 1

    run_dir = RESULTS_DIR / args.name
    run_dir.mkdir(parents=True)
    write_run_json(run_dir, cfg, vars(args), len(tasks), git_rev())
    summary = asyncio.run(execute(cfg, jobs, run_dir))
    print_summary(summary, pricing, cfg)
    return 2 if summary.killed else 0


if __name__ == "__main__":
    sys.exit(main())
