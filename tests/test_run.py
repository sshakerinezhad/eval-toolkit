import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import llm
import run


# ---------- helpers ----------

def _write_yaml(tmp_path, **over):
    cfg = {
        "tasks": str(tmp_path / "tasks.jsonl"),
        "models": ["openai:gpt-4.1-mini"],
        "system_prompt": "sys",
        "temperature": 0.5,
        "max_tokens": 100,
    }
    cfg.update(over)
    if not (tmp_path / "tasks.jsonl").exists():
        _write_tasks(tmp_path, [{"id": "seed", "prompt": "seed"}])
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _write_tasks(tmp_path, rows):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _cfg(tmp_path, **over):
    return run.load_config(_write_yaml(tmp_path, **over), {})


def _ok(response="r", tin=10, tout=5):
    return llm.CallResult(response, tin, tout, "stop", params_sent={"temperature": 0.5, "max_tokens": 100})


def _fail(status, retryable=False, fatal=False):
    r = llm.CallResult(None, None, None, None, fatal=fatal)
    r.errors.append(llm.Attempt(1, status, "Err", None, f"status {status}", retryable))
    return r


@pytest.fixture
def scripted_call(monkeypatch):
    """Replace llm.call with a scripted fake; returns the list of (provider, model, prompt) calls made."""
    state = {"script": [], "calls": []}

    async def fake_call(client, provider, model, system_prompt, prompt, temperature, max_tokens, max_retries):
        state["calls"].append((provider, model, prompt))
        item = state["script"].pop(0) if state["script"] else _ok()
        return item() if callable(item) else item

    monkeypatch.setattr(run.llm, "call", fake_call)
    monkeypatch.setattr(run.llm, "make_client", lambda provider, timeout: NS(provider=provider))
    return state


class _NoBar:
    def __init__(self, *a, **k): pass
    def update(self, n=1): pass
    def set_postfix_str(self, s): pass
    def close(self): pass


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(run, "tqdm", _NoBar)


# ---------- load_config ----------

def test_load_config_defaults(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.models == ["openai:gpt-4.1-mini"]
    assert (cfg.n_samples, cfg.workers, cfg.max_retries, cfg.timeout) == (1, 4, 4, 60)


def test_load_config_cli_overrides_win(tmp_path):
    cfg = run.load_config(_write_yaml(tmp_path, timeout=30), {"timeout": 90, "max_retries": 9})
    assert (cfg.timeout, cfg.max_retries) == (90, 9)


def test_load_config_temperature_optional(tmp_path):
    p = _write_yaml(tmp_path)
    import yaml
    d = yaml.safe_load(p.read_text())
    del d["temperature"]
    p.write_text(yaml.safe_dump(d))
    assert run.load_config(p, {}).temperature is None


@pytest.mark.parametrize("bad", [
    {"models": []},
    {"models": ["nope:x"]},
    {"models": "openai:gpt-4.1-mini"},
    {"max_tokens": 0},
    {"n_samples": 0},
    {"workers": 0},
    {"tasks": "does/not/exist.jsonl"},
])
def test_load_config_rejects_bad_values(tmp_path, bad):
    with pytest.raises(SystemExit):
        _cfg(tmp_path, **bad)


def test_load_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(SystemExit, match="unknown"):
        _cfg(tmp_path, temprature=0.5)


# ---------- load_tasks ----------

def test_load_tasks_extracts_id_prompt_and_metadata(tmp_path):
    p = _write_tasks(tmp_path, [{"id": "a", "prompt": "hi", "difficulty": 3, "tags": ["x"]}])
    [t] = run.load_tasks(p)
    assert t == {"id": "a", "prompt": "hi", "metadata": {"difficulty": 3, "tags": ["x"]}}


def test_load_tasks_skips_blank_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":1,"prompt":"a"}\n\n{"id":2,"prompt":"b"}\n')
    assert [t["id"] for t in run.load_tasks(p)] == [1, 2]


@pytest.mark.parametrize("rows,msg", [
    ([{"id": "a"}], "prompt"),
    ([{"prompt": "x"}], "id"),
    ([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}], "duplicate"),
    ([{"id": "a", "prompt": 5}], "prompt"),
])
def test_load_tasks_rejects_bad_rows(tmp_path, rows, msg):
    with pytest.raises(SystemExit, match=msg):
        run.load_tasks(_write_tasks(tmp_path, rows))


# ---------- build_jobs ----------

def test_build_jobs_is_tasks_x_models_x_samples(tmp_path):
    cfg = _cfg(tmp_path, models=["openai:a", "anthropic:b"], n_samples=3)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(2)]
    jobs = run.build_jobs(cfg, tasks)
    assert len(jobs) == 12
    assert {(j.provider, j.model) for j in jobs} == {("openai", "a"), ("anthropic", "b")}
    assert sorted({j.sample_index for j in jobs}) == [0, 1, 2]
    assert len({j.key for j in jobs}) == 12


def test_build_jobs_identical_prompts_share_cache_key(tmp_path):
    """Cache is keyed on the call, not the task id: two tasks with the same prompt hit the same entry."""
    cfg = _cfg(tmp_path)
    jobs = run.build_jobs(cfg, [{"id": 1, "prompt": "same", "metadata": {}}, {"id": 2, "prompt": "same", "metadata": {}}])
    assert jobs[0].key == jobs[1].key


def test_build_jobs_marks_cache_hits(tmp_path):
    cfg = _cfg(tmp_path)
    tasks = [{"id": 1, "prompt": "p", "metadata": {}}]
    key = llm.cache_key("openai", "gpt-4.1-mini", "sys", "p", 0, 0.5, 100)
    llm.cache_put(key, {"response": "c", "tokens_in": 1, "tokens_out": 2, "finish_reason": "stop"})
    [j] = run.build_jobs(cfg, tasks)
    assert j.key == key and j.cached == {"response": "c", "tokens_in": 1, "tokens_out": 2, "finish_reason": "stop"}


# ---------- estimate ----------

def test_estimate_costs_only_uncached_jobs_using_input_and_output_prices(tmp_path):
    cfg = _cfg(tmp_path, max_tokens=1000, workers=2)
    tasks = [{"id": i, "prompt": "x" * 400, "metadata": {}} for i in range(4)]  # ~100 tokens + sys
    jobs = run.build_jobs(cfg, tasks)
    jobs[0].cached = {"response": "c", "tokens_in": 1, "tokens_out": 1, "finish_reason": "stop"}
    pricing = {"openai:gpt-4.1-mini": {"input_per_m": 1.0, "output_per_m": 10.0}}
    smoke = [llm.SmokeResult("openai:gpt-4.1-mini", True, 2.0, None, "OK")]
    est = run.estimate(cfg, jobs, smoke, pricing)
    m = est["per_model"]["openai:gpt-4.1-mini"]
    assert (m["calls"], m["cached"]) == (4, 1)
    in_tok = 3 * (len("sys") + 400) / 4
    assert m["cost_max"] == pytest.approx(in_tok / 1e6 * 1.0 + 3 * 1000 / 1e6 * 10.0)
    assert m["time_s"] == pytest.approx(2.0 * 3 / 2)
    assert est["cost_max"] == pytest.approx(m["cost_max"])
    assert est["time_s"] == pytest.approx(m["time_s"])


def test_estimate_unknown_price_gives_none_cost(tmp_path):
    cfg = _cfg(tmp_path)
    jobs = run.build_jobs(cfg, [{"id": 1, "prompt": "p", "metadata": {}}])
    est = run.estimate(cfg, jobs, [llm.SmokeResult("openai:gpt-4.1-mini", True, 1.0, None, "")], {})
    assert est["per_model"]["openai:gpt-4.1-mini"]["cost_max"] is None
    assert est["cost_max"] is None


# ---------- run dir ----------

def test_resolve_run_name_returns_free_name_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    assert run.resolve_run_name("r1", yes=True) == "r1"
    assert not (tmp_path / "results" / "r1").exists()  # nothing created yet


def test_resolve_run_name_refuses_existing_under_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results" / "r1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="exists"):
        run.resolve_run_name("r1", yes=True)


def test_resolve_run_name_prompts_until_free(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results" / "r1").mkdir(parents=True)
    (tmp_path / "results" / "r2").mkdir()
    answers = iter(["r2", "r3"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert run.resolve_run_name("r1", yes=False) == "r3"


def test_resolve_run_name_blank_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results" / "r1").mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(SystemExit, match="aborted"):
        run.resolve_run_name("r1", yes=False)


def test_write_run_json_records_config_and_args(tmp_path):
    cfg = _cfg(tmp_path, n_samples=2)
    run.write_run_json(tmp_path, cfg, {"name": "x", "yes": True}, n_tasks=7, git_rev="abc")
    d = json.loads((tmp_path / "run.json").read_text())
    assert d["config"]["n_samples"] == 2 and d["config"]["models"] == ["openai:gpt-4.1-mini"]
    assert d["cli_args"] == {"name": "x", "yes": True}
    assert d["n_tasks"] == 7 and d["git_rev"] == "abc" and d["started_at"]


# ---------- execute ----------

def _run(cfg, jobs, run_dir):
    return asyncio.run(run.execute(cfg, jobs, run_dir))


def _lines(run_dir):
    return [json.loads(l) for l in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def test_execute_writes_one_full_line_per_job(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path)
    tasks = [{"id": "t1", "prompt": "p1", "metadata": {"k": "v"}}]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_ok("answer", 11, 7)]
    summary = _run(cfg, jobs, tmp_path)
    [line] = _lines(tmp_path)
    assert line["task_id"] == "t1" and line["provider"] == "openai" and line["model"] == "gpt-4.1-mini"
    assert line["system_prompt"] == "sys" and line["prompt"] == "p1" and line["sample_index"] == 0
    assert line["params"] == {"temperature": 0.5, "max_tokens": 100}
    assert line["params_sent"] == {"temperature": 0.5, "max_tokens": 100}
    assert line["cached"] is False and line["response"] == "answer" and line["finish_reason"] == "stop"
    assert (line["tokens_in"], line["tokens_out"]) == (11, 7)
    assert line["errors"] == [] and line["metadata"] == {"k": "v"}
    assert line["started_at"] and line["ended_at"] and line["total_time_s"] >= 0
    assert (summary.ok, summary.cached, summary.failed, summary.killed) == (1, 0, 0, False)


def test_execute_caches_successes(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path)
    [job] = jobs = run.build_jobs(cfg, [{"id": 1, "prompt": "p", "metadata": {}}])
    scripted_call["script"] = [_ok("ans", 3, 4)]
    _run(cfg, jobs, tmp_path)
    assert llm.cache_get(job.key) == {"response": "ans", "tokens_in": 3, "tokens_out": 4, "finish_reason": "stop"}


def test_execute_cache_hit_skips_call_and_marks_cached(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path)
    jobs = run.build_jobs(cfg, [{"id": 1, "prompt": "p", "metadata": {}}])
    jobs[0].cached = {"response": "c", "tokens_in": 1, "tokens_out": 2, "finish_reason": "stop"}
    s = _run(cfg, jobs, tmp_path)
    [line] = _lines(tmp_path)
    assert scripted_call["calls"] == []
    assert line["cached"] is True and line["response"] == "c" and line["total_time_s"] == 0
    assert line["tokens_in"] == 1 and line["started_at"] == line["ended_at"]
    assert s.cached == 1


def test_execute_failed_job_writes_line_with_errors_and_no_cache(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path)
    [job] = jobs = run.build_jobs(cfg, [{"id": 1, "prompt": "p", "metadata": {}}])
    scripted_call["script"] = [_fail(429, retryable=True)]
    s = _run(cfg, jobs, tmp_path)
    [line] = _lines(tmp_path)
    assert line["response"] is None and line["errors"][0]["status"] == 429
    assert llm.cache_get(job.key) is None
    assert s.failed == 1 and s.killed is False


def test_execute_kills_immediately_on_fatal(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(5)]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_fail(401, fatal=True)]
    s = _run(cfg, jobs, tmp_path)
    assert s.killed is True and "401" in s.kill_reason
    assert len(scripted_call["calls"]) == 1
    assert len(_lines(tmp_path)) == 1


def test_execute_kills_after_four_consecutive_nonretryable(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(10)]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_fail(400)] * 4
    s = _run(cfg, jobs, tmp_path)
    assert s.killed is True and "4 consecutive" in s.kill_reason
    assert len(scripted_call["calls"]) == 4


def test_execute_success_resets_consecutive_counter(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(7)]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_fail(400)] * 3 + [_ok()] + [_fail(400)] * 3
    s = _run(cfg, jobs, tmp_path)
    assert s.killed is False and s.failed == 6 and s.ok == 1


def test_execute_retryable_failures_do_not_count_toward_kill(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(6)]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_fail(429, retryable=True)] * 6
    s = _run(cfg, jobs, tmp_path)
    assert s.killed is False and s.failed == 6


def test_execute_respects_per_model_worker_limit(tmp_path, monkeypatch, quiet):
    cfg = _cfg(tmp_path, models=["openai:a", "anthropic:b"], workers=2)
    tasks = [{"id": i, "prompt": "p", "metadata": {}} for i in range(6)]
    jobs = run.build_jobs(cfg, tasks)
    active = {"a": 0, "b": 0}
    peak = {"a": 0, "b": 0}

    async def fake_call(client, provider, model, *a, **k):
        active[model] += 1
        peak[model] = max(peak[model], active[model])
        await asyncio.sleep(0.01)
        active[model] -= 1
        return _ok()

    monkeypatch.setattr(run.llm, "call", fake_call)
    monkeypatch.setattr(run.llm, "make_client", lambda provider, timeout: None)
    _run(cfg, jobs, tmp_path)
    assert peak == {"a": 2, "b": 2}


def test_execute_counts_truncated_responses(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(3)]
    jobs = run.build_jobs(cfg, tasks)
    cut = llm.CallResult("", 10, 5, "length", params_sent={})
    jobs[0].cached = {"response": "", "tokens_in": 1, "tokens_out": 5, "finish_reason": "length"}
    scripted_call["script"] = [cut, _ok()]
    s = _run(cfg, jobs, tmp_path)
    assert (s.ok, s.cached, s.truncated) == (2, 1, 2)  # one live cut + one cached cut


def test_execute_summary_groups_errors_by_type_and_status(tmp_path, scripted_call, quiet):
    cfg = _cfg(tmp_path, workers=1)
    tasks = [{"id": i, "prompt": f"p{i}", "metadata": {}} for i in range(3)]
    jobs = run.build_jobs(cfg, tasks)
    scripted_call["script"] = [_fail(429, True), _fail(429, True), _fail(400)]
    s = _run(cfg, jobs, tmp_path)
    assert s.error_counts == {"Err/429": 2, "Err/400": 1}
    assert s.tokens_in == 0 and s.tokens_out == 0


# ---------- main ----------

def test_main_exits_1_when_smoke_fails(tmp_path, monkeypatch, capsys):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, False, 0.1, llm.Attempt(1, 404, "NotFoundError", None, "no model", False), "") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    assert run.main([str(cfg_path), "--name", "x", "--yes"]) == 1
    out = capsys.readouterr().out
    assert "404" in out and "no model" in out
    assert not (tmp_path / "results" / "x").exists()


def test_main_existing_run_name_refused_before_smoke(tmp_path, monkeypatch, capsys):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results" / "taken").mkdir(parents=True)
    smoked = []

    async def fake_smoke(specs, timeout):
        smoked.extend(specs)
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    with pytest.raises(SystemExit, match="exists"):
        run.main([str(cfg_path), "--name", "taken", "--yes"])
    assert smoked == []


def test_main_existing_run_name_prompts_before_smoke_and_uses_new_name(tmp_path, monkeypatch, scripted_call, quiet):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results" / "taken").mkdir(parents=True)
    answers = iter(["fresh", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    assert run.main([str(cfg_path), "--name", "taken"]) == 0
    assert (tmp_path / "results" / "fresh" / "raw.jsonl").exists()
    assert not (tmp_path / "results" / "taken" / "raw.jsonl").exists()


def test_main_full_flow_with_yes(tmp_path, monkeypatch, scripted_call, quiet):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}, {"id": 2, "prompt": "q"}])
    cfg_path = _write_yaml(tmp_path)

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    assert run.main([str(cfg_path), "--name", "x", "--yes"]) == 0
    assert len(_lines(tmp_path / "results" / "x")) == 2
    assert (tmp_path / "results" / "x" / "run.json").exists()


def test_main_exits_2_when_killed(tmp_path, monkeypatch, scripted_call, quiet):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    scripted_call["script"] = [_fail(401, fatal=True)]
    assert run.main([str(cfg_path), "--name", "x", "--yes"]) == 2


def test_main_declined_confirmation_exits_1_without_running(tmp_path, monkeypatch, scripted_call, quiet):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)

    async def fake_smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(run.llm, "smoke", fake_smoke)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert run.main([str(cfg_path), "--name", "x"]) == 1
    assert scripted_call["calls"] == []


def test_main_ctrl_c_during_smoke_exits_130_cleanly(tmp_path, monkeypatch, capsys):
    _write_tasks(tmp_path, [{"id": 1, "prompt": "p"}])
    cfg_path = _write_yaml(tmp_path)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")

    async def interrupted_smoke(specs, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(run.llm, "smoke", interrupted_smoke)
    assert run.main([str(cfg_path), "--name", "x", "--yes"]) == 130
    assert "interrupted" in capsys.readouterr().out
