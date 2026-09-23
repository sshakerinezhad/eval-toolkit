# Inference Runner — plan + completion record (archived 2026-09-23)

# Inference Runner — Design + Implementation Plan

## Context

Toolkit needs a batch inference runner: take a jsonl of prompts, fan them out to one or more
models across providers, and produce an append-only jsonl of results with full provenance
(tokens, timing, errors, params). Must be crash-safe and rerun-safe (disk cache = $0/0s on
rerun), must fail fast on config/key errors, and must show cost/time before spending money.

Repo today is a scaffold: `smoke.py` (native SDK per provider), `requirements.txt`, empty
plan files. Nothing reusable except the smoke idea. `.cache/` and `results/` already gitignored.

## Decisions (from Q&A — do not re-litigate)

| Topic | Decision |
|---|---|
| SDK | `openai` only. Providers = base_url + env key. |
| Providers | `openai` (default base), `anthropic` (`https://api.anthropic.com/v1/`, `ANTHROPIC_API_KEY`), `openrouter` (`https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`). Gemini not included. |
| Run config | YAML file. One system prompt + temp + max_tokens for ALL models in run. |
| Model spec | `provider:model` strings, e.g. `anthropic:claude-haiku-4-5`, `openrouter:openai/gpt-5-mini`. |
| Output | New folder per run: `results/<run-name>/raw.jsonl` (results, append-only, one line per (task, model, sample)) + `run.json` (resolved config snapshot: yaml contents + CLI overrides + tasks path + task count + models + start time + git rev). User supplies `--name`; if folder exists → prompt for new name (abort under `--yes`). `run.json` written before first call so a crashed run still records its setup. |
| Cache | `.cache/<sha256>.json`; key = provider, model, system_prompt, prompt, sample_index, temperature, max_tokens. Stores response + tokens only. Successes only. |
| Cache-hit line | `cached: true`, started=ended=now, `total_time_s: 0`, tokens from cache. |
| Workers | `workers: N` = per-model semaphore. |
| Retries | Retryable: 408, 429, ≥500 (incl. Anthropic 529), timeout, connection error. `max_retries: 4` default (yaml + `--max-retries`). Exp backoff 2s→64s + jitter, honor `Retry-After`. `timeout: 60` default (yaml + `--timeout`). SDK's own retries disabled (`max_retries=0`) so we log every attempt. |
| Kill rules | 401/403 → kill immediately. 4 consecutive non-retryable final failures (400/404/422/other 4xx) → kill. Counter resets on real success only (cache hits don't count). Smoke failure → kill before run. |
| Temperature | Optional. `temperature: null` or omitted = param not sent, model default. Cache key uses requested value (null included). |
| Reasoning models | OpenAI `gpt-5*`/`o*` prefixes: drop `temperature`, send `max_completion_tokens`. Line records `params` (requested) and `params_sent` (actual). Unknown reasoning model elsewhere → 400 → kill; user sets `temperature: null`. No silent retry-without-temp. |
| finish_reason | Recorded per line (`stop` / `length` / other). Truncation must be visible for splicing. |
| Smoke pass | HTTP 200 + `usage` present. Content printed, not required (reasoning models may spend whole tiny budget thinking). `max_tokens` 256 for smoke. |
| Estimate | Cost = Σ over uncached calls of (`len(system+prompt)/4 × input_per_m` + `max_tokens × output_per_m`), labelled "up to". Time = smoke latency × uncached calls ÷ workers, per model, max across models. Unknown price → "unknown", still runs. |
| Pricing | `pricing.json` in repo, researched + filled during implementation (WebSearch official price pages). |
| Concurrency | asyncio + `AsyncOpenAI` + per-model `asyncio.Semaphore`. |
| Smoke | Thin wrapper over runner's own client. `python smoke.py config.yaml` or `--model provider:model`. Runner calls same fn pre-run. Old native-SDK `smoke.py` replaced. |

## File layout

Two modules on the natural seam: **one call** vs **one run**. `llm.py` is reusable later by
graders/judges (same retry+cache); `run.py` is batch orchestration only.

```
llm.py                     ONE CALL. PROVIDERS registry; make_client(provider, timeout) -> AsyncOpenAI(max_retries=0);
                           cache_key()/cache_get()/cache_put() (atomic write); call() w/ retry loop, error
                           classification, reasoning adapt; smoke(models, timeout) -> list[SmokeResult]
run.py                     ONE RUN. load_config, load_tasks, build jobs, pricing/estimate, dashboard, confirm,
                           execute (asyncio + per-model semaphore), output writer, kill-switch, summary, CLI.
                           python run.py configs/x.yaml --name x [--yes] [--max-retries N] [--timeout S]
smoke.py                   ~25-line CLI over llm.smoke(): python smoke.py configs/x.yaml | --model provider:model
pricing.json               {"provider:model": {"input_per_m": float, "output_per_m": float}}
configs/example.yaml
tests/test_llm.py          cache key determinism/roundtrip; retry classification; backoff; reasoning adapt (mocked client)
tests/test_run.py          config/task validation; kill logic (401 immediate, 4 consecutive); cache-hit path;
                           output line schema; run.json written; existing folder refused
```

Add to `requirements.txt`: `pyyaml`, `tqdm`, `pytest`, `pytest-asyncio`.

## Data shapes

**Config yaml**
```yaml
tasks: data/tasks.jsonl
models:
  - openai:gpt-5-mini
  - anthropic:claude-haiku-4-5
system_prompt: |
  You are ...
temperature: 0.7
max_tokens: 1024
n_samples: 1        # default 1
workers: 4          # per model
max_retries: 4      # optional
timeout: 60         # optional, seconds per call
```

**Task jsonl row** — `id` (str/int, required, unique), `prompt` (str, required). Every other key → `metadata` dict, passed through untouched.

**Job** = (task, provider, model, sample_index). Total jobs = tasks × models × n_samples.

**Output line** (one per job, written on completion, completion order)
```json
{
  "task_id": "...", "provider": "anthropic", "model": "claude-haiku-4-5",
  "system_prompt": "...", "prompt": "...", "sample_index": 0,
  "params": {"temperature": 0.7, "max_tokens": 1024},
  "params_sent": {"max_completion_tokens": 1024},
  "cached": false,
  "response": "..." | null,
  "finish_reason": "stop" | "length" | null,
  "tokens_in": 123 | null, "tokens_out": 45 | null,
  "errors": [ {"attempt": 1, "status": 429, "type": "RateLimitError", "code": "rate_limit_exceeded"|null, "message": "...", "retryable": true} ],
  "started_at": "ISO8601", "ended_at": "ISO8601", "total_time_s": 1.23,
  "metadata": { ...passthrough... }
}
```
`errors` is always present (empty list on clean success). Failed job after retries: `response: null`, tokens null, errors non-empty. Retried-then-succeeded: response set AND errors lists the failed attempts.

**Cache file** `.cache/<key>.json`: `{"response": str, "tokens_in": int, "tokens_out": int, "finish_reason": str}`.

## Interfaces (signatures executor must honour)

```python
# llm.py
PROVIDERS = {"openai": (None, "OPENAI_API_KEY"),
             "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
             "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")}
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")          # openai provider only
RETRYABLE_STATUS = {408, 429} | set(range(500, 600))
FATAL_STATUS = {401, 403}

def parse_model(spec: str) -> tuple[str, str]              # "anthropic:claude-haiku-4-5" -> ("anthropic", "claude-haiku-4-5"); ValueError on bad provider
def make_client(provider: str, timeout: float) -> AsyncOpenAI   # max_retries=0; KeyError-style error if env key blank
def build_params(provider, model, temperature, max_tokens) -> dict   # reasoning adapt lives here; returns exactly what is sent
def cache_key(provider, model, system_prompt, prompt, sample_index, temperature, max_tokens) -> str   # sha256 of json.dumps(sort_keys=True)
def cache_get(key) -> dict | None
def cache_put(key, record: dict) -> None                    # write .tmp then os.replace

@dataclass
class Attempt: attempt: int; status: int | None; type: str; code: str | None; message: str; retryable: bool
@dataclass
class CallResult: response: str | None; tokens_in: int | None; tokens_out: int | None; finish_reason: str | None; errors: list[Attempt]; params_sent: dict; fatal: bool   # fatal = 401/403 seen
async def call(client, provider, model, system_prompt, prompt, temperature, max_tokens, max_retries) -> CallResult
    # loop attempts 1..max_retries+1; classify exception; backoff min(64, 2*2**(attempt-1)) + random(); honor Retry-After
    # empty choices / None content -> Attempt(type="EmptyResponse", retryable=False)

@dataclass
class SmokeResult: spec: str; ok: bool; latency_s: float; error: Attempt | None; content: str
async def smoke(specs: list[str], timeout: float) -> list[SmokeResult]

# run.py
@dataclass
class RunConfig: tasks: str; models: list[str]; system_prompt: str; temperature: float | None; max_tokens: int; n_samples: int = 1; workers: int = 4; max_retries: int = 4; timeout: float = 60
def load_config(path, overrides: dict) -> RunConfig       # yaml + CLI overrides; validation errors -> SystemExit with message
def load_tasks(path) -> list[dict]                         # each: {"id", "prompt", "metadata": {...rest}}; dup id / missing field -> SystemExit
@dataclass
class Job: task: dict; provider: str; model: str; sample_index: int; key: str; cached: dict | None
def build_jobs(cfg, tasks) -> list[Job]
def load_pricing() -> dict                                  # pricing.json
def estimate(cfg, jobs, smoke_results, pricing) -> dict     # per-model {calls, cached, est_in_tokens, cost_max|None, time_s}; totals
def print_dashboard(cfg, name, jobs, est) -> None
def prepare_run_dir(name, yes: bool) -> Path                # results/<name>/; exists -> prompt new name (or SystemExit under --yes)
def write_run_json(run_dir, cfg, cli_args, n_tasks, git_rev) -> None
async def execute(cfg, jobs, run_dir) -> Summary            # per-model semaphores; tqdm; kill-switch; appends raw.jsonl line per job
def main(argv) -> int                                       # 0 ok, 1 pre-run failure (config/smoke/refused), 2 killed, 130 Ctrl-C
```

Test fixture note: openai exceptions need a real `httpx.Response`:
`openai.RateLimitError("rate", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)`.
Mock client = simple object with `chat.completions.create` async fn scripted to raise/return in sequence.

## Control flow (runner.py)

1. Load config, load tasks, validate. Fail with clear message on bad yaml/missing fields/dup ids.
2. Build job list. For each job compute cache key, mark hit/miss.
3. Smoke: one call per distinct `provider:model` in run via `runner.smoke.smoke()` (prompt "Reply with the single word OK.", `max_tokens` 256 so reasoning models don't return empty). Any failure → print model, error type, status, message → exit 1. Record latency per model.
4. Dashboard: tasks file, n tasks, models, n_samples, total jobs, cached, to-run, workers, temp, max_tokens, retries, timeout, output path, per-model est cost + time, totals.
5. Confirm `Proceed? [y/N]` unless `--yes`.
6. Execute: `asyncio.gather` over jobs; each job acquires its model's semaphore; cache hit → write line immediately (no semaphore needed); miss → `client.call()` → on success `cache.put` then write line; on failure write line with errors.
7. Kill-switch: shared state `consecutive_nonretryable`, `killed` flag. On 401/403 or counter ≥ 4 → set flag, cancel pending, let in-flight finish, print grouped error summary + reason, exit 2. Ctrl-C handled same way (exit 130).
8. Summary: total/ok/cached/failed, error counts grouped by `type` and `status`, tokens in/out totals, actual cost (from pricing), wall time.

Progress bar: tqdm over jobs; postfix `err=N (P%) rate=X/s`; tqdm supplies ETA.

Time estimate caveat: smoke latency is for a ~1-token answer; real calls generate up to `max_tokens`. Dashboard labels it "rough floor". Cost is an upper bound. Both honest, neither precise.

## Review pass findings (writing-plans, 2026-09-23)

- Added `finish_reason` to line + cache (truncation visibility).
- Defined smoke pass criterion (200 + usage), not content.
- `temperature` nullable; cache key includes requested value → switching 0.7→null is a cache miss even where both send nothing. Accepted.
- Kill "consecutive" = completion order across concurrent workers. Accepted.
- Test fixtures for openai exception classes documented above.

## client.py detail

- `call(client, model, system, prompt, temperature, max_tokens, max_retries) -> CallResult(response, tokens_in, tokens_out, errors, params_sent)`.
- Messages: `[{"role":"system",...},{"role":"user",...}]`. Anthropic compat supports system role.
- Classification via openai exception classes: `RateLimitError`(429), `APITimeoutError`, `APIConnectionError`, `InternalServerError`(≥500) → retryable. `AuthenticationError`(401), `PermissionDeniedError`(403) → fatal. `BadRequestError`(400), `NotFoundError`(404), `UnprocessableEntityError`(422), other `APIStatusError` → non-retryable. Extract `status_code`, `code` (from `body.error.code` if present), message.
- Backoff: `min(64, 2 * 2**attempt) + uniform(0,1)`; if `Retry-After` header present use it.
- Empty `choices` or `content is None` → treat as non-retryable error type `EmptyResponse`.

## Implementation order (TDD per step; run straight through, no pausing between steps)

1. `llm.py` + `tests/test_llm.py`: providers, client, cache, call() w/ retries/classification/reasoning adapt, smoke().
2. `pricing.json`: WebSearch official price pages for gpt-5, gpt-5-mini, gpt-5-nano, claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-1; OpenRouter entries for same. Real numbers only; unknown → omit.
3. `run.py` + `tests/test_run.py`: config, tasks, jobs, estimate, dashboard, execute, output folder (`raw.jsonl` + `run.json`), kill, summary, CLI.
4. `smoke.py` CLI (replaces old native-SDK version), `configs/example.yaml`, `requirements.txt` (+ pyyaml, tqdm, pytest, pytest-asyncio).
5. `pytest` green. Copy this plan to `.claude/work-plan.md` (repo convention).
6. **STOP and ask** before any live API call (smoke / live run below). Everything above is free.

## Verification

- `pytest` green (mocked; free).
- **[costs money — ask first]** `python smoke.py --model openai:gpt-5-mini` → OK line with latency.
- **[costs money — ask first]** Tiny jsonl (3 tasks), config with 2 models, `n_samples: 2`, `--name t1` → dashboard shows 12 jobs, 0 cached; run; `results/t1/raw.jsonl` 12 lines `cached: false`, `results/t1/run.json` present.
- Rerun same config `--name t2` → dashboard 12 cached / 0 to run / $0; instant; 12 lines `cached: true`, `total_time_s: 0`. (free)
- Rerun `--name t1` → refuses, prompts for new name. (free)
- Bad key in `.env` → smoke fails, run never starts, error shows 401 + message.
- Bogus model name → smoke fails with 404.
- Config with `max_tokens: 1` on a task set → forced weird outputs, confirm errors/empties recorded not crashed.
- Ctrl-C mid-run → partial file intact, cache has completed calls, rerun picks up.


---

# Completion record (2026-09-23)

Plan above executed in full. Session: design Q&A → plan → TDD build → mocked tests → live verification (user ran main path, Claude ran edge cases).

## What was built

| File | Role |
|---|---|
| `llm.py` | ONE CALL: provider registry (openai / anthropic-compat / openrouter, all via OpenAI SDK), `build_params` (reasoning adapt), sha256 cache in `.cache/`, `call()` with retry loop + `Attempt` records, `smoke()` |
| `run.py` | ONE RUN: yaml config, jsonl tasks (`id`, `prompt`, rest → `metadata`), jobs = tasks×models×samples, pricing estimate, dashboard, y/N confirm, `results/<name>/run.json` + `raw.jsonl`, asyncio + per-model semaphore, tqdm, kill-switch, summary |
| `smoke.py` | thin CLI over `llm.smoke()`; old native-SDK smoke replaced |
| `pricing.json` | 73 models, USD/1M tokens, source URL per entry, caveats in `_notes` |
| `configs/example.yaml`, `configs/livetest.yaml` | reference configs (livetest = n_samples 2) |
| `data/example_tasks.jsonl` | 3 tiny tasks with extra fields (metadata passthrough) |
| `tests/` | 79 tests, fully mocked, ~2s, $0 |

## Deviations from plan (all improvements found during verification)

1. **Run name resolved before smoke.** Original order ran smoke first; a name clash wasted 2 paid calls. Now `resolve_run_name()` runs right after config load; folder is created only after confirm.
2. **Truncation surfaced.** `finish_reason == "length"` counted (live + cached) → `Summary.truncated` + WARNING line. Found when `max_tokens: 5` made gpt-5-mini return `''` and it was counted as plain "ok".
3. **Ctrl-C anywhere.** Original handler only wrapped execute; Ctrl-C during smoke gave a raw traceback. `main()` now wraps `_main()` → clean message, exit 130.
4. **`gpt-6` added to reasoning prefixes** after pricing research showed that family exists.
5. **10 files → 2 modules** (user pushback on fragmentation).
6. **`SmokeResult.line()`** so both CLIs format identically.

## Tests run

### Mocked (`python -m pytest tests -q` → 79 passed)
- `test_llm.py` (32): model spec parsing; reasoning adapt for gpt-5.x / gpt-6 / o-series, openai-provider only; null temperature omitted; cache key deterministic + sensitive to every field; atomic cache write; retry on 429/5xx/timeout/connection; backoff 2→64s cap; Retry-After honoured; give-up after max_retries; 400 not retried; 401/403 fatal; EmptyResponse; smoke ok/fail/empty-content/missing-key.
- `test_run.py` (44): config defaults, CLI overrides, unknown-key rejection, bad-value rejection; task parsing, blank lines, missing/dup id, bad prompt; jobs count and keys; identical prompts share key; cache hit detection; estimate math (input+output priced separately, cached excluded); unknown price → None; resolve_run_name free/refuse/prompt/blank-abort; run.json contents; full output line schema; success caching; cache-hit line (`cached`, t=0); failed line no cache; fatal kill; 4-consecutive kill; success resets counter; retryable doesn't count; per-model semaphore peak; truncation count live+cached; error grouping; main exit codes 0/1/2/130; name clash refused before smoke; smoke fail → no folder; declined confirm → no calls.
- `test_smoke.py` (3): CLI exit codes, error print, models from config.

### Live (real APIs, ~$0.15 total)
| Scenario | Result |
|---|---|
| `smoke.py configs/livetest.yaml` (user) | 2/2 OK |
| `run.py livetest --name live1` (user) | 12 fresh calls, all `stop`, 0 errors; gpt-5-mini `params_sent={max_completion_tokens}`, haiku has temp |
| `--name live2` (user) | 12/12 cached, $0, instant, `total_time_s: 0` |
| bogus model names | 404 `model_not_found` / `not_found_error`, exit 1 |
| bad keys via env override | 401 `invalid_api_key` / `authentication_error`, exit 1 |
| `max_tokens: 5` | all `finish_reason: length`, no crash, WARNING in summary |
| existing name `--yes` | refused before smoke, exit 1 |
| existing name interactive, blank | prompt → abort, exit 1 |
| 401 mid-run (key swapped after smoke) | in-flight jobs 401, killed, 10 not run, exit 2 |
| 4× consecutive 404 (model renamed after smoke) | killed on 4th, 5th in-flight still logged, 7 not run, exit 2 |
| SIGINT 12s into execute | 2 completed lines valid, cache +2, no `.tmp`, exit 130 |
| resume after SIGINT | dashboard `2 cached + 10 to run`, clean, exit 0 |

## Known limits / gotchas for next agent
- **Cache key = call signature, not task id.** Two tasks with the same prompt share one cache entry.
- **Anthropic pricing ids inferred** from display names (`claude-<family>-<major>-<minor>`); unknown id → "unknown (no price)", run proceeds.
- **Time estimate is a floor** (smoke latency × calls / workers); real calls with long outputs take longer.
- **Kill "consecutive" = completion order** across concurrent workers.
- **Reasoning prefix list is hardcoded** (`llm.REASONING_PREFIXES`); a new reasoning family → 400 on temperature → kill → add prefix or set `temperature: null`.
- Bash heredocs with apostrophes break in this environment; use the Write tool for source files.
