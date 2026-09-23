# Scratchpad — session handoff

## Last session (2026-09-23): inference runner built, verified, shipped

Archived full plan + completion record (what was built, every test run, live results, deviations) to
`.claude/changelog/2026-09-23-inference-runner.md`. Read that first.

**State:** done and committed to main. `llm.py` (one call) + `run.py` (one run) + `smoke.py` CLI +
`pricing.json` + `configs/`, `data/`, `tests/` (79 mocked tests, `python -m pytest tests -q`).
Live-verified: happy path, cache reruns, 401/404/truncation, kill-switch (fatal + 4-consecutive),
Ctrl-C + resume. User's real runs live in `results/live1`, `results/live2` (gitignored).

**Usage:**
```
python smoke.py configs/x.yaml              # one tiny call per model, same client path as runner
python run.py configs/x.yaml --name NAME    # dashboard + estimate, y/N, results/NAME/{run.json,raw.jsonl}
```

**Key decisions (why):**
- OpenAI SDK only; Anthropic via its OpenAI-compat endpoint. One client type, three base URLs.
- SDK retries OFF (`max_retries=0`); own retry loop so every attempt is logged in `errors`.
- Cache key = provider+model+system+prompt+sample_index+temp+max_tokens (NOT task id). Successes only.
- Per-model semaphore (rate limits are per provider). Kill: 401/403 instant; 4 consecutive non-retryable.
- `temperature` nullable; OpenAI gpt-5*/gpt-6*/o* auto-drop temp + use `max_completion_tokens`.
- One results folder per run, append-only `raw.jsonl`, `run.json` snapshot written before first call.

**Pending / not done:**
- archive-plan skill step 5 (CLAUDE.md learnings) — proposed to user in chat, NOT applied, awaiting approval.
- No grader/judge yet. `llm.call()` + cache designed to be reused for that.
- `pricing.json` Anthropic ids were inferred from display names; verify when a new model is used.

**User prefs learned (also in auto-memory):** few files over many; after plan approval run all steps
without pausing; never execute paid API calls, user runs them and watches; ask, don't invent.
