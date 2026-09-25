# Scratchpad — session handoff

## This session (2026-09-24): APEX-v1 runner for gpt-5.6-luna / gpt-5.6-sol

**Goal:** run APEX-v1 `apex-v1/data/train.csv` tasks through `openai:gpt-5.6-luna` + `openai:gpt-5.6-sol`,
save task id, domain, prompt sent, full response, errors. Rubric deliberately dropped (grading later).
Plan: `C:\Users\User\.claude\plans\we-re-gonna-set-up-vivid-bee.md` (user-approved CUT scope).

**State: built, tests green (124 passed), live 2-task test done. NOT committed** (branch `mock2`; also
pre-existing uncommitted `.gitignore` change adding `apex-v1/`).

**Built (why in parens):**
- `run.py`: `RunConfig.task_ids` subset filter; `load_tasks(path, task_ids)` dispatches `.csv` ->
  `_load_apex_csv`, `.jsonl` -> `_load_jsonl` (old logic); shared `_select()` (unknown id -> die).
  Filter runs BEFORE attachments are read (63/100 rows reference PDFs not on disk; first draft died on them).
  Prompt = task prompt + `\n\n==== Attached files content: ====\n\n` + `=== <bare filename> ===\n<text>`
  blocks joined by blank lines (user-specified Mercor harness format; bare name because 6/30 prompts cite
  files by bare name). Attachments read `utf-8-sig` (some BOM) + read_text normalises CRLF. Missing or
  non-text attachment (.pdf/.docx/.xlsx) -> die naming task+file. metadata = {domain, prompt_raw, attachments}.
- `llm.py`: `FATAL_CODES = {"insufficient_quota"}` -> fatal, not retried (it's a 429 that never recovers).
- `configs/apex.yaml` (30 ids, reserve 7 in comment), `configs/apex_test3.yaml` (145, 2205, 2302).
  No temperature; max_tokens 32000 (includes hidden reasoning); timeout 600; workers 8; retries 4;
  system prompt "You are a helpful assistant." (placeholder, user never confirmed wording).
- `tests/test_run.py`: +2 tests (prompt-then-docs order incl. BOM/CRLF + out-of-subset missing file ignored;
  missing attachment dies).

**Cache:** unchanged key (provider+model+system+prompt+sample+temp+max_tokens). Docs are inlined into
`prompt`, so file contents are in the key automatically: edit a CSV -> only that task reruns.

**Data facts:** train.csv 100 rows (Task ID, Domain, Prompt, Rubric JSON, File Attachments = newline paths
relative to `apex-v1/data/`). Only 40 CSV attachments on disk; 63 tasks have zero files present (PDFs etc).
Subset 30 + reserve 7 are all CSV-only, all present. Subset = Consulting 20 + Finance 10, NO Legal/Medicine.
Reserve: 804, 1122, 1150, 1169, 2121, 2266, 2315.

**Live test (user said "just run it"):** 2205 + 2302 x both models, NO cache, via scratch script calling
`llm.call` directly (scratch file lives in temp dir, will be gone). Output: `results/tests/apex_test2/`
(`raw.jsonl` + one `.md` per response). All 4 ok, finish=stop, 0 retries, ~$0.57 total.
Findings: output mostly hidden reasoning (luna 4.4k tokens -> 915 visible chars); chars/4 estimate ~2x LOW
for numeric CSVs (2302: est 17.6k, real 38,450 in) -> dashboard underestimates input cost; sol 218s on 2302.

**Pending / next:**
- `--no-cache` flag on run.py: proposed, user REJECTED the edit mid-way; don't add unless asked.
- Full 30-task run: `python run.py configs/apex.yaml --name <name>` (user decides/runs).
- Commit this work (user hasn't asked yet).
- Deferred by user ("not now"): error categories, `turns` field, attachment sha256, more tests,
  work-plan update, PDF native file parts (needed if non-CSV tasks ever used).
- Maybe: better token estimate for CSV-heavy prompts.

**User prefs this session:** wants terse, action over questions; got frustrated at repeated confirmation
asks. Explicitly told me to run the paid live test myself this time (memory says user runs live calls;
treat "just run it" as override for that instance). Pasted cut-list = approval format they use.

## Previous session (2026-09-23): inference runner
Details in `.claude/changelog/2026-09-23-inference-runner.md`. `llm.py` (one call: providers via OpenAI SDK,
own retry loop, disk cache `.cache/`, smoke) + `run.py` (one run: dashboard/estimate, per-model semaphore,
kill-switch 401/403 instant or 4 consecutive non-retryable, append-only `results/<name>/raw.jsonl` + run.json).
Since then also committed: `metrics.py` (pass@k, pass^k, paired bootstrap) and `judge.py` (LLM-judge tasks
from traces). `pricing.json` Anthropic ids inferred from display names; verify on use.
