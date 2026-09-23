# Scratchpad — session handoff (2026-09-23)

## Working-tree state (branch mock1, nothing committed this session)
Untracked: `passk.py`, `tests/test_passk.py` (session B), `grader.py`, `traces.py`, `tests/helpers.py`,
`tests/test_grader.py`, `tests/test_traces.py` (session C). Modified: `llm.py`, `tests/test_llm.py`.
Full `pytest`: 140 pass. Two sessions edited grader.py/traces.py concurrently — diff before committing.

## Session C: LLM-judge grader (built, tested, RUN on 397B — results in results/grade_q397b_sonnet/)
Plan file: `~/.claude/plans/claude-scratchpad-md-pasted-content-id-cuddly-sprout.md`.
- `traces.py`: `load_variant(dir)` → `Run`; `features()` (termination, dead-call streaks, confirm prompts,
  backed_off, sleep/repeat cmds, giveup, stalled); `pre_label()` INFRA_ERROR / CONTEXT_OVERFLOW_LOOP /
  SERVING_STALL / PASS / None; `render_transcript()` numbered turns, dead streaks collapsed, tests footer.
- `grader.py`: 13-category taxonomy + 1-5 rubrics in SYSTEM_PROMPT; prompt BLIND to variant; strict JSON
  parse + 1 repair retry; `judge_run` = cache_key→cache_get→llm.call→cache_put; asyncio semaphore +
  kill switch; `--fails-only` (judge only pre_label None), `--max-prompt-tokens N` (skip monsters; skipped
  runs still recorded as UNJUDGED so pass rates/pre-labels show); `--dry-run`, `--report-only`, `--task`.
  Output `results/grade_<name>/{judgments.jsonl (append, last wins), scores.csv, summary.md, tasks/*.md}`.
- `llm.py`: optional `extra_body` (reasoning_effort) on build_params/cache_key/call; old cache keys valid.
- Cost lesson: terminal text tokenizes at ~2.2 chars/token (CHARS_PER_TOKEN=2.2), judge output ~2.5k
  tok/run with thinking. Full Opus 5.5 run would be ~$90 — user capped at $10.
- RUNS DONE: `grade_pilot` (Opus 5.5, qemu-alpine-ssh, 6 runs, excellent judgments, ~$1.2);
  `grade_q397b` (Opus, killed after 19 runs, ~$1.9, cached); `grade_q397b_sonnet` = Sonnet 5,
  `--fails-only --max-prompt-tokens 50000` → 135 judged (T 58 / U 77), 64 ambiguous fails skipped as
  too long, actual $10.21, 9 min, 0 parse errors.
- Headline (judged fails only, biased toward short/declared-done runs): T vs U primary category —
  IMPL_BUG 19/34, WRONG_APPROACH 14/19, VERIFIED_BUT_WRONG 14/10, UNVERIFIED_CLAIM 5/7, MISREAD 4/5.
  false_completion 90% vs 92%. verification mean 2.81 vs 2.53. Secondary UNVERIFIED_CLAIM 16 vs 27.
  Untrained fails more by IMPL_BUG; trained's remaining fails skew to VERIFIED_BUT_WRONG (tested own way,
  hidden tests disagree) → next: hand-read those 14 in tasks/*.md; check for spec-ambiguity tasks.
- Next: (1) hand-read `results/grade_q397b_sonnet/tasks/*.md` for the 23 contested tasks; (2) optionally
  judge the 64 skipped monsters with a turn-window (head/tail) renderer (not built; ~$6 Sonnet); (3) 35B
  folders same command; (4) commit.

## Session B: pass@3 script (done, verified)
Built `passk.py` — one file, stdlib only, ~100 lines. Tests `tests/test_passk.py` (10 pass).
- `load_variant(root)` globs `tasks/*/traj_*.json`, score = `trajectory.trajectory_output.score`.
  Groups by task dir. Counts `trajectory_status != completed`.
- `pass_at_k(scores,k)` unbiased estimator `1 - C(n-c,k)/C(n,k)` (n=k=3 ⇒ any-pass). `--k 1` = pass@1.
- `make_draws` + `bootstrap_ci`: **task-level** resample, B=10000, seed=0, percentile 95%. Same draws
  reused for trained, untrained, and **paired diff** (same 89 tasks in both).
- `main()` prints per-task table + summary, writes `results/passk_<model>_k<k>.csv` (gitignored).
- Flags: `--model` (default Qwen3p5-397B; works for Qwen3p6-35B), `--k`, `--boot`, `--seed`.

Decisions: score source = traj file, not index (index `final_score` None for 3 infra runs; file 0.0;
counted as fail, reported as `errored_trajs`). Bootstrap unit = task not trajectory (3 samples/task
correlated). Paired diff CI is the real trained-vs-untrained test.

Results Qwen3p5-397B (89 tasks × 3):
| | pass@3 | CI | pass@1 | CI |
|---|---|---|---|---|
| trained | 0.674 | [0.573,0.764] | 0.554 | [0.461,0.644] |
| untrained | 0.640 | [0.539,0.742] | 0.506 | [0.412,0.599] |
| diff paired | +0.034 | [-0.022,+0.090] | +0.049 | [+0.004,+0.097] |
k=3 flips: 5 trained-only, 2 untrained-only, 55 both, 27 neither. pass@1 matches raw 148/267, 135/267.
Not yet run on Qwen3p6-35B (`python passk.py --model Qwen3p6-35B`).

## Session A: trained vs untrained analysis (earlier, analysis only)
Compared Qwen3.5-397B **trained** (`mercor/qwen35-397b-a17b-gs264`, batch `qwen-397b-post-trained`) vs
**untrained** (`qwen35-397b-a17b-base`, batch `qwen-397b-baseline`) on Terminal-Bench 2.1 traces in
`tbench-traces/Qwen3p5-397B_{trained,untrained}_traces/`. Groundwork for grader.

### Data layout (verified)
- `trajectories_index.json`: list of 267 rows per variant (89 tasks x 3 trials). world_name = category.
- `tasks/<task>__task_<id>/traj_*.json` → `{"trajectory": {...}}`: `trajectory_messages`
  (user/assistant/tool; first user msg = Terminus system prompt + "Task Description:"), `final_answer`,
  `trajectory_output` (score, test_statuses, tests_passed/total, duration_seconds, command_history,
  exception_type, error_message, `test_summary_metadata.test_stdout_tail`, `usage_metrics.call_log`).
- Harness = Terminus: model returns JSON {analysis, plan, commands[keystrokes,duration], task_complete};
  stored assistant content keeps ONLY "Analysis/Plan" text — commands + task_complete flag dropped.

### Key results
- Pass@1: trained 55.4% vs untrained 50.6% (+4.9pp). Task-level: 16 T-better, 7 U-better, 66 tie
  (30 both 3/3, 27 both 0/3). Wilcoxon p≈0.019. 12/16 T wins by 1 trial → borderline.
- Behavioral gap: trained handles interactive processes (qemu/tmux/vim/heredoc) better + verifies
  externally (qemu-alpine-ssh 3/3 vs 0/3, qemu-startup, fix-ocaml-gc). False-completion rate T 34.5% vs
  U 42.0% (p=0.12). Model never admits failure.
- Efficiency on 30 both-pass tasks: no difference. Trained failures run ~20% longer / more tokens.

### Confounders (verified)
1. **Context-overflow death loop**: prompt ≥~176-180k (262k ctx − 81,920 max_tokens) → 0-token calls →
   harness injects "Technical difficulties. Please continue with the task." → loops to 3h cap. T 24 / U 22.
2. **3h wall cap** (≥10,800s): T 44 / U 35 runs; ~75% of all tokens.
3. **Stalled trained runs** (≥10,000s, <60 calls): 5, all trained — polyglot-rust-c x3, write-compressor,
   circuit-fibsqrt. Likely serving stall → polyglot-rust-c (1/3 vs 3/3) probably infra.
4. Infra zeros: regex-chess U ConflictError, rstan-to-pystan T bb315822 (HTTP 504), feal-differential T
   VerifierTimeoutError.
- Suspect tasks: mteb-leaderboard, mteb-retrieve, install-windows-3.11, sanitize-git-repo,
  model-extraction-relu-logits, build-pov-ray, torch-pipeline-parallelism.

### Grader detection rules (seeded; check traces.py — may already implement)
- Confirm prompt: tool msg containing "Are you sure you want to mark the task as complete".
- Dead call: assistant content starts "Technical difficulties" == call_log completion_tokens 0.
  call_log aligns 1:1 with assistant msgs. Some assistant content None — null-guard.
- Cap: duration ≥10,800. Stall: ≥10,000s & <60 calls. Infra: exception_type set or tests_total below norm.
- Unreliable: index reasoning_tokens/tool_calls (always 0), empty_response_loop, compaction_count,
  outcome, command_history (lossy), final_answer (placeholder ~38 runs), last-role==tool.
- Best failure signal: `test_summary_metadata.test_stdout_tail`.
- Reset fixtures before scoring (make-doom edits vm.js; break-filter wrote /tests/filter.py).

### Old artifacts (session temp dir, may be gone)
`C:\Users\User\AppData\Local\Temp\claude\C--Users-User-eval-toolkit\c770b045-1056-4e79-96aa-4611531250ec\scratchpad\`
— extract.py, behavior.py, trajs.csv, paired.csv, behavior.csv, calls.csv. Superseded by traces.py if
that covers the same features.

## Next steps
1. Decide commit: passk.py + test alone, or together with grader/traces work. Run full `pytest` first
   (llm.py modified — verify test_llm still green).
2. Run `python passk.py --model Qwen3p6-35B` if 35B numbers wanted.
3. Grader: read grader.py/traces.py docstrings, reconcile with detection rules above, then brainstorm
   per CLAUDE.md before further implementation. User runs any paid judge calls.
4. Consider re-runs (user runs): 5 stalled trained runs + 3 infra failures.
