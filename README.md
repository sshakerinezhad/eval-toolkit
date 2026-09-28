# eval-toolkit

A small harness for evaluating LLMs on rubric-graded tasks: run models, judge every criterion with an LLM,
validate the judge against hand labels, report numbers with task-level bootstrap intervals, read and
classify failures, plot, deliver. Plain Python, one file per step, every step a command you can rerun.

## The flow, file to file

```
data (csv / jsonl)            -> run.py        -> results/<run>/raw.jsonl        one line per (task, model, sample)
raw.jsonl + rubrics [+ gold]  -> grade.py build -> results/<judge>/tasks.jsonl   one judge question per criterion
tasks.jsonl                   -> run.py        -> results/<judge>/raw.jsonl      the judge's yes/no per criterion
tasks.jsonl                   -> grade.py label -> judge_labels.csv               your own yes/no on 20 to 40 of them
judge raw.jsonl [+ labels]    -> grade.py score -> scores.jsonl, metrics.json     scores per sample, means, CI, judge vs you
judge raw.jsonl               -> grade.py failures -> failures.md, failures.jsonl, failure_labels.csv
failures.jsonl                -> run.py -> classifier.py score --labels               taxonomy counts, agreement with you
metrics.json / scores.jsonl   -> plot.py       -> plot_*.png + plot_*.json        the picture and the numbers it drew
results/<judge>/              -> deliver.py    -> deliverables/, zip, email body
```

Two entry points: with inference to run, start at `run.py`; with responses already given, write them in
`raw.jsonl`'s shape (the loader prompt's fourth shape) and start at `grade.py build`.

## Commands, in order

```bash
source .venv/bin/activate                     # or: pip install -r requirements.txt
cp .env.example .env                          # keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
python smoke.py configs/apex.yaml             # one tiny call per model; exit 0 = keys and models alive
python -m pytest tests -q                     # green before anything else
python peek.py data/tasks.jsonl --n 5         # five random rows: fields, lengths, first 200 chars each

python run.py configs/apex.yaml --name full   # dashboard, cost estimate, confirm, then the run
python grade.py build --runs results/full/raw.jsonl --data apex-v1/data/train.csv --out results/judge/tasks.jsonl [--gold gold.json]
python run.py configs/apex_grade.yaml --name judge          # tasks: results/judge/tasks.jsonl
python grade.py label results/judge/tasks.jsonl --n 40      # judge running or not; y / n / q
python grade.py score results/judge/raw.jsonl --tasks results/judge/tasks.jsonl --baseline <model> --labels results/judge/judge_labels.csv
python grade.py failures results/judge/raw.jsonl --prompt prompts/classify_criteria.md --label --n 20
python run.py configs/<classifier>.yaml --name classify      # tasks: results/judge/failures.jsonl
python classifier.py score results/classify/raw.jsonl --labels results/judge/failure_labels.csv
python plot.py --kind means results/judge/metrics.json
python plot.py --kind paired results/judge/scores.jsonl
python plot.py --kind taxonomy results/classify/scores.json
python deliver.py results/judge                              # deliverables/, zip, email body printed
python clock.py "minute 25 launch"                           # empty commit with the mark; git log is the timeline
python demo.py                                               # the whole chain on toy tasks, fake model and judge, no keys
```

A fresh repo for a new evaluation: `./new-eval.sh NAME` copies the toolkit, inits git, runs smoke and
tests, prints the clock start line.

## Shapes between steps

- **Tasks in** (`run.py`): jsonl `{"id", "prompt", ...}` (every other field rides along as metadata), or APEX `train.csv` (attachments inlined
  under the prompt as `=== <file> ===` blocks; pdf, xlsx, docx converted by `attachments.py`; anything else stops).
- **raw.jsonl** (`run.py`): one line per call: `task_id, provider, model, sample_index, prompt, response,
  finish_reason, tokens_in, tokens_out, errors[], cached, metadata`. Failed calls have `response: null` and the
  error attempts in `errors`. Finished calls are cached by the full call signature; a rerun with a new
  `--name` only pays for what is new.
- **Judge tasks** (`grade.py build`): id `task|model|sample|criterion`; prompt = judge prompt + `<TASK_PROMPT>` +
  `<RESPONSE>` + `<CRITERION>`; fields `task_id, model, sample_index, criterion_id` travel as metadata.
- **Judge reply**: `{"rationale", "is_criteria_true"}`; a fence is tolerated; anything else is a judge error.
- **scores.jsonl** (`grade.py score`): `task, variant, sample, score` (met / total, errors as fail),
  `score_errors_dropped`, `errors`. **metrics.json** (`metrics.py`): per variant `mean`, `pass@k`, `pass^k`
  with 95% task-bootstrap intervals; `diff` (B minus A, paired, with up/down/same) when two variants;
  `judge_errors` (timeouts, api errors, unparsed; mean both ways per model).
- **judge_labels.csv**: `id, task_id, model, sample_index, criterion_id, label` (y/n). `score --labels`
  prints accuracy, precision and recall on Met, Cohen's kappa, n, base rate, disagreements.
- **failures.md**: one block per not-met criterion. **failures.jsonl**: classifier tasks in `classifier.py`'s row
  shape (`trajectory_id`, `variant`, `categories`). **failure_labels.csv**: `trajectory_id, category`.
- **plot_*.json**: the exact series each PNG was drawn from.

## Code walk, one line per file

- `llm.py`: one call. Provider registry (all through the OpenAI SDK), disk cache, retries with exact error
  records, `smoke()`.
- `run.py`: one run. Config, tasks, jobs, per-model semaphore, kill switch, dashboard and cost estimate, raw.jsonl.
- `attachments.py`: file to text (pdf, xlsx, docx), cap, per-task character report.
- `grade.py`: build judge questions per criterion and sample; score them (errors bucketed, gold apart);
  hand labels and agreement; failures and their labels.
- `classifier.py`: trajectory classifier (agent traces to a category) and the category scorer used by the
  failure taxonomy.
- `metrics.py`: pass@1 / pass@k / pass^k, task-level bootstrap, paired diff.
- `plot.py`: means, paired, taxonomy figures with their data.
- `peek.py`: look at a data file before touching it. `deliver.py`: package and email body. `clock.py`: timeline
  marks. `smoke.py`: keys and models alive. `demo.py`: the chain end to end with fakes, in results/demo/.
- `prompts/`: `apex_judge.md` (Mercor's APEX judge prompt, verbatim), `classify_failures.md` (trajectories),
  `classify_criteria.md` (rubric failures), `loader.md` (builder prompt for a new data shape). Every prompt
  file opens with a provenance header line that `load_prompt` strips.
- `metrics.py` resamples **tasks**, never criteria or samples: criteria within a task are correlated, and the
  task is the unit the claim is about.

## Conventions

- Finished calls are cached; new `--name` per rerun. Judge errors are counted apart and the mean is reported
  both ways. Gold answers, when the data has them, are graded as a check on the judge and never compared as
  a model. Nothing here calls a model except `run.py` and `smoke.py`.
