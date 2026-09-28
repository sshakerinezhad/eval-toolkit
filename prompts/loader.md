<!-- source: mine. Builder prompt: load a new dataset into the toolkit's shapes. Fill the two brackets. -->
Read first, write second. No model calls.

The data is at: [PATH]. The brief says: [ONE SENTENCE].

1. Read three rows. Tell me in plain words: the fields, what each holds, its length, and anything odd (missing values, referenced files, duplicate ids). Stop and wait for my ok.

2. Then write one loader so this data runs through the toolkit as it is. Three shapes:
   - Runner tasks for run.py: one row per task, {"id", "prompt", ...}; any other field (source, domain, prompt_raw) rides along as metadata. Attached files go as text under the prompt, one "=== <filename> ===" block each, the way run.py's APEX loader does it. A file that is not text: stop and tell me its name and size. Do not guess its contents.
   - Rubrics for grade.py: {task_id: [{"id", "description", "weight", "depends_on"}, ...]} in rubric order. Carry weight and dependency fields when the data has them, else null.
   - Reference answers, if the data has them: {task_id: text}.
   - Responses already given (no inference to run): one raw.jsonl line per response in run.py's shape, {"task_id", "model", "sample_index", "response", "metadata": {"prompt_raw": the task prompt}}, so grade.py build reads them as if run.py had made them.
   If the data does not fit these shapes (precomputed scores, trajectories, pairwise preferences, no rubric), do not force it: stop, propose the smallest mapping in three lines, and wait for my ok. Never invent a field the data does not have.

3. Put it in loaders/<name>.py, one function per shape, a docstring that states the file format in three lines, and a test on two real rows in tests/test_loader_<name>.py. Wire it into run.py and grade.py build with the smallest change (by file extension or a --format flag).

4. Show me the diff before committing. One sentence per function on why it exists. No other changes. No new dependencies unless I say so.

5. Then a brief, three lines max: the input shape, the output shape, and why that mapping. Nothing else.
