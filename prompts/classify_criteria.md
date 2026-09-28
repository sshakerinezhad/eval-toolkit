<!-- source: mine. Failure taxonomy over rubric criteria. Edit the category list for the dataset (from the task, or from your own reading of failures.md); the code reads categories from the '- name: description' lines. -->
You are given one failed rubric check: the criterion, the judge's reason it was not met, and the model's response.
Pick exactly one category from the list below.
Reply with one JSON object and nothing else:
{"category": "<name>", "why": "<one sentence, quoting the response or the criterion>"}

Categories:
- missing: the response never addresses what the criterion asks for
- wrong_value: the response gives a value, and it is not the expected one
- precision_or_unit: right quantity, wrong unit, rounding or precision
- scattergun: several different values given for the one quantity the criterion names
- unsupported: a claim made with no working or source when the criterion asks for one
- format: the content is there but not in the form the criterion requires (table, section, file)
- judge_doubt: the response looks correct to you; the judge or the criterion may be wrong

If multiple categories apply, pick the one higher in the list.
