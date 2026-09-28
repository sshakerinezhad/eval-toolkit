"""DEMO. The whole chain on toy tasks with a fake model, a fake judge and a fake classifier. No keys, no
network, nothing cached outside results/demo/.  python demo.py

12 toy tasks (each asks for a count and a total), 2 fake models (one right 80% of the time, one 60%,
one scattergun answer), 2 samples each, gold answers, a fake judge that checks the number is in the response
(one reply unparseable, one timed out), 10 hand labels, the failures, a fake classifier, plots, deliverables.
Every step is the real command; only llm.call, llm.smoke and llm.make_client are replaced.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import classifier
import deliver
import grade
import llm
import plot
import run

HERE = Path(__file__).resolve().parent
DEMO = Path("results/demo")
MODELS = ["openai:demo-a", "openai:demo-b"]
JUDGE = "openai:demo-judge"
CLASSIFIER = "openai:demo-classifier"


def toy_tasks() -> tuple[list[dict], dict[int, tuple[int, int]]]:
    truth = {i: (i * 3 + 1, i * 40 + 7) for i in range(1, 13)}
    tasks = [{"id": i, "prompt": f"Item {i}: report the count and the total in one sentence. The ledger shows a count of {c} and a total of {t}.",
              "prompt_raw": f"Item {i}: report the count and the total.", "truth": [c, t]}  # extra fields ride as metadata
             for i, (c, t) in truth.items()]
    return tasks, truth


def fake_model(model: str, prompt: str) -> str:
    i = int(re.search(r"Item (\d+)", prompt).group(1))
    c, t = int(re.search(r"count of (\d+)", prompt).group(1)), int(re.search(r"total of (\d+)", prompt).group(1))
    good = (i * 7) % 10 < (8 if model == "demo-a" else 6)
    if model == "demo-b" and i == 3:
        return f"The count is {c}. The total is {t}, or possibly {t + 10}."  # scattergun
    return f"The count is {c} and the total is {t}." if good else f"The count is {c + 1} and the total is {t - 5}."


def fake_judge(prompt: str, n: int) -> str | None:
    if n == 7:
        return "I think this one is met."          # unparseable, once
    if n == 13:
        return None                                # timeout, once
    crit = re.search(r"<CRITERION>\n(.*?)\n</CRITERION>", prompt, re.S).group(1)
    resp = re.search(r"<RESPONSE>\n(.*?)\n</RESPONSE>", prompt, re.S).group(1)
    want = re.search(r"(\d+)", crit).group(1)
    met = want in resp and not ("or possibly" in resp and "total" in crit)
    return json.dumps({"rationale": f"looked for {want}", "is_criteria_true": met})


def label_from_truth(shown: str) -> str:
    """What a careful human would type for one judge call: y when the criterion's number is in the response."""
    crit = re.search(r"CRITERION: (.*)", shown).group(1)
    resp = shown.split("chars):\n", 1)[1]
    want = re.search(r"(\d+)", crit).group(1)
    return "y" if want in resp and not ("or possibly" in resp and "total" in crit) else "n"


def fake_classifier(prompt: str) -> str:
    cat = "scattergun" if "or possibly" in prompt else "wrong_value"
    return json.dumps({"category": cat, "why": "the number differs"})


def install_fakes():
    """Replace the three network functions in llm with fakes. Returns a function that puts them back."""
    saved = (llm.call, llm.smoke, llm.make_client, llm.CACHE_DIR)
    counter = {"judge": 0}

    async def call(client, provider, model, system_prompt, prompt, temperature, max_tokens, max_retries):
        if model == "demo-judge":
            counter["judge"] += 1
            text = fake_judge(prompt, counter["judge"])
            if text is None:
                return llm.CallResult(None, None, None, None, errors=[llm.Attempt(1, None, "APITimeoutError", None, "demo timeout", True)])
        elif model == "demo-classifier":
            text = fake_classifier(prompt)
        else:
            text = fake_model(model, prompt)
        return llm.CallResult(text, len(prompt) // 4, len(text) // 4, "stop")

    async def smoke(specs, timeout):
        return [llm.SmokeResult(s, True, 0.01, None, "ok") for s in specs]

    llm.call = call
    llm.smoke = smoke
    llm.make_client = lambda provider, timeout: object()
    llm.CACHE_DIR = DEMO / ".cache"

    def restore():
        llm.call, llm.smoke, llm.make_client, llm.CACHE_DIR = saved
    return restore


def write_inputs(tasks: list[dict], truth: dict) -> None:
    DEMO.mkdir(parents=True)
    (DEMO / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    with open(DEMO / "rubrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task ID", "Rubric JSON"])
        for i, (c, t) in truth.items():
            w.writerow([i, json.dumps({"c1": {"description": f"States the count {c}"}, "c2": {"description": f"States the total {t}"}})])
    (DEMO / "gold.json").write_text(json.dumps({str(i): f"The count is {c} and the total is {t}." for i, (c, t) in truth.items()}))
    (DEMO / "run.yaml").write_text(f"tasks: {DEMO / 'tasks.jsonl'}\nmodels: {MODELS}\nsystem_prompt: ''\nmax_tokens: 100\nn_samples: 2\nworkers: 4\n")
    (DEMO / "judge.yaml").write_text(f"tasks: {DEMO / 'judge_tasks.jsonl'}\nmodels: ['{JUDGE}']\nsystem_prompt: ''\nmax_tokens: 100\nworkers: 8\n")
    (DEMO / "classify.yaml").write_text(f"tasks: {DEMO / 'judge' / 'failures.jsonl'}\nmodels: ['{CLASSIFIER}']\nsystem_prompt: ''\nmax_tokens: 100\n")


def step(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    tasks, truth = toy_tasks()
    write_inputs(tasks, truth)
    restore = install_fakes()
    try:
        return _chain()
    finally:
        restore()


def _chain() -> int:
    step("1. run.py: two fake models, 12 tasks, 2 samples")
    assert run.main([str(DEMO / "run.yaml"), "--name", "demo/full", "--yes"]) == 0
    step("2. grade.py build (+ gold): one judge question per criterion per sample")
    grade.build(DEMO / "full/raw.jsonl", DEMO / "rubrics.csv", HERE / "prompts/apex_judge.md", DEMO / "judge_tasks.jsonl", gold=DEMO / "gold.json")
    step("3. run.py: the fake judge")
    assert run.main([str(DEMO / "judge.yaml"), "--name", "demo/judge", "--yes"]) == 0
    step("4. grade.py label: 10 hand labels (answered here from the truth, no keyboard)")
    shown: list[str] = []
    grade.label(DEMO / "judge_tasks.jsonl", 10, DEMO / "judge/judge_labels.csv", ask=lambda _: label_from_truth(shown[-1]), say=shown.append)
    step("5. grade.py score --labels: scores, metrics, judge errors, gold apart, judge vs labels")
    grade.score(DEMO / "judge/raw.jsonl", DEMO / "judge_tasks.jsonl", baseline="demo-a", labels=DEMO / "judge/judge_labels.csv")
    step("6. grade.py failures --prompt --label: failures.md, failures.jsonl, 5 failure labels")
    shown = []
    grade.failures(DEMO / "judge/raw.jsonl", prompt=HERE / "prompts/classify_criteria.md", label_n=5,
                   ask=lambda _: "scattergun" if "or possibly" in shown[-1] else "wrong_value", say=shown.append)
    step("7. run.py: the fake classifier; classifier.py score --labels")
    assert run.main([str(DEMO / "classify.yaml"), "--name", "demo/classify", "--yes"]) == 0
    classifier.score(DEMO / "classify/raw.jsonl", DEMO / "judge/failure_labels.csv")
    step("8. plot.py: means, paired, taxonomy")
    for p in (plot.means(DEMO / "judge/metrics.json"), plot.paired(DEMO / "judge/scores.jsonl"), plot.taxonomy(DEMO / "classify/scores.json")):
        print(f"wrote {p}")
    step("9. deliver.py: deliverables, zip, email body")
    deliver.main([str(DEMO / "judge"), "--out", str(DEMO / "deliverables")])
    print(f"\ndemo ok -> {DEMO}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
