"""Synthetic Terminal-Bench trace fixtures shared by test_traces / test_grader."""
import json

from traces import DEAD

A = "assistant"
T = "tool"


def msg(role, content):
    return {"role": role, "content": content}


def write_run(root, task="terminal_bench_foo", tid="aaa", *, score=0.0, msgs=None, duration=100.0,
              n_calls=None, exception=None, tests_passed=1, tests_total=2, tail=None, cmd_history="",
              task_text="Do the thing.", world="Terminal-Bench 2.1 / Sci"):
    """Write one traj file + append to trajectories_index.json. msgs excludes the harness user msg."""
    # tid in the default text keeps sibling trials byte-distinct: identical prompts share one cache entry
    msgs = msgs if msgs is not None else [msg(A, f"Analysis: x {tid}\nPlan: y"), msg(T, "New Terminal Output:\nroot@modal:/app#")]
    full = [msg("user", "harness prompt ... Task Description:\n" + task_text)] + msgs
    call_log = [{"prompt_tokens": 100, "completion_tokens": 0 if (m["content"] or "").startswith(DEAD) else 50}
                for m in full if m["role"] == A]
    if n_calls is not None:
        call_log = call_log[:n_calls] + [{"prompt_tokens": 1, "completion_tokens": 1}] * max(0, n_calls - len(call_log))
    out = {
        "score": score, "tests_passed": tests_passed, "tests_total": tests_total,
        "test_statuses": {"test_outputs.py::t_a": "pass", "test_outputs.py::t_b": "fail" if score < 1 else "pass"},
        "duration_seconds": duration, "exception_type": exception, "error_message": "boom" if exception else None,
        "command_history": cmd_history,
        "test_summary_metadata": {"reward_source": "reward.txt", **({"test_stdout_tail": tail} if tail else {})},
        "usage_metrics": {"call_log": call_log, "prompt_tokens": 100 * len(call_log), "completion_tokens": 50,
                          "total_tokens": 150, "max_prompt_tokens": 100},
    }
    traj = {"trajectory_id": f"traj_{tid}", "trajectory_status": "completed", "task_name": task,
            "task_id": "task_1", "initial_messages": [msg("system", "sys"), msg("user", task_text)],
            "trajectory_messages": full, "trajectory_output": out}
    d = root / "tasks" / f"{task}__task_1"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"traj_{tid}.json").write_text(json.dumps({"trajectory": traj}), encoding="utf-8")
    idx_p = root / "trajectories_index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else []
    idx.append({"trajectory_id": f"traj_{tid}", "task_name": task, "world_name": world,
                "trajectory_status": "completed"})
    idx_p.write_text(json.dumps(idx), encoding="utf-8")
    return d


