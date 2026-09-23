import json

import pytest

import passk


def _traj(dir_, tid, score, status="completed"):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"traj_{tid}.json").write_text(json.dumps({
        "trajectory": {
            "trajectory_id": f"traj_{tid}",
            "trajectory_status": status,
            "trajectory_output": {"score": score},
        }
    }))


# ---------- pass_at_k ----------

def test_pass_at_k_all_fail():
    assert passk.pass_at_k([0.0, 0.0, 0.0], 3) == 0.0


@pytest.mark.parametrize("scores", [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
def test_pass_at_k_any_success_is_one_when_k_equals_n(scores):
    assert passk.pass_at_k(scores, 3) == 1.0


def test_pass_at_1_is_mean():
    assert passk.pass_at_k([1.0, 0.0, 0.0], 1) == pytest.approx(1 / 3)


def test_pass_at_k_rejects_k_gt_n():
    with pytest.raises(ValueError):
        passk.pass_at_k([1.0], 2)


# ---------- load_variant ----------

def test_load_variant_groups_by_task_and_counts_errors(tmp_path):
    root = tmp_path / "X_trained_traces"
    _traj(root / "tasks" / "tb_a__task_1", "a1", 1.0)
    _traj(root / "tasks" / "tb_a__task_1", "a2", 0.0)
    _traj(root / "tasks" / "tb_a__task_1", "a3", 0.0, status="error")
    _traj(root / "tasks" / "tb_b__task_2", "b1", 0.0)
    _traj(root / "tasks" / "tb_b__task_2", "b2", None)   # ungraded -> 0.0
    _traj(root / "tasks" / "tb_b__task_2", "b3", 0.0)
    tasks, errored = passk.load_variant(root)
    assert tasks == {"tb_a__task_1": [1.0, 0.0, 0.0], "tb_b__task_2": [0.0, 0.0, 0.0]}
    assert errored == 1


def test_load_variant_empty_exits(tmp_path):
    with pytest.raises(SystemExit):
        passk.load_variant(tmp_path)


# ---------- bootstrap ----------

def test_bootstrap_degenerate_inputs():
    draws = passk.make_draws(5, 200, seed=0)
    assert passk.bootstrap_ci(draws, [1.0] * 5) == (1.0, 1.0)
    assert passk.bootstrap_ci(draws, [0.0] * 5) == (0.0, 0.0)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    d1 = passk.make_draws(len(vals), 2000, seed=7)
    d2 = passk.make_draws(len(vals), 2000, seed=7)
    assert d1 == d2
    lo, hi = passk.bootstrap_ci(d1, vals)
    assert lo <= sum(vals) / len(vals) <= hi
    assert lo < hi
