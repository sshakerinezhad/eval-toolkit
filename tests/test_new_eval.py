import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def test_new_eval_makes_a_fresh_repo(tmp_path):
    env = {**os.environ, "NEW_EVAL_ROOT": str(tmp_path)}
    r = subprocess.run([str(HERE / "new-eval.sh"), "fresh", "--no-smoke"], capture_output=True, text=True, env=env, cwd=HERE)
    assert r.returncode == 0, r.stderr + r.stdout
    d = tmp_path / "fresh"
    assert (d / "grade.py").exists() and (d / "tests").is_dir() and (d / "results").is_dir() and (d / ".gitignore").exists()
    assert not (d / ".venv").exists() and not (d / ".claude").exists() and not (d / "prompts" / "local").exists() and not (d / "demo.py").exists()
    log = subprocess.run(["git", "-C", str(d), "log", "--oneline"], capture_output=True, text=True).stdout
    assert log.count("\n") == 1 and "init from eval-toolkit" in log
    assert "passed" in r.stdout and "ready:" in r.stdout and 'python clock.py "start"' in r.stdout
    # a second run on the same name refuses
    r2 = subprocess.run([str(HERE / "new-eval.sh"), "fresh", "--no-smoke"], capture_output=True, text=True, env=env, cwd=HERE)
    assert r2.returncode == 1 and "exists" in r2.stdout
