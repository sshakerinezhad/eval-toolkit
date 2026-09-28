#!/bin/sh
# NEW EVAL. A fresh repo for one evaluation, made from this toolkit:  ./new-eval.sh NAME [--no-smoke]
#
# Copies the toolkit (code, tests, configs, prompts, .env) to ~/NAME; never prompts/local/ (that is prep, not the work),
# leaves behind git history, results, caches, datasets, venvs. Inits git on main, first commit, empty
# results/, runs the tests and a smoke call, prints the clock start line and how long all this took.
# Python: the toolkit's venv, activated by path (no new venv to build). Nothing is pushed: the gh line
# is printed for you to run when the repo should exist on GitHub.
set -e
NAME="$1"
[ -n "$NAME" ] || { echo "usage: ./new-eval.sh NAME [--no-smoke]"; exit 1; }
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${NEW_EVAL_ROOT:-$HOME}/$NAME"
[ ! -e "$DEST" ] || { echo "ERROR: $DEST exists"; exit 1; }
START=$(date +%s)

rsync -a \
  --exclude .git --exclude .venv --exclude results --exclude .cache --exclude __pycache__ --exclude .pytest_cache \
  --exclude tbench-traces --exclude apex-sample --exclude apex-v1 --exclude 'deliverables*' --exclude .claude --exclude prompts/local \
  "$SRC/" "$DEST/"
mkdir -p "$DEST/results"
cd "$DEST"
git init -q -b main
git add -A
git commit -q -m "init from eval-toolkit $(git -C "$SRC" rev-parse --short HEAD)"

. "$SRC/.venv/bin/activate"
python -m pytest tests -q | tail -1
if [ "$2" != "--no-smoke" ]; then
  python smoke.py configs/apex_test3.yaml
fi

echo
echo "ready: $DEST   ($(( $(date +%s) - START ))s)"
echo "next:  cd $DEST && source $SRC/.venv/bin/activate"
echo "       python clock.py \"start\""
echo "       gh repo create $NAME --public --source . --push     # when the link must be real"
