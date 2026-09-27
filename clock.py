"""CLOCK. Stamp a mark into git history: python clock.py "minute 70 stop"

Empty commit, message = mark + local time. The git log becomes the run timeline.
"""
import subprocess
import sys
from datetime import datetime

mark = " ".join(sys.argv[1:]).strip()
if not mark:
    raise SystemExit('usage: python clock.py "minute 25 launch"')
msg = f"clock: {mark} [{datetime.now():%H:%M}]"
subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", msg], check=True)
print(msg)