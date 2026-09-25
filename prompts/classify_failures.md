<!-- source: mine, 2026-09-24, drafted with Claude Code -->
You are given one agent trajectory: the task it was given, its last turns, and the test output.
Pick exactly one category from the list below.
Reply with one JSON object and nothing else:
{"category": "<name>", "why": "<one sentence, quoting the trajectory>"}

Categories:
- environment_failure: Environment error: missing package, broken tool, sandbox died, etc. Failure is not caused by the agent's own edits.
- misread_task: the agent did not understand the task. the tests check for X; the Agent built Y
- wrong_approach: The agent understood the task, and chose a method that cannot pass the tests even if done correctly
- implementation_bug: the agent understood the task and chose a workable approach, but the code had one or more bugs the tests caught
- declared_done_unchecked: the agent claimed it finished the task without any check of its own on its work
- verified_still_wrong: the agent claimed it finished the work, ran its own tests, they passed or it misread them, and the tests still failed.

If multiple categories apply, pick the one higher in the list