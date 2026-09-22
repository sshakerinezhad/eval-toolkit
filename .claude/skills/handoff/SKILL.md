---
name: handoff
description: Capture current session context into .claude/scratchpad.md for the next session. Use at the end of a session or whenever context must be preserved for handoff.
---

# Handoff

Capture current session context for the next session.

## Instructions

1. Read the current scratchpad at `.claude/scratchpad.md`
2. Summarize the current session's context:
   - What was worked on
   - Current state/progress
   - Any blockers or pending items
   - Key decisions made
   - Next steps
3. If the existing scratchpad content is relevant to current work, merge the new context with it
4. If the existing scratchpad content is stale/unrelated, replace it with the new context
5. Write the updated scratchpad

Keep the scratchpad concise but complete enough for a fresh session to pick up where this one left off.
