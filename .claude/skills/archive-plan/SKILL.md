---
name: archive-plan
description: Archive the completed masterplan to .claude/changelog/ and prepare for a new project. Use when a masterplan is complete and a new project phase is starting.
---

# Archive Plan

Archive the completed masterplan and prepare for a new project.

## Instructions

1. Read `.claude/masterplan.md`

2. Generate a short descriptive title (2-4 words, kebab-case) based on the masterplan content

3. Copy the full masterplan.md content verbatim to `.claude/changelog/YYYY-MM-DD-<title>.md`

4. Review the completed masterplan and identify any learnings worth adding to CLAUDE.md:
   - New principles or guidelines discovered
   - Technical gotchas or patterns that should be remembered
   - Mistakes made and how to avoid them
   - Architecture decisions and their rationale

5. Draft proposed additions to CLAUDE.md or AGENTS.md and present them to the user for review. Wait for approval before making any changes.

6. After user approval, apply the approved changes to CLAUDE.md or AGENTS.md

7. Reset masterplan.md to:
   ```
   # Master Plan

   No active master plan. Use this file for long-term multi-step plans.
   ```

8. Update scratchpad.md to note: "Archived previous plan to changelog/YYYY-MM-DD-<title>.md"
