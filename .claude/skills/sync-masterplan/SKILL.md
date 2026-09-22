---
name: sync-masterplan
description: Update .claude/masterplan.md with this session's accomplishments, run the handoff skill, and propose a one-line commit message. Use at the end of a working session.
---

# Sync Masterplan

Update the masterplan with accomplishments from this session, run the handoff command, and then generate a proposed one-line commit message for the changes. In that order.

## Instructions

1. Read `.claude/masterplan.md`, `.claude/work-plan.md`, and `.claude/scratchpad.md`

2. Identify what was accomplished this session:
   - Completed items from workplan
   - Work documented in scratchpad
   - Any additional changes made during the session

3. Cross-reference with masterplan to identify:
   - Items already marked complete (skip these - no duplicates)
   - Items that need to be marked complete
   - New work that should be added (new phases, sub-phases)
   - Files created/modified that should be documented

4. Draft proposed updates to masterplan.md:
   - Mark completed items with `[x]`
   - Add new phases or sub-phases if scope expanded
   - Update "Files created" and "Files modified" lists
   - Include reasoning traces per CLAUDE.md guidelines:
     - **Why** decisions were made (not just what)
     - **Gotchas** encountered and how they were resolved
     - **Bugs** that arose during implementation
     - **Mistakes** made and corrections applied

5. Present the proposed changes to the user for review. Wait for approval before applying.

6. After approval, apply the changes to masterplan.md

7. Review the completed work for learnings worth adding to CLAUDE.md. Only propose additions that are:
   - **High-level context** every future agent should know (tech stack, architecture decisions that arent obvious and will come up often)
   - **Decision-making guidance** (when to use X vs Y)
   - **Critical gotchas** that would cause significant problems if forgotten

   Do NOT add:
   - Library-specific implementation details (those belong in the archived masterplan)
   - Niche or one-off fixes
   - Anything too lengthy or specific to one feature

   If proposing CLAUDE.md changes, present them for user approval before applying.

8. Run the /handoff command.

9. Determine and present the user with a breif commit message if they decide to commit and push these changes. it should fit in one line and be natural sounding.
