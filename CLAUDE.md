# AGENTS.md

# Behavioural Compass
1. Step outside the issue at hand and see the big picture at EVERY STEP
2. See what's implicit/needed and differentiate from what's asked or assumed
3. consider all layers of the problem
4. Do not be aftaid to rip something to shreds, tear the guts out, and rebuild better. If something can be done better, tear it down and start fresh.

## Golden Rules
1. Never assume. Read the code, don't guess.
2. NO BANDAIDS EVER. Simplest solution wins. If it's a bandaid, re-evaluate.
3. The first solution is rarely the best, be critical and compare every option shrewdly
4. Challenge your own biases, think several layers of abstraction deep
5. The WHY matters as much as the WHAT. Include reasoning in decisions and documentation.
6. Before implementing a plan, critique it. Does it make sense? What could go wrong? What does it interact with?
7. My words are NOT gospel. They are a starting point. Push back.
8. Do not start implementation until i explicitly confirm.


## File Conventions
- `masterplan.md` — long-range architecture and goals
- `workplan.md` — current implementation steps
- `scratchpad.md` — context for session handoffs
- `changelog/` — archived masterplans

### Explaining Things Protocol
- Explain things simply, avoid jargon, use simple language and avoid buzzwords, and provide simple examples to illustrate complex ideas. 
- When understanding systems I need to understand the whole thing hollistically and be able to trace the flow.
- I need to understand what things in which files do what, and how.

### Reasoning Context
When updating plans be sure to include what was done and why in *just* enough detail that context is preserved for the next agent.

### Two-Pass Thinking
When making decision critique your own work. Does this make sense? What could go wrong?
When things break: after fixing, briefly explain why it happened and what I should know to catch it earlier.


## Learned
- Bash heredocs with apostrophes break here; use Write tool for source files.
- Live-test failure paths, not just happy path; mocks passed before every real bug was found.
- Cache is the recovery mechanism; any model call goes through `llm.call()` + cache.

## Timeline rule
- Commit after every working change, small commits, plain short human messages. The git log is the record of the session.
- Mark run-sheet minutes with empty commits: `git commit --allow-empty -m "clock: minute 70 stop"`. Also: "clock: start", "clock: build start", "clock: email sent".
- Debrief reads: `git log --format='%ad %s' --date=format:%H:%M`
