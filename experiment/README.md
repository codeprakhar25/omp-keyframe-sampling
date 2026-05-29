# When Does Persistent Context Help Coding Agents?

A controlled ablation study on how repository-level context files (`AGENTS.md`)
affect the behavior of production coding agents. We reconcile two contradictory
2026 findings by separating **efficiency** effects from **correctness** effects,
and by varying *how* context is injected — not just *whether* it exists.

## Motivation

Two recent papers disagree:

- **Paper 1** (arXiv:2601.20404) — *"On the Impact of AGENTS.md Files on the
  Efficiency of AI Coding Agents."* Measures **efficiency only** (wall-clock time,
  token usage). Finds `AGENTS.md` reduces both. It has **no task-correctness
  metric** — "success" means the agent produced output, not that it solved the task.
- **Paper 2** (arXiv:2602.11988) — *"Evaluating AGENTS.md: Are Repository-Level
  Context Files Helpful for Coding Agents?"* Finds **no significant effect**.

This study adds a **correctness dimension** (test-based pass/fail) on top of the
efficiency metrics, and tests whether the effect of context is **moderated by
injection strategy and task complexity**.

## Hypotheses

- **H1 (Injection strategy):** *how* context is injected matters more than whether
  it exists. Three strategies:
  - **none** — no context file
  - **always_on** — full `AGENTS.md` injected every turn
  - **selective** — `AGENTS.md` split into searchable wiki topic files; the agent
    retrieves on demand
- **H4 (Quality vs. efficiency):** context changes *where* the agent spends effort
  (exploration depth, turns, cache traffic) even when the final outcome is unchanged.

## Method

For each `(task, strategy)` cell we run a production coding agent (Claude Code CLI)
in an isolated workspace, then evaluate the result:

1. Clone the repo, check out the task's `base_sha`.
2. Inject context per strategy (the **only** independent variable).
3. Run the agent (`--bare` isolation; context enters solely via system prompt).
4. Capture per-turn instrumentation (tokens, cache, tool calls, files, duration).
5. **Tier C evaluation (SWE-bench style):** the agent never sees the gold tests.
   After it finishes, we inject the gold PR's test files and run them. The task
   passes iff those tests pass against the agent's code.

### Why `--bare`

The agent runs with `claude --bare`, which isolates it from the host's global
config (no user `CLAUDE.md`, no hooks, no auto-memory) and forces strict
`ANTHROPIC_API_KEY` billing. This removes contamination and makes the injection
strategy the sole independent variable.

## Repository layout

```
harness/                 Experiment harness (Python package)
  config.py              ContextStrategy enum, TaskConfig, RunConfig
  logger.py              Per-turn instrumentation (tokens, tools, files, timing)
  context.py             Context injection + AGENTS.md -> wiki splitter
  agent.py               ClaudeCodeAgent (wraps `claude` CLI, parses stream-json)
  evaluate.py            Tier C test-based evaluation
  runner.py              Orchestration: clone -> inject -> run -> evaluate -> store
  db.py                  SQLite results storage (runs + turns)
  task_generator*.py     Build tasks from real merged PRs
  generate_wiki.py       Split AGENTS.md into wiki topic files
tasks/pilot.json         Task definitions (real merged PRs + gold diffs)
run_pilot.py             Resumable pilot driver (skips completed cells)
calibrate_eval.py        Validates the eval without any agent/API calls
analyze.py               Results analysis over the SQLite DB
```

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # required for --bare billing

# Validate the evaluator on a task (no agent calls):
python3 calibrate_eval.py --task-id firebase__firebase-admin-python__940

# Single run:
python3 -m harness --task-file tasks/pilot.json \
  --strategy always_on --agent claude \
  --task-id firebase__firebase-admin-python__940

# Full resumable pilot:
python3 run_pilot.py --repeats 3            # all repos
python3 run_pilot.py --repeats 1 --repo firebase

# Analyze:
python3 analyze.py --agent claude_code
```

## Tasks

12 pilot tasks from real merged PRs across three repos
(`pdm-project/pdm`, `firebase/firebase-admin-python`, `OpShin/opshin`), spanning
simple and medium complexity. Each task carries the issue/PR prompt, `base_sha`,
gold diff (used for test extraction), and metadata.

## Evaluation tiers

- **Tier C (default):** run the gold PR's test files against the agent's code.
  Agent test edits are reverted to base (`base_sha`-anchored); new gold test files
  are created fresh. Pass iff tests are green.
- **Fallback:** for tasks whose gold PR ships no tests, a relaxed line-overlap
  comparison against the gold diff.

## Status

Harness and Tier C evaluation are validated across all three repos. Pilot data
collection is in progress.

## License

Research code, released for reproduction. See repository for details.
