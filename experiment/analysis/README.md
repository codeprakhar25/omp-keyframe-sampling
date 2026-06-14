# Exploratory Analysis — Context-Files Study Runs

For our own understanding (not a paper). Mining `experiment_full.db` (291 runs,
5609 turns). Regenerate: `/tmp/edaenv/bin/python analysis/explore.py`
(matplotlib venv at `/tmp/edaenv`). Numbers dump in `SUMMARY.md`; graphs below.

## Headline reads

1. **More effort → LOWER success (both agents).** `06_effort_vs_pass.png` — quick
   solves pass ~74-76%; the heaviest-tool-call quartile passes **26% (Claude) /
   46% (Codex)**, monotonic decline. Long grinds are mostly doomed thrashing, not
   productive effort. *(Descriptive, confounded: hard tasks need more tools AND
   fail more — not proof that stopping early wins. But the signal is stark.)*
2. **Failed runs burn more tools than passes.** Median tool calls pass→fail:
   Claude **41→58**, Codex **27→35**. Same thrashing story at the median.
3. **The two agents work differently.**
   - Claude: Bash-heavy but mixed — Bash 3238, Read 1588, Edit 626; reads ~8
     files/run via the explicit Read tool.
   - Codex: routes **everything** through `run_bash` (3030/3030) — file edits and
     reads happen inside shell, so our file counters undercount it (reads show 0.0,
     a measurement artifact, not behavior).
4. **Token economics are inverted between agents.** Claude: tiny input (470) +
   huge **cache_read 2.66M** + output 19.6k (context cached, generates a lot).
   Codex: huge **input 853k** (re-sends context, less caching) + small output 7.2k
   + reasoning 1.9k. Same task, opposite token shape → why cross-agent token
   comparison needs the portable metric.
5. **Correctness null holds in the cut too** — pass rate flat across strategies
   (Claude 53/54/56%, Codex 59/57/53%); see `03_passrate_heatmap.png`.
6. Claude avg **18.7 min/run**, Codex **15.2 min/run**.

## Tie-back
Read #1+#2 are exactly the empirical premise behind the "teach the agent to quit"
idea — long runs mostly fail. Our own data supports *why* someone wants an
early-stop lever, even if that paper space is crowded.

## Graph index
| File | Shows |
|---|---|
| `01_tool_use.png` | top tools per agent |
| `02_toolcalls_outcome.png` | tool-calls/run distribution, pass vs fail |
| `03_passrate_heatmap.png` | pass rate, repo × strategy, per agent |
| `04_time_tokens.png` | wall-time boxplot + token composition |
| `05_files.png` | files read/written per run |
| `06_effort_vs_pass.png` | tool-call quartile vs pass rate (the headline) |

## Deeper pass (explore2.py → SUMMARY2.md, graphs 07–08)

**The effort→fail signal SURVIVES clean eval (tests-only)** — not a line_overlap
artifact:
- Claude: pass% by tool-call quartile = **56→65→48→23%**
- Codex: **77→74→22→13%** (a cliff — heavy-effort Codex runs almost always fail)

Fail runs burn more tools (Claude 51→55, Codex 27→37 avg). Codex's "top tool =
100%" just reflects everything routing through `run_bash`; Claude's top tool is
~56-63% of calls (Bash-dominant, mixed). Bottom line: **on clean eval, a run that's
grinding through many tool calls is a run that's losing.**

## Caveats
- Codex `unique_files_read` ≈ 0 is an instrumentation gap (reads via bash, not a
  Read tool) — don't read it as "Codex doesn't read files."
- Effort→outcome is **correlational**; task difficulty confounds it.
- 906/926/932 graded by `line_overlap` (no gold tests) are still in these cuts
  except where the analysis is outcome-specific; for clean outcome stats use
  `eval_method='tests'`.
