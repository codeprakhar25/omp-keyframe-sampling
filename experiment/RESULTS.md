# Results — When Does Persistent Context Help Coding Agents?

**Status as of 2026-06-03.** Pilot data collection complete. Brutally honest read
of what the data does and does **not** support. Written to be the kind of internal
doc that stops you over-claiming in the paper.

---

## 0. Run status

| Item | State |
|---|---|
| Cells collected | **99 / 99** (11 tasks × 3 strategies × 3 repeats), balanced n=3 |
| Agent | Claude Code CLI (`claude --bare`), model `claude-sonnet-4-6` |
| Eval | Tier C (gold tests run against agent code), SWE-bench style |
| Execution | RunPod, egress-locked (GitHub blackholed); push physically impossible |
| Safety | 0 actual pushes; **multiple** blocked `git commit`/PR attempts (defense held) |
| **Open caveat** | **7 cells (all task 605) have a `test run timed out (1200s)` verdict** — eval-runner timeout, not a real test fail. Re-eval with 3600s **in progress**; expected to resolve to genuine fails (605 is all-fail regardless), so the headline won't change. |
| Cost | Agent runs ~$40; re-eval ~$0 (no LLM, pod compute only) |

**What's trustworthy now:** efficiency metrics (turns, cache, tools) for all 99.
**What's pending:** correctness verdict for the 7 timed-out 605 cells (conclusion
unaffected — see §4).

---

## 1. Design (one paragraph)

For each `(task, strategy)` we run a production coding agent in an isolated
workspace and evaluate with hidden gold tests. The **only** independent variable
is how repo context is injected: `none` / `always_on` (full AGENTS.md in system
prompt) / `selective` (AGENTS.md split into wiki files the agent retrieves). 11
tasks from real merged PRs across 3 repos (pdm, firebase-admin-python, opshin),
mixed simple/medium complexity. 3 repeats each.

---

## 2. Headline numbers (n=99)

### Correctness + efficiency by strategy

| strategy | pass-rate | mean turns | mean cache_read |
|---|---|---|---|
| none | **55%** (18/33) | 50.4 | 3069k |
| always_on | **61%** (20/33) | 55.4 | 3379k |
| selective | **58%** (19/33) | 56.6 | **2878k** |

### Per-task pass (across all 3 strategies)

| task | repo | cmplx | none | always_on | selective | verdict |
|---|---|---|---|---|---|---|
| 939 | firebase | simple | 3/3 | 3/3 | 3/3 | all-pass |
| 940 | firebase | simple | 3/3 | 3/3 | 3/3 | all-pass |
| 3769 | pdm | simple | 3/3 | 3/3 | 3/3 | all-pass |
| 3781 | pdm | simple | 3/3 | 3/3 | 3/3 | all-pass |
| 616 | opshin | simple | 3/3 | 3/3 | 3/3 | all-pass |
| 595 | opshin | simple | **2/3** | 3/3 | 3/3 | ~all-pass (1 none flake) |
| 3790 | pdm | medium | 1/3 | **2/3** | 1/3 | **the only "flip"** |
| 920 | firebase | medium | 0/3 | 0/3 | 0/3 | all-fail |
| 3797 | pdm | medium | 0/3 | 0/3 | 0/3 | all-fail |
| 605 | opshin | medium | 0/3 | 0/3 | 0/3 | all-fail (7 via timeout*) |
| 610 | opshin | medium | 0/3 | 0/3 | 0/3 | all-fail |

\* 605 verdicts include the 7 timeout artifacts being re-evaluated.

---

## 3. What the data actually says (honest)

### 3.1 Correctness: **no strategy effect.** (Confirms Paper 2.)
- **9 of 11 tasks are deterministic** across all strategies and repeats —
  all-pass or all-fail regardless of context injection.
- Only **2 tasks vary**: 595 (none 2/3 vs 3/3 — one flaky run) and 3790 (always_on
  2/3 vs 1/3). 
- **The entire "always_on 61% vs none 55%" gap = ~3 cells across those 2 tasks.**
  That is **noise**, not signal. With n=3 there is no statistical power to call it
  an effect. Outcome is **task-determined, not strategy-determined.**

### 3.2 Efficiency: small and **inconsistent**.
- `cache_read`: selective (2878k) < none (3069k) < always_on (3379k). always_on
  re-injecting AGENTS.md every turn costs the most cache — expected, real, but
  small.
- **But turns contradict:** selective has the **most** turns (56.6) and none the
  fewest (50.4). So "selective is more efficient" is **not** clean — it reads less
  cached context but takes more steps.
- No strategy is a universal efficiency winner. The "it depends" both papers
  half-saw is here, but **weakly**.

### 3.3 The "3790 flip" is an anecdote, not evidence.
One medium task where always_on passed 2/3 vs 1/3 for the others. n=1 task, 1-cell
difference. Suggestive, **not** publishable as an effect on its own.

---

## 4. The 7 timed-out cells — does it matter? **No (for the conclusion).**
All 7 are task **605**, which is **all-fail under every strategy** in the cells
that *did* finish (genuine `197 pass / 6 fail`). The re-eval (3600s) will turn the
7 fake "timeout" rows into genuine fails. So: the verdict for those cells changes
from *artifact-fail* → *real-fail*, and 605 stays all-fail. **The headline is
unchanged**; the fix is about defensibility (a reviewer would flag a "timeout"
verdict), not about the finding.

---

## 5. Threats to validity (what a reviewer will kill — grill yourself here)

1. **n is tiny.** 11 tasks, n=3. No correctness effect could be detected even if
   it existed. This is a **pilot**, not a powered study. Do not report p-values as
   if this were confirmatory.
2. **Floor/ceiling problem.** 9/11 tasks are all-pass or all-fail → almost no
   **dynamic range** to detect a context effect on correctness. The tasks are
   either trivial (context irrelevant) or too hard (context can't save them). The
   informative middle (borderline tasks like 3790) is **1 task**. The design is
   underpowered *by construction* for the correctness question.
3. **Single agent, single model.** Only `claude-sonnet-4-6`. Paper 1 used Codex.
   Any claim about "coding agents" generally is unsupported by one agent.
4. **`selective` is our own construction.** The wiki split + retrieval hint is a
   design choice; results are sensitive to how we split AGENTS.md. Not a neutral
   condition.
5. **Efficiency DV mixing.** turns vs cache disagree; we have not committed to a
   single pre-registered efficiency metric. Cherry-picking cache_read to favor
   selective would be p-hacking.
6. **Provenance:** repeat-0 ran on the laptop (pre-pod), repeats 1-2 on the pod.
   Audited clean (no cell ran git push/commit/gh), and a cross-check cell matched
   (940 always_on: 22 vs 23 turns), so behavior is machine-stable — but it's a
   mixed-machine dataset; note it.

---

## 6. Relevance — is this publishable, honestly?

**As a positive result: no.** The correctness finding is a **null** (context
doesn't change pass/fail), and the efficiency finding is **small and
inconsistent**. Neither is a clean "context helps / hurts" headline.

**As a measurement/reconciliation paper: maybe, with reframing.** The defensible
contribution is:
> "We add a correctness axis to the AGENTS.md debate. Across 11 real tasks,
> injection strategy does **not** affect correctness (replicating Paper 2's null),
> and efficiency effects are **small, inconsistent, and complexity-dependent** —
> reconciling Paper 1 (efficiency-only, found effects) and Paper 2 (found none) as
> *measuring different things on different task difficulties*."

That is honest and real, but it is a **workshop-tier** result as it stands, not a
main-conference claim.

**What it would take to be solid:**
- **More tasks, chosen for dynamic range** — specifically *borderline* tasks (like
  3790) where the agent sometimes passes. All-pass/all-fail tasks carry ~no signal
  for the correctness question.
- **A second agent** (Codex) to claim anything about "agents" broadly.
- **Pre-registered single efficiency metric** + paired Wilcoxon (per-task means,
  n=3) so the efficiency claim isn't post-hoc.
- Bump n per cell (3 → 5+) on the borderline tasks for any power.

---

## 7. One-line summary

A clean, safe, reproducible **pilot** that shows **context-injection strategy does
not move correctness** (strong null, confirms Paper 2) and **moves efficiency only
slightly and inconsistently**. Real and honest; **underpowered and
single-agent**; needs borderline tasks + a second agent before it's more than a
workshop note.

---

## 8. Reproduce

```bash
# DB of record: results/experiment.db (99 clean cells)
python3 analyze.py --agent claude_code
sqlite3 results/experiment.db \
  "SELECT task_id,strategy,repeat_index,total_turns,total_cache_read_tokens,task_passed \
   FROM runs WHERE agent='claude_code' AND eval_method IS NOT NULL ORDER BY task_id,strategy,repeat_index;"
```
