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
| ~~Open caveat~~ **RESOLVED** | The 7 timed-out 605 cells were re-eval'd at 3600s. 3 resolved to a genuine `197 pass / 6 fail`; the other ~~4~~ still exceeded 1h of pytest (the agent's change makes opshin compilation pathologically slow → effectively fail). **605 is all-fail, confirmed**, no `timed out` labels remain. |
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

### 3.2 Efficiency: one robust signal, the rest underpowered.

**Stats done correctly** — unit of analysis = **task** (the 3 repeats of a task
are correlated, so they are *averaged* within (task, strategy), giving one value
per task). Paired Wilcoxon across **n=11 tasks**, **Holm-Bonferroni** corrected
(family = 12 tests). Script: `efficiency_stats_correct.py`.

> ⚠️ An earlier analysis treated the 3 repeats as independent (n=15) and reported
> p=0.0001–0.0006. **Those were pseudoreplication artifacts and are discarded.**
> At the correct unit (n=11) almost all of that significance disappears.

| comparison | metric | direction | raw p | **Holm p** |
|---|---|---|---|---|
| none vs selective | cache_create | selective lower **11/11** | 0.0010 | **0.012 \*** |
| always_on vs selective | cache_create | selective lower 9/11 | 0.042 | 0.46 ns |
| always_on vs selective | duration | selective lower 8/11 | 0.054 | 0.54 ns |
| none vs selective | cache_read | selective lower 9/11 | 0.067 | 0.61 ns |
| none vs selective | turns | — | 0.15 | 1.0 ns |
| (all 7 others) | turns/dur/cache | — | >0.27 | 1.0 ns |

- **One result survives correction:** `selective` uses significantly less
  **cache-creation** than `none` (p=0.001, unanimous 11/11; Holm p=0.012).
- **Directional but ns after correction:** selective also has lower cache-read
  (9/11, p=0.067) and the always_on/selective duration gap (8/11, p=0.054) — both
  **right at the edge**, i.e. underpowered, not absent.
- **No effect** on turns or (corrected) duration.
- The "always_on is expensive" story is **mostly mechanical** (it re-injects the
  whole AGENTS.md every turn → more cache by construction) and does not survive as
  a clean scientific claim.

**Honest read:** selective has a **leaner context footprint** — one metric
significant, two more trending. Real, but thin (1 of 12 tests survives), and the
*user-facing* meaning of a cache-creation reduction is not obvious.

### 3.3 The "3790 flip" is an anecdote, not evidence.
One medium task where always_on passed 2/3 vs 1/3 for the others. n=1 task, 1-cell
difference. Suggestive, **not** publishable as an effect on its own.

---

## 4. The 7 timed-out cells — RESOLVED, conclusion unchanged.
All 7 were task **605**. Re-evaluated at a 3600s pytest cap (re-eval from the
stored agent diff — no agent re-run, so turns/cache untouched):
- **3** resolved to a genuine `197 pass / 6 fail`.
- **4** still exceeded **1 hour** of pytest — the agent's change makes opshin
  compilation pathologically slow. These are labeled
  `fail: test runtime exceeds cap >3600s`. A change whose tests can't complete in
  an hour is, for our purposes, a fail.

Either way **605 is all-fail under every strategy**, now with clean verdicts and
**no `timed out` artifacts left**. Efficiency metrics (turns/cache) for these
cells were always valid and are unaffected. (Side observation worth a sentence in
the paper: some agent edits *blow up compile time* — an efficiency failure mode
distinct from a correctness one.)

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

**Not yet.** After correct stats the result is:
- **Correctness:** clean **null** (even the one "flip", 3790, dissolves into
  coin-flip noise across repeats).
- **Efficiency:** **1 of 12 tests** survives correction (selective ↓ cache-create),
  plus two edge-of-significance trends. That is **real but thin** — too thin to
  hang a paper on, and the user-facing meaning of "fewer cache-creation tokens" is
  not self-evident.

A reviewer sees 1/12 survived + a null and asks "why this metric, and was the
study powered?" — both fair, both currently lose.

**Why thin ≠ wrong:** the efficiency signals are **underpowered, not absent**.
selective is lower-footprint in 9–11 of 11 tasks on the cache metrics; with n=11
that lands at p=0.001–0.07. The direction is consistent; the n is just small.

**What would make it solid (and it's the SAME lever for both axes):**
- **Add ~12–15 tasks.** For **efficiency**, *any* difficulty adds power — the
  trending cache effects (9/11 @ p=0.067) would very likely cross into
  significance at n~25. For **correctness**, add **borderline** tasks (~30–70%
  baseline pass, like 3790) so the outcome can actually move. → one task-collection
  effort buys progress on **both** axes.
- **A second agent** (Codex) — needed for any claim about "coding agents" broadly,
  but with a **portable** efficiency metric (turns/wall-time), since `cache_*` is
  Claude-specific and won't transfer.
- **Pre-register the primary efficiency metric** (none-vs-selective on cache,
  per-task) so it isn't post-hoc.

**Verdict:** a clean, safe, honest **pilot** with one real (thin) efficiency
signal and a correctness null. **Workshop-tier only after ~12–15 more tasks**
power the efficiency trend; main-conference needs that **plus** a second agent.

---

## 7. One-line summary

A clean, safe, reproducible **pilot**: **context-injection strategy does not move
correctness** (strong null, confirms Paper 2), and the only surviving efficiency
effect is that **`selective` has a leaner context footprint** (cache-creation
↓, p_Holm=0.012; cache-read trending) — **real but thin (1/12 tests), underpowered,
single-agent**. ~12–15 more tasks would power the efficiency trend (and, if
borderline, reopen the correctness axis); a second agent is needed for any general
claim.

---

## 8. Reproduce

```bash
# DB of record: results/experiment.db (99 clean cells)
python3 analyze.py --agent claude_code
sqlite3 results/experiment.db \
  "SELECT task_id,strategy,repeat_index,total_turns,total_cache_read_tokens,task_passed \
   FROM runs WHERE agent='claude_code' AND eval_method IS NOT NULL ORDER BY task_id,strategy,repeat_index;"
```
