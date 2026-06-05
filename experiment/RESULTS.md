# Results — When Does Persistent Context Help Coding Agents?

**Status as of 2026-06-05.** Pilot complete on **two agents** (Claude + Codex, 99
cells each). See **§4B** for the Codex second-agent arm. Brutally honest read
of what the data does and does **not** support. Written to be the kind of internal
doc that stops you over-claiming in the paper.

---

## 0. Run status

| Item | State |
|---|---|
| Cells collected | **99 / 99** (11 tasks × 3 strategies × 3 repeats), balanced n=3 |
| Agent | **Two arms:** Claude Code (`claude-sonnet-4-6`) **+ Codex** (`gpt-5.5`, ChatGPT-auth) — 99 cells each, see §4B |
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

## 4B. Second agent — Codex (closes the §5.3 single-agent kill)

**Added 2026-06-05.** Replicated the full 99-cell design on a second agent:
**OpenAI Codex CLI** (`codex exec`, model `gpt-5.5`, ChatGPT-plan auth, free pool).
Same 11 tasks × 3 strategies × 3 repeats, same Tier-C eval, same egress-locked pod
(GitHub blackholed; git/gh PATH-shims + scrub remotes — push physically impossible).
**99/99 clean cells, 0 errors, 0 pushes.** DB of record `experiment.db` (`agent='codex'`).

### Correctness: the null **replicates** in Codex.

| strategy | claude pass | codex pass |
|---|---|---|
| none | 55% (18/33) | 52% (17/33) |
| always_on | **61%** (20/33) | **55%** (18/33) |
| selective | 58% (19/33) | 48% (16/33) |

- `always_on` is numerically best for **both** agents — the *direction* (context
  helps) is **agent-robust**. But for codex too the gap is **~1–2 cells on a single
  task** → **noise, not signal** (same n=3 power problem).
- **9/11 tasks are deterministic for codex as well** (all-pass or all-fail across
  every strategy/repeat). Same floor/ceiling problem (§5.2), independently confirmed.

### The informative task is **agent-specific** (key cross-agent finding).

| task | claude (n/a/s) | codex (n/a/s) | note |
|---|---|---|---|
| 3790 (pdm med) | 1/3 · **2/3** · 1/3 | **0/3 · 0/3 · 0/3** | claude's only "flip"; codex **can't do it at all** |
| 3769 (pdm simple) | 3/3 · 3/3 · 3/3 | **2/3 · 3/3 · 1/3** | trivial for claude; **codex's borderline** (and it *moves* with strategy) |

→ Each agent has a **different borderline task**. Dynamic range is a property of
`task × agent`, not the task alone. This both (a) explains why single-agent
screening is insufficient and (b) is itself a reportable result: *whether context
helps correctness is gated by whether the task sits in that agent's borderline band.*

### Efficiency: portable metrics (cache_* is Claude-only → dropped).

Pre-registered cross-agent DV = **tool_calls, wall-time, output tokens** (cache
accounting is not comparable — see caveat). Averaged over all clean cells:

| agent | tool_calls | wall-time | output_tok |
|---|---|---|---|
| claude | 54 | 1247s | 20,551 |
| **codex** | **31** | **909s** | **6,605** |

- **Codex is leaner on every portable axis**: ~40% fewer tool calls, ~27% faster,
  ~3× fewer output tokens — at a slightly lower pass-rate. A genuinely different
  efficiency profile, robust across all 99 cells (not a per-task artifact).
- **Within codex, strategy does *not* move efficiency** (tools 30–31, dur 887–942s
  flat across none/always_on/selective) — unlike claude, where `always_on` mechanically
  inflated cache. Codex has no equivalent always-reinject cost.

### ⚠️ Token-cost caveat (the pre-registered trap held).
Raw `total_tokens` (codex 794k vs claude 21k) is **apples-to-oranges**: codex's
`input_tokens` *includes* cached input; claude books cache in separate columns.
Normalized: **fresh** (non-cached in+out) = codex 68k vs claude 21k; **total incl
cache** = claude **3.29M** vs codex 794k (claude re-reads ~3.1M cached tokens/cell).
→ Report **output / tool_calls / wall-time** as the clean efficiency DVs; any dollar
comparison must use per-provider pricing, not a raw token count.

**Honest read of the second arm:** the **correctness null replicates** (context
doesn't move outcome at this power, on either agent), the **always_on direction is
agent-robust**, and Codex is **uniformly leaner** on portable efficiency. The
single-agent kill (§5.3) is now **closed**; the floor/ceiling kill (§5.2) is
**re-confirmed**, not fixed — and the fix (borderline tasks) is now **agent-aware**:
the task-expansion screener runs on **Codex** to find codex-borderline candidates
(claude's are already known from this pilot).

### 4C. Task-expansion screen — DONE (2026-06-05)

28 new candidates (pdm/firebase/opshin, 12/repo minus pilot overlap) screened on
**Codex** at `strategy=none`, repeats=3. Spread: **4 BORDERLINE** · 9 all-pass · 15
too_hard (0/3). **pdm wiped out** (all 6 codex-0/3) → Codex is floored on pdm; per
§4B those are most likely *claude*-borderline (difficulty = task×agent). Borderline
keepers: **598** (opshin), **906 / 926 / 932** (firebase).

> **Classifier artifact (pre-registration lesson):** the screener's `medium_effort`
> rule split on `turns≥30`, but **Codex emits exactly 1 `turn.completed` per session**
> (§4B), so *every* Codex all-pass collapses to `too_easy` — 8 genuinely high-effort
> tasks (18–51 tool calls; e.g. 614=51 tools/1220s, 609=47/1083s) were silently
> dropped. Reclassifying on the **pre-registered portable metric (`tool_calls`)**
> recovers them. This is exactly why turns were dropped from the cross-agent DV; the
> screener should use `tool_calls` for `agent=codex`.

**Action taken:** merged the **4 BORDERLINE only** (conservative; correctness-pure,
keeps the $150 pool intact) → **pilot grows 11 → 15** (firebase 6 / opshin 5 / pdm 4).
The 8 recovered effort-tasks were deferred (running both arms on them ≈ $130 Claude).

---

## 5. Threats to validity (what a reviewer will kill — grill yourself here)

1. **n is tiny.** 11 tasks, n=3. No correctness effect could be detected even if
   it existed. This is a **pilot**, not a powered study. Do not report p-values as
   if this were confirmatory.
2. **Floor/ceiling problem.** 9/11 tasks are all-pass or all-fail → almost no
   **dynamic range** to detect a context effect on correctness. The tasks are
   either trivial (context irrelevant) or too hard (context can't save them). The
   informative middle (borderline tasks like 3790) is **1 task**. The design is
   underpowered *by construction* for the correctness question. **Partially addressed
   (§4C):** the Codex screen added **4 borderline** tasks → pilot now **15** (more
   dynamic range), but the powered claim still needs the deferred effort-tasks and a
   Claude re-screen of the pdm wipeout set.
3. ~~**Single agent, single model.**~~ **CLOSED (§4B).** Added a **Codex / `gpt-5.5`**
   arm (99 cells). Correctness null **replicates**; always_on direction is
   agent-robust; Codex is uniformly leaner on portable efficiency. Two agents now,
   one Claude-family + one OpenAI — reconciles Paper-1(Codex)/Paper-2(Claude).
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
- ~~**A second agent** (Codex)~~ **DONE (§4B).** Codex arm collected with a
  **portable** efficiency metric (tool_calls/wall-time/output-tokens), since
  `cache_*` is Claude-specific and didn't transfer. Null replicated; efficiency
  profile differs (Codex leaner). Remaining lever for a *powered* claim is the
  task count, not the agent count.
- **Pre-register the primary efficiency metric** (none-vs-selective on cache,
  per-task) so it isn't post-hoc.

**Verdict:** a clean, safe, honest **two-agent pilot** with one real (thin)
efficiency signal and a correctness null that **replicates across Claude and Codex**.
The second-agent kill is closed; **the remaining gate is task count/dynamic range** —
~12–15 more (agent-aware) **borderline** tasks would power the efficiency trend and
reopen the correctness axis. That single task-expansion effort is now the only thing
between this and workshop-tier.

---

## 7. One-line summary

A clean, safe, reproducible **two-agent pilot**: **context-injection strategy does
not move correctness** (strong null, **replicated on Claude and Codex**), and the
only surviving Claude efficiency effect is that **`selective` has a leaner context
footprint** (cache-creation ↓, p_Holm=0.012). Cross-agent: **Codex is uniformly
leaner** (portable tools/wall-time/output), and the one *borderline* task is
**agent-specific** (3790 for Claude, 3769 for Codex). Still **underpowered (n=11)** —
~12–15 more agent-aware borderline tasks is the one remaining lever.

---

## 8. Reproduce

```bash
# DB of record: results/experiment.db (claude 99 + codex 99 clean cells)
python3 analyze.py --agent claude_code
python3 analyze.py --agent codex
# cross-agent portable efficiency (cache_* dropped — not comparable):
sqlite3 results/experiment.db \
  "SELECT agent,strategy, COUNT(*) n, SUM(task_passed) pass, \
          AVG(total_tool_calls) tools, AVG(total_duration_s) dur, AVG(total_output_tokens) out \
   FROM runs WHERE eval_method IS NOT NULL GROUP BY agent,strategy ORDER BY agent,strategy;"
```
