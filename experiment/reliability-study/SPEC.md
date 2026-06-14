# Reliability Study — How Reproducible Are Coding-Agent Benchmarks?

**Working title:** *Run-to-Run: Quantifying Nondeterminism in Coding-Agent
Benchmarks and Its Impact on Rankings.*

Status: **design / pre-registration** (no new runs yet). Started 2026-06-14.
Builds on the context-files harness (locked-pod, 2-agent, Tier-C gold-test eval).

---

## 1. Motivation (seeded from existing 291-run data)

Single-run SWE-bench-style scores are reported as if deterministic. They are not.
From `results_from_pod_codex/experiment_full.db` (context-files study, n=3/cell),
**restricted to `eval_method='tests'` runs only** (219/291; see line_overlap
caveat below):

- **Fixed-config flip rate** (identical task+strategy, 3 repeats):
  Claude **13%** of cells, Codex **10%** of cells disagree.
- Per-task nondeterministic (pooled across strategies): Claude 18%, Codex 15%.
- **Single-run benchmark wobble** (bootstrap from per-task rates):
  Claude SD **5.4pp**, Codex SD **5.1pp** — i.e. a single benchmark run carries
  ≈±5pp noise; two papers reporting "48%" vs "54%" could be the same agent.

**Eval-method caveat (a sub-finding, feeds RQ4).** The *contaminated* numbers
(Claude 12% / Codex **24%** flip; "Codex 2× noisier") were an **artifact** of 72
runs graded by the fuzzy `line_overlap` fallback (tasks 906/926/932, whose gold
PRs touch no test files — e.g. 906 = `AGENTS.md` only). On clean gold-test eval
the agent gap **vanishes** (~10–13% both). Lesson: **weak eval methods inflate
apparent nondeterminism** → this study uses *only* test-grounded tasks, and the
"Codex noisier" idea is now an open RQ to *measure*, not a presumed result.

Caveat: n=3 *undercounts* (a p=0.5 task reads unanimous 25% of the time); per-task
`p_i` estimated from ~9 runs is itself noisy; floor/ceiling tasks add zero
variance. So the above are **motivating lower bounds**, not results. This study
estimates the real magnitudes with high K on medium-difficulty tasks.

## 2. Research questions

- **RQ1 (magnitude).** Distribution of per-task pass-probability `p_i`; true
  flip rate; share of tasks that are nondeterministic.
- **RQ2 (stability).** Benchmark-score sampling SD/CI as a function of repeats K.
  → "repeats-needed" curve + recommended reporting N.
- **RQ3 (rankings).** With two agents on the **same** task set: P(single-run
  ranking flip) vs the true (high-K) order; how K shrinks it; how big a true gap
  must be to survive K=1.
- **RQ4 (source/framing).** The CLIs do not expose temperature → variance is
  **as-deployed**. Quantify it under the exact config a user runs; attribute to
  serving-level nondeterminism (batching / MoE routing / non-associative float),
  not user-set sampling. Headline: deployed agents are assumed reproducible; they
  are not.

## 3. Design

- **Unit:** (task, agent). Outcome: binary Tier-C pass (gold tests).
- **Fixed IV:** strategy = **`none`** for all runs (strip context entirely →
  removes the context-files variable; pure run-to-run noise).
- **Agents / models:**
  - **Codex** (gpt-5.5, ChatGPT-auth, **free**) — heavy arm.
  - **Claude Code** (**Haiku-class**, budget arm) — 2nd vendor.
- **Repeats K:**
  - Codex: **K=25** on **~50** medium-difficulty tasks (≈1250 runs, $0).
  - Claude-Haiku: **K=10** on a **~20-task shared subset** (≈200 runs, ≈$87).
  - Shared subset enables RQ3 (same tasks, both agents).
- **Task selection (critical):** oversample **medium difficulty** — variance only
  exists where `0 < p < 1`. Floor/ceiling tasks are uninformative. Pick tasks
  with prior pooled pass-rate in **[0.2, 0.8]** where known (reuse
  `experiment_full.db` + `tasks/candidates.json` + screener output), top up with
  fresh tasks from the 3 repos via `harness.task_generator`. Target a difficulty
  spread, not all-borderline (need some easy/hard anchors for the curve).

## 4. Pre-registered metrics (lock now — no post-hoc swapping)

1. **Per-task variance:** `p_i = passes/K`; report histogram, % with `0<p_i<1`,
   mean Bernoulli variance `E[p(1-p)]`.
2. **Flip rate:** P(two random runs of the same task disagree) = `2·E[p(1-p)]`.
3. **Benchmark SD vs K:** for K∈{1,3,5,10,25}, bootstrap-resample runs → SD and
   95% CI of the mean score. Primary stability metric = **SD at K=1**.
4. **Repeats-needed:** smallest K s.t. 95% CI width ≤ 5pp (and ≤2pp).
5. **Ranking flip:** P(sign(score_A−score_B) wrong vs high-K truth) at each K.
6. **Agent contrast:** Codex SD vs Haiku SD (variance-by-agent), with bootstrap CI.

Analysis: task-clustered bootstrap (10k); Wilson per-task CIs; everything scripted
in `analyze_variance.py` (to write). No new statistical tests beyond these.

## 5. Harness reuse (minimal change)

The existing engine already does (task × agent × strategy × repeat). Needed:
- Set `strategy=none`, bump `repeats` to K (runner/run_pilot already parameterized).
- Claude arm: point model at Haiku (config `EXP_CLAUDE_MODEL` / equivalent — verify
  flag in `harness/config.py` + `agent.py`).
- New DB `reliability.db` (do **not** pollute `experiment_full.db`).
- Same locked-pod safety stack (egress lock, deny-hooks, verify_lock) — unchanged.
- **Cost probe first:** run K repeats of one task back-to-back; check whether the
  5-min prompt cache stays warm (cache_read reuse) → could cut the $1.64→ lower.

## 6. Threats / honesty

- **Cache-warmth confound on timing** (not on correctness) — keep repeats
  independent for the *correctness* metric; note timing separately.
- **Task contamination** — orthogonal to variance, but note.
- **Haiku ≠ frontier** — framed as model-tier point, not a frontier claim.
- **gpt-5.5 / Haiku snapshots** can change under us mid-study → pin dates, run each
  agent's full sweep in one window.
- Medium-difficulty selection is **conditioned on prior runs** → mild selection
  effect on `p_i`; mitigate by including fresh (unseen) tasks and reporting both.

## 7. First concrete steps

1. Assemble the medium-difficulty task set (mine `experiment_full.db` rates +
   `candidates.json`; generate fresh) → `tasks/reliability_tasks.json`.
2. Wire Haiku model flag; write `analyze_variance.py` against a fixture.
3. Pod bring-up + verify_lock; **cost/cache probe** (1 task × K=10, both agents).
4. Codex heavy sweep (free) → then Claude-Haiku shared subset.
