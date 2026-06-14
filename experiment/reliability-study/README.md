# Reliability Study — Orientation & Alignment

Read this first. It's the map: where we came from, what data we have *right now*,
what that data says, and the exact path through the experiment. The formal
pre-registration is `SPEC.md`; the frozen current-data analysis is
`seed_analysis_existing_data.txt`.

---

## Why this matters (plain terms)

**One sentence:** when people test AI coding agents they run the test *once* and
report a score like "solves 49% of tasks" — we show that number is shaky (run the
same test again, get a different score), measure *how* shaky, and give a rule for
fixing it.

**The analogy.** Grading a student on a *single* exam question. If they'd flip
between right and wrong depending on the day, one question tells you almost
nothing — you'd make them retake it and average. Coding benchmarks today give the
"one question, one try" grade, then rank agents on it.

**Why the score moves with nothing changed.** Same agent, same task, same settings,
different outcome — because of the machinery underneath (how requests are batched
on the servers, tiny math-ordering effects, internal routing). The user can't even
turn it off; the tools expose no "make it deterministic" knob. So **deployed agents
are quietly nondeterministic, and everyone reports numbers as if they aren't.**

**What the old data already shows (before spending anything):**
- ~**1 in 15** re-runs of the same task flips pass↔fail.
- A single benchmark run wobbles **±5 points** → "48%" and "54%" in two papers
  could be the *same* agent, the gap just noise.
- Rank A vs B from one run each → **wrong winner ~18% of the time**.

**Why it's valuable (the payoff):**
1. **Corrects the literature** — many "our agent wins by 3 points" claims are
   inside the noise, i.e. not real. A result people must cite.
2. **Gives a fix, not just a complaint** — a concrete rule: *report benchmarks with
   at least K repeats; here's the K you need.* Reusable, becomes a standard.
3. **A free second finding** — re-checking our own data, the "Codex is 2× noisier"
   result turned out fake, caused by a *weak grading method*, not the agent.
   Lesson: sloppy evals invent noise that isn't there.

**Cost reality:** money is fine (Codex free, $90 covers Claude). The real cost is
**time** — ~15 min/run, hundreds of runs = days; two pods in parallel is why it's
tractable.

---

## 0. The arc (how we got here)

1. **Context-files paper** (`../paper/`, public repo `context-files-coding-agents`)
   — "Do context files (AGENTS.md) help coding agents?" Answer: a **correctness
   null** on both agents (TOST-bounded), one efficiency/process effect. *Shipped.*
2. While analyzing that data we noticed identical re-runs sometimes **flip**
   pass↔fail. That side-observation is this study.
3. **This study:** *How reproducible are coding-agent benchmarks?* Quantify
   run-to-run variance and its impact on reported scores and rankings.

We are at the **pre-pod** stage: design + tooling done, no new runs yet.

---

## 1. What data we have RIGHT NOW

| Asset | What it is | Use here |
|---|---|---|
| `../results_from_pod_codex/experiment_full.db` | **291 runs** from the context-files study (Claude 138 + Codex 153, n=3/cell, 2–3 strategies) | **Seed** — proves the phenomenon exists; NOT this study's data |
| `reliability_tasks.json` | **32 tasks, 4 repos**, test-touching, dedup | The benchmark we will repeat-run |
| `seed_analysis_existing_data.txt` | frozen output of `analyze_variance.py` on the seed | read this for the numbers |
| `reliability.db` | **does not exist yet** | will hold THIS study's runs |

The seed DB is *borrowed evidence*. It has only ~9 runs/task and mixes eval
methods — enough to motivate, not to conclude. This study collects purpose-built
high-repeat data into a fresh `reliability.db`.

### The 32-task benchmark
- **Repos (4):** `OpShin/opshin` (10), `pdm-project/pdm` (9),
  `firebase/firebase-admin-python` (7), `jlowin/fastmcp` (6, NEW 4th repo for
  generalization).
- **Filter:** only PRs whose gold diff **touches test files** (so Tier-C gold-test
  eval is valid). Docs/CI/changelog PRs were dropped on purpose.
- **Difficulty:** a mix — known floor/ceiling/medium anchors + 19 unknown (the
  Codex sweep will measure them). Variance lives in the *medium* band (0<p<1).

---

## 2. What the current data SAYS (and one correction)

Run `python3 analyze_variance.py ../results_from_pod_codex/experiment_full.db --eval tests`.

**Clean (test-eval only, the honest seed):**
- **Flip rate** (P two identical-config runs of a task disagree): **~6–7%**.
- **Nondeterministic tasks:** 18% (Claude), 15% (Codex).
- **Single-run benchmark SD:** **~5pp** for both agents. → a benchmark reported as
  "49%" from one run is really 49±5pp. Two papers at "48%" vs "54%" could be the
  *same* system.
- **Repeats needed:** even K=25 only tightens the 95% CI to ~4pp at our small task
  count — task-count and K *both* matter.
- **Ranking flip** (which agent is "better"): **18% at K=1**, falling to 8% at K=10.
- **Agent variance contrast:** difference **≈0**, CI spans zero → *no* agent is
  reliably noisier on clean eval.

**The correction (a real lesson, not a footnote).** Our first pass said "Codex is
2× noisier (24% vs 12%)." That was an **eval artifact**: tasks 906/926/932 have no
gold tests (906 is literally just an `AGENTS.md` edit) and were graded by the
fuzzy `line_overlap` fallback (72 of 291 runs). On clean test-eval the agent gap
**disappears**. → *Weak eval methods manufacture apparent nondeterminism.* This
becomes a sub-result (SPEC RQ4) and the reason we run **test-grounded tasks only**.

> Skeptic's note: n=3 *undercounts* variance (a true 50/50 task looks unanimous 25%
> of the time). Real flip rates are **higher** than the seed shows. Good for us —
> the effect is at least this big.

---

## 3. The plan (two arms, your two pods)

Fixed knob for everything: **strategy = `none`** (strip context entirely → pure
run-to-run noise, no context variable). Outcome = binary Tier-C gold-test pass.

| Arm | Agent / model | Cost | K | Tasks | Runs | Where |
|---|---|---|---|---|---|---|
| **Heavy** | Codex `gpt-5.5` (ChatGPT-auth) | **FREE** | 20 | 32 | ~640 | **Pod A** (and shard to Pod B) |
| **Budget** | Claude **Haiku** | ~$87 of $90 | 10 | ~20 medium | ~200 | after Codex picks the medium subset |

**Sequencing (important):** Codex runs **first** because it's free and its results
*define each task's difficulty*. We then spend the scarce $90 Haiku budget **only**
on the ~20 tasks Codex shows to be medium (0.2–0.8), where variance actually lives.
Same tasks on both agents → clean ranking analysis (RQ3).

**The binding constraint is WALL-TIME, not money.** Codex ≈ 15 min/run. 640 runs
serial ≈ 160 hr. The runner parallelizes by repo (4 tmux lanes) → ~40 hr. **Two
pods halves it again** → shard the task file by repo across Pod A / Pod B.

### Commands (run on pod, after `verify_lock` is GREEN)
```bash
# Codex heavy sweep (free). Shard by repo across pods with --repo if desired.
EXP_LIVE=1 python3 run_pilot.py --agent codex \
  --task-file reliability-study/reliability_tasks.json \
  --strategies none --repeats 20 \
  --db reliability-study/reliability.db
# e.g. Pod A: add  --repo opshin   /  --repo pdm
#      Pod B: add  --repo firebase /  --repo fastmcp

# Claude-Haiku budget arm (LATER, on the medium subset we pick from Codex data)
EXP_LIVE=1 EXP_CLAUDE_MODEL=claude-haiku-4-5-20251001 python3 run_pilot.py \
  --agent claude --task-file reliability-study/reliability_medium.json \
  --strategies none --repeats 10 --db reliability-study/reliability.db

# Analyze (anytime, locally or on pod)
python3 reliability-study/analyze_variance.py reliability-study/reliability.db --eval tests
```

### Before the heavy sweep — a 1-task cost/cache probe
Run **1 task × K=10** on each agent first. Confirms: streaming + eval work, Haiku
model flag takes, deny-hooks fire, and whether back-to-back repeats keep the 5-min
prompt cache warm (could cut Haiku cost well below $0.44/run).

---

## 4. The six pre-registered metrics (what we'll report)

Locked in `SPEC.md §4` and implemented in `analyze_variance.py`. No post-hoc swaps.

1. **M1 per-task variance** — distribution of pass-prob `p_i`, % nondeterministic.
2. **M2 flip rate** — P(two runs of a task disagree) = `2·E[p(1-p)]`.
3. **M3 benchmark SD vs K** — how much a K-repeat benchmark score wobbles run-to-run.
4. **M4 repeats-needed** — smallest K for 95% CI width ≤5pp / ≤2pp.
5. **M5 ranking flip** — P(single-run agent ranking is wrong vs high-K truth) by K.
6. **M6 agent variance contrast** — is one agent noisier? bootstrap CI over tasks.

All via task-clustered bootstrap (10k), stdlib only.

---

## 5. Risks / things to watch (be honest in the paper)

- **Model snapshots drift** — gpt-5.5 / Haiku could change mid-sweep. Pin the run
  window per agent; record model id per run (DB already does).
- **Wall-time tail runs** — a stuck agent eats hours. Watchdog timers +
  `EXP_MAX_TURNS` cap exist; monitor the first batch.
- **Medium-task selection bias** — picking the Haiku subset from Codex difficulty
  is a mild selection effect; report it, and the fastmcp repo is fully fresh.
- **fastmcp integration** — new repo; the cost/cache probe doubles as its
  validation (deps install, `uv run pytest` works, gold-test restore). If it
  doesn't run clean, drop it — Codex being free makes that cheap.
- **Determinism source** — CLIs don't expose temperature; frame variance as
  *as-deployed* (serving-level: batching/MoE/float), not user sampling.

---

## 6. Glossary
- **Tier-C eval** — apply the agent's patch, restore + run the gold PR tests; pass
  iff all gold tests pass with 0 failures.
- **Flip / nondeterministic** — same task, same config, different outcome across repeats.
- **K** — number of repeats per (task, agent).
- **line_overlap** — fuzzy diff-match eval fallback (no tests); a variance
  *contaminant*, excluded here.
- **strategy=none** — context file stripped; the controlled baseline.

---

## 7. Status checklist
- [x] Design + pre-registration (`SPEC.md`)
- [x] 32-task benchmark, 4 repos (`reliability_tasks.json`)
- [x] Haiku model flag (`EXP_CLAUDE_MODEL`)
- [x] Analysis tool, validated on seed (`analyze_variance.py`)
- [x] Seed analysis frozen (`seed_analysis_existing_data.txt`)
- [ ] Pod A/B bring-up + `verify_lock` GREEN
- [ ] Cost/cache probe (1 task × K=10, both agents)
- [ ] Codex heavy sweep → `reliability.db`
- [ ] Pick medium subset → Claude-Haiku arm
- [ ] Run M1–M6 → write-up
