# Paper skeleton — "Do AGENTS.md Files Help Coding Agents?"

**Target:** REALM workshop (~6-week clock). **Type:** rigorous naturalistic null +
power analysis. **Working title:** *Do Context Files Help Coding Agents? A Controlled
Ablation of AGENTS.md on Real Repositories.*

**One-sentence thesis:** Injecting real-world repository context files (AGENTS.md /
CLAUDE.md) does **not** measurably improve coding-agent **correctness** across two
frontier agents on real GitHub tasks; the only detectable effect is a narrow
context-footprint efficiency change in one agent — and we **quantify** exactly how
large an effect our design could have detected (TOST/power), so the null is
characterized, not just asserted.

---

## Section map (and where current RESULTS.md content goes)

### Abstract
Null across 2 agents; naturalistic real-repo ablation; power-bounded (<15pp);
practitioner-relevant takeaway. ~150 words.

### 1. Introduction
- AGENTS.md/CLAUDE.md are now ubiquitous; agents auto-load them. Practitioners spend
  effort writing them. **Do they actually help?**
- The contradiction: Paper-1 (efficiency framing) vs Paper-2 — RESULTS.md §0/§1.
- Contributions (bullet):
  1. Controlled ablation of context-injection *strategy* (none / always_on / selective)
     on real issue→PR tasks, **two agents** (Claude Code, Codex).
  2. SWE-bench-style **gold-test (Tier-C) correctness** eval + portable efficiency metrics.
  3. A **rigorous null**: TOST equivalence + Monte-Carlo power (the thing most ablations skip).
  4. Agent-specific difficulty is real (borderline tasks differ by agent) — methodological finding.

### 2. Related Work
- Agentic coding / SWE-bench. Context & memory files. Retrieval-augmented agents.
- Paper-1 / Paper-2 positioning. Ablation & equivalence-testing methodology.

### 3. Method  (RESULTS.md §2, §3, §8)
- **Tasks:** 3 repos (pdm, firebase-admin-python, opshin), issue→PR, base-SHA checkout,
  future-history stripped. Tier-C eval = gold PR tests vs agent code.
- **IV — three injection strategies:** define none (strip context, no system prompt),
  always_on (full AGENTS.md injected), selective (wiki split + retrieval hint). Fig 1.
- **Agents:** Claude Code (sonnet-4-6), Codex (gpt-5.5). Cross-agent portable metric
  (tool_calls, wall-time, output tokens; drop turns + cache — §4B caveat).
- **Borderline screening:** difficulty = task×agent property; keep 0<pass<n (§4C).
- **Stats:** within-task design, Wilson CIs, task-clustered bootstrap, sign-flip +
  omnibus permutation, TOST, Monte-Carlo power (`power_analysis.py`).
- **Safety/repro harness** → Appendix (egress lock, scrub remotes, deny hooks).

### 4. Results  (RESULTS.md §4, §4B, §4D, §4E, §5.2)
- 4.1 Determinism / floor-ceiling (§5.2) — motivates borderline screening.
- 4.2 **Correctness null, Claude** (§4) + **Codex replication** (§4B, §4D).
  Table: pass-rate × strategy × agent. **Headline: null holds even WITH dynamic range.**
- 4.3 **Efficiency:** Claude selective leaner cache (p_Holm=0.012, the one signal);
  Codex flat (all |dz|<0.2). Table.
- 4.4 **Power & equivalence (§4E):** omnibus p=0.66; TOST bounds <15pp; MDE>30pp;
  ~120-200 tasks for 10pp; repeats don't help. **Present as methodological rigor.**
- 4.5 Agent-specific borderline (3790 Claude vs 3769 Codex; pdm codex-floored).

### 5. Discussion
- **Naturalistic interpretation (the load-bearing framing — use near-verbatim):**
  the AGENTS.md in our sample are representative real-world context files — predominantly
  style/architecture guides. If that's what practitioners write, the null is directly
  informative: *the context files practitioners actually create don't measurably improve
  correctness.* Whether purpose-built, task-specific (load-bearing) context would help is
  an explicit open question for future work.
- Process vs outcome: strategy doesn't move correctness OR (on Codex) efficiency.
- Practitioner takeaway: effort on generic AGENTS.md may not pay off in agent success.

### 6. Limitations  (RESULTS.md §5)
- Power: bounded to 15pp; n=3; 10pp effect undetectable (§4E).
- 3 repos, all Python; generality.
- **Arm asymmetry:** Claude 15 tasks vs Codex 17; not identical task sets.
- Mixed-machine provenance (§5.6); Codex user-turn vs system-channel injection confound (§4B).
- selective is our construction (§5.4).

### 7. Conclusion + Future Work
- Restate scoped null. Future: load-bearing-context probe; scale to ~120 tasks for a
  powered equivalence; more languages/repos.

### Appendix
- Safety harness; full per-task tables; reproduce steps (§8); prompt templates.

---

## Figures / Tables to produce
- Fig 1: the 3 injection strategies (diagram).
- Tab 1: pass-rate × strategy × agent (Claude 15 / Codex 17).
- Tab 2: efficiency × strategy × agent (tools/tokens/duration + Claude cache).
- Tab 3: power/equivalence summary (TOST bounds, MDE, n-needed).
- Fig 2: per-task pass-rate dot plot by strategy (shows overlap / no separation).
- Fig 3 (optional): power curve (power vs n_tasks for Δ=10pp).

## Writing TODO (6-week)
- [ ] Wk1: finish Claude borderline run (in flight) → fold into Tab 1/2. Outline → prose.
- [ ] Wk1: regenerate Tab 1-3 + Fig 2-3 from experiment_merged.db + claude_borderline.db.
- [ ] Wk2-3: draft Method + Results (RESULTS.md is ~80% raw material).
- [ ] Wk2-3: Intro + Related Work (the hard, load-bearing 20%).
- [ ] Wk4: Discussion + Limitations (own the inert-manipulation point), polish, figures.
- [ ] Wk5: internal review pass; tighten claims to match power bounds.
- [ ] Wk6: format for REALM, submit.

## Open decisions
- Merge Claude borderline (4) into the master? → analysis db = experiment_merged.db +
  claude_borderline.db (or merge into one). Decide before Tab 1.
- Report Claude arm as 15 tasks (pilot 11 + 4 borderline) vs Codex 17 — own the gap.
- Include pdm-codex-floored as a finding (agent-specific saturation) or footnote.
