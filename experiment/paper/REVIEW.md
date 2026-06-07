# Simulated Peer Review — "Do Context Files Help Coding Agents?"

**Venue:** REALM @ EMNLP 2026 (workshop) · **Format:** ACL · **Type:** empirical null-result, ML agents / empirical SE
**Mode:** full (5 reviewers + editorial synthesis) · **Date:** 2026-06-06
**Manuscript:** `paper/main.tex` (read-only review; paper not modified)

---

## Phase 0 — Field Analysis & Panel Configuration

- **Primary field:** ML / LLM coding agents (evaluation). **Secondary:** empirical software engineering, experimental methodology.
- **Paradigm:** quantitative, controlled ablation, within-task paired design, frequentist + equivalence testing.
- **Maturity:** complete first draft; workshop-scope (short/long). Honest null with methodological framing.
- **Target tier (calibration):** workshop at a top-tier NLP venue → bar is *soundness + a transportable lesson*, not SOTA.

**Panel:**
1. **EIC** — workshop area chair, agent-evaluation & reproducibility focus.
2. **R1 Methodology** — experimental statistician (equivalence testing, power, multiple comparisons).
3. **R2 Domain** — coding-agents researcher (SWE-bench lineage, agent memory/context).
4. **R3 Perspective** — practitioner-facing / HCI-of-dev-tools, external validity.
5. **DA Devil's Advocate** — core-argument challenger.

---

## Phase 1 — Reviews

### R0 · Editor-in-Chief — overall

**Summary.** A controlled ablation of context-injection *strategy* (none/always_on/selective) for two frontier coding agents on real issue→PR tasks, gold-test evaluated. Headline is a power-characterized null on correctness, two narrow process-level efficiency signals, and a mechanism (skill-not-knowledge) supported by a failure triage + a manipulation-validity probe.

**Strengths.**
- Genuinely **two agents from two vendors** + **gold-test (not LLM-judge) correctness** — above the bar for most context-file claims.
- **Equivalence + power analysis** instead of "p>0.05 therefore no effect." Rare and welcome.
- Unusually **honest** about scope, confounds, and underpowering.
- The **agent-specific difficulty** result (Tab 3) and the **manipulation-validity probe** are the most transportable ideas.

**Concerns (gating).**
- **C1 (blocking): placeholder citations.** `paper1`/`paper2` are the entire framing ("reconcile the contradiction") yet are unresolved `[PLACEHOLDER]`. The paper's central motivation cannot be evaluated, and it cannot be published with placeholder references. Must be filled and the reconciliation claims checked against what those papers actually report.
- **C2: contribution altitude.** For a workshop this is fine, but the "so what" leans on rigor + mechanism, not a positive finding. Foreground the *transportable* lessons (agent-specific screening; turn-metric non-portability; the probe protocol) earlier.

**Provisional:** Major Revision.

---

### R1 · Methodology Reviewer

Evidence-based, specific. Scores at end.

**M1 — TOST vs MDE tension (MAJOR).** The paper claims equivalence "≤10pp (Claude)" (Tab 2, §4.2) yet also reports MDE >30pp at n=15/reps=3 (§4.4: Δ=30pp → 40–57% power). On its face these look contradictory — "we can't detect a 30pp effect, but we bound the effect to 10pp." They are reconcilable (TOST bounds the *realized* estimate's bootstrap CI, which is narrow because the point estimate ≈0; MDE concerns *power to detect a hypothesized* effect), but the paper never reconciles them and a careful reader will distrust the equivalence claim. **Fix:** add one paragraph explaining the distinction, and report the actual TOST bounds (the bootstrap CI half-width), not just the binary "Equivalent."

**M2 — bootstrap at n=15 (MAJOR).** Task-clustered bootstrap CIs on 15 (Claude) / 17 (Codex) clusters are themselves unstable; percentile CIs under-cover at this n. The ≤10pp equivalence rests on a small-cluster bootstrap. **Fix:** report the bootstrap CI numerically, add a sensitivity check (BCa or a sign/permutation-based equivalence), and soften "Equivalent" to "bootstrap CI within ±δ (n=15, interpret with caution)."

**M3 — opshin efficiency effect is exploratory + untested (MAJOR).** §4.3 Tab 4: the full-suite counts (3.67/2.44/1.67) are presented as the mechanism but **no inferential test is run on the counts themselves** — only the duration (p=0.125, n=5). Was this effect pre-registered or found during a post-hoc dig? The text reads exploratory. **Fix:** (a) label it explicitly exploratory; (b) run a test on the per-cell full-suite counts (e.g., within-task paired / ordinal trend test across the 3 strategies) rather than only describing means; (c) keep the honest "p=0.125, underpowered" framing.

**M4 — multiple comparisons / garden of forking paths (MODERATE).** The two surviving signals (cache p_Holm=0.012; opshin process effect) are survivors among many tested metrics × pairs × repos. The cache result is Holm-corrected within its family (good), but the opshin effect is a separate, repo-specific dig not in the same correction family. **Fix:** state the full set of efficiency tests run and which were pre-registered vs exploratory; present a single accounting so the reader can judge selection.

**M5 — probe over-reach from one task (MAJOR, shared w/ DA).** §5.2 "where correctness has range, context trends *down*" is supported by **exactly one task** (Claude 907: 2/3→1/3→0/3, n=3/cell). That is 9 runs total behind a directional claim. **Fix:** demote to "in the single task with cross-agent range, the trend was non-positive (2/3→1/3→0/3); we do not claim a general downward effect." Remove "trends down" from the abstract or hedge it heavily.

**M6 — averaging binary→rate then permuting (MINOR).** Collapsing 3 binary repeats to a task rate before the omnibus permutation is defensible but discards within-task variance; a mixed-effects logistic (task random intercept) would be the standard and would also give a CI on the strategy effect directly. **Fix:** add a GLMM as a robustness check or justify the rate-permutation choice.

**Reproducibility:** strong — dbs + scripts released, harness documented (App A). Good.

**Scores (0–10):** Rigor 6 · Reproducibility 9 · Stats reporting 5 (under-reported CIs, MDE/TOST tension) · Overall methodology **6**.

---

### R2 · Domain Reviewer (coding agents)

**D1 — related work too thin (MAJOR).** One paragraph per topic. For REALM (memory/retrieval) the paper must engage: context/memory for agents beyond MemGPT/Voyager (e.g., retrieval-augmented code generation, repo-level context selection), SWE-bench *Verified*, and any prior AGENTS.md/CLAUDE.md-specific studies beyond the two placeholders. **Fix:** expand to position the contribution against repo-context-selection work specifically.

**D2 — content-type not varied (MODERATE).** The IV is *injection channel*, not *content*. The audit (firebase "Excellent" etc.) is used to argue files aren't junk, but the study never varies content type (style-guide vs task-specific). The honest conclusion is "channel doesn't matter," yet the practitioner claim (§5.3) generalizes to "context files don't help" — a content claim the design can't fully support. **Fix:** tighten the claim to injection-channel, or explicitly scope the content generalization as conjecture.

**D3 — agent-specific difficulty under-developed (MAJOR, but an opportunity).** Tab 3 (3790/3769) is the paper's most novel, transportable result but rests on 2 example tasks. This is the finding most likely to be *cited*. **Fix:** quantify it across the whole task set (e.g., rank-correlation of per-task pass rates between agents; how many tasks are borderline for exactly one agent). That turns an anecdote into a result and strengthens the "reconciles prior work" argument.

**D4 — injection-channel confound (MAJOR).** Claude gets context via system prompt, Codex via user-turn prepend (no system-prompt flag). The cross-agent *comparison* of efficiency profiles (§4.3 "Codex uniformly leaner") is then partly confounded by channel + tokenizer + accounting differences. The within-agent strategy comparison is clean; the between-agent comparison is not. **Fix:** restrict between-agent claims to qualitative, and flag the cache-accounting asymmetry (Codex folds cache into input) wherever cross-agent token counts appear.

**Scores:** Lit coverage 4 · Domain contribution 6 · Framing accuracy 6 · Overall **5**.

---

### R3 · Perspective Reviewer (practitioner / external validity)

**P1 — practitioner overgeneralization (MAJOR).** §5.3 "generic context files do not measurably improve coding agent correctness" → "practitioners may be better served investing effort elsewhere." From 3 Python repos, 15/17 tasks, underpowered. Defensible as *scoped* but currently reads as a broad prescription. **Fix:** bound to "in our sample of 3 Python repos with naturalistic style-guide-type AGENTS.md, we detect no correctness benefit; we cannot rule out benefits for other languages, larger repos, or task-specific context."

**P2 — the useful positive message is buried.** The process-effect (context cuts wasted full-suite runs) is *actionable*: context can make agents cheaper/faster even when it doesn't change pass/fail. That's a practitioner-relevant nuance worth elevating from a §4.3 paragraph.

**P3 — external validity (MODERATE).** All-Python, 3 repos, two specific model versions that will age. State the model-version snapshot explicitly as a validity caveat (already partly in Limitations).

**Scores:** Practical impact 6 · External validity 4 · Clarity 7 · Overall **6**.

---

### DA · Devil's Advocate

**Strongest counter-argument (≈250 words).**
The paper's core risk is that it is an *underpowered* study wearing the costume of a *characterized* null. The design cannot detect anything below a ~30pp correctness effect (§4.4), which is enormous — almost no real intervention moves correctness by 30pp. So "no detectable effect" is close to preordained by the sample size, and the equivalence machinery (TOST ≤10pp) is doing rhetorical work the data may not support: a percentile bootstrap on 15 clusters can easily produce a deceptively narrow CI around a near-zero point estimate, which then gets reported as "Equivalent ≤10pp" while the honest power analysis on the same page says a 30pp effect would be missed 40–60% of the time. A skeptic reads these two facts together and concludes the study simply *could not have found* a moderate effect, making both the null and the equivalence bound weak evidence rather than strong. Layered on top: the two "surviving" signals are survivors of many comparisons (cache + a repo-specific post-hoc dig), and the headline-grabbing "context trends *down*" rests on **nine runs of one task**. Strip those, and the paper's positive claims reduce to "we couldn't detect much, on a small sample, with a confound between the two agents we compare."

**Issue list.**
- **CRITICAL — placeholder citations (paper1/paper2).** The paper *is* a reconciliation of two studies it does not cite. Unverifiable core framing. (Also blocks Accept per panel rule.)
- **CRITICAL — single-task "trends down" overclaim** in abstract + §5.2. One task, n=3, presented as a directional finding. Must be hedged before acceptance.
- **MAJOR — TOST/MDE rhetorical tension** (see M1): the equivalence claim and the power claim are presented without reconciliation; together they can be read as self-undermining.
- **MAJOR — selection of surviving signals** not presented as a single multiple-comparison accounting (M4).
- **MAJOR — between-agent confound** used for a "Codex leaner" claim (D4).
- **MINOR — "291 cells" framing** sounds large but is 15–17 tasks × 3 × 3 × 2; the effective n is *tasks*, which the paper elsewhere admits. Don't let the cell count imply more power than exists.

**Ignored alternatives.** (a) The null may be *channel-specific*: a different selective design (true semantic retrieval, not wiki-split) might help — the paper's selective is one construction (own Limitation, but the practitioner claim ignores it). (b) The opshin duration effect could be driven by 2–3 slow tasks, not a uniform shift (the means hide the distribution).

**Missing stakeholders.** Agent/tool *vendors* (who ship the auto-load behavior) — the paper's "don't bother" message has product implications it doesn't engage.

**"So what?" test.** Passes *if* reframed around the transportable lessons (agent-specific screening, turn-metric non-portability, the probe protocol, process-not-outcome efficiency). Fails if sold as "context files don't help."

---

## Phase 2 — Editorial Decision

**Decision: MAJOR REVISION.** (Devil's Advocate raised CRITICAL issues → Accept is precluded per panel rule. Both CRITICALs are fixable without new large-scale experiments.)

**Consensus across ≥4 reviewers:**
1. Placeholder citations must be resolved (EIC C1, DA CRITICAL). **Blocking.**
2. Hedge/soften two overclaims: "context trends down" (single task) and the broad practitioner prescription (R1·M5, R3·P1, DA). **Blocking-ish.**
3. Reconcile TOST vs MDE and report numeric CIs; treat the opshin effect as exploratory + add an inferential test (R1·M1/M2/M3).
4. Develop the agent-specific-difficulty finding into a quantified result, not 2 examples (R2·D3) — the highest-upside revision.

**Disagreement / arbitration:** R2 wants substantial related-work expansion; for a *workshop* the EIC arbitrates this to "targeted expansion (1–2 paragraphs positioning vs repo-context-selection work)" rather than a full survey. Not blocking.

---

## Revision Roadmap (prioritized — drop-in for revision mode)

**P0 — blocking (must fix before acceptance)**
1. Replace `paper1`/`paper2` with real citations; verify the reconciliation claims (§1, §5.2) against what those papers actually report. *(owner: authors — external input)*
2. Abstract + §5.2: remove/heavily hedge "trends *down*." Reframe to "in the one task with cross-agent range, the trend was non-positive (n=3); we make no general directional claim."
3. §5.3 + abstract: scope the practitioner claim to "3 Python repos, naturalistic style-guide context, channel not content."

**P1 — major (expected for a strong revision)**
4. §4.2/§4.4: one paragraph reconciling TOST equivalence vs MDE>30pp; report bootstrap CI half-widths numerically; add a non-bootstrap equivalence robustness check.
5. §4.3: label the opshin process-effect exploratory; add an inferential test on the full-suite counts; present a single accounting of all efficiency tests (pre-registered vs exploratory) for the multiple-comparisons concern.
6. §4.5: quantify agent-specific difficulty across all tasks (cross-agent per-task pass-rate rank correlation; count of single-agent-borderline tasks). Promote toward a headline contribution.
7. §4.3 + §6: restrict between-agent ("Codex leaner") claims to qualitative; flag cache-accounting asymmetry at every cross-agent token comparison.

**P2 — minor (polish)**
8. Targeted related-work expansion (repo-context selection / RAG-codegen / SWE-bench Verified).
9. Elevate the process-not-outcome efficiency message as the actionable practitioner takeaway (R3·P2).
10. Add model-version snapshot caveat; soften "291 cells" so it doesn't imply more power than the task-level n provides.
11. Optional: per-task dot-plot (overlap visualization) + GLMM robustness check.

**Strengths to preserve (do not over-correct into self-flagellation):** two-agent gold-test design, equivalence+power rigor, the probe protocol, and the overall honesty. The revision is about *precision of claims*, not redoing the study.

---

### Dimension scores (panel mean, 0–100 → workshop scale)
| Dimension | Score | Note |
|---|---|---|
| Originality | 62 | Null + method; novelty in agent-specific difficulty & probe protocol |
| Methodological rigor | 62 | Strong design; CI under-reporting, TOST/MDE tension, exploratory-vs-confirmatory blur |
| Evidence sufficiency | 55 | Underpowered (own admission); single-task overclaim; placeholder cites |
| Argument coherence | 66 | Clear; a few claims outrun the data |
| Writing quality | 78 | Clean, well-structured, honest |
| **Overall** | **64 / Major Revision** | Sound, fixable; raise claim-precision and resolve citations |

*Per skill IRON RULE: this review is a separate document; `main.tex` was not modified. CRITICAL findings (placeholder citations; single-task overclaim) preclude an Accept decision until addressed.*
