# Response to reviewers — external audit of 2026-08-30

Reviewer: independent model-based audit (Codex, `gpt-5`-class, read `paper/arxiv.tex`
in full plus `RESULTS_2026-08-30_AKS_CORRECTION.md`). Four questions were posed:
unsupported claims, which arm to run next, per-venue rejection risk, and
contribution framing.

Status vocabulary: **ACCEPTED** (changed as asked) · **PARTIAL** (changed, with a
stated boundary) · **DISAGREE** (not changed, with evidence) · **DEFERRED**
(agreed in principle, scoped out by the author).

---

## Q1 — Unsupported or overstated claims

**Status: ACCEPTED, in full.** Every line-level item was changed. A closing sweep
confirms none of the flagged phrasings survives in either file.

| Reviewer item | Change |
|---|---|
| "on hour-long videos", "dominant lever" | Scoped to the LongVideoBench bin carrying the evidence; the lever claim is replaced by the sharper and better-supported statement that *which* rule is used matters more than whether it is query-aware, which is what Video-MME shows |
| "That ordering survives a scorer swap" | Scoped to the three rules and one bin the test covers; states explicitly that the six-rule ordering is not shown to be scorer-invariant |
| "The three interventions rank cleanly" | Restated as a synthesis across separate contrasts, not an identified effect ranking |
| "which frames … buys more than how many" | Scoped to the answerer visual-token budget, with the unmeasured selection-stage cost named at first use rather than 200 lines later |
| "two independent cross-budget contrasts" | "two disjoint LongVideoBench duration bins", naming the shared benchmark, scorer, answerer and implementation |
| "Explicit boundaries on all three claims" | Softened, and the unevenness of those boundaries stated |
| **"gain grows with video duration"** | **Corrected.** The caption contradicted the body one line below it (7.8 points at 600 s, 7.5 at 3600 s). Recast as a threshold that switches on when the pool outgrows the budget, with the bin contrast marked observational |
| Post-hoc TOST margins | "statistically equivalent" / "formal equivalence" / "establishing equivalence" removed throughout. The interval and both margins are reported, the post-hoc choice is stated, and the result is called descriptive. Multiplicity non-correction noted |
| Causal and mechanistic language (8 sites) | "rules out" → "weakens"; "shows the stability is a property of" → "is consistent with"; "carries no usable signal" → "did not improve accuracy detectably", with the absent equivalence test named; the residual account explicitly labelled a hypothesis with no intervention on residual geometry; "model capacity is not the explanation" → capacity-by-benchmark interaction left open; "entirely" removed; variance-apportionment claim withdrawn |
| Failure audit | Described as purposive and failure-conditioned. "Dominant OMP-specific pattern" → "most frequent tag", with the absent blinding, inter-rater agreement, sampling frame and matched control stated |
| "confirms the diversity term carries no signal" | Restated as what one query-blind configuration looks like |
| "bounds how much weight any cross-paper number can carry" | Two harnesses cannot bound this in general; restated as evidence that deltas of this size are fragile |
| "a lower floor leaves more room" | Labelled a hypothesis we did not test |
| "OMP works here for a reason" | "appears to work … in a way its original derivation does not predict" |
| "worth close to four accuracy points" | "roughly two to four points across the two long bins", now supported by the 3600 s replication rather than the 600 s figure alone |

**Beyond the review.** Two defects the audit did not catch were found while acting
on it: `\ref{tab:t3}` had no matching label in the ACL build, so the submitted PDF
would have read "Tables 1--??" in three places; and the Limitations section sat
*before* the Conclusion, which violates ARR's required order and hid a countable
section from our own page measurement.

---

## Q2 — Which arm to run next

**Status: ACCEPTED for the ranked first choice; DEFERRED for the rest.**

The reviewer's top-ranked arm was a fused-query replication of the prompt-boundary
result on the 3600 s bin, on the ground that the effect was measured on 600 s alone
while the headline concerns hour-long video. We pre-registered the contrast before
running it (`prereg/2026-08-30_fused_query_3600s.md`) and ran it.

> OMP $k{=}8$, LVB-3600 s, $n{=}564/564$: stem .5461 → fused .5691, **+2.30 points**,
> 46 rescued against 33 broken, exact McNemar $p{=}.18$. Manipulation check: 3 of 564
> frame sets identical, mean Jaccard 0.31.

Replicates on the criterion fixed in advance, which was direction and magnitude
rather than significance: $n{=}564$ cannot resolve a three-point shift, and the
600 s contrast was itself $p{=}.134$. The prompt boundary is now stated as a level
shift on long video rather than a property of one bin.

Ranked arms 2–4 (SigLIP at 3600 s, a text-only floor on LVBench, a SigLIP matrix on
Video-MME) are **DEFERRED**. Each would widen a claim rather than repair one, and
the reviewer's own position was that the remaining problem is claim scope, not
missing evidence. The claims were narrowed instead.

---

## Q3 — Per-venue rejection risk

**Status: ACCEPTED on length; PARTIAL DISAGREE on ACL fit.**

*Length.* The reviewer's predicted desk-reject cause for ARR was body length. It
was worse than either of us thought: the Conclusion counts toward the limit and
Limitations does not, and our section order had them reversed. After reordering and
cutting, the body is **exactly eight pages** with no unresolved references. Detail
moved to appendices behind one-sentence body pointers; every relocated table cell
is still stated numerically in the surrounding prose, so nothing needed to assess a
central claim now sits only in an appendix.

*Fit.* We **partially disagree** that this is a poor ACL-family fit. Checking the
calls directly rather than reasoning from venue reputation, ARR's CFP names this
paper's contribution types explicitly: "reproduction study", "negative results",
"model analysis papers", and negative results covering "non-reproducibility or
non-generalizability of previously published results". ICLR's call, by contrast,
does not mention negative results or reproducibility at all and asks for "your most
complete and most exciting work". This reversed our own earlier ICLR-first
inclination. Venue is locked to **ARR October 12 2026**.

*NeurIPS.* The reviewer's claim that the Evaluations & Datasets 2026 deadline has
passed is confirmed by the author. The track rename could not be independently
verified here and is recorded as unverified.

---

## Q4 — Contribution framing

**Status: PARTIAL.** This is the item where the response is incomplete, and we
prefer to say so rather than overstate it.

The reviewer's verdict was that the paper "pretends its main contribution is the
prescription select/compress/reinvest, while its strongest original result is
evaluation instability", and that OMP should be a baseline rather than the
protagonist.

**Accepted and done:** the AKS porting failure is now a named result rather than a
silent correction, in the abstract, a contribution bullet, and a Discussion
paragraph, framed as the reviewer suggested — the finding is not the bug but how
completely aggregate accuracy hid it (99.5% of frames changed, 0.07 points moved on
LVBench). **The abstract now leads with the three evaluation results** — the port
failure, the 0.07-to-3.74-point cross-harness disagreement, and the prompt-boundary
shift — before any selector number, with every previous number and caveat retained.

**Not done:** the paper has not been restructured around that lead. Section order,
the Results ordering, and the title still follow the select/compress/reinvest
sequence. This is an author scope decision for the October 12 cycle, not a
disagreement with the reviewer: a full reframe means a new introduction, a
reordered Results, and probably a new title. It is recorded as the largest known
open item.

---

## Author-side items the review did not cover

- **MDP3 and Q-Frame (R9)** appear in LDDR's matched table and not in ours. Both
  would be from-scratch ports with no published number at our budget to validate
  against, which is the exact failure class that produced the AKS bug. Dropped
  deliberately; the reasoning is now in the source. A reviewer may still ask.
- `PAPER_DRAFT.md:20` carries a stale pre-panel venue table, superseded by
  `VENUE_TARGETS_POSSIBLE.md` but not yet deleted.
- arXiv posting has not happened. ARR has no anonymity period, so it costs no
  optionality and would timestamp the concealment result.
