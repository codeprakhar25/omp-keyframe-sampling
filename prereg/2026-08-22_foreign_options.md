# Pre-registration — foreign-options control arm

Written 2026-08-22, BEFORE generating picks or running the arm. Follows
`2026-08-22_fused_query.md`, whose two CPU analyses refuted the modality-gap
mechanism (H1) and found no support for the low-content-stem mechanism (H2).

## What this arm tests

The fused query (question + its own options) gained +3.88 pt for OMP/LongCLIP on
LongVideoBench-600s. The residual trace showed this is NOT better query-image
alignment: the fused query is MORE orthogonal to the image subspace (explained
variance after 8 picks 5.91% fused vs 6.26% stem, t=+7.35 on resid_frac).

The post-hoc reading from that trace is that a fused string carries ~6 semantic
directions where a stem carries ~1, so OMP keeps finding relevant directions past
the pick-5 point where a stem's residual is dead.

**If that is the whole story, the answer set is irrelevant and any six plausible
directions will do.** This arm supplies exactly that: the item's own stem, with the
options block replaced by another item's options.

## Construction

For each of the 412 items: `stem(self)` + the verbatim options block of a donor item,
plus the same trailing answer instruction, preserving the exact surface format.

Donor constraints, all applied before any outcome is seen:
- different `video_file` (LongVideoBench has multiple questions per video),
- identical option count, so the A)/B)/C)... letters line up,
- nearest fused-token length to the item's own fused string, deterministic tie-break.

Achieved length match is reported. Selector OMP, scorer LongCLIP, k=8, bin 600s,
n=412, run in the same environment as the stem and fused arms already there.

## Pre-registered contrasts

Both paired McNemar on the same 412 ids:
- **C1 foreign vs stem** — is a length- and structure-matched query with the WRONG
  options as good as the stem?
- **C2 fused vs foreign** — do the item's OWN options beat foreign ones?

## Decision rule, fixed here

- **C1 clearly positive and C2 null** => the gain is structural (more query
  directions), not the answer set. The "answer-aware selection" framing is dead.
  The finding survives only as an evaluation-practice point about query construction.
- **C1 null and C2 clearly positive** => the option CONTENT carries the gain. The
  direction is worth replication (3600s, n=564) and the gold-removed arm.
- **Anything in between, including both underpowered** => close the direction. Do
  NOT spend more on it.

## Power, stated before the run

This is the binding caveat. The entire fused-vs-stem effect is a 16-item margin
(33 rescued, 17 broken, 50 discordants of 412). C2 asks for a difference between two
arms that are far closer to each other than fused was to stem, so it is expected to
be underpowered. A null C2 is therefore WEAK evidence and must not be read as
"content does not matter" on its own.

C1 is the informative contrast: it is the same size of comparison as the original
fused-vs-stem test, against the same stem baseline (.6311, OMP/LongCLIP).

The third outcome above is pre-committed precisely so an ambiguous result does not
turn into a request for more compute.

## Cost

One GPU arm, L40S, ~412 items at k=8. ~$2.

---

# Construction check (recorded BEFORE the GPU arm returned)

`modal_siglip.py::picks_foreign`, run 2026-08-22.

- Donor option count differs for **1/412** items (the single 3-option item in the
  bin has no same-count donor on a different video). Option letters are rewritten to
  run contiguously from A), so the block stays well-formed. The answerer's prompt is
  untouched by any of this — only the scorer's query text changes — so a count
  mismatch costs nothing.
- Length match: mean |delta tok| **13.90**, median **10.0**, max 143.
- Donor pool: **343 distinct donors** across 412 items, max reuse 3.
- Spot check, item 0: own options are `watch / doll / earrings / potted plant /
  necklace`; donor options are `Faced the camera and bit their tongue / Sat on the
  stage and stretched their back / ...`. Semantically unrelated, structurally identical.

**Pick perturbation is matched to the real fused query**, which is what makes C1 a
fair test rather than a weaker intervention:

| query | OMP picks shared with stem /8 | % changed |
|---|--:|--:|
| fused (own options) | 3.74 | 53% |
| foreign (donor options) | 3.68 | 54% |

foreign vs fused directly: 3.74/8 shared, 53% changed — the two perturbed queries
disagree with each other as much as either disagrees with the stem.

---

# RESULTS (2026-08-22)

`modal_siglip.py::run_arm_foreign` (L40S, 412/412 unique qids, coverage gate passed)
then `::analyze_foreign`. Raw: `slm-lab-sig:/results/prereg/foreign.json`.

| arm | acc |
|---|--:|
| stem | .6311 |
| foreign (donor options) | **.6529** |
| fused (own options) | .6699 |

Paired McNemar on the same 412 ids:

| contrast | delta | rescued | broken | p |
|---|--:|--:|--:|--:|
| **C1 foreign vs stem** | +2.18 | 27 | 18 | **.2327** |
| **C2 fused vs foreign** | +1.70 | 31 | 24 | **.4188** |
| (ref) fused vs stem | +3.88 | 33 | 17 | .0328 |

## Verdict: the third branch of the decision rule. DIRECTION CLOSED.

Foreign options land almost exactly halfway. **A query built from another video's
answer options — semantically unrelated, length matched, perturbing 54% of OMP's
picks against the real options' 53% — recovers 56% of the fused-query gain
(2.18 of 3.88 pt).** Whatever the item's OWN options contribute beyond that is
1.70 pt and is not distinguishable from zero.

Neither C1 nor C2 is significant. So this does not cleanly establish the structural
story either; what it rules out is the clean content story. The defensible statement
is the negative one: **the fused-query gain is not specific to the item's own answer
options.** "Answer-aware selection" has no support.

Per the rule fixed before the run, this closes the direction. No replication at
3600s, no gold-removed arm, no Video-MME run.

## What survives

1. The measurement itself. Scorer query construction moves LongVideoBench-600s
   OMP accuracy by ~3.9 pt (.6311 -> .6699), which is larger than most published
   selector gaps. That is an evaluation-practice finding and needs no mechanism.
2. It is NOT a method and must not be written as one. The gain does not come from
   the answer set, so there is nothing "answer-aware" to propose, and the MCQ-only
   objection would have applied anyway.
3. Consistent with the field: KeyVideoLLM (arXiv 2407.03104) defines exactly this
   axis (CLIP-Q / CLIP-A / CLIP-QA) and uses question-only at inference.

## Cumulative evidence against the mechanism hypotheses

| hypothesis | test | outcome |
|---|---|---|
| closes the modality gap | residual trace, n=412 | REFUTED, wrong direction, t=+7.35 |
| helps where stems are low-content | pre-registered interaction | ns, p=.7568; Spearman +0.02 |
| the item's own options carry it | foreign-options arm | ns, C2 p=.4188; 56% recovered by wrong options |

Three pre-registered mechanism tests, three negatives. Total spend across all of
them: one GPU arm, ~$2, plus CPU minutes.
