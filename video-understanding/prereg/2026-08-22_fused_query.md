# Pre-registration — fused-query (question+options) scorer effect

Written 2026-08-22, BEFORE inspecting any per-item outcome data for the analyses
below. Motivation: the 2026-08-21 fused-vs-stem result on LongVideoBench-600s
(OMP/LongCLIP .6311 -> .6699, +3.88 pt, p=.033, n=412) is uncorrected, single-bin,
and contradicts the prior expectation that answer options act as distractors.

Existing per-item results are already on the Modal volume `slm-lab-sig`
(`results/omp_lc_600s`, `results/omp_lc_fused_600s`, and the top-k counterparts).
The accuracy deltas above are known. What is NOT yet known, and is what these two
analyses test, is (A) the geometric mechanism and (B) which items the gain sits on.

## Analysis 1 — residual trace under the fused query (CPU, no GPU)

`scripts/omp_residual_trace.py` math, run twice on the same 412 LongVideoBench-600s
items and the same LongCLIP image embeds, changing only the text query
(stem vs fused).

**Hypothesis H1.** The fused query is less orthogonal to the image subspace than the
stem, i.e. it partially closes the modality gap documented on 2026-07-19
(stem: residual fraction 0.972 -> 0.967 across 16 picks, `cos_orig` max 0.23,
OMP explains ~3.3% of query norm ever).

**Pre-registered predictions, fused vs stem, same items:**

- P1.1 `cos_orig` at pick 1 is HIGHER under fused.
- P1.2 `resid_frac` after 8 picks is LOWER under fused (more query mass drained).
- P1.3 total drained mass over picks 1-8 is HIGHER under fused.

**Falsification.** If all three are flat (within noise) then the accuracy gain is NOT
explained by better query-image alignment, and the leading alternative becomes the
input-length/distribution confound (LongCLIP was fine-tuned on long captions;
stem median is 50 tokens, fused median 102). That alternative is settled by the
`foreign-options` GPU arm, not by this trace.

## Analysis 2 — where the gain sits (CPU, no GPU)

Paired per-item correctness, OMP/LongCLIP, stem arm vs fused arm, n=412.

**Hypothesis H2.** The fused-query gain is concentrated on items whose STEM carries
little visual content, because for those items the options supply most of the
groundable nouns.

**Primary split (fixed here, computed from text alone, outcome-blind):**

    added_frac = (tok(fused) - tok(stem)) / tok(fused)

split at its median over the 412 items. "High-added" = above median.
Tokenisation: the LongCLIP tokenizer, the same one that produced the embeddings.

**Primary test.** The INTERACTION between split half and query type, on paired
correctness. Not the two within-half McNemar p-values. Reported as a 2x2 table of
discordant counts (stem-only-correct vs fused-only-correct) per half, tested with
Fisher's exact test. alpha = .05, one pre-registered test.

**Pre-registered prediction P2.1:** the fused-minus-stem gain is LARGER in the
high-added half than the low-added half.

**Explicitly secondary / exploratory, not claimable:**

- LongVideoBench `question_category`, including the T* (temporal) group.
- Stem length in tokens as an alternative continuous predictor.

**Why the interaction and not the subgroup pair.** On 2026-08-03 the question-type
regime claim was withdrawn for exactly this error: two subgroup tests, one
significant and one not, were read as an interaction. The direct test came back
p=.087 (600s) and p=.73 (3600s), with the sign reversing between bins, and none of
34 per-category tests survived Bonferroni. Same discipline applies here.

## What these two analyses cannot establish

Neither analysis can show the effect is real. Both are conditional on the +3.88 pt
observation, which remains uncorrected for multiple comparisons (4 arms; Bonferroni
alpha = .0125, or .025 across the two LongCLIP arms — it fails both) and unreplicated
outside LongVideoBench-600s at k=8. Replication (3600s, n=564) and the two control
arms (gold-removed, foreign-options) are separate GPU work and are NOT covered here.

---

# RESULTS (2026-08-22, same day, both CPU on Modal `p-khatri`)

Raw artifacts: `slm-lab-sig:/results/prereg/{residual_trace_fused,gain_split}.json`.
Code: `modal_siglip.py::residual_trace_fused`, `::gain_split`.

## Analysis 1 — H1 REFUTED, and refuted in the wrong direction

| contrast | stem | fused | delta | t | predicted | outcome |
|---|--:|--:|--:|--:|---|---|
| P1.1 cos_orig pick 1 | .2312 | .2204 | -.0108 | -9.80 | higher | **FAILS** |
| P1.2 resid_frac after 8 | .9682 | .9700 | +.0018 | +7.35 | lower | **FAILS** |
| P1.3 sum abs(drained) 1-8 | .4093 | .4325 | +.0231 | +12.67 | higher | passes, but see below |

Sanity check on the harness: stem cos_orig pick 1 = .2312 reproduces the "cos_orig
maxes ~0.23" figure from the 2026-07-19 trace.

**P1.3 passing is an artefact of a badly chosen statistic, not support for H1.**
Queries are L2-normalised, so explained mass is sum(drained^2) = 1 - resid_frac^2,
not sum(|drained|). Explained variance after 8 picks: **stem 6.26%, fused 5.91%**.
The fused query explains LESS. sum(|drained|) is larger only because the fused
query's mass is spread more evenly across picks rather than concentrated in pick 1.
P1.3 as pre-registered should be read as FAILED alongside P1.2.

**Conclusion.** The fused query is MORE orthogonal to the image subspace than the
stem, not less. The +3.88 pt accuracy gain is not a modality-gap effect. H1 is dead.

**Post-hoc observation, flagged as post-hoc and NOT tested here.** The trace shows a
different structure: the fused query's relevance decays far more slowly. `cos_resid`
under the stem is drained by pick ~5 (.2312 -> .0028 by pick 8, going NEGATIVE from
pick 13). Under the fused query it is still positive at pick 16 (.2204 -> .0054 at
pick 8, .0002 at 16), and per-pick drained mass is higher at every pick from 2 on.

Candidate mechanism: a stem is roughly one semantic direction; question + 5 options
is ~6. OMP on a multi-direction query keeps finding new relevant directions past the
point where a stem's residual is dead. That would explain why OMP gains more than
top-k (+3.88 vs +3.40, and 53.3% vs 41.9% of picks change), and why SigLIP — capped
at 64 tokens, so it sees one or two options — gains almost nothing. It is a
hypothesis generated from this data and cannot be tested on it.

## Analysis 2 — H2 NOT SUPPORTED, and the test was underpowered by construction

added_frac median .4666; high-added n=206, low-added n=206.
Stem length: median 50 tokens overall, 42 in the high-added half, 60 in the low.

| half | n | stem | fused | delta | rescued | broken | McNemar p |
|---|--:|--:|--:|--:|--:|--:|--:|
| high-added | 206 | .5437 | .6019 | +5.83 | 23 | 11 | .0576 |
| low-added | 206 | .7184 | .7379 | +1.94 | 10 | 6 | .4545 |

**Primary pre-registered test — interaction, Fisher exact on discordants:
p = .7568.** Direction of P2.1 is as predicted (+5.83 vs +1.94) but there is no
evidence for it. Continuous version agrees: Spearman(added_frac, per-item gain)
= **+0.0201**, n=412, i.e. nothing.

**Confound in the direction that did appear.** The high-added half has a much lower
stem baseline (.5437 vs .7184), so it is simply the harder half with more headroom.
The larger gain there is at least as consistent with headroom as with the mechanism.

**Power.** Only 50 discordant items exist across both halves (33 rescued, 17 broken
overall — the entire +3.88 pt effect is a 16-item margin). Simulating at the observed
rescue fractions (.676 high vs .625 low): power is .05 at the current n, .10 at 4x,
.15 at 8x, and **.30 at 16x (~6,600 items)**. The test could not have succeeded.
Pre-registering the split was right; pre-registering this test at this n was not.

## Effect on the decision

The stated go/no-go was "if both point the right way, spend the $4 on controls."
They did not. H1 is refuted, H2 is unsupported. The fused-query gain remains a real
but uncorrected, unreplicated, 16-item margin with no established mechanism.
