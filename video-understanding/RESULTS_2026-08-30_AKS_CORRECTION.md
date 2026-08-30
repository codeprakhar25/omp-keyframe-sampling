# 2026-08-30 — AKS correction, benchmark by benchmark

Continues `RESULTS_2026-08-29_R11_R12_AKS.md`, which found that our `aks_indices`
port ended in a padding branch absent from the official `ncTimTang/AKS`
implementation. At `k < 2^all_depth` every leaf contributed `int(k/2^depth) = 0`
frames, so the pad filled the whole selection from global top-k. Fix = `all_depth=2`,
preserving the official two-frames-per-leaf ratio.

All arms: Qwen3-VL-8B-Instruct, subtitles OFF, `batch_size=1`, LongCLIP scorer,
question stem only, k=8, one AWS g6e.xlarge (L40S). Coverage-gated on unique qids;
no `.done` unless rc=0 AND the exact expected count.

## Infrastructure note (this was the blocker)

The Azure storage account `slmlab89c4a8` was **deleted** in the 2026-08-04 billing
migration; everything lives on **`slmlabsponsored`** (same container `slm-lab`,
same prefixes). A dead account presents as a DNS resolve failure, not a 404, which
is why this read as a network fault and as possible data loss on 2026-08-29.
`AZURE_BACKUP.md` and four other files carried the stale name and have been swept.
Bench videos are under `hf/videomme/` and `hf/lvbench/`, not the bare prefixes.

## Set 1 — LongVideoBench, COMPLETE

The 600 s bin was measured 2026-08-29; 15/60/3600 s were run today in one model
load from a merged picks file (qids verified disjoint).

| bin | n | corrected AKS |
|---|---:|---:|
| 15 s | 189 | .7196 |
| 60 s | 172 | .7267 |
| 600 s | 412 | .5922 |
| 3600 s | 564 | .5071 |
| **micro-avg (official val)** | **1337** | **.5916** (791/1337) |

Published Table 1 cell was **.6021**. Corrected = **.5916**, a **−1.05 pt** drop.

### Table 1, LongVideoBench column, after correction

| rule | acc | vs OMP |
|---|---:|---:|
| LDDR-select | .6320 | +0.97 |
| **OMP** | **.6223** | — |
| top-$k$ | .6028 | −1.95 |
| AKS (corrected) | **.5916** | **−3.07** (was −2.02) |
| FOCUS$^\star$ | .5819 | −4.04 |
| uniform | .5654 | −5.69 |

**The ordering does not change.** AKS was 4th and stays 4th, between top-$k$ and
FOCUS$^\star$. The correction moves the cell, not the ranking, and it widens OMP's
margin rather than narrowing it — so it does not favour us in the direction that
would matter for the headline claim.

## Paper edits this forces

1. `arxiv.tex:345` / `acl.tex:289` — AKS LongVideoBench cell
   `.6021 (−2.02)` → `.5916 (−3.07)`.
2. `tab:published` (`arxiv.tex:630` / `acl.tex:613`) — AKS$^\dagger$ LongVideoBench
   "ours" gain over uniform `+3.67` → `+2.62`. (Published stays `+5.01`; our
   reproduction gap on that cell widens from 1.34 to 2.39 pt.)
3. **`arxiv.tex:607` — a claim BREAKS and must be rewritten.** The sentence reads
   "on LongVideoBench and LVBench it [the room between subset-selection rules] is
   smaller than the distance from uniform sampling to plain top-$k$." Using the
   R14 comment's own convention of excluding FOCUS$^\star$:

   | | between-rule span | uniform→top-$k$ | verdict |
   |---|---:|---:|---|
   | old AKS | 2.99 | 3.74 | holds |
   | corrected AKS | **4.04** | 3.74 | **breaks** |

   R14 already recorded that this claim fails on Video-MME. With the corrected AKS
   it now fails on LongVideoBench too, leaving LVBench as the only benchmark where
   it holds. The sentence cannot survive as written and should not be patched by
   quietly dropping AKS from the span.

## Set 2 — LVBench, COMPLETE

n=1549/1549 (gate PASS). **Corrected AKS = .4287**; published cell was **.4280**.
Movement: **+0.07 pt** — a wash.

Cross-checked two ways: our gate's own tally and lmms-eval's `results.json`
(`lvbench_score = .42866`) agree.

### This is a genuine null, not a silent no-op

The near-identical accuracy is suspicious on its face, so the picks were compared
directly before the number was believed (mean Jaccard on rounded timestamps):

| comparison | mean Jaccard | identical items |
|---|---:|---:|
| buggy AKS vs top-$k$ | **.9968** | — |
| corrected AKS vs top-$k$ | .3479 | — |
| buggy vs corrected AKS | .3478 | **5 / 1549 (0.3%)** |

The first row re-confirms the bug on LVBench independently. The third says 99.7% of
items received a *different* frame set after the fix and the aggregate still landed
within 0.07 pt. Different frames, same accuracy.

### Table 1, LVBench column, after correction

| rule | acc | vs OMP |
|---|---:|---:|
| LDDR-select | .4693 | +0.58 |
| **OMP** | **.4635** | — |
| top-$k$ | .4319 | −3.16 |
| AKS (corrected) | **.4287** | **−3.48** (was −3.55) |
| FOCUS$^\star$ | .3983 | −6.52 |
| uniform | .3454 | −11.81 |

Ordering unchanged; AKS stays 4th. `tab:published` AKS$^\dagger$ LVBench "ours"
gain moves `+8.26` → `+8.33`. The item-3 claim **holds** on LVBench: between-rule
span 4.06 vs uniform→top-$k$ 8.65, as predicted.

## Set 3 — Video-MME, COMPLETE

n=900/900/900 across the three duration bins (gate PASS). Short .7311, medium
.5733, long .5133. Bins are equal-sized, so the mean is the micro-average.

**Corrected AKS = .6059**; published cell was **.5785**. Movement: **+2.74 pt UP**.

This is the opposite direction from LongVideoBench (−1.05) and LVBench (+0.07).
Verified against lmms-eval's own `results.json` (60.5926) and, as on LVBench,
against the picks themselves:

| bin | buggy==corrected | J(buggy, corrected) | J(buggy, top-$k$) | J(corrected, top-$k$) |
|---|---:|---:|---:|---:|
| short | 4 / 900 (0.44%) | .4349 | **.9985** | .4353 |
| medium | 5 / 900 (0.56%) | .3752 | **.9986** | .3757 |
| long | 5 / 900 (0.56%) | .3511 | **.9990** | .3511 |

The `J(buggy, top-k) ≈ .999` column re-confirms the original bug on Video-MME
independently of the LongVideoBench evidence.

### Paired tests (exact two-sided McNemar, n=2700)

| contrast | Δ | b | c | p |
|---|---:|---:|---:|---:|
| AKS(corrected) vs uniform | +4.22 | 359 | 245 | **4.0e−06** |
| AKS(corrected) vs top-$k$ | +3.56 | 238 | 142 | **9.7e−07** |
| AKS(corrected) vs AKS(buggy) | +2.74 | 223 | 149 | **1.5e−04** |
| OMP vs AKS(corrected) | +1.63 | 241 | 197 | .0398 |

### Table 1, Video-MME column, after correction

| rule | acc | vs OMP |
|---|---:|---:|
| **OMP** | **.6222** | — |
| LDDR-select | .6193 | −0.29 |
| AKS (corrected) | **.6059** | **−1.63** (was −4.37) |
| top-$k$ | .5704 | −5.18 |
| uniform | .5637 | −5.85 |
| FOCUS$^\star$ | .5578 | −6.44 |

**Ordering CHANGES here.** AKS moves from 4th to 3rd, above top-$k$ and uniform.
This is the only benchmark where the correction reorders Table 1.

## Consolidated result

| benchmark | n | published | corrected | Δ |
|---|---:|---:|---:|---:|
| LongVideoBench | 1337 | .6021 | **.5916** | −1.05 |
| Video-MME | 2700 | .5785 | **.6059** | **+2.74** |
| LVBench | 1549 | .4280 | **.4287** | +0.07 |

The correction is not a uniform shift. It hurts AKS on one benchmark, helps it
decisively on another, and does nothing on the third — while changing ~99.5% of
the selected frames everywhere.

## Paper edits required

### Mechanical

| # | location | change |
|---|---|---|
| 1 | Table 1 LVB cell | `.6021 (−2.02)` → `.5916 (−3.07)` |
| 2 | Table 1 V-MME cell | `.5785 (−4.37)` → `.6059 (−1.63)` |
| 3 | Table 1 LVBench cell | `.4280 (−3.55)` → `.4287 (−3.48)` |
| 4 | `tab:published` AKS$^\dagger$ LVB "ours" | `+3.67` → `+2.62` |
| 5 | `tab:published` AKS$^\dagger$ V-MME "ours" | `+1.48` → `+4.22` |
| 6 | `tab:published` AKS$^\dagger$ LVBench "ours" | `+8.26` → `+8.33` |

### Claims that no longer hold

**(a) `arxiv.tex:371` / `acl.tex:311` — BREAKS.** "On Video-MME, top-$k$, AKS, and
FOCUS$^\star$ do not separate detectably from uniform sampling, while OMP and
LDDR-select do." Corrected AKS separates from uniform at **p=4.0e−06**. AKS must
move to the other side of that sentence.

**(b) `arxiv.tex:607` — BREAKS on LongVideoBench.** "the room between
subset-selection rules ... on LongVideoBench and LVBench it is smaller than the
distance from uniform sampling to plain top-$k$."

| benchmark | span (excl. FOCUS$^\star$) | uniform→top-$k$ | old | new |
|---|---:|---:|---|---|
| LongVideoBench | 4.04 | 3.74 | holds | **breaks** |
| Video-MME | 5.18 | 0.67 | breaks | breaks |
| LVBench | 4.06 | 8.65 | holds | holds |

Now survives on LVBench only — one benchmark of three. It should be restated at
that scope, not repaired by dropping AKS from the span.

**(c) `arxiv.tex:647` / `acl.tex:688` — FULLY REVERSES.** "the rule whose native
encoder we did \emph{not} substitute (LDDR-select, $-0.40$ on Video-MME)
reproduces closest there but not on LongVideoBench, where AKS is closer despite a
substituted scorer."

| benchmark | AKS gap (old → new) | LDDR gap | closest, old → new |
|---|---:|---:|---|
| Video-MME | 2.81 → **0.07** | 0.40 | LDDR → **AKS** (flipped) |
| LongVideoBench | 1.34 → **2.39** | 2.24 | AKS → **LDDR** (flipped) |
| LVBench | 3.81 → 3.74 | 2.59 | LDDR → LDDR |

Both halves of the sentence invert. Note the corrected AKS reproduces the
published Video-MME gain to within **0.07 pt** (+4.22 ours vs +4.29 published) —
strong independent corroboration that `all_depth=2` is the right fix, since
nothing in our pipeline was tuned toward that number.

### Claims that survive

- **OMP still leads AKS, FOCUS$^\star$ and top-$k$ on all three benchmarks.** The
  narrowest margin is now Video-MME, OMP vs corrected AKS **+1.63, p=.0398** —
  still detectable, but much tighter than the +4.37 the buggy row implied.
- Table 1 ordering is unchanged on LongVideoBench and LVBench (AKS stays 4th).

## Provenance

Runs: `inputs/run_aks_lvb.sh`, `run_aks_lvbench.sh`, `run_aks_vmm.sh`;
gate `inputs/gate_gen.py`; paired tests `inputs/mcnemar_vmm.sh`,
`mcnemar_vmm2.sh` — all in `s3://prakhar-ml-research-044b36d8/`.
Results under `results_i2/aksd2_{lvb_1560_3600,lvbench,vmm}/`.

Gate discipline: every arm required rc=0 AND an exact unique-id count before
`.done`. It refused twice on parser bugs (LVBench stores a bare bool, Video-MME
uses `pred_answer` not `parsed_pred`) rather than reporting a false zero; both
were fixed by reading the actual row shape, and no eval was re-run for either.
