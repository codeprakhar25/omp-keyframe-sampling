# Runbook — SigLIP scorer ablation (paper item 1)

**Goal.** The paper's central claim is that selector gains live in the *scorer*, not the
subset rule — but every result is LongCLIP. This ablation swaps the scorer and holds
everything else fixed, converting that claim from an inference into a measurement.

**Either outcome is publishable.** If the selector ranking changes under SigLIP, that is a
direct evaluation-practice finding: rankings are not scorer-invariant, so cross-paper
selector tables are not comparable. If the ranking holds, the confound is bounded and
Table 1 is hardened. There is no losing branch — do not pre-commit to a preferred result.

## What already exists (do not rebuild)

The pipeline is already implemented; this is not new engineering.

| step | tool | notes |
|---|---|---|
| image embeds | `scripts/dump_embeds.py` | emits `siglip (N,1152)` — GPU, **already run for much of LVB** |
| scorer resolution | `harness/embeds.py` | `KEY = {"sig": "siglip", "lc": "longclip"}` |
| stem-only text embeds | `scripts/score_from_embeds.py --scorer sig --text-embeds-out` | seconds, CPU-capable |
| picks | `scripts/gen_omp_picks.py --scorer sig`, `gen_topk.py`, `gen_aks_picks.py` | CPU, minutes |
| eval | lmms-eval task reading `$LVB_PICKS_*` | **the only GPU step** |

Cached embeds live on Azure at `slm-lab/results/embeds/` (1337 blobs, the complete
LongVideoBench val set). Coverage is **mixed**: some items carry `siglip`+`dinov2`, others
only `longclip`. Run the coverage gate below before planning any GPU spend.

## Measured coverage (2026-08-21, `modal_siglip.py::gate`)

| bin | total | siglip | longclip | verdict |
|---|---|---|---|---|
| 15s | 189 | 189 | 189 | both towers |
| 60s | 172 | 172 | 172 | both towers |
| **600s** | **412** | **412** | 6 (+412 in `embeds_lc/`) | **usable** |
| 3600s | 564 | **33** | 564 | needs GPU re-encode — out of scope |

Target bins are **600s + 60s**. 3600s would need 531 items re-encoded on a GPU and roughly
triples the experiment; skip it. 15s is excluded because uniform/top-k/OMP already tie there
(.7249), so a scorer swap has nothing to move.

**Pick divergence (index-based, `overlap_v2`).** SigLIP and LongCLIP select largely different
frames, well above chance — so the intervention is real, not cosmetic:

| bin | selector | shared /8 | random | vs chance |
|---|---|---|---|---|
| 600s | top-k | 2.62 | 0.19 | 14.1x |
| 600s | OMP | 1.32 | 0.19 | 7.1x |
| 60s | top-k | 5.08 | 2.22 | 2.3x |
| 60s | OMP | 3.58 | 2.22 | 1.6x |

OMP diverges most (only 16% shared at 600s): greedy decorrelation compounds small scorer
differences through the pick chain.

## Two traps that cost time here

**1. Compare frame INDICES, never timestamps.** Pick files carry different float precisions
(`283.5` in one file vs `283.5199890136719` in another for the same frame). A raw `set()`
intersection reports near-zero overlap for identical picks. `_to_idx()` resolves timestamps
against the embeds grid. The tell that something is wrong: within-scorer `|topk & omp|`
below 1.0 is *impossible*, since OMP's first pick is top-k's argmax by construction.

**2. Fan out with `spawn()`, not `map()` from a `@local_entrypoint`.** A local entrypoint runs
on the CLIENT. `--detach` keeps the App alive but not that driver, so a client network drop
killed six in-flight arms mid-run. `run_arm.spawn()` hands each arm to the server as an
independent FunctionCall. Killed arms wrote *zero* lines (lmms-eval flushes at the end), so
gate on unique qids via `audit_results` — a populated results dir proves nothing.

## Step 0 — coverage gate (CPU, free) — MANDATORY

```bash
python scripts/check_siglip_coverage.py --embeds-dir results/embeds_azure/flat --bin 600
```

Do not rent a GPU until this reports full coverage for the target bin. If coverage is
partial, re-encoding the missing items is a GPU step and must be budgeted. Per the
"Non-Empty Is Not Coverage" lesson: assert the join over every id the run will consume,
not that files merely exist.

## Step 1 — stem-only SigLIP text embeds (CPU, minutes)

```bash
PYTHONPATH=. python scripts/score_from_embeds.py \
    --scorer sig \
    --embeds-dir results/embeds_azure/flat \
    --text-embeds-out results/scores/text_sig_stem.npz \
    --out results/scores/scores_sig_600.jsonl \
    --bins 600
```

**Stem only.** Answer options must never enter selection — the fused string moved ~40% of
selected frames on LVB-600s and can reverse the ordering. `harness/text.py::question_stem`
is the single source of truth for that boundary.

## Step 2 — picks, one per selector (CPU, minutes)

```bash
for SEL in omp topk aks; do
  PYTHONPATH=. python scripts/gen_${SEL}_picks.py \
      --scorer sig \
      --scores results/scores/scores_sig_600.jsonl \
      --text-embeds results/scores/text_sig_stem.npz \
      --embeds-dir results/embeds_azure/flat \
      --k 8 --out results/picks_lmmseval/picks_${SEL}_sig_600_k8.json
done
```

Uniform sampling is scorer-independent — do **not** regenerate it.

## Step 3 — eval (GPU; the only paid step)

Run **all** arms in one environment, LongCLIP included. Do not compare SigLIP arms run
today against the banked LongCLIP numbers from the primary stack: that is exactly the
mistake that produced the item-3 confound (see `slm-lab/CORRECT_FINDINGS.md`,
2026-08-21). Same-environment pairing is what makes the ranking comparison valid.

Arms: `{longclip, siglip} x {topk, omp, aks}` + one uniform control = 7 runs x 412 items.

Hardware: Qwen3-VL-8B measured ~23.9 GB at k=8, so **48 GB (L40S / `g6e.xlarge`)**.
A10G/L4 at 24 GB will OOM. Gate every new pod with the =<90 s CUDA smoke before launching;
`rc==0` from lmms-eval is **not** success — gate on artifacts produced.

## Step 4 — analysis (CPU, free)

Report Spearman rank correlation between the two scorers' selector orderings, plus paired
McNemar per selector. The headline is whether the *ordering* survives the scorer swap.

## Cost

Steps 0-2 and 4 are CPU and free — run them on Azure (`slmlabsponsored`, same region as
the blob store, so egress is free). Only step 3 needs a rented GPU: ~8-12 GPU-hours,
roughly $8-22 on `g6e.xlarge` spot.


## Result (2026-08-21) — the selector ranking survives the scorer swap

All arms re-run together in ONE environment (Modal L40S), paired on identical ids.
Banked Table-1 numbers were deliberately NOT reused as the LongCLIP side; see the drift
note below for why that mattered.

**600s (n=412)**

| arm | acc | vs uniform | p |
|---|--:|--:|--:|
| uniform | .5534 | — | — |
| top-k LongCLIP | .6068 | +5.34 | .039 |
| OMP LongCLIP | .6311 | +7.77 | .0024 |
| top-k SigLIP | .6068 | +5.34 | .033 |
| OMP SigLIP | .6408 | +8.74 | .00026 |

Ranking identical under both scorers: `uniform < top-k < OMP`, despite the scorers sharing
only 1.32/8 frames for OMP (84% of the answerer's input differs). Scorer x selector
interaction: top-k +0.00 pt (p=1.0), OMP +0.97 pt (p=.73) — no detectable scorer effect.

**60s (n=172):** all null (best +3.49 pt, p=.31); under SigLIP top-k and OMP tie exactly.
Consistent with the scarcity story — median pool 36 frames, so k=8 covers 22% of the video.

**DO NOT claim "OMP prefers SigLIP".** OMP-vs-top-k is significant under SigLIP (+3.40,
p=.049) and not under LongCLIP (+2.43, p=.33), but the interaction is p=.73. Two subgroup
tests, one significant and one not, are NOT an interaction — the same error that killed the
question-type claim in the paper's SS5.5.

**Reading:** this is the LESS dramatic outcome. There is no "published selector tables are
incomparable" finding; the evidence points the other way. What it does establish is that
Table 1's ranking is not an artifact of choosing LongCLIP — which is the first question a
reviewer asks of a single-scorer study.

### top-k .6068 vs banked .6141 — CLOSED, environment drift

Not a pick-file bug. `topk_provenance` showed the two candidate pick files agree 412/412
(100%) and both match top-k recomputed from banked `scores_lc_600.jsonl` at 406/412 (the 6
are argsort tie-breaks, identical in both). Same picks, 253 vs 250 correct = **3 items**,
-0.73 pt — squarely inside the +0.36/-0.71 pt Modal-vs-primary drift measured in the item-3
work. Mechanism is known: greedy decoding is deterministic within an environment, but
padding plus non-associative float flips near-ties across hardware/torch builds.

Three independent drift readings, consistent: uniform 0/412, OMP 0/412, top-k 3/412.

### Is a @50% resolution pass needed for SigLIP?

Not on current evidence. Claim A/B are about pixels per frame, not frame provenance, and
the generality control already exists: uniform-8-half gave +0.10 pt (p=1.000), so
compression is free for uniform-chosen frames as well as OMP-chosen ones. A SigLIP arm
would be a third frame source on an axis already shown insensitive to frame source.
~$3-4 if a reviewer asks for the end-to-end recipe under a second scorer.

---

## Item 5 (2026-08-21) — prompt boundary: options in the scorer HELP, not hurt

Fused (question+options) vs stem-only picks, LVB-600s n=412, paired against stem arms
run in the SAME environment. `modal_siglip.py::picks_fused / run_fused / analyze_fused`.

### Pick divergence (index-based)

| scorer | selector | shared/8 | % frames changed |
|--|--|--:|--:|
| LongCLIP | top-k | 4.65 | **41.9%** <- the paper's "roughly 40%" |
| LongCLIP | OMP | 3.74 | **53.3%** |
| SigLIP | top-k | 5.58 | 30.2% |
| SigLIP | OMP | 5.20 | 35.0% |

The paper's ~40% describes **top-k**. OMP is MORE sensitive (53.3%), as its own docstring
predicts (each pick explains a residual component of the query, so query perturbation
compounds through the greedy chain). Ordering OMP>top-k replicates under both scorers.

### Accuracy — direction is OPPOSITE to the paper's framing

| scorer | selector | stem | fused | delta | p |
|--|--|--:|--:|--:|--:|
| LongCLIP | top-k | .6068 | .6408 | **+3.40** | .087 |
| LongCLIP | OMP | .6311 | **.6699** | **+3.88** | **.033** |
| SigLIP | top-k | .6068 | .6117 | +0.49 | .87 |
| SigLIP | OMP | .6408 | .6529 | +1.21 | .50 |

All four positive. Fused is a STRONGER configuration, not a contaminated one — the options
are a bag of visually-groundable nouns that enrich the query for a CLIP-family scorer.

### "Could reverse the ordering between rules" — NOT SUPPORTED

OMP beats top-k under every query and scorer; its lead is slightly LARGER under fused
(lc 2.43->2.91, sig 3.40->4.13). That clause in `arxiv.tex` L147 has no support and
should be removed.

### MANDATORY caveat — SigLIP truncation confound

SigLIP-so400m's text tower caps at 64 tokens. Measured on these 412 items:

| query | median tok | >64 tok | fraction SigLIP sees |
|--|--:|--:|--:|
| stem | 50 | 24% | 96% |
| fused | 102 | **93%** | **63%** |

SigLIP's fused arm saw ~63% of the string. Its near-null is **not** scorer robustness — it
is a weaker intervention. LongCLIP (248-token context) fits the median 102, though the
longest items (max 566) exceed even that. Do NOT present the lc-vs-sig gap as a scorer
property; it is confounded by context length.

### Consequence for the paper

Stem-only remains defensible (matches AKS's official implementation, which passes
question-only to its CLIP/BLIP branches) but it is **conservative, not protective**: it
costs ~3.9 pt for OMP under LongCLIP. The defensible claim is that the scorer's prompt is
an uncontrolled axis worth ~3.9 points, so papers using fused queries hold an unstated
advantage over papers using stems — which supports the paper's thesis more concretely than
the current unquantified assertion.
