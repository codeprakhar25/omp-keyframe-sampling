# Region-recall gate — spec (Fork B redesign, go/no-go before building coarse-to-fine)

**Created:** Jul 7, 2026 · **For:** pod agent, run directly on `/workspace/slm-lab`
**Cost:** free (echo answerer, local GPU scoring; optional small text call for the grounding arm)
**Owner metric:** `region_recall@m` (new) — no frontier answerer needed.

---

## 0. The one question this answers

We know **frame-level** recall on long video is bad and *selector-limited* (SigLIP hit@k ≈ 0.36 @600s,
0.24 @3600s; k=6→40 barely moves it). Coarse-to-fine only makes sense if the scorer can at least find
the right **neighborhood** (chunk) even when it misses the exact frame.

> **Question:** if we split a video into M chunks and score each chunk by its best frame, does the
> **gold chunk land in the top-m chunks**? And is that **region-recall meaningfully higher than the
> frame-level hit@k**?

- **If yes** → the hard part (finding the region) works; coarse-to-fine + zoom + draft-verify is worth
  building. **PROCEED.**
- **If no** (region-recall ≈ frame-level) → the neighborhood itself is unfindable cheaply; no
  partition/zoom scheme can help. **STOP**, and the honest close is the ≤10-min adaptive-k tool.

**Second purpose — isolate scorer vs task.** Run the gate with **multiple scorers**. If *SigLIP* can't
find the region but a *grounding/detection* scorer can, the Fork-B wall was a scorer choice, not the
task. If **no** cheap scorer finds the region, it's the task.

---

## 1. Setup (reuse what exists)

- **Data:** the existing LongVideoBench manifest (`data/manifest.lvb.frames.local.json` or equivalent),
  1-fps pre-extracted JPEG frames, `gold_evidence_seconds` per item.
- **n = 100/bin (NON-NEGOTIABLE).** The go/no-go gate compares two rates whose gap decides everything;
  at n=25 the Wilson CI is ±~0.19 and the gate is self-contradictory. It's echo/free, so pay the sample.
  **Gold-validation caveat:** the original "100/100 reliable" was the 100-total set (25 items × 4 conds).
  Expanding to 100 *distinct items* per bin (~400 total) means the new ~300 items must be re-validated —
  keep only `gold_reliable == True` (spans resolved from native fps, `fetch_lvb_subset.py`). Region-recall
  measured against unvalidated gold is measured against noise. Report n_reliable/bin actually used.
- **Bins:** focus on the broken ones — **600s and 3600s** (label them by real length too; the "3600"
  bucket is actually ~15–52 min, median ~19 min, mean ~23.5 min). Include **60s** as a sanity check
  (should be near ceiling).
- **No answerer.** Everything is scoring + `region_recall`. Echo only. $0 — but **not free in wall-clock**:
  the `ground` arm at n=100 over ~1,000-frame videos is ~10⁵+ detector forward passes per bin. Hours on
  one GPU. Run it deliberately (batch, checkpoint per item), don't assume instant.

---

## 2. Definitions

- **Chunks:** split each video's frame list into `M` equal, contiguous, non-overlapping time chunks.
- **Gold chunk(s):** any chunk that contains ≥1 frame inside a `gold_evidence_seconds` span.
- **Chunk score:** aggregate the per-frame relevance scores within a chunk. Aggregations to compare:
  - `max` (peak — primary; matches "does the best frame in the chunk stand out"),
  - `mean` (baseline — the one that fails the peak-vs-average case),
  - `ucb` = `mean + c·std` (c=1.0; a cheap stand-in for FOCUS's Bernstein optimism).
- **region_recall@m:** fraction of items where at least one gold chunk is in the top-`m` chunks by
  chunk score. Report m ∈ {1, 2, 3}. **Lead with region_recall@1 at M=4** — that's the literal design
  (quadrant, single-chunk descent).
- **Random-chunk null (the real anchor):** expected region_recall@m if chunks were ranked randomly.
  Per item with `G` gold chunks out of `M`: `null@m = 1 − C(M−G, m)/C(M, m)`. Average over items.
  The gate is **significantly above this null AND above frame-level hit@k** — *not* an arbitrary absolute
  (drop the old 0.60). At n=100 the CIs can actually separate rate-from-null.
- **Probe budget `p`:** how many frames per chunk are actually scored (sparsity test):
  `p ∈ {1, 2, 4, all}`. `all` = dense (upper bound, no compute win); small `p` = FOCUS-style cheap
  probe (evenly-spaced frames within the chunk, deterministic). Tells us whether a *cheap sparse* coarse
  pass suffices or dense scoring is required.

---

## 3. Procedure (per scorer × bin)

1. For each item: load 1-fps frames, compute per-frame relevance to the question with the scorer.
   (Reuse `EmbeddingSelector._score` / `PESelector` batched scoring; grounding scorer below.)
2. Split into `M` chunks; for each chunk aggregate its frame scores under each `agg` and each probe
   budget `p` (for `p<all`, sample `p` frames uniformly within the chunk *before* aggregating).
3. Rank chunks; record whether a gold chunk is in top-{1,2,3}.
4. Aggregate `region_recall@m` over the reliable items, with **Wilson 95% CI**. Also compute the
   **random-chunk null@m** per item and average it (the anchor).
5. Sweep **`M ∈ {4, 8, 16, 32}`** (chunk granularity — M=4 is the quadrant design, added). Report the grid.

Also print, for reference, the **frame-level hit@k (k=6)** already known for that scorer/bin, so the
region-vs-frame gap is visible in one table.

---

## 4. Scorers to run (arms)

| arm | scorer | notes |
|---|---|---|
| `siglip` | SigLIP so400m (`EmbeddingSelector`) | baseline / incumbent |
| `pe` | PE-Core-L14-336 (`PESelector`) | best cheap encoder so far (won medium band) |
| `ground` | **question→target → open-vocab detector** | the untried lever — detection, not similarity |

**Grounding/detection arm (`ground`) — build this new:**
1. **Decompose the question into a target phrase** (+ optional cues). Either a cheap one-time text call
   (gpt-4.1, ~$0) per question, or a small local VLM. Cache to disk (`data/targets.json`) so it's paid
   once. Example: "What color is the bin the person opens?" → target `"bin"`.
2. **Per-frame score = max open-vocab detection confidence** for the target phrase, using
   **OWLv2** (`google/owlv2-base-patch16-ensemble`) or **GroundingDINO** (`IDEA-Research/grounding-dino-tiny`)
   — both HF, local, cheap, no frontier answerer.
3. Feed those per-frame scores through the same chunk aggregation. (This arm tests T\*'s actual
   mechanism — grounding on a decomposed target — under our honest metric.)

**`ground` is MANDATORY, not optional.** `siglip` and `pe` are the *same* cosine-similarity signal — if
similarity is the wall, both fail region-recall too and we learn nothing new. `ground` (detection on a
decomposed target) is the only arm testing a *different* signal — it's the whole point of the gate.
**Lock one detector** (default **GroundingDINO-tiny** `IDEA-Research/grounding-dino-tiny`; OWLv2 as the
alt) and **cache** the question→target decompositions to `data/targets.json`.

**Concrete-noun trap — log and segment (Fix 5).** Detection needs a groundable noun. Many LongVideoBench
questions are event/temporal ("what happens after…") with no object to detect — there `ground` fails for
a reason unrelated to detection power. So:
- Log **% of questions that decompose to a concrete groundable noun** (flag per item in `targets.json`).
- **Segment region_recall by `has_concrete_target` vs not.** Positive framing: if `ground` wins on
  object-questions but not event/temporal ones, that *is* the finding — it tells us *which query types*
  coarse-to-fine can serve. Don't average the two together into a muddy single number.

---

## 5. Decision gate

Anchor everything on the **random-chunk null**, not an absolute. For each scorer, at 600s and 3600s,
compare **region_recall@1 (M=4, agg=max)** and **region_recall@2** against both the **null@m** and the
scorer's **frame-level hit@6**:

- **PROCEED (build coarse-to-fine)** if **any** scorer's region_recall is **significantly above its
  random null** (Wilson CIs separate) **AND ≥ +0.20 above its frame-level hit@6** — it finds the
  neighborhood much more reliably than the exact frame and better than chance. Build with *that* scorer;
  add zoom (re-extract higher fps inside the winning chunk) + optional draft-verify.
- **STOP (coarse-to-fine is dead on this slice)** if **all** scorers sit at/near their random null, or
  within ~0.10 of their frame-level hit@6 (no neighborhood advantage). Close Fork B; ship the ≤10-min
  adaptive-k tool.
- **Recursion-compounding read (M=4).** Multi-level descent multiplies error: single-chunk descent needs
  **region_recall@1 ≳ 0.8** for 2 levels to survive (0.8²≈0.64). If @1 is weak but a small **beam (m=2)**
  clears it, that's the honest hedge — report the **@1-vs-@2 gap at M=4** as the "how aggressive can we
  descend" tradeoff, and the extra zoom cost m=2 buys.
- **Sparsity read:** if PROCEED, check the probe-budget `p` sweep — if small `p` (1–2) already gives most
  of the region-recall, the cheap coarse pass is justified; if only `p=all` works, there's no compute win
  over flat top-k (honesty check).
- **`ground` segmentation:** read the PROCEED/STOP call **separately** for `has_concrete_target` vs not.

Report all numbers with Wilson CIs at n=100; treat overlapping CIs as directional, not significant.

---

## 6. Implementation notes

- Add `region_recall(frame_scores, frames, item, M, agg, probe, ms)` + `random_null(frames, item, M, ms)`
  to `harness/metrics.py` (pure functions over precomputed per-frame float scores + `gold_evidence_seconds`).
- Add a small driver `scripts/region_recall.py`:
  `--manifest --scorer {siglip,pe,ground} --bins 60 600 3600 --M 4 8 16 32 --agg max mean ucb --probe 1 2 4 all`
  → writes `results/region_recall_<scorer>.json` and prints the grid + CIs. Scores each video ONCE, sweeps
  M/agg/probe/m in-memory (no re-scoring). Filters to `gold_reliable`.
- Reuse existing batched scoring (`_score`) so a whole video is scored once; aggregation is cheap and
  swept in-memory (no re-scoring per M/agg/p).
- `ground` arm: new `scorers`/selector entry; cache decomposed targets to `data/targets.json`.
- Leakage: scorer never sees `gold_evidence_seconds`. Gold is used only to *score* the result.
- Env: same S2 stack (torch 2.11+cu128). OWLv2/GroundingDINO are standard HF `AutoModelForZeroShotObjectDetection`.

---

## 7. Deliverable to report back

One table per bin: rows = scorers (`siglip`/`pe`/`ground`), columns = **random null@m**, `frame hit@6`,
`region_recall@1/2/3` (agg=max), across **M ∈ {4,8,16,32}**, plus the probe-budget sweep summarized, plus
the `ground` **concrete-noun segmentation** (has-target vs not) and the **% concrete targets**. Report
n_reliable/bin actually used. Then the **PROCEED / STOP** call per §5, with Wilson CIs at n=100 — and if
PROCEED, which scorer, whether a sparse probe suffices, and the @1-vs-@2 (M=4) descent-aggressiveness gap.

**Honest framing reminder:** this is scored on `region_recall` / `hit@k` only (uncontaminated). Do NOT
use MCQA accuracy here — we already showed it's guess-dominated on this benchmark and can't discriminate
selectors.
