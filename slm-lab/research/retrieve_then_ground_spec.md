# Retrieve-then-Ground — Fork B, reopened as a *two-kind* pipeline (Jul 8)

**Thesis:** long-video evidence finding = **search** (find the ~60s region) + **localize** (pick the
exact evidence inside it). These are **two different *kinds* of model**. Beam failed because it did both
stages with the *same* per-frame cosine (SigLIP), which goes flat inside a single scene. Fix the second
stage: cheap cosine finds the region (region@2 ≈ .82–.84, already validated), then a **generative VLM**
reads the region and localizes the needle.

---

## 0. TL;DR — what to build & run

- **Stage 1 (search, KEEP):** SigLIP so400m dense per-frame scoring → top-K windows (±pad, NMS),
  K & pad **adaptive to video length**. This is the retriever we already have (region-recall confirmed).
- **Stage 2 (localize, NEW):** feed the **union of retrieved windows** to **`Qwen2.5-VL-3B-Instruct`**,
  which reads the frames jointly (with timestamps) and returns the best evidence frame(s).
- **Metric first, money later:**
  - **Run 1 (free, decisive):** `--answerer echo`, measure **hit@6** of the two-stage output vs SigLIP
    flat top-k and vs beam. This is the direct fix-test for the inert-beam result.
  - **Run 2 (paid, only if Run 1 wins):** end-to-end **accuracy + tokens** with a frontier answerer on
    the localized clip, vs flat-top-k and full-dump.
- **Bins:** 600s + 3600s (decision bins) + 60s (sanity). **n = 100/bin** (paper-grade, matches
  region_recall). Reuse the same manifest IDs as the region-recall run so numbers are comparable.

---

## 1. Why now (evidence, honest)

Two independent things point at the *same* design:

1. **Our own inert beam (Jul 7).** region@2 = .82–.84 (the right quarter is findable) but beam hit@6 ≈
   flat (.34/.20 @600/3600). The loss is entirely in the **last step** — ranking the exact frame inside
   the found region — and beam couldn't fix it because it re-ranks the *same* SigLIP scores. Within one
   scene the cosine is flat (region@ decays to null at fine M). **Narrowing the pool doesn't change which
   frame the same scorer ranks first.**

2. **ExtremeWhenBench (NAVER, arXiv 2606.12300, Jun 2026)** tested *this exact regime* (mean 75.7 min,
   9s median event) and found:
   - Monolithic Video-LLMs collapse; **plain CLIP frame-retrieval beats all of them** (0.269 vs Qwen 0.11).
   - **85% of failures are *search*, 11% localize** → the region-finding IS the wall, matching us.
   - Cheap cosine region recall = **81.6%** (top-3 ±1min) → **the same ~0.8 we independently measured.**
   - A **retrieve-then-ground hybrid** (CLIP top-K → *Video-LLM* localizes within the union) recovers
     **6.7×** over the monolith, using only ~6 min of VLM context. **Stage 2 is a VLM, not another CLIP.**

The paper's Stage-2 grounder is 9B. Our contribution is orthogonal to theirs: **how small can Stage 2
be** while holding localization — so the frontier answerer only ever sees the compressed clip. That cost
story is not in the paper.

---

## 2. The pipeline (two *kinds*)

| Stage | Job | Model *kind* | This spec | Why this kind |
|---|---|---|---|---|
| 1 · search | find the ~60s window(s) (region) | per-frame **cosine encoder** | SigLIP so400m (PE-Core-L14 alt) | one similarity number/frame — great for "which region", useless for "which of 60 look-alikes" |
| 2 · localize | read the window, pick exact evidence | **generative VLM** | Qwen2.5-VL-3B-Instruct | reads frames jointly + reasons — the signal cosine cannot give inside a scene |

**Multi-needle (e.g. 2:20 *and* 8:10) falls out for free:** Stage 1 is top-**K** windows + NMS (K>1), so
disjoint regions both survive into the Stage-2 union. Paper uses top-3 ±1min.

**Adaptive (your "dynamic by length"):** K and pad scale with duration — e.g. 60s→1 window, 600s→top-3
±30s, 3600s→top-3–5 ±60s. Cap the Stage-2 union at a fixed frame budget so cost is bounded.

---

## 3. Stage-2 model — decision & fallbacks

**Primary: `Qwen2.5-VL-3B-Instruct`.**
- Strongest small open VLM for multi-image + temporal; timestamp-aware; the ExtremeWhenBench pipeline
  itself uses the Qwen-VL family.
- Small = on-thesis (the compression claim). Fits the 4050 at 4-bit (`bitsandbytes`, ~3–4GB); use bf16 if
  a bigger GPU is available. If VRAM is tight, drop to `Qwen2-VL-2B-Instruct`.

**Cheap ablation: `SmolVLM2-2.2B`** — already wired (`SmolVLMSelector`, per-frame P(yes)). Weaker signal
(scores frames independently, no joint temporal read) but a useful floor: if even the joint Qwen read
doesn't beat this, the VLM-localize idea is weak.

**Ceiling reference (optional, paid): `Gemini-3.5-flash`** — the paper's best model on this exact task
(0.115 mIoU vs GPT-5.4 0.013; native auto-fps video). Use ONLY to learn the upper bound; not the thesis
model. (GPT-5.5 is already wired as an answerer if no Gemini key.)

---

## 4. What to build — `TwoStageVLMSelector`

New class in `harness/selectors.py`, registered in `run.build_selector` as `--selector twostage`.
Subclass `EmbeddingSelector` (reuse `._score` for Stage 1).

```python
class TwoStageVLMSelector(EmbeddingSelector):
    """Stage 1: SigLIP dense scores -> top-K windows (±pad, NMS) -> union candidate frames.
       Stage 2: a generative VLM reads the union (with timestamps) and returns the best k frames.
    Falls back to flat top-k if the VLM stage yields nothing. Stage-2 model is lazy-loaded."""
    name = "twostage"

    def __init__(self, model_id="google/siglip-so400m-patch14-384", device=None,
                 vlm_id="Qwen/Qwen2.5-VL-3B-Instruct", vlm_4bit=True,
                 top_windows=3, pad_sec=30.0, union_cap=48):
        ...
```

Stage 1 (region retrieval):
1. `scores = self._score([f.image for f in frames], question)` — dense, once (same as beam).
2. Group frames into windows around the top peaks: take the top score, expand ±`pad_sec`, NMS
   (suppress overlaps), repeat until `top_windows` disjoint windows. Union = frames in those windows,
   capped at `union_cap` (uniform-subsample within the union if over budget). **This is region retrieval,
   not final selection** — union may be 30–60 frames.

Stage 2 (VLM localize within union):
3. Build a single multi-image prompt: the union frames **in temporal order, each tagged with its
   timestamp** (`[t=142s]`), and ask the VLM to name the timestamp(s) that contain the visual evidence
   for the question. Parse timestamp(s) → map to nearest union frame(s) → return top-`k`.
4. Robustness: if parse fails / VLM returns nothing usable → fall back to SigLIP top-k **within the
   union** (never worse than beam-in-union).

Wiring:
- Add `twostage` to `build_selector` and to the `--selector` choices in `run.main`.
- `run.run_condition` already sets `selector._item = item` — use it for the **zoom** path (§9).
- `hit_at_k` / `recall_at_k` in `metrics.py` already score the returned frames against
  `gold_evidence_seconds` — no metric changes needed for Run 1.

---

## 5. Metrics & baselines

**Primary (Run 1, free): hit@6** — does the returned set contain ≥1 gold-span frame. This is the
uncontaminated number that beam couldn't move; it's the whole question.

**Secondary (Run 2, paid): end-to-end accuracy + mean_input_tokens** through a frontier answerer.

Baselines on the **same IDs**, all at k=6:
| arm | selector | what it isolates |
|---|---|---|
| flat top-k | `embedding` (SigLIP) | the incumbent Stage-1-only number (.34/.20) |
| beam | `beam` | narrow-pool-same-scorer (inert control) |
| **twostage-Qwen** | `twostage` | **the test: VLM localize in region** |
| twostage-SmolVLM | `twostage --vlm SmolVLM2-2.2B` | cheap-VLM floor |
| uniform | `uniform` | non-semantic floor |

---

## 6. Runs

**Run 1 — free hit@k (decisive).** Per bin (600, 3600, and 60 sanity):
```bash
# incumbent + inert control (re-confirm on these IDs)
python -m harness.run --manifest data/manifest.lvb_600.json --conditions C \
    --answerer echo --selector embedding --k 6 --label flat
python -m harness.run --manifest data/manifest.lvb_600.json --conditions C \
    --answerer echo --selector beam --k 6 --label beam

# the test
python -m harness.run --manifest data/manifest.lvb_600.json --conditions C \
    --answerer echo --selector twostage --k 6 --label twostage_qwen
```
Read `mean_hit_at_k` per arm from `results/summary.json`; compare with Wilson CI (`metrics.wilson_ci`).

**Run 2 — paid accuracy (only if Run 1 wins @600 or @3600).**
```bash
python -m harness.run --manifest data/manifest.lvb_600.json --conditions A C \
    --answerer openai --model gpt-5.5 --selector twostage --k 6 --judge \
    --judge-model gpt-4.1 --label twostage_e2e
```
Report accuracy + token_reduction_vs_A vs flat-top-k and full-dump.

---

## 7. Decision gate

Let `h_ts` = twostage hit@6, `h_flat` = SigLIP flat hit@6, at the decision bins.

- **PROCEED (Fork B revived):** `h_ts` beats `h_flat` at 600 **or** 3600 by a margin whose Wilson CIs
  **don't overlap** (n=100 → ~±0.09). → run Run 2; the two-stage design is the product.
- **DEEPER NEGATIVE (close for good):** `h_ts` ties `h_flat` at both decision bins. Then even a reasoning
  VLM can't localize the 1–2s needle inside a correctly-retrieved region → the wall is the *needle grain
  on this adversarial slice*, not the pipeline. Ship adaptive-k for ≤10-min; stop here.

Sanity: at 60s all arms should be high (~.7–.9); if twostage *loses* at 60s something is wired wrong
(bad timestamp parse / union too small).

---

## 8. Honest caveats / risks

1. **The decomposition itself is no longer novel** (ExtremeWhenBench, Jun 2026). Replicating it is not a
   contribution; **the small-Stage-2 cost frontier is.** Frame the result as "how cheap can localize be,"
   not "retrieve-then-ground works."
2. **Our slice is harder than the paper's** — 1–2s single-frame needle vs their 9s median. Stage-1 region
   recall transfers (we measured it); Stage-2 on a 2s needle is the real risk. A tie is a plausible,
   publishable negative.
3. **VLM parse fragility.** Timestamp extraction from free text is brittle; always keep the SigLIP-in-union
   fallback so twostage ≥ beam by construction.
4. **VRAM.** Qwen2.5-VL-3B may need 4-bit on the 4050; verify it loads before the full run (smoke on 3
   items). Fall back to Qwen2-VL-2B / SmolVLM2 if OOM.
5. **MCQA guess contamination (Run 2 only).** Accuracy may not move even if hit@6 does — that's why Run 1
   (hit@k) is the gate, and accuracy is secondary.

---

## 9. Zoom (optional, second-order)

Once a window is retrieved, re-decode it from the **source mp4** at >1fps (e.g. 4fps) so a 2s needle
becomes ~8 frames instead of ~2, then run Stage 2 on the denser frames. Needs the source video:
gate on `item.get("media_path")` via `selector._item` (we run on pre-extracted 1fps JPEGs, so this is a
separate code path). **Do NOT bundle zoom into the first run** — test VLM-localize vs cosine on the same
1fps frames first, so any lift is attributable to the model *kind*, not the extra frames. Gold spans are
median 2.0s, so zoom is expected to be a small effect; keep it as an ablation after the primary result.

---

## Artifacts
- Code: `harness/selectors.py::TwoStageVLMSelector`, `build_selector` wiring in `harness/run.py`.
- Results: `results/rtg_*` (mirror the region-recall naming); keep durable local copies.
- Compare against: `results/rr_*` (region-recall), Fork-B flat/beam numbers in `FINDINGS.md`.
