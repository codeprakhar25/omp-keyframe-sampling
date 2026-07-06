# Scorer-swap experiment (Fork B close-out / "is it SigLIP or is it cheap selection?")

**Created:** Jul 5, 2026 · **For:** pod agent · **Cost:** free (echo + local GPU; no gpt-5.5)
**Depends on:** the Fork-A/B harness, LongVideoBench manifests (n=25/bin × {15,60,600,3600}s),
`gold_evidence_seconds`, `hit@k`, `recall_vs_k.py`.

## Why
Fork B concluded "the wall is small-model localization capacity → need a stronger (frontier) scorer."
But every arm used **SigLIP so400m — an image-text *similarity* model**, the wrong tool for **temporal
moment localization**. This experiment tests whether the wall is *SigLIP* or *cheap selection itself*,
by swapping ONLY the scorer for a model built for video/localization, everything else held fixed.

**Question:** does a cheap **video-native / localization** scorer lift long-video hit@k above so400m's
`.36 (600s) / .24 (3600s)` at equal k=6 budget?

## Encoders to test (arms)
All local, cheap (embedding pass, no frontier answerer):

| Arm | Model | Why | Setup |
|---|---|---|---|
| `so400m` (baseline) | `google/siglip-so400m-patch14-384` | Fork-A/B incumbent | done |
| `uniform` (floor) | — | dumb baseline | done |
| **`videoret` (PRIMARY)** | **LanguageBind_Video_FT** (`LanguageBind/LanguageBind_Video_FT`) | video-text shared space; scores short clips, not stills → real temporal localization | HF, moderate |
| `siglip2` (control) | `google/siglip2-so400m-patch14-384` | isolates "just a better *image* encoder" from "video-native" — if siglip2 alone closes the gap, it was never a video-modeling problem | drop-in |
| `internvideo2` (stretch) | InternVideo2 CLIP | SOTA video retrieval; try only if LanguageBind underwhelms | heavier repo |
| `marengo` (optional ceiling, PAID) | TwelveLabs Marengo via API | the purpose-built commercial retriever = "why not just use the incumbent" answer + an upper bound | API key, small $ |

Run `so400m` + `uniform` + `videoret` + `siglip2` first (all free). `internvideo2`/`marengo` only if the
free arms are promising or ambiguous.

## Equal-budget protocol (the part that keeps it honest)
1. **Identical everything except the scorer:** same manifests, same 4 length bins, same n=25/bin, same
   `gold_evidence_seconds`, same **hit@k tolerance** as Fork A/B (don't change the tolerance — it must be
   comparable to the `.24/.36` numbers).
2. **Equal DOWNSTREAM budget: every selector outputs exactly k=6 frames.** The scorer may *internally*
   look at more frames / short clips to compute scores (that's cheap scorer compute, the whole point) —
   but it must **return 6 frames**. This mirrors how so400m already works (scores all candidates → top-6).
   ⚠️ Do not let a video model leak >6 frames to the answerer, or the comparison is void.
3. **Primary metric: `hit@k` via echo** (free). No gpt-5.5 needed to answer the question.
4. **Also run `recall_vs_k`** (k=6/12/20/40) for the video-native arm. The tell: if it flips 3600s from
   **selector-limited** (so400m: .24→.32 at 6.7× budget) to **budget-limited** (recall climbs with k),
   the scorer *can* localize and the earlier wall was SigLIP.

### Clip-window scoring (for the video-native arm)
Frames are pre-extracted 1 fps JPEGs. A video encoder should score a **short window**, not a still:
- For each candidate center frame *t*, form a window of its ±w neighbors (e.g. w=2 → 5 frames ≈ ±2 s).
- Embed that window with the video tower; score vs the query text; assign the score to center frame *t*.
- Take top-6 center frames (dedup overlapping windows). Log the effective input resolution (avoid the
  SmolVLM downscale trap — if the model shrinks frames, note it).

## Success gate (decide the fork)
- **PASS (cheap selection alive):** `videoret` (or `siglip2`) lifts **3600s hit@k ≥ ~.40** (from .24) **or**
  `recall_vs_k` shows 3600s is now budget-limited (climbs clearly with k). → the wall was the *model
  choice*, not cheap selection. Next: scale n≥50, make the video-native scorer the product.
- **FAIL (wall confirmed):** all cheap arms stay ≈ so400m at 3600s and 3600s remains selector-limited.
  → cheap frame-scoring genuinely can't localize hour-scale needles. Honest pivot: ship the cheap
  selector as a **≤10-min tool** (adaptive-k) and drop the hour-scale claim, OR step to a stronger
  (frontier) scorer and explicitly leave the "cheap" thesis. Either way the negative is now *earned*.
- **`siglip2` beats `videoret`:** the gap was image-encoder quality, not video modeling — cheapest win.

## Implementation notes
- Add `--selector videoret` (+ `--selector-model <hf_id>`) to `build_selector`; reuse the top-k-output
  interface. Put the clip-window helper next to frame loading.
- Verify env compat first: LanguageBind / InternVideo2 may pin torch/transformers versions vs the S2
  stack (torch 2.11 / transformers 4.57). If it conflicts, use a separate venv for the scorer arm.
- Keep leakage discipline: scorer never sees `gold_evidence_seconds`.
- Everything is echo/free except the optional `marengo` API arm.

## Deliverable to report back
The hit@k table (uniform / so400m / videoret / siglip2 × 4 bins) + the `recall_vs_k` row for the
video-native arm at 3600s, and the PASS/FAIL call against the gate above.
