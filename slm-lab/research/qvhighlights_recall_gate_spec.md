# QVHighlights — format + SigLIP recall gate (the fair rematch)

**Compiled:** Jul 9, 2026. **Why:** LVB killed compression on a *rigged* task (1–2 s needle in
visually-homogeneous 10-min video). QVHighlights is the honest rematch: **gold spans**, moments
**visually distinctive by design**, and it's a **recognized leaderboard TwelveLabs benchmarks Marengo
on**. Question: does cheap frame-selection recall the moment when the target actually looks different?

## Dataset format (NeurIPS 2021, moment retrieval + highlight detection)

- ~10k videos, each trimmed to **150 s**; segmented into **2-second clips** (75 clips/video).
- Splits: `train` / `val` / `test`. **test hides labels** → use **`val`** for our gate (labeled).
- Annotation = JSONL, one dict/row. Example:
  ```json
  {"qid": 8737,
   "query": "A family is playing basketball together on a green court outside.",
   "duration": 126,
   "vid": "bP5KfdFJzC4_660.0_810.0",
   "relevant_windows": [[0, 16]],
   "relevant_clip_ids": [0,1,2,3,4,5,6,7],
   "saliency_scores": [[4,1,1],[4,1,1],[4,2,1],...]}
  ```
  - `relevant_windows` = **gold moment span(s)** in seconds ← this is the target (LVB gave a frame; this
    gives a clip). Multiple windows possible.
  - `relevant_clip_ids` = the 2-s clip indices inside the windows (start at 0). Clip-level gold.
  - `saliency_scores` = 3 annotators × {0=Very Bad … 4=Very Good} per gold clip → per-clip relevance.
  - `vid` = `{ytid}_{start}_{end}`.

**Contrast with LVB (why this should un-rig it):** LVB target = 1–2 s / 600 s = **0.3%** of video,
near-duplicate of neighbors. QVH moment ≈ tens of seconds / 150 s = **~15%**, and it *depicts a
described event* ("basketball on a green court") that looks different from the rest. Bigger + distinct
= the exact condition a similarity selector needs. If SigLIP recall here ≫ LVB's floor (.30), that's
the finding: **compression works where the evidence is visually separable.**

### Gold-window distribution (val n=1550, measured Jul 9)

- duration ~120–150 s; segmented into 2-s clips.
- **moment span: median 30 s / 15 clips / 0.20 of video** (p10–p90 = 16–88 s; min 12, max 150).
- **# needles (≤6 s moment): 0%.** broad (≥50% of video): 12%. mean 1.81 windows/query.

**This reframes "peak" → "plateau."** QVH moments are broad, so the cosine curve won't spike — it should
sit **elevated across the ~30 s moment** vs background. So the confirmation metric is **separability**,
not sharp-peak detection:
- **clip-AP** = average precision ranking gold clips by cosine (the real number).
- **plateau contrast** = `mean(cos | gold clips) − mean(cos | background clips)`.
- **any-hit@k / clip-recall@k** — will be high (fat target); report but not discriminating.
- **span R@1@0.5 (proxy)** = top-scoring contiguous run as predicted window, tIoU vs gold.

**Spectrum placement (honest):** QVH = broad+distinct end; LVB long-bin = fine-needle+homogeneous end.
A QVH win does **not** revive the LVB needle — it establishes the *good* end. The science is locating
where on span-width × homogeneity compression stops working. Two datapoints (LVB fail, QVH ?) start the
curve.

## Standard metrics (for reference / later full eval)

- **Moment Retrieval:** R@1@0.5, R@1@0.7 (top-1 predicted span, tIoU≥thr); mAP@0.5, mAP@0.75, and
  **mAP avg** over IoU [0.5:0.05:0.95].
- **Highlight Detection:** mAP, HIT@1 (saliency).

## The cheap recall gate (first move — mostly no GPU)

Mirror the LVB union_ceiling gate, on spans. **Do NOT build a span predictor yet** — just test recall.

**Shortcut that makes it near-free:** the released `moment_detr_features.tar.gz` (8 GB) ships
**precomputed CLIP ViT-B/32 features per 2-s clip + CLIP text feature per query**. Use those directly
— cosine(query_text, each clip) → rank clips. No video download beyond features, no forward pass. (CLIP
≠ our SigLIP, but for a go/no-go gate it answers "does clip-text similarity recall the moment." Swap in
SigLIP on raw frames only if the gate is green.)

**Procedure (val split):**
1. For each query: cosine(query_feat, clip_feat[i]) for all 75 clips → rank.
2. Metrics vs `relevant_clip_ids` (gold clips):
   - **any-hit@k** = ≥1 gold clip in top-k (parallels LVB any-hit).
   - **clip-recall@k** = |top-k ∩ gold| / |gold|.
   - **span R@1@0.5 (proxy)** = take the top-scoring contiguous run of clips as the predicted window,
     tIoU vs `relevant_windows` ≥ 0.5. Rough (no boundary model) but comparable to the leaderboard shape.
3. Baselines: **uniform top-k** (no selector) and **saliency-oracle** (rank by gold saliency = ceiling).
4. Read k ∈ {1,3,5,10}. Headline compare: any-hit@5 / R@1@0.5 here **vs LVB floor .30**.

**Decision rule:**
- SigLIP/CLIP recall ≫ LVB floor and → saliency-oracle ⇒ **selection works on distinctive moments.**
  Compression thesis lives; coarse-to-fine + adaptive-k has a real home. Proceed to SigLIP-on-frames +
  boundary refinement.
- recall ≈ uniform ⇒ even distinctive moments aren't separable by frame similarity ⇒ the wall is
  deeper than LVB's homogeneity; reconsider the whole selection premise.

## Where "coarse-to-fine" + "make the retriever better" fit (staying in the cost thesis)

The user's core: not beat a SOTA retriever, but make the *effective* retrieval better **by allocating
compute**, not by training a bigger model. QVHighlights is the ideal testbed:
- **Coarse:** cheap wide scan — SigLIP over 2-s clips (or low-fps) across the whole 150 s. Finds the
  approximate window. (This is the recall gate above.)
- **Fine:** zoom only into the top coarse window(s) — higher fps / higher res — to sharpen the
  **boundary** (R@1@**0.7** needs tight edges that a 2-s coarse grid can't give).
- This is "making the retriever better" *without* a new model: the win is precision-per-dollar (cheap
  everywhere, expensive only where it matters) = the compression-buys-cost thesis, now on a task where
  the target is real. See [[coarse_to_fine_briefing]] (research/coarse_to_fine_briefing.md).

## Cost / feasibility
- Annotations: free, tiny (val ≈ 1550 queries, one JSONL in the moment_detr repo `data/`).
- Features: 8 GB tar (or a val-subset). Gate runs on CPU (cosine over 75×512 vectors × 1550 queries).
- No pod needed for the CLIP-feature gate. SigLIP-on-frames step (if green) needs the raw videos + GPU.

## Sources
- Repo/format: github.com/jayleicn/moment_detr, data/README.md (fields above).
- Paper: Lei et al., "Detecting Moments and Highlights in Videos via Natural Language Queries," NeurIPS 2021.
- Adoption: listed in Marengo 3.0's 14-benchmark retrieval composite (QVHighlight).
