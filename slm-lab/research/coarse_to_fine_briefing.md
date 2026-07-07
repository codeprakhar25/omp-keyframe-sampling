# Coarse-to-fine (Fork B, reopened) — briefing for the design/framing agent

**Project:** slm-lab — visual evidence compression for agents.
**Date:** Jul 7, 2026. **Status:** region-recall gate PASSED on n=25/bin (directional); n=100/bin + grounding
arm running now to make it paper-grade. Beam selector built.

This doc is self-contained: what we're doing, why, the exact 25/bin results, what they mean, and what's
in flight. Read it as the current state of the coarse-to-fine thread before we harden the numbers.

---

## 1. The one-line thesis of the whole project

A **cheap selector** (small image-text model, SigLIP) picks the top-k frames of a long video, so a
**frontier answerer** (gpt-5.5) can answer a question about the video **cheaply** — seeing ~6 frames
instead of hundreds. The selector is a *filter*, not a smarter model. Compression buys **cost** (≈25–34×
fewer tokens at ≥600s video), at roughly iso-accuracy.

## 2. Where Fork B was, and why it was stuck

Fork A established the wall: on long video the cheap selector's **selection recall collapses** — it lands
the exact answer frame only **36% of the time at 600s, 24% at 3600s** (metric = `hit@k`, "is ≥1 gold
frame in the k I picked"). The whole game is lifting that.

The **first** Fork B tried to fix it two ways (hierarchical re-ranking, transcript-gating). Both were
**thin wrappers around the same SigLIP scores** and both **failed** — they re-ordered the same broken
signal. We closed that as a principled negative. (A scorer-swap study confirmed it: swapping SigLIP for
SigLIP2, X-CLIP, even TwelveLabs' commercial Marengo, none escaped ~0.2–0.3 recall at an hour — the
hour-scale *fine-needle* task is just hard for everyone cheap.)

## 3. The reframe that reopened Fork B (the key idea)

The old question was wrong. Coarse-to-fine descent **does not need the exact frame** at the first step.
It needs the right **chunk** — "the answer is somewhere in the 3rd quarter" — and *then* it zooms in.

So we invented and measured a new metric:

> **region_recall@m:** split a video into `M` equal time-chunks; score each chunk by its **best** frame;
> is a **gold chunk** in the **top-m** chunks? Anchored against a **random-chunk null** (the recall you'd
> get ranking chunks by coin-flip: `1 − C(M−G,m)/C(M,m)`).

Plain version: *even if the selector can't pin the exact frame, can it at least point at the right region?*
If yes, coarse-to-fine is worth building. If region-recall is no better than the exact-frame recall (or no
better than chance), no partition/zoom scheme can help and we stop.

This is measured **free** (no frontier answerer, just local GPU scoring + the hit/region metrics — $0).

## 4. The result on n=25/bin (SigLIP so400m, M=4 = quadrant, best-frame aggregation, dense scoring)

Wilson 95% CIs in brackets. "hit@6" = old exact-frame recall (the wall). "null" = random-chunk baseline.

| bin    | frame hit@6 | region@1 (top quarter) | region@2 (top-2 of 4) | random null@2 |
|--------|-------------|------------------------|-----------------------|---------------|
| 60s    | 0.72        | 0.68                   | 0.92                  | 0.67          |
| **600s**   | **0.36**    | 0.64 [.45–.80]         | **0.84 [.65–.94]**    | 0.57          |
| **3600s**  | **0.24**    | 0.56 [.37–.73]         | **0.88 [.70–.96]**    | 0.53          |

**What this says, in one sentence:** where the selector finds the exact frame only 24–36% of the time, it
finds the right **quarter (top-2 of 4)** about **84–88%** of the time — and that clears the random-chunk
null with separated confidence intervals. **The neighborhood is findable even when the frame isn't. The
premise of coarse-to-fine holds — on plain cheap SigLIP, before any fancier scorer.**

### Four design decisions the data locks (not opinions — measured)

1. **M=4 is the sweet spot.** As chunks get finer (M = 4 → 8 → 16 → 32), region@1 decays 0.56 → 0.36 →
   0.28 → 0.20, sinking toward the null. Big, few chunks win. So recurse with a **small branching factor
   per level**, never one fine partition.
2. **Keep 2 of 4 (beam), not the single best (greedy).** region@1 (0.56–0.64) is too low to survive
   multi-level descent (0.56² ≈ 0.31 over two levels — barely beats the wall). region@2 (0.84–0.88)
   survives two levels (0.88² ≈ 0.77). The jump from @1 to @2 is huge → **beam width 2**.
3. **Best-frame (max) aggregation, not average.** max (0.84/0.88) far beats mean (0.68/0.64). One hot
   frame should light up its chunk.
4. **Score densely.** A cheap sparse probe leaves recall on the table (region@2 at 600s: 0.52 with 1
   frame/chunk → 0.84 scoring all). That's fine: scoring every 1fps frame with SigLIP costs *seconds*;
   the compression payoff was never in the selector — it's the **answerer** seeing 6 frames not 500.

## 4b. The method number (Jul 7) — beam ≈ flat, trending NEGATIVE

§4 validated the *premise*. This tests the *method*. `BeamCoarseToFineSelector` (beam-2·M=4·max·dense),
**echo answerer — no gpt-5.5, pure hit@6 selection recall, $0**, n=25/bin. Mechanism as run: score all 1fps
frames once, recurse (split window into 4, keep best 2, recurse into *both*, … depth 1–4 levels by length)
down to a ~24-frame pool, then flat SigLIP top-6. No zoom.

| bin | beam hit@6 | flat hit@6 | Δ |
|---|---|---|---|
| 15s | .96 | .96 | 0 |
| 60s | .68 | .72 | −.04 |
| 600s | .40 | .36 | +.04 |
| 3600s | .24 | .24 | 0 |

**Beam ≈ flat** — every Δ inside n=25 noise, exact tie at the hard bin, beam even loses at 60s. The
re-ranking mechanism is **inert**: pruning distant distractors didn't lift hit@k, so the frames outranking
the needle are near/similar look-alikes *inside its own region*, not distant — beam removes cold frames, not
competitors. recall_vs_k's ranking-capacity wall **survives pool-narrowing**. Combined with the 2s-needle
fact (zoom dead), the method has no lever left except a **different scorer** (the grounding arm). If
grounding's hit@6 also ≈ SigLIP, Fork B closes as a clean **principled negative** and the honest ship is the
adaptive-k ≤10-min tool. This is a legitimate, publishable negative — not a failure — but it is not the win.

## 5. What we built (code)

- **Metric + driver:** `region_recall` / `random_null` / `wilson_ci` in `harness/metrics.py`; sweep driver
  `scripts/region_recall.py` (scorers: siglip / pe / **ground**; sweeps M, aggregation, probe-budget, m).
  Spec: `research/region_recall_spec.md`.
- **The selector:** `BeamCoarseToFineSelector` in `harness/selectors.py` (`--selector beam`). Implements
  the locked design: **beam-2 · M=4 · max · dense**, one score pass, recursive descent, then global top-k
  within the surviving windows. Unit-tested (needle survives descent; beam keeps two regions on a two-peak
  video). Currently echo-validated end-to-end on the pod.

## 6. Honest caveats (state these — do NOT overclaim)

- **n=25/bin, SigLIP only — directional.** At n=25 the confidence intervals are ±~0.19, so the gaps are
  suggestive, not settled. The n=100/bin rerun (in progress) is what makes it paper-grade.
- **Beam *without zoom* CAN beat flat top-k — but only by re-ranking.** (Correcting an earlier claim that
  it can't "by construction".) Beam takes top-k *within the surviving windows*, a subset of all frames;
  pruning frames can only improve or hold the needle's rank. So beam beats flat exactly when the false
  positives outranking the needle sit in *distant* chunks that descent prunes. If they cluster near the
  needle, beam ≈ flat. This is the actual go/no-go, and it is an open empirical question — recall_vs_k
  showed the wall is ranking capacity, but it varied *k* (budget on the full pool), NOT pool size, so it
  does not settle whether pruning distant distractors helps. The beam-echo hit@k number tests it directly.
- **Zoom is nearly a red herring on THIS benchmark.** Gold spans here are median/mean 2.0s (158/161 ≤2s),
  so the needle is already ~2 frames at 1fps and rarely missed by undersampling. Denser resampling adds
  little. The lever on this data is *re-ranking via descent*, not resampling. (Zoom would matter on
  sub-second needles or coarser benchmarks — not here.) So the method's whole bet reduces to: does beam
  descent lift the already-captured needle's rank above flat top-k.
- **This is T\*'s mechanism.** T\* (Stanford, LV-Haystack) already does temporal+spatial adaptive zoom and
  reports better public numbers (e.g. GPT-4o 47.1→51.9% at 8 frames on LongVideoBench XL) — but with a
  **stronger question-guided detector scorer** on gated Ego4D data. Our novelty is **not** the tree; it's
  the **cheap-scorer cost-quality frontier** — how close a cheap per-frame scorer + zoom gets to T\*, at a
  fraction of the cost. Frame it as a frontier result, not "we beat SOTA."
- **The "3600s bin" is not one-hour videos** — median ~19 min, mean ~23.5 min, only one video >50 min.
  Claims are about *long* video, not specifically an hour.

## 7. What's running right now (the hardening chain)

1. **Fetching ~300 more videos** → 100 distinct videos/bin (pools confirmed ≥100/bin: 15s 107, 60s 102,
   600s 264, 3600s 342). Fast aria2 stream of the LongVideoBench tar.
2. Auto-chained on fetch completion: **validate gold** (resolve each video's gold timestamp from native
   fps, keep only `gold_reliable`) → **extract frames** (1fps JPEGs) → **rerun region_recall at n=100**.
3. **Then the grounding arm:** GroundingDINO-tiny scoring each frame by detection confidence on a
   question-derived target phrase. This is the decisive scorer-vs-task test — SigLIP and PE are the *same*
   cosine-similarity signal; grounding is a *different* signal. We'll log the % of questions that yield a
   concrete groundable noun and segment region-recall by it (detection can only help object-questions;
   event/temporal questions are a known limit, and that segmentation is itself a finding).

## 8. What we'd want from the design/framing agent

- A skeptic pass on the **"region ≠ frame, so coarse-to-fine is worth building"** narrative and the
  **beam-vs-greedy** argument — the same service the earlier review did for the Marengo result (it
  corrected "cheap beats SOTA" → "parity, nobody solves it").
- A clean **results figure**: 3 bins × [frame hit@6, region@1, region@2, random null] — the story is
  visual and one bar chart carries it. (Best done once n=100 numbers are final.)

Raw numbers: `results/rr_siglip_all.json` (n=25). Full findings section: `FINDINGS.md` → "Region-recall
gate — Fork B REOPENED, PROCEED (Jul 7, 2026)".
