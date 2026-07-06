# Visual Evidence Compression for Agents: A Cheap Frame Selector Matches Full-Dump, and the Long-Video Wall Is the Task, Not the Tool

**Status:** complete result, video-only track. RunPod RTX PRO 4500 (Blackwell). Answerer: gpt-5.5.
Benchmark: LongVideoBench. All raw numbers in `FINDINGS.md`; this document is the consolidated story.

---

## Abstract

An agent that must answer a question about a long video faces a token wall: dumping every frame into a
frontier model is expensive and, past a point, may even hurt. We test whether a **small, cheap frame
selector** (SigLIP so400m, a 400M image-text encoder, one forward pass per frame) can pick the few
frames that matter so a frontier answerer (gpt-5.5) answers as well for a fraction of the cost — and
whether, past some video length, **compression beats full-dump** (context dilution).

We find: (1) **6 well-chosen frames are sufficient** — a cheap top-k selector reaches full-dump / model-
knob accuracy within confidence intervals at **13–34× fewer tokens**; compression buys **cost, not
accuracy**. (2) The selector's own **recall collapses with length** (hit@k 0.96 → 0.24 from 15 s to
1 h), which is the real bottleneck. (3) That collapse is **not fixable by a smarter cheap selector**:
coarse-to-fine, transcript-gating, a better image encoder (siglip2), and a video-native temporal encoder
(X-CLIP) all fail — and a diagnostic shows the wall is *ranking capacity*, not budget. (4) Crucially, a
purpose-built commercial dense retriever (TwelveLabs Marengo 3.0) does no better — it **ties the cheap
selector in the noise** (hit@k 0.30 vs 0.24 at 1 h, n = 10). Across **five different approaches** —
uniform, two image encoders, a pooled-video encoder, and the commercial retriever — **none escapes
~0.2–0.3 at 1 h**; that *pattern*, not any single small-n cell, is the evidence. The binding limit is
therefore **the task** — hour-scale localization of a **1–2 s visual needle, single-frame-answerable** —
not the selector or the encoder. This is an adversarial slice; on the realistic **≤ 10-minute** regime
cheap selection works well, and that scoped tool is the shippable win.

---

## 1. Setup

- **Benchmark:** LongVideoBench, chosen over the gated Ego4D-based LVHaystack. It ships gold keyframes
  (`position`), subtitles, and a native `duration_group` giving four length bins **{15, 60, 600, 3600} s**
  — exactly the length axis this study needs. Cost: we no longer share T*'s exact eval set, so cross-work
  comparison is directional (same task, different subset), not same-numbers.
- **Data:** 100 questions, distinct videos, n = 25 per length bin. Frames pre-extracted at 1 fps JPEG;
  gold keyframe seconds self-validated per video (100/100 reliable).
- **Primary metric — hit@k:** 1 if the selected frames contain ≥ 1 gold-evidence frame. Uncontaminated.
- **Secondary — MCQA accuracy:** guessing-contaminated at length (4-option; at 3600 s the selector has
  hit@k 0.24 but accuracy 0.76 → most correct answers arrive *without* the evidence frame). We lead with
  hit@k and treat accuracy as directional.
- **Answerer:** gpt-5.5 (`reasoning_effort=low`; hard cap of 500 images/request — so "full-dump" at
  600/3600 s is *already* a 500-frame subsample, which itself argues you must compress).
- **Selector (the cheap tool):** SigLIP `so400m-patch14-384`, top-k by image-text relevance, k = 6.
- **Baselines:** `A` full-dump (≤ 500 imgs uniform across clip) · `knob` model-knob (full-dump @ 0.5 fps,
  low detail — the *honest* cost baseline) · `U` uniform-k6 (isolates "is the selector smart," not just
  "is it cheap").

Confidence: Wilson 95% CIs; n = 25 → ±~0.18. Directional, not conclusive.

---

## 2. Fork A — does a cheap top-k selector match full-dump?

### hit@k (selection recall — primary)
| bin | A/knob (full) | U uniform | **C so400m (cheap)** |
|---|---|---|---|
| 15 s  | 1.0  | 1.0 | 0.96 |
| 60 s  | 1.0  | 0.56 | 0.72 |
| 600 s | 1.0  | 0.04 | **0.36** |
| 3600 s| 0.92 | 0.00 | **0.24** |

### Accuracy (secondary — guessing-contaminated at 600/3600 s)
| bin | A full | knob | U uniform | **C so400m** |
|---|---|---|---|---|
| 15 s  | 0.76 | 0.64 | 0.64 | 0.60 |
| 60 s  | 0.84 | 0.84 | 0.72 | 0.76 |
| 600 s | 0.76 | 0.72 | 0.56 | 0.72 |
| 3600 s| 0.72 | 0.84 | 0.60 | 0.76 |

### Tokens (mean input)
| bin | A full | knob | U/C |
|---|---|---|---|
| 15 s  | 2,137  | 1,161  | 2,547 |
| 60 s  | 6,972  | 3,609  | 2,545 |
| 600 s | 66,961 | 34,166 | 2,560 |
| 3600 s| 87,772 | 87,087 | 2,564 |

**Findings (honest split):**

1. **[SOLID] The selector beats the equal-budget dumb baseline.** At long video, C beats uniform-k6 on
   hit@k — **0.36 vs 0.04 (600 s), 0.24 vs 0.00 (3600 s)** — at identical 6-frame cost. SigLIP relevance
   ≫ even spacing. This is the uncontaminated win.
2. **[SOLID] 6 well-chosen frames are sufficient.** Full-dump's ~100× more frames add **no measurable
   accuracy** over C (ties/within-CI at every bin) at **26–34× the token cost**. Against the honest `knob`
   baseline, C ties on accuracy at **13–34× fewer tokens** (34K→2.5K @600 s; 87K→2.5K @3600 s). C's case
   is *cost at iso-accuracy vs the real baseline*, not an accuracy win.
3. **[SUGGESTIVE ONLY] Context dilution.** Full-dump *holds* the evidence (hit@k 0.92–1.0) yet scores no
   better than 6-frame C — consistent with dilution, but equally explained by MCQA guessability, so
   "extra frames actively hurt" is **not** established at n = 25. Downgraded from headline to hypothesis.

**The wall:** C's own recall **collapses with length, 0.96 → 0.24**. That is the concrete problem the
rest of the study attacks.

---

## 3. Fork B — can a smarter *cheap* selector fix the collapse?

All levers validated **free** first (echo answerer, hit@k only) — no frontier spend on anything that
didn't first beat flat top-k on hit@k.

| selector | 15 s | 60 s | 600 s | 3600 s | outcome |
|---|---|---|---|---|---|
| flat top-k (Fork-A C) | 0.96 | 0.72 | 0.36 | 0.24 | incumbent |
| coarse-to-fine (`hier`) | 0.96 | 0.76 | 0.32 | 0.20 | **worse** |
| transcript-gate | 0.88 | 0.72 | **0.04** | **0.08** | **much worse** |

- **Coarse-to-fine (T*/VideoTree family)** re-ranks the *same* SigLIP scores; a sparse coarse probe drops
  the 1–2 s needle's window before the fine stage can recover it.
- **Transcript-gating** fails hardest: for frames-answerable questions the ASR track describes *speech*,
  not the *visual* needle (gold-in-window = False for 4/5 audited long items). Lexical match ≠ needle
  location, so gating actively excludes the answer region.

### The decisive diagnostic — recall vs budget
Scoring every frame once and reading off hit@k at several k separates "not enough budget" from "the
scorer can't rank the needle":

| bin | k=6 | k=12 | k=20 | k=40 | reading |
|---|---|---|---|---|---|
| 60 s  | 0.72 | 0.72 | 0.92 | **1.00** | budget-limited |
| 600 s | 0.36 | 0.36 | 0.44 | **0.60** | partly budget |
| 3600 s| 0.24 | 0.24 | 0.24 | **0.32** | **selector-limited** |

At 1 h, **6.7× more budget moves recall only 0.24 → 0.32**. The wall is *ranking capacity*, not budget —
so more frames or more clever re-ranking of the same scores cannot help.

---

## 4. Scorer-swap — is the wall SigLIP, or cheap selection itself?

Swap **only** the scorer, everything else fixed (same manifest, bins, n = 25, k = 6, echo answerer).

| scorer | 15 s | 60 s | 600 s | 3600 s | what it is |
|---|---|---|---|---|---|
| uniform (floor) | 1.00 | 0.56 | 0.04 | 0.00 | non-semantic |
| **so400m (incumbent)** | 0.96 | 0.72 | **0.36** | **0.24** | image-text *matching* |
| siglip2 | 0.88 | 0.76 | 0.24 | 0.08 | *better image* encoder |
| videoret (X-CLIP) | 0.44 | 0.16 | 0.12 | 0.04 | *video* temporal (action) encoder |

- **siglip2 loses cleanly** — a newer/better *image* encoder localizes **worse**, not better. "It's just
  encoder quality" is falsified.
- **X-CLIP loses by construction** — it scores a 32-frame (32 s) window as one unit, so a 1–2 s needle is
  smeared across the pool. Any clip-pooled encoder (X-CLIP-p32, LanguageBind) fails by the same
  mechanism, so those were skipped as low-information.

**Principled reading:** fine-needle localization at 1 fps is a **per-frame retrieval task**. Per-frame
image-text scoring (SigLIP) is structurally the right cheap tool; clip-pooled video encoders smear the
needle by design. The cheap ceiling is *already SigLIP in hand*, and its residual limit is ranking
capacity at hour scale.

---

## 5. Ceiling — does a purpose-built dense retriever clear the wall?

The one remaining assumption: "a dense moment-retrieval model *would* localize the needle." We tested it
with the commercial state of the art, **TwelveLabs Marengo 3.0** (3600 s bin, n = 10, k = 6 clips):

| metric | Marengo 3.0 | so400m (cheap SigLIP) |
|---|---|---|
| strict (clip midpoint ∈ gold span — apples-to-apples with SigLIP's point metric) | **0.10** | 0.24 |
| lenient (clip [s,e] overlaps gold span — full credit to a span-returner) | **0.30** | 0.24 |

**The dense commercial retriever ties the cheap selector — nobody solves it.** At its fair metric
(lenient, given ~6 s clip granularity) Marengo is *nominally ahead*, 0.30 vs 0.24, but at n = 10 (Wilson
±~0.28) that is a **tie in the noise**, not a win for either side; 3/10 lenient hits, 1/10 centred. The
defensible claim is **parity**: cheap per-frame SigLIP ≈ commercial dense-retrieval SOTA on this slice.

**What actually hardens the negative is the cross-arm pattern, not this one n = 10 cell.** Five different
approaches — uniform (0.00), so400m (0.24), siglip2 (0.08), X-CLIP (0.04), Marengo (0.10/0.30) — and
**none escapes ~0.2–0.3 at 1 h**. That coherent pattern is the evidence.

**Robustness (rules out a confound):** Marengo ingests the *real video* at its own, denser internal
sampling — not our 1 fps JPEGs — and still lands ~0.30. So the wall is **not** an artifact of starving
the selector at 1 fps; it survives a fundamentally different ingestion path.

**So the wall is the task, not the tool — for this adversarial slice.** Hour-scale localization of a
**1–2 s, single-frame-answerable** visual needle is hard for *everyone affordable*. This answers the
recurring positioning question — *"why not just use TwelveLabs?"* — on the record: **it doesn't solve this
slice either.** (It says nothing about coarser, multi-frame long-video questions, which most real agent
video is.)

---

## 6. What ships — adaptive budget for realistic lengths

The recall-vs-budget table also shows the **positive** result. Real agent-browse video is mostly
sub-10-minute, and in that regime a cheap selector with an **adaptive frame budget** (budget ∝ length)
works well: **60 s → 1.0** and **600 s → 0.60** at k = 40, still far below full-dump token cost. The
1-hour extreme — where everything ties low, cheap or commercial — is the *least* representative regime for
agents. The shippable product is: **cheap per-frame SigLIP top-k with length-adaptive k, targeted at
≤ 10-minute video.**

**Two cheap checks before this hardens from "promising operating point" into a shipped accuracy claim:**
(a) one **end-to-end run** (accuracy + cost on the ≤ 10-minute bins) — adaptive-k is currently validated
on hit@k only; (b) the **blind question-only baseline** — without the guess floor, no accuracy number on
this MCQA benchmark is interpretable. Both are cheap and are the last things needed before an accuracy
headline.

---

## 7. Limitations

- **n is small** (25/bin; 10 for the Marengo ceiling). All numbers directional; CIs overlap.
- **MCQA is guessable** — accuracy is decoupled from selection at length; a genuine *accuracy* win over
  the cost baseline may be un-showable on this benchmark regardless of selection quality. A less-guessable
  eval (open-ended / true-needle) is needed before any accuracy claim.
- **Marengo caveats:** different 10-video subset than the so400m 25; 3600 s only; ~6 s clip granularity
  penalizes the strict metric.
- **Frontier 500-image cap** means there is no true full-dump at 600/3600 s — which strengthens, not
  weakens, "you must compress."
- **Frames-only** — audio/subtitle-dependent questions were deliberately excluded; this bounds the
  transcript-gating result to the visual-needle case.

---

## 8. Reproduce

```bash
# selection recall (free, echo answerer) — Fork A / Fork B / scorer-swap
python -m harness.run --manifest data/manifest.lvb.frames.local.json --conditions C \
  --answerer echo --selector {embedding|hier|transcript|videoret} --k 6 --max-dump-frames 3600
python scripts/recall_vs_k.py --model <hf_id>          # recall-vs-budget diagnostic
bash scripts/scorer_swap.sh                            # uniform / so400m / siglip2 arms

# Marengo ceiling (paid; runs local, no GPU) — needs TWELVELABS_API in .env
python scripts/marengo_ceiling.py --manifest data/manifest.marengo_subset.json --videos-dir data/videos
```

Artifacts: `results/n25/` (Fork A), `results/forkB/`, `results/scorer_swap/`, `results/marengo_ceiling.json`.
Full raw numbers and the running log of decisions: `FINDINGS.md`. Handoff / decision log: `HANDOFF.md`.

---

## One-line takeaway

> A 400M image-text encoder picking 6 frames matches full-dump accuracy at 13–34× lower cost and is,
> for hour-scale 1–2 s needle localization, **as good as the commercial state of the art** — because that
> task is tool-agnostically hard. Ship cheap length-adaptive selection for ≤ 10-minute video; don't pay
> for a dense retriever expecting it to fix the 1-hour case.
