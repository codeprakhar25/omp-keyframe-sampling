# Findings — visual evidence compression for agents

**As of:** Jul 4, 2026 · RunPod RTX PRO 4500 Blackwell · **current result = Phase 0 / Fork A on
LongVideoBench (bottom of this doc)**. The current answerer is **gpt-5.5**; earlier sections used
gpt-4.1.
**TL;DR (current, video):** On real long video, a cheap so400m top-k selector **beats the equal-budget
uniform baseline on selection recall** (hit@k .36 vs .04 @600s, .24 vs 0 @3600s) and reaches full-dump/
model-knob accuracy **within CI at 13–34× fewer tokens** — i.e. 6 well-chosen frames are *sufficient*,
compression buys **cost, not accuracy**. **Open problem:** the selector's own recall collapses with
length (1.0 → .24), and the MCQA accuracy signal is guessing-contaminated at length. Read **hit@k as
primary**. Full detail: "Phase 0 — Fork A on LongVideoBench" below.

> **Sections 1–4 below are the EARLIER image/GUI track — SUPERSEDED.** They conclude "model quality is
> the binding axis" on a stitched-GUI slice that we later dropped (video-only scope). Kept for history;
> do not carry their absolute numbers or takeaway forward. Jump to **Phase 0 — Fork A** for the live result.

---

## 1. Experiments & numbers  *(earlier GUI/image track — superseded)*

### Toy needle (synthetic, 12×40, distinct coloured needle)
| Cond | acc | hit/recall@k | tokens | reduction |
|---|---|---|---|---|
| A full-dump | 1.0 | 1.0 | 10236 | 1.0x |
| C SigLIP | 1.0 | 1.0 | 1566 | 6.5x |

PROCEED — but **misleading**: the synthetic needle is trivially separable. Motivated real data.

### Real screen needle — stitched GUI (8 items, ~80 frames each, 1 target clip among same-app distractors)
Built by `scripts/stitch_needle.py`: concatenate 1fps frames of 6 real GUI-World clips; question
targets one clip → its frame range is the gold needle, the other 5 are hard distractors. Metric
`hit@k` (≥1 gold frame in top-k) — the right diagnostic when the answer persists across the span.

| Selector | hit@k | acc | reduction |
|---|---|---|---|
| C SigLIP (embedding) | **0.50** | 0.50 | 13.5x |
| C SmolVLM-500M (generative) | **0.375** | — (echo) | 13.5x |

**STOP.** Both selectors ~coin-flip (0.375 vs 0.50 within n=8 noise). Generative reasoning at 500M
does **not** beat embedding on near-identical screens.

### Real video needle — whispr.mp4 (103s talking-head, needle = thumbs-up gesture @frame 100)
| Cond | acc | hit@k | pred | tokens |
|---|---|---|---|---|
| A full-dump (104 fr) | **0.0** | 1.0 | "wave" ❌ | 44232 |
| C SigLIP (6 fr) | **1.0** | 1.0 | "thumbs-up" ✅ | 2582 (17x) |
| C SmolVLM (6 fr) | — | 1.0 | — (echo) | — |

Both selectors find the distinct needle. **Notably, compression BEAT full-dump on accuracy**:
A saw all 104 frames and answered "wave" (context dilution → latched on the wrong gesture); C's 6
focused frames → correct. Full-dump is *not* always the accuracy ceiling. (n=1, question mildly
ambiguous if the person also waved.)

### S2: resolution-vs-model test (Jul 2, stitched set, echo answerer, free)

Question from S1: is near-identical-screen failure a *resolution* problem or a *model* problem?
Six arms on the same 8 stitched items, k=6 (`results/res_test/`). New env: torch 2.11+cu128,
transformers 4.57.6 (5.x drops SmolVLM2; needs torchvision; so400m tokenizer needs
sentencepiece+protobuf).

| Arm | Selector | proc res | hit@k | recall@k |
|---|---|---|---|---|
| 1 | SmolVLM2-500M | 512 (default) | 0.25 | 0.044 |
| 2 | SmolVLM2-500M | 1920 | 0.375 | 0.039 |
| 3 | SmolVLM2-2.2B | 1536 (default) | 0.625 | 0.113 |
| 4 | SmolVLM2-2.2B | 1920 | 0.625 | 0.166 |
| 5 | **SigLIP-so400m-patch14-384** | 384 | **0.875** | **0.259** |
| 6 | SigLIP-base-patch16-224 | 224 | 0.50 | 0.088 |

(Arm 6 = in-env re-run of the S1 baseline, reproduces 0.50 exactly. Arm 1 re-run gave 0.25 vs
S1's 0.375 — one item flipped under the new torch/transformers stack; treat 500M numbers as ±1 item.)

Reading the axes:
- **Resolution** (1→2, 3→4): +1 item for 500M, none for 2.2B (recall depth up 0.113→0.166 only).
  Real but weak lever.
- **Model scale, generative** (2→4 at same res): 0.375 → 0.625. Stronger lever.
- **Embedding quality** (6→5): 0.50 → **0.875 (7/8, only stitch-005 missed)** — the biggest jump,
  from a 400M encoder at just 384px, single forward pass per frame (~100x cheaper than generative
  scoring).

**Verdict: model problem, not resolution problem — and the S1 conclusion "embedding-vs-generative
both fail" was really "weak checkpoints fail".** A modern shape-optimized SigLIP at 384px separates
near-identical GUI screens fine (n=8 caveat stands). Practical selector for the tool track =
siglip-so400m-patch14-384: best hit@k AND cheapest inference of everything tested.

stitch-005 (the one so400m miss): gold's best rank = 13 of 74 (k=6 misses it); top-6 all come from
one *distractor* Todoist clip (frames 65–72) that matches "expand to view existing projects" better
than the true needle. Genuine hard-distractor confusion, not noise.

### S2b: scale-out + paid end-to-end confirm (Jul 2)

**Website category, 12 new stitched items** (GUI-World `website.jsonl`, 30 clips, same
stitch_needle.py recipe; `results/res_test/web_*`), echo answerer:

| Selector | hit@k (web n=12) | recall@k | hit@k combined n=20 |
|---|---|---|---|
| SigLIP-so400m-384 | **0.917** (11/12) | 0.442 | **0.90** (18/20) |
| SigLIP-base-224 | 0.75 (9/12) | 0.285 | 0.65 (13/20) |

Website screens are more distinct than same-app software screens (base jumps 0.50→0.75), but
so400m still adds +2 items. so400m's 0.875 was not an n=8 fluke.

**Paid end-to-end (software stitched n=8, answerer gpt-4.1, LLM judge,
`results/res_test/paid_confirm/`):**

| Cond | acc | mean input tok | reduction | hit@k |
|---|---|---|---|---|
| A full-dump (~83 fr) | 0.75 | 35,230 | 1x | 1.0 |
| C so400m k=6 | **0.75** | **2,612** | **13.5x** | 0.875 |

**Equal accuracy at 13.5x fewer tokens on the hardest slice — harness GO/NO-GO: PROCEED.**
Per-item: A misses 003+004, C misses 000+003 (not the same items). Two caveats worth keeping
honest: C answers stitch-005 correctly *despite* hit@k=0 (gold answer "B" is guessable —
GUI-World B-bias; accuracy overstates selection quality there), and C beats A on 004 —
context-dilution again, matching the whispr result.

## 2. Synthesis  *(earlier GUI/image track — SUPERSEDED; see Phase 0 — Fork A at bottom)*

The axis that decides success is **needle distinctiveness + frame resolution**, NOT embedding-vs-
generative:

- **Distinct needle** (a gesture in a video) → both selectors win, and compression can *improve*
  accuracy by removing distracting frames.
- **Near-identical distractors** (which Todoist screen shows X) → both cheap selectors fail ~50%.
  The job needs reading fine UI detail; SmolVLM-500M downscales frames (512px longest edge) → UI
  text becomes unreadable → it can't tell the screens apart.

**H2 (an off-the-shelf small selector suffices) FAILS on the hard GUI slice — for both families.**

## 3. Honest caveats

- n=8 (stitched) and n=1 (whispr): directional, not conclusive; wide error bars.
- GUI-World MCQA gold is position-biased (~8/10 answer "B") → trust `hit@k`, not absolute accuracy.
- SmolVLM-500M is tiny and downscales; a bigger / higher-res VLM is untested.
- Stitched needles are real frames but synthetic *layout* (concatenation).

## 4. Open lever (next)

~~Resolution-vs-model test~~ — **done, see S2**: model quality wins, so400m-384 = 0.875 hit@k.

Remaining levers for the tool track:
- Scale n past 8 + GUI-World `website` category (dilute B-bias); is so400m's 0.875 stable?
- stitch-005 failure analysis: what does so400m still miss?
- End-to-end paid confirm: so400m selector + gpt-4.1 answerer, accuracy + token reduction.
- Package as MCP evidence-compression tool (paper gate failed 2026-07-02 — build track only).

## 5. Repro

```bash
cd /workspace/slm-lab && set -a && . ./.env && set +a   # HF_HOME=/workspace/hf (persistent volume)
# stitched needle, free hit@k:
python3 -m harness.run --manifest data/manifest.s1_stitch.json --conditions C \
    --answerer echo --selector {embedding|smolvlm} --k 6 --max-dump-frames 100
# real video, paid accuracy:
python3 -m harness.run --manifest data/manifest.real.json --conditions A C \
    --answerer openai --model gpt-4.1 --selector embedding --k 6 --max-dump-frames 110 --judge
```

---

# Phase 0 — Fork A on **LongVideoBench** (Jul 4, 2026)

**Setup:** answerer **gpt-5.5** (reasoning, `detail:low`/`high`), judge gpt-4.1, selector k=6.
Data = **LongVideoBench** (see decision note below — NOT LVHaystack), 100 questions, 25 per
`duration_group` bin {15, 60, 600, 3600}s, one question per distinct video, frames-answerable only
(dropped `T` subtitle-dependent categories). Frames pre-extracted at 1 fps to JPEG; gold keyframe
`position` → `gold_evidence_seconds` via each video's native fps. Wilson 95% CI (n=25 → ±~0.18).

**Four operating points on the accuracy-vs-cost curve:**
`A` full-dump (≤500 imgs, uniform across whole clip) · `knob` model-knob (full-dump @0.5fps, low
detail) · `U` uniform-k6 · `C` so400m top-k6.

**Read hit@k as primary.** Accuracy at long bins is guessing-contaminated (4-option MCQA): at 3600s C
has hit@k .24 but accuracy .76 → most correct answers arrive **without** the evidence frame, so
accuracy is largely decoupled from selection quality at length. Lead with the uncontaminated number.

### hit@k (selection recall — PRIMARY metric)
| bin | A/knob (full) | U unif | **C so400m** |
|---|---|---|---|
| 15s  | 1.0  | 1.0 | .96 |
| 60s  | 1.0  | .56 | .72 |
| 600s | 1.0  | .04 | **.36** |
| 3600s| .92  | 0   | **.24** |

### Accuracy (secondary — guessing-contaminated at 600/3600s)
| bin | A full | knob | U unif | **C so400m** |
|---|---|---|---|---|
| 15s  | .76 | .64 | .64 | .60 |
| 60s  | .84 | .84 | .72 | .76 |
| 600s | .76 | .72 | .56 | .72 |
| 3600s| .72 | **.84** | .60 | .76 |

### Tokens (mean input) — the frontier that matters is **C vs knob**
| bin | A full | **knob** | U/C |
|---|---|---|---|
| 15s  | 2137  | 1161  | 2547 |
| 60s  | 6972  | 3609  | 2545 |
| 600s | 66961 | 34166 | 2560 |
| 3600s| 87772 | 87087 | 2564 |

**Findings (honest split — sufficiency solid, dilution only suggestive):**
1. **[SOLID] The selector beats the equal-budget dumb baseline (U) on recall.** At long video C beats
   uniform-k6 on hit@k: **.36 vs .04 (600s), .24 vs 0 (3600s)** — same 6-frame cost, SigLIP relevance
   >> uniform. This is the uncontaminated win. (Pass-1 n=10 missed it because U wasn't run; U is the
   bar that isolates "is the selector *smart*," not just "is it cheap.")
2. **[SOLID] 6 well-chosen frames are *sufficient*.** Full-dump's ~100× more frames add **no
   measurable accuracy** over C (ties/within-CI at every bin) at **26–34× the token cost**. knob
   ties/beats C on accuracy but costs **13–34× more tokens at 600/3600s** (34K vs 2.5K @600s; 87K vs
   2.5K @3600s) — so C's case is *cost at iso-accuracy vs the real baseline (knob)*, not an accuracy win.
3. **[SUGGESTIVE ONLY] Context dilution.** Full-dump *holds* the evidence (hit@k .92–1.0) yet scores no
   better than 6-frame C — consistent with dilution, BUT the flat accuracy is equally explained by
   MCQA guessability, so "extra frames actively *hurt*" is NOT established at n=25 (only the whispr n=1
   case is clean; here it's within CI). Downgraded from a headline to a hypothesis.

**Caveats (numbers-first):** arm accuracy-CIs overlap (n=25, ±~0.18 — directional). **C's own recall
collapses with length (1.0 → 0.24)** = the concrete thing Fork B must fix. **Gating risk:** the
accuracy story is capped by a guessable MCQA benchmark — a real *accuracy* win over knob may be
un-showable here regardless of selection quality; would need a less-guessable eval (open-ended / true
needle). gpt-5.5 has a **500-image/request hard cap** → "full-dump" at 600/3600s is *already* ~500
frames sub-sampled (no true full-dump at length — which itself strengthens "you must compress"). `knob`
quirk: at 3600s a 0.5-fps sample of an hour is still >500 frames → caps to 500 = ~same 87K as A. Results:
pod `results/p0_{A,knob,U,C}/`, `results/forkA_{accuracy,hit_at_k}.json`, local `results/n25/`.

**media.py correctness fix applied (matches the reviewer's flag):** the frame-cap now **uniformly
subsamples across the whole clip** instead of keeping the first N — both the images path
(`_load_image_dir`, the path we actually run) and the video path (`_load_video_frames`). Old first-N
made full-dump on a 3600s video see only its first ~500s and silently miss late needles, corrupting
exactly the crossover Fork A measures. (Extracted frames are zero-padded `f%06d.jpg` → lexicographic ==
temporal, so ordering/`seconds` were already correct — no scramble bug.)

# Fork B — coarse-to-fine + transcript (Jul 4) — **NEGATIVE, closed**

**Target:** the one broken number — C's long-video selection recall (hit@k .24 @3600s). All levers
validated **free** first (echo answerer, hit@k only) before any paid run — none beat flat top-k, so
**no gpt-5.5 was spent on Fork B.**

### hit@k by lever (echo, n=25/bin)
| selector | 15s | 60s | 600s | 3600s |
|---|---|---|---|---|
| flat top-k (Fork-A C) | .96 | .72 | **.36** | **.24** |
| coarse-to-fine (`hier`) | .96 | .76 | .32 | .20 |
| transcript-gate | .88 | .72 | **.04** | **.08** |
| uniform (floor) | 1.0 | .56 | .04 | 0 |

**Levers NOT run (deprioritized by the recall-vs-k diagnostic, below — not executed):**
- **high-res zoom** (re-encode the winning window at bigger pixels): `hier` ran coarse-to-fine but
  re-ranks the *same* SigLIP scores at the *same* 384px — the zoom lever from the Fork-B design was
  never built.
- **draft-verify / backtrack:** never built.
- **video-native retrieval scorer** (InternVideo2 / LanguageBind / moment-retrieval CLIP): never
  tried. This is the real gap — see Conclusion. Everything above sits downstream of, or on the wrong
  modality from, the *same* SigLIP scorer, so recall-vs-k predicts they inherit its ceiling; a
  *different* scorer does not.

### recall-vs-budget diagnostic (`scripts/recall_vs_k.py`, score once, read off k)
| bin | k=6 | k=12 | k=20 | k=40 | reading |
|---|---|---|---|---|---|
| 60s | .72 | .72 | .92 | **1.0** | budget-limited |
| 600s | .36 | .36 | .44 | **.60** | partly budget |
| 3600s | .24 | .24 | .24 | **.32** | **selector-limited** |

**Findings:**
1. **Coarse-to-fine (T*/VideoTree family) fails here** — *worse* on the long bins it targets. It
   re-ranks the same SigLIP scores; a sparse coarse probe (4 frames / ~112s window) drops the 1–2s
   needle's window before the fine stage can recover it.
2. **Transcript-gate fails harder** (600s .36→.04, 3600s .24→.08). Verified not a bug: clocks align,
   but for frames-answerable questions the ASR track describes *speech*, not the *visual* needle —
   gold_in_window = False for 4/5 audited long items (gold @200s, top text-window @12s). Lexical match
   to question words ≠ needle location, so gating actively excludes the answer region. (Consistent with
   the literature: subtitles help *subtitle-dependent* questions — which we deliberately excluded.)
3. **The bottleneck is THIS scorer, not selection strategy.** recall-vs-k shows 3600s is selector-
   limited: 6.7× more budget moves recall only .24→.32. **SigLIP so400m is an image-text *matching*
   model, not a temporal-localization model** — it cannot rank a 1–2s visual needle among ~2400
   hour-video frames. The recall-vs-k argument proves every lever *downstream of the same scores*
   inherits this ceiling; it says **nothing** about a *different* scorer.
4. **One bankable positive: adaptive-k** (budget ∝ length) recovers 60s fully (→1.0) and 600s
   substantially (→.60) at k=40 — still far below full-dump tokens. Useless at 3600s. **This is a
   shippable operating point for the sub-10-min regime, which is what real agent-browse video mostly
   is; the 1-hour extreme may be the least important regime.**

**Conclusion (scoped honestly):** we proved **SigLIP (image-text matching) can't localize** at
hour-scale — NOT that *cheap selection* can't. The on-thesis escape hatch is untested: **cheap
video-native retrieval encoders** (InternVideo2 / LanguageBind / Marengo-family) are purpose-built for
temporal moment retrieval and are still small/cheap (embed pass, no frontier answerer). The real
close-out test — before conceding to a frontier/expensive scorer — is **one free swap-the-scorer run**
on the same echo/hit@k harness (see `scorer_swap_spec.md`). If a video-native encoder *also* flatlines
at 3600s, then "need a stronger scorer" is earned; until then Fork B is **negative for SigLIP, open on
scorer choice.** Artifacts: local `results/forkB/`, pod `results/fb_echo_*`,
`results/recall_vs_k.json`. Code shipped: `--selector hier|transcript` in `harness/selectors.py`.

> **RESOLVED by the scorer-swap below (Jul 5): the "open on scorer choice" question is now answered.**

# Scorer-swap — Fork B close-out (Jul 5) — **NEGATIVE, now PRINCIPLED**

**Question:** is the long-video wall *SigLIP's*, or *cheap selection itself*? Swap ONLY the scorer,
everything else fixed (same manifest, 4 bins, n=25, k=6, echo answerer = $0). Spec: `research/scorer_swap_spec.md`.

### hit@k by bin (echo, k=6)
| scorer | 15s | 60s | 600s | 3600s | what it is |
|---|---|---|---|---|---|
| uniform (floor) | 1.00 | 0.56 | 0.04 | 0.00 | non-semantic |
| **so400m** (incumbent) | 0.96 | 0.72 | **0.36** | **0.24** | SigLIP image-text *matching* |
| siglip2 | 0.88 | 0.76 | 0.24 | 0.08 | *better image* encoder |
| videoret (X-CLIP) | 0.44 | 0.16 | 0.12 | 0.04 | *video* temporal (action) encoder |

Fresh so400m reproduces Fork-A/B exactly (`.96/.72/.36/.24`) → harness is deterministic. **Both swaps lose.**

### Read them differently (this is the point)
1. **siglip2 (image) — clean loss.** A newer/better *image* encoder localizes **worse**, not better →
   "it's just encoder quality" is **falsified cleanly**. Nice supporting nail.
2. **videoret (X-CLIP) — loses, but by construction.** X-CLIP is video **action-recognition** and scores
   a **32-frame (32 s) window as one unit** → a 1–2 s needle is **smeared across the pool**. So it can't
   localize *by architecture*, not by tuning — running smaller-window / other clip-pooled encoders
   (X-CLIP-patch32, LanguageBind_Video_FT) would fail by the **same mechanism** and was skipped as
   low-information.

### The principled conclusion (the actual deliverable)
**Fine-needle localization at 1 fps is a per-frame retrieval task.** Per-frame image-text scoring
(SigLIP) is *structurally the right cheap tool* — clip-pooled video encoders smear the needle by
construction and can't help. And `recall_vs_k` already showed the residual wall isn't the *encoder* but
**ranking capacity at hour scale** (so400m 3600s: .24→.32 at 6.7× budget). So **the cheap ceiling is
already SigLIP in hand**; the only thing that plausibly beats it is a **dense moment-retrieval head
(Marengo / TwelveLabs-class)** — precisely the paid/commercial stack the "cheap selector" thesis was
trying to avoid. This reframes the negative from *"we couldn't find a better scorer"* to *"the task
structure says the cheap right tool is already in hand, and its limit is ranking capacity, not encoder
choice."*

**Open assumption — now TESTED and FALSE (Marengo ceiling, Jul 6).** The bet was "a dense moment-
retriever *would* localize a 1–2 s needle in an hour." Ran TwelveLabs **Marengo 3.0** (the purpose-built
commercial retriever) on the 3600 s bin, n=10, k=6 clips:

| metric | Marengo 3.0 | so400m (cheap SigLIP, k=6) |
|---|---|---|
| strict (clip midpoint ∈ gold span — apples-to-apples with SigLIP's point metric) | **0.10** | 0.24 |
| lenient (clip [s,e] OVERLAPS gold span — full credit to a span-returner) | **0.30** | 0.24 |

**Marengo TIES the cheap selector — nobody solves this slice.** At its fair metric (lenient, given ~6 s
clips) Marengo is *nominally ahead*, 0.30 vs 0.24, but at n=10 (Wilson ±~0.28) that is a **tie in the
noise**, not a win either way (3/10 lenient, 1/10 centred). Defensible claim = **parity**: cheap SigLIP ≈
commercial dense-retrieval SOTA here. **The evidence that hardens the negative is the CROSS-ARM PATTERN,
not this one n=10 cell:** uniform .00 / so400m .24 / siglip2 .08 / X-CLIP .04 / Marengo .10–.30 — five
approaches, **none escapes ~0.2–0.3 at 1 h**. **Robustness:** Marengo ingests the *real video* at its own
denser sampling (not our 1 fps) and still ~0.30 → the wall is **not** a 1-fps-starvation artifact.
**Scope:** "fundamentally hard" is earned for the *adversarial* slice — a 1–2 s, single-frame-answerable
visual needle in a 1-hour video — NOT long video broadly (most real agent video is coarser/multi-frame).
"Why not TwelveLabs?" → doesn't solve *this slice* either. Caveats: n=10 subset (≠ so400m 25), 3600 s only,
~6 s granularity penalizes strict. Runner `scripts/marengo_ceiling.py` (SDK 1.2.8, marengo3.0); `results/marengo_ceiling.json`.

**Second point — Marengo 600 s (Jul 6, n=10, apples-to-apples same 10 IDs).** Added the 10-min bin to
see if Marengo pulls ahead when the haystack shrinks. It does **not**: Marengo 600 s lenient **0.20** /
strict **0.10** vs cheap so400m hit@6 **0.30** (same 10 videos; hit@40 0.60). At 10-min the **cheap
per-frame selector edges *ahead*** of the commercial dense retriever at equal budget; at 1-hour it's a
tie. Across *both* lengths Marengo never beats cheap SigLIP → hardens "nobody solves the fine needle, and
cheap is at least as good." (Marengo lenient 600 s .20 < 3600 s .30 is small-n wobble, don't over-read.)
`results/marengo_600s.json`.

**Net close of Fork B:** on the hour-scale fine-needle slice, cheap per-frame SigLIP ≈ commercial SOTA
(nobody solves it cheaply); the shippable win is **adaptive-k for ≤10-min** video. No further scorer work
is warranted. **Before any accuracy headline:** (a) one end-to-end adaptive-k run on ≤10-min bins
(currently hit@k-only), (b) the blind question-only baseline (guess floor) — both cheap, both unrun.

**Shippable positive stands:** adaptive-k for the **≤10-min** regime (60s→1.0, 600s→.60 at k=40) — the
length range real agent-browse video actually lives in.

Artifacts: local `results/scorer_swap/{uniform,so400m,siglip2,videoret}_{by_bin.json,runs.jsonl}`. Code:
`--selector videoret` (X-CLIP) in `harness/selectors.py`; `recall_vs_k.py --model` for scorer sweeps.

---

## Perception Encoder (PE-Core) scorer-swap — the medium-band DOES move (Jul 6)

**Question left open by the scorer-swap:** the 3600 s wall is task-inherent (Marengo ties cheap SigLIP),
but the **≤10-min band is partly budget-limited** (recall_vs_k: 600 s .36→.60 as k 6→40), i.e. the scorer
isn't maxed there. Does a *stronger per-frame encoder* lift the operational-k (k=6) medium band? Tested
Meta's **Perception Encoder (PE-Core)** — current SOTA image-text encoder (beats SigLIP2 on image,
InternVideo2 on video) — as a drop-in scorer via new `PESelector` (`harness/selectors.py`),
`recall_vs_k.py --pe-config`. Same 25 videos/bin, free (scoring + hit@k, no answerer).

**hit@k, same 25 IDs/bin — encoder scaling at operational k:**

| bin | so400m (400M) | **PE-Core-L14-336 (320M)** | PE-Core-G14-448 (1.9B) |
|---|---|---|---|
| 60 s @6 | 0.72 | **0.88** | 0.80 |
| 600 s @6 | 0.36 | **0.48** | 0.32 |
| 600 s @40 | 0.60 | 0.56 | 0.56 |

**Two findings:**
1. **PE-Core-L14 lifts the shippable band** — 60 s @6 0.72→**0.88**, 600 s @6 0.36→**0.48**. The medium
   regime is *not* encoder-independent; a better per-frame encoder front-loads the needle into the top-6.
   And L14 wins **despite a handicap**: PE-Core caps text at **32 tokens** vs SigLIP so400m's 64 (Marengo
   500), so question-style queries truncate harder — the gain is real, not a query-length artifact.
2. **Bigger is NOT better** — G14 (1.9 B, L/B are distilled from it) is *worse* at k=6 than its own 320 M
   distillation (600 s @6 0.32, even below so400m). All three converge at k=40 (~0.56–0.60): **same
   ceiling, L14 just ranks the needle highest at low k.** Shippable pick = **PE-Core-L14-336**; scaling up
   buys nothing at operational budget while costing 6× params.

**Caveats:** n=25 (600 s @6 0.48-vs-0.32 is a ~4-item swing, Wilson ±~0.19 — suggestive, not locked);
deep-budget ceiling unchanged, so this helps *operational-k selection*, not the fundamental 600 s ceiling;
3600 s not run (task-inherent wall, skip). Env: Blackwell sm_120 needs torch 2.11+cu128 (full deps, not
`--no-deps`); PE needs `perception_models` on `PE_REPO` + ftfy; no xformers required for CLIP inference.
Artifacts: `results/recall_vs_k_pe_{L14,G14}.json`.

**Revised shippable story:** ≤10-min adaptive-k **with PE-Core-L14 as the scorer** — a strictly better
cheap selector than so400m in exactly the band that ships (60 s @6 .88, 600 s @6 .48). so400m still fine
at deep k; PE-Core-L14 wins where budget is tight.

### Attribution (siglip2 control) + end-to-end reality-check (Jul 6)

**siglip2 control — the lift is PE-specific, not "any newer encoder."** Ran google/siglip2-so400m as a
modern non-PE control, same 25 IDs/bin, free hit@k:

| bin | so400m @6 | PE-Core-L14 @6 | siglip2 @6 |
|---|---|---|---|
| 60 s | 0.72 | **0.88** | 0.76 |
| 600 s | 0.36 | **0.48** | **0.24** |

siglip2 is *worse* than so400m at 600 s @6 (0.24) and ~flat at 60 s. So a newer SigLIP-family model does
**not** reproduce PE's low-k gain — combined with G14 (bigger, also worse), the operational-k lift is
specific to **PE-Core-L14**, not a generic "newer/bigger encoder" effect. `results/recall_vs_k_siglip2.json`.

**End-to-end at k=6 (600 s, n=25, gpt-5.5 answerer, gpt-4.1 judge) — accuracy is guess-dominated.**
Condition C, full 3600-frame pool, both selectors on the same 25 items:

| selector | hit@k | accuracy | input tok | latency |
|---|---|---|---|---|
| **PE-Core-L14** | 0.48 | 0.76 | 2560 | 4.6 s |
| so400m (SigLIP) | 0.36 | 0.72 | 2560 | 5.0 s |

PE's **+0.12 hit@k** (0.48 vs 0.36, replicating the free result) buys only **+0.04 accuracy** (0.76 vs 0.72)
— **inside n=25 noise (Wilson ±~0.17)**. The sharper tell: an earlier PE run on a 64-frame subsampled pool
scored **0.80 accuracy at hit@k 0.28** — *more* needle-in-frame (0.28→0.48) did **not** raise accuracy
(0.80→0.76). On this 4-option MCQA slice gpt-5.5 answers ~0.72–0.80 whether or not the gold frame is
present (language priors + partial visual). **Conclusion: end-to-end MCQA accuracy does NOT track hit@k
and cannot discriminate selectors here — hit@k stays the primary metric, and the still-unrun blind
(question-only) baseline is the needed control, not more end-to-end accuracy runs.**
`results/e2e_{pe_L14,so400m}/`. Live gotcha: run.py default `--max-dump-frames 64` silently subsamples the
candidate pool *before* the selector, pre-dropping the 1 s needle — use 3600 to score the full 1 fps pool,
else end-to-end hit@k understates the selector.

# Region-recall gate — Fork B REOPENED, **PROCEED** (Jul 7, 2026)

**Reframe that reopened Fork B:** the old Fork B (hier/transcript) asked "find the exact frame" and failed.
Wrong question. Coarse-to-fine only needs the right **chunk** at each level, then zoom. New metric
`region_recall@m` (`harness/metrics.py`): split a video into `M` contiguous chunks, chunk score = **max**
of its frame scores; is a **gold chunk** in the top-`m`? Anchored on a **random-chunk null**
(`1 − C(M−G,m)/C(M,m)`), not an absolute. Echo/GPU-only, $0. Driver `scripts/region_recall.py`.

**Setup:** SigLIP so400m, `data/manifest.lvb.frames.json`, n=25/bin all `gold_reliable`, 1-fps frames.
Result `results/rr_siglip_all.json`. **Directional (n=25, SigLIP only).**

**Headline (M=4 = quadrant, agg=max, dense), Wilson 95% CI:**

| bin | frame hit@6 | region@1 | region@2 | null@2 |
|---|---|---|---|---|
| 60s   | 0.72 | 0.68 [.48–.83] | 0.92 [.75–.98] | 0.67 |
| 600s  | 0.36 | 0.64 [.45–.80] | **0.84 [.65–.94]** | 0.57 |
| 3600s | 0.24 | 0.56 [.37–.73] | **0.88 [.70–.96]** | 0.53 |

**The premise holds:** where SigLIP lands the exact frame only 24–36 % of the time, it lands the right
**quarter (top-2 of 4)** ~84–88 %, CIs clear of the random null. **The neighborhood is findable even when
the frame isn't. SigLIP alone passes the gate — before the grounding arm.**

**Design locked by the grid:**
1. **M=4 is the sweet spot.** region@1 decays 0.56→0.36→0.28→0.20 as M=4→8→16→32 (sinks toward null).
   Big/few chunks win → recurse with **small M per level**, never one fine partition.
2. **Beam-2, not greedy-1.** region@1 (0.56–0.64) is below the ≳0.8 needed for 2-level single-chunk descent
   (0.56²≈0.31, ~hit@6). region@2 = 0.84–0.88 → beam-2 over 2 levels ≈ 0.77, **survives**. The @1→@2 jump
   (0.56→0.88 @3600) mandates keeping **2 of 4 per level**.
3. **max ≫ mean** (0.84/0.88 vs 0.68/0.64); ucb mixed. Peak aggregation confirmed.
4. **Dense coarse pass.** probe sparsity rr@2 @600: p1=.52 p2=.56 p4=.68 **all=.84** — sparse leaves recall
   on the table. Fine: SigLIP over all 1-fps frames costs seconds; the compute win is the **answerer** seeing
   6 frames not 500, not a cheaper selector. Score dense, descend cheap.

**Build:** coarse-to-fine = **beam-2 · M=4 · max · dense**, then zoom (re-decode winning chunk at higher fps)
+ optional draft-verify. New selector class in `harness/selectors.py`.

**Not yet paper-grade (pending hardening):**
- **n=25/bin, SigLIP only.** Fix-1 = rebuild **100/bin** (re-validate `gold_reliable` on the new ~300) and
  run the **`ground` arm** (GroundingDINO-tiny on decomposed targets) — could lift region@1 enough to permit
  greedy descent, or confirm beam is mandatory. Log %concrete-noun targets, segment region_recall by it.
- Pod cold-start (Jul 7): Blackwell sm_120 needs **torch 2.11+cu128** (2.4+cu124 = "no kernel image");
  uninstall stale torchaudio, add transformers+sentencepiece+protobuf. Long ssh runs drop (exit 255) → run
  detached with `nohup`, poll result file over short ssh calls.

## Beam coarse-to-fine — the METHOD number (Jul 7, 2026) — ≈ flat, trending NEGATIVE

The region-recall gate validated the *premise* (right chunk findable). This tests the *method*: does beam
descent actually lift selection hit@k? `BeamCoarseToFineSelector` (beam-2 · M=4 · max · dense), **echo
answerer (NO gpt-5.5 — pure hit@6 selection recall, $0)**, n=25/bin, same LongVideoBench frames.

**Mechanism as run:** score all 1fps frames once (SigLIP), then recurse — split window into 4, keep best 2
(by max frame score), split *each* kept window into 4 again, keep best 2, ... until windows ≤~24 frames,
then flat SigLIP top-6 on that pool. Descent depth: 60s 1 level, 600s 2, 3600s 3–4. **No zoom** (unbuilt;
and a red herring here — see gold spans).

**Result — beam vs flat hit@6:**

| bin | beam | flat | Δ |
|---|---|---|---|
| 15s | .96 | .96 | 0 |
| 60s | .68 | .72 | −.04 |
| 600s | .40 | .36 | +.04 |
| 3600s | .24 | .24 | 0 |

**Beam ≈ flat.** Every Δ inside n=25 noise (±.19); exact tie at the hard 3600s bin; beam even *loses* at
60s (over-pruning drops needles flat keeps). `results/beam_echo/`.

**Why (the real finding):** pruning distant distractors *should* raise the needle's rank, but hit@k didn't
move. So the frames that outrank the needle are **not distant — they're near/similar look-alikes inside the
needle's own region.** Beam removes *cold* frames, not the *competitors*. This is recall_vs_k's
"ranking-capacity wall" **surviving pool-narrowing** — narrowing the haystack doesn't fix a per-frame
ranking problem when the false positives are local.

**Zoom won't rescue it:** gold spans on this set are **median/mean 2.0s, 158/161 ≤2s** — the needle is
already ~2 frames at 1fps, rarely missed by undersampling, so denser resampling adds ~nothing. The lever
was re-ranking, and re-ranking is inert.

**Status:** region@2 = .88 (find the half) but beam hit@6 ≈ flat (can't rank inside the half). Premise real,
method inert. **One lever left = the grounding arm** (different signal, not cosine — the only thing that can
move ranking not just pool; queued in n=100 chain). If grounding hit@6 also ≈ SigLIP → Fork B closes as
**principled negative #2**, honest ship = adaptive-k ≤10min. n=100 will *harden* the negative, not reverse a
~0 effect. My briefing §6 error corrected: beam CAN beat flat in principle (re-ranking); it just doesn't here.

## Region-recall — n=100/bin CONFIRMATION (Jul 7, 2026) — premise paper-grade

Rebuilt to 100 distinct videos/bin (fetch 400/400, gold reliable **400/400**, exactly 100/bin), same
SigLIP/M=4/max/dense. `results/rr_siglip_n100.json`. Premise holds at ±0.07–0.09 CIs:

| bin | hit@6 | region@1 (M=4) | region@2 (M=4) | null@2 |
|---|---|---|---|---|
| 60s   | 0.72 | 0.67 [.57–.75] | 0.91 [.84–.95] | 0.66 |
| 600s  | 0.34 | 0.59 [.49–.68] | 0.82 [.73–.88] | 0.57 |
| 3600s | 0.20 | 0.61 [.51–.70] | 0.84 [.76–.90] | 0.56 |

region@2 lower-CI (.73/.76) far above null (.57/.56) AND hit@6 (.34/.20) — separated, no overlap.
Barely moved from n=25 (.84/.88). **"Neighborhood findable, exact frame not" is confirmed, not directional.**
M=4 remains the sweet spot (M=8 worse at every m). hit@6 held at .34/.20 = the ranking wall is real.

**Net Fork-B state:** premise CONFIRMED paper-grade; **method (beam≈flat) inert** — the tree can't exploit
the found region with the same scorer (arithmetic: same numbers reshuffled). Zoom dead (2s spans; and the
±1s gold window is OUR 1fps tolerance, not LVB's annotation — LVB gold = single native-fps keyframe, we
match a ±1s temporal window). **Only remaining lever = a different SCORER** (grounding/detection, frame-level,
concrete-noun segmented). Decision: build the grounding scorer-swap, or close Fork B as principled negative #2
and ship adaptive-k ≤10min.

## Grounding scorer-swap — DECISIVE (Jul 8, 2026) — Fork B closes, negative #2

The last lever (a *different scorer* — open-vocab detection, frame-level, concrete-noun segmented) ran on
n=100/bin, RTX PRO 6000. Two strong, independent detector families in one pass — deliberately going strong
from exp 1 to remove the "weak-checkpoint" doubt from the S2 SigLIP lesson:
`IDEA-Research/grounding-dino-base` (phrase grounding, DETR cross-modal fusion) and
`google/owlv2-large-patch14-ensemble` (ViT + contrastive, cross-family control). Per-frame score = max box
logit for the item's cached target phrase from `data/targets.json` (327/400 concrete). Both smoke-validated
before the run (score spread, gold-above-median) so a flat/mis-wired scorer couldn't fake a tie.

**Read metric: frame-level hit@6 on the *concrete* subset vs SigLIP.**

| bin | SigLIP (ref) | grounding-dino-base | owlv2-large |
|---|---|---|---|
| 600s  | **0.34** | 0.12 [.067–.208] (n=83) | 0.24 (n=83) |
| 3600s | **0.20** | 0.04 [.014–.111] (n=75) | — (skipped) |

`results/rr_ground_base_n100.json` (full), `results/rr_ground_owlv2_n100.partial.json` (600s only).

Both detector families lose to cheap SigLIP even at the *easier* 600s bin; base CIs don't touch SigLIP's
point estimate at either decision bin. owlv2 3600s was intentionally not run — it already trails SigLIP at
600s and the signal only decays with length (SigLIP .34→.20, base .12→.04), so a 3600s revival was
near-impossible and not worth the credits. Detection scores the *object*, not the *event/relation* the
question asks about, and long-video needle recall is dominated by the temporal-localization wall, not by
what open-vocab thing sits in a frame.

**Fork B CLOSED — conclusive negative #2.** Premise (neighborhood findable, exact frame not) is paper-grade;
every method lever — beam re-ranking (inert), zoom (dead), scorer-swap (loses to SigLIP) — fails to convert
the found region into frame-level recall with any scorer. Consistent with the Marengo ceiling (cloud span
recall never beats cheap SigLIP either): **the hour-scale fine-needle wall is the TASK, not the cheap
selector.** Honest ship = **adaptive-k ≤10-min** (compression buys COST at iso-accuracy where the selector
actually works), not a hierarchical selector for hour-scale needles.

OWLv2 wiring note: text tower caps at 16 positions — long target phrases crash
(`tensor a (34) must match tensor b (16)`); scorer truncates (`truncation=True, max_length=16`). Architectural
property of that family, honest to report.

---

## Retrieve-then-ground (Fork B v2) — DECISIVE (Jul 8) — negative #3

**Idea:** keep the cheap SigLIP per-frame retriever for Stage-1, but replace the *scorer* with a
generative frontier VLM (GPT-5.5) that reads the retrieved frames JOINTLY (timestamp-tagged) and
localizes the needle — the joint read being the lever cosine/detection structurally can't do.
Two questions, both answered offline-first before spend.

### Q1 — retriever union algorithm (free offline gate, `scripts/union_ceiling.py`, n=100 @600s)

The designed **peak-NMS** union (adaptive-τ peaks → temporal NMS → pad-windows, frame-budgeted) is
**DOMINATED by plain flat top-k** at every equal frame budget:

| frames (=imgs to VLM) | peak_nms any-hit | flat_top_k any-hit |
|---|---|---|
| 30 | 0.49 | **0.56** |
| 50 | 0.55 | **0.66** |
| 86 | 0.63 | **0.72** |
| 100 | 0.64 | **0.77** |

peak-NMS gets *lower recall at lower cost* (not the thesis of equal recall at lower cost) — it just
runs a smaller budget, and temporal spreading actively *hurts* because SigLIP ranking already
concentrates on gold; padding around secondary peaks wastes budget. **peak-NMS killed; Stage-1 = flat
top-k.** (bin 60 is degenerate — videos <100 frames, top-100 = whole video → trivial 1.00; ignore it.)

### Q2 — does GPT-5.5 recover the rank headroom? (`scripts/gpt_probe.py`, @600s, K=50, n=30)

| probe | floor (SigLIP top-6) | GPT-5.5 hit@6 | ceiling (gold-in-union) |
|---|---|---|---|
| 512px / effort=low    | 0.367 | **0.30** | 0.60 |
| 768px / effort=medium | 0.367 | **0.30** | 0.60 |

**GPT-5.5 lands BELOW the SigLIP top-6 floor**, identically at both resolutions/efforts — the
resolution/effort confound is rejected. Item-level (n=30): 12 recall-fail (gold never in union), 18
gold-in-union of which 11 already in SigLIP top-6; GPT recovered 3/7 rank-headroom items but **broke
5/11 floor items** → net 9 < floor 11.

### Root cause — the wall is the TASK, confirmed visually

Both stages are gated by the *same* signal, and the gate is task structure, not scorer quality:

1. **Stage-1 recall wall (40% miss, not sampling):** for all 12 recall-fail items a gold frame WAS
   sampled at 1fps — SigLIP cosine just ranked it 55–323 (median >130) out of ~200–590. Not a
   sampling/fps problem; a *ranking* problem.
2. **Stage-2 can't beat cosine** even reading 50 frames jointly at 768px/medium.

Eyeballing the frames (`results/probe_frames/`) shows why: LVB long-bin questions carry a **verbose
compositional preamble** ("woman in dark-red floral top holding the craft with her *right hand*…")
that pins ONE exact frame inside a video that is **visually homogeneous** — the same person, same
setting, whole clip. Gold vs top-cosine frames are **near-identical** (e.g. `LYvStKy8iAc_0`: gold s96
and cosine-top s348 are the same talking-head pose/background; the "pressing foundation onto a sponge"
action is sub-second in a ~10-min constant image). A global CLIP embedding cannot separate the gold
frame from ~300 near-duplicates, and a frontier VLM given 50 near-duplicates can't pin the transient
instant either.

**Fork B v2 CLOSED — negative #3.** peak-NMS dominated by flat top-k; GPT-5.5 joint-read below the
cosine floor at iso-config. Same wall as Fork B v1 / the Marengo ceiling: **hour-/10-min-scale fine-
needle localization in visually-homogeneous video is a TASK wall, not a selector-quality gap.** The
honest ship remains **adaptive-k ≤10-min** (compression buys COST at iso-accuracy where the selector
works), not a two-stage or hierarchical selector for the needle regime.

Artifacts: `results/scores/scores.jsonl` (Stage-1 cache, n=200), `results/gpt_probe_600.json` +
`_hi.json`, `results/union_ceiling_k1.json`, `results/probe_frames/` (95 gold/pick/cosine frames).
