# Relevance keyframe-sampling for long-video MLLMs — timeline & landscape

Survey of query-relevant frame selection / compression for long-video MLLMs, Feb 2025 → Jun 2026.
Compiled 2026-07-09. Linked from `FINDINGS.md`. Our work (slm-lab) = frozen SigLIP so400m + plain
top-k, GPT answerer, thesis = **compression buys cost at iso-accuracy** (not accuracy gain).

## Two tiers
- **Selection tier** — *which* frames to keep (AKS → AdaRD → Focus → Evidential). Scored by BLIP/CLIP/SigLIP.
- **Compression tier** — *how many tokens per frame* (AdaCodec, LDDR). Our cost thesis lives here.

## The core idea (read this first)
Every method starts identically: a **scorer (SigLIP/BLIP/CLIP) gives one relevance number per frame**. The
scorer is commodity — SigLIP vs BLIP is a ±2-pt detail (Focus Table 10). **All the intellectual content is in
how you turn scores → a picked subset.** Our method = plain **top-k** (take the highest numbers). Everything
else fixes one of top-k's two weaknesses:
- **W1 redundancy** — top-k grabs near-duplicate frames clustered on one moment.
- **W2 blind/expensive** — scores every frame, trusts the raw score.

Four families of fix:

**A. Spread the picks (coverage / diversity)** — fix W1.
- **AKS**: recursive **time-binning** → force picks across the timeline (temporal coverage).
- **AdaRD-Key**: `R(i) + λ·log det(Gram)` → **determinant** big only if picks point different directions;
  near-duplicate adds ~0 gain (the DPP idea, feature-space diversity).
- **Adaptive Greedy**: `α·SigLIP-relevance + β·DINOv2 facility-location coverage`, greedy (1−1/e). Two encoders —
  SigLIP judges relevance, DINOv2 (pure vision) judges "is the video visually covered."
- **LDDR**: Linear-**DPP** diversity + **dynamic resolution** (important frames hi-res, others lo-res) → spend
  tokens as pixels, not just frame count.

**B. Score cheaply (search efficiency)** — fix W2-cost.
- **Focus**: clips = bandit **arms**. Coarse: glance every clip. Fine: spend scoring budget only on top clips by
  **Bernstein UCB** (undersampled clip gets wide band → optimistically kept, no premature drop). Coarse-to-fine.

**C. Score the RIGHT thing (rethink relevance)** — fix W2-blind.
- **Evidential**: cosine picks frames that *look like* the query; they redefine goal as **max I(S;O∣Q)** — keep
  frames that *change the MLLM answer*. Can't compute combinatorially → **train** a ~10M scorer to predict
  evidential value. (= our blind-baseline similarity-vs-answer gap, formalized.)
- **Swift**: score = **temporal surprise** (Taylor-series derivative vs neighbors) = novelty. **Query-free** —
  finds *change*, not *your question's* answer.

**D. Rethink the output (compression, not selection)**.
- **AdaCodec**: per frame decide **keep-full / compress / drop** — a variable-rate **codec** before the MLLM;
  every frame contributes at cost ∝ its information.

**One-line map:** scorer is commodity; the unique idea is *spread* (A), *score cheaply* (B), *score the right
thing* (C), or *don't pick — compress* (D). Our top-k is the honest baseline all are defined against; our
unexplored levers = **C (is the frame the answer?)** and **D (compress)** — A and B are now crowded.

## Master table

| date | paper | arXiv | tier | pick / score mechanism | encoder | train-free | headline result |
|---|---|---|---|---|---|---|---|
| Feb 25 | **AKS** (Adaptive Keyframe Sampling, CVPR25) | 2502.21271 | select | relevance + recursive-bin **coverage** | BLIP/CLIP | yes | LVB 62.7 vs 58.9 uniform (LLaVA-Video-7B) — THE baseline everyone beats |
| Jun 25 | Q-Frame | 2506.22139 | select | query-aware + **multi-resolution**, Gumbel-Max | CLIP | yes | query-aware res adaptation |
| Oct 25 | **AdaRD-Key** | 2510.02778 | select | relevance + **log-det diversity** (Gram) | BLIP-2 ITM | yes | LVB 62.9 vs AKS 62.7; +1.3 on 180–600s |
| Oct 25 | From-Frames-to-Clips | 2510.02262 | select | training-free adaptive **key-clip** | — | yes | — |
| Oct 25 | K-frames | 2510.13891 | select | scene-driven **any-k** | — | — | — |
| **Nov 25** | **Focus** (FOCUS) | 2510.27280 | select | **coarse-to-fine bandit** (CPE, Bernstein UCB) | BLIP ITM | yes | LVB 63.5 vs 58.9 uniform; **+11.9 on >20min**; <2% frames, 5.5 GPU-h |
| **Mar 26** | **Adaptive Greedy** | 2603.20180 | select | relevance + **DINOv2 facility-location coverage**, submodular (1−1/e) | **SigLIP** | yes | MLVU K=10: **64.48** vs AKS 62.31 (+1.98 avg); oracle 66.38 |
| **Apr 26** | **Query-Conditioned Evidential** | 2604.01002 | select | **info-bottleneck** max I(S;O∣Q) — *evidence not similarity* | CLIP + **trained 10M scorer** | **NO** (Seek-173K, 0.6 GPU-h) | LVBench 32f **47.7** vs 37.6 uniform (+10.1); VideoMME 63.6 vs 60.7; long 55.0 vs 50.6 |
| May 26 | **Swift Sampling** (temporal surprise) | 2605.22678 | select | **Taylor-series temporal derivative** = novelty; **QUERY-FREE** | CLIP | yes | up to **+12.5** on long videos, limited frame budget; vs AdaRD/UniComp/FastVid |
| May 26 | **LDDR** (Linear-DPP dynamic-res) | 2605.11477 | **compress** | **Linear-DPP** diversity + **dynamic resolution** (hi-res important frames) | CLIP | yes | **+2.5** budget-constrained / +1.6 high-budget; **3× faster** than DPP; 4 benches |
| Jun 26 | **AdaCodec** (predictive visual code) | 2606.02569 | **compress** | **predictive codec** — retain/compress/drop per frame | — | yes | **1/7 token budget** (32k vs 224k) beats baseline; TTFT **9.26s→1.62s** (82%); 11 benches vs Qwen3-VL-8B |

## Per-paper notes (mechanism + relevance to us)

### AKS (Feb 25) — the reference selector
BLIP/CLIP prompt-frame matching score + recursive binning for temporal coverage (ADA algorithm: threshold
compare, recursively split segments, prioritize high-score vs enforce spread). Plug-and-play, no MLLM retrain.
**The shared baseline** both AdaRD and Focus beat; we lack it in our ladder (blind/uniform/topk/full).

### AdaRD-Key (Oct 25) — diversity added
R(f) = BLIP-2 ITM probability (temperature-free). Greedy marginal gain `Δ(i|F)=R(i)+λ·log((1+ε)−rᵀ(G_F+εI)⁻¹r)`
= relevance + **log-det diversity**. Adaptive λ; gate to diversity-only if max R<0.4. **No encoder ablation.**
Fixes near-duplicate picks — the exact failure we saw (600s bimodal distractors).

### Focus (Nov 25) — coarse-to-fine bandit
Clips = arms (fixed **16s**). Coarse: pull each arm q times. Fine: extra pulls to top-α arms by Bernstein UCB
(zoom promising). Pick top-m by empirical mean. **Encoder ablation (Table 10, same pipeline):**
Uniform 58.9 / CLIP 60.2 / **SigLIP 60.9** / **BLIP 63.5** → BLIP > SigLIP +2.6 (selection-quality gain,
ITM cross-attn > dual-tower cosine). **Failure mode:** i.i.d. clip-mean **dilutes a 1-frame answer** inside a
16s clip; UCB guards unlucky sampling, not a genuinely diluted mean (Appendix H, admitted). = our single-needle
edge (top-k on 2s clips, no clip-mean).

### Adaptive Greedy (Mar 26) — SigLIP + DINOv2, submodular
`F(S)=α·R(S)+β·C(S)`; R = **SigLIP** query cosine (modular), C = **DINOv2** facility-location coverage.
Greedy (1−1/e) guarantee. 1-fps pool ≤1000 frames. Training-free (only light question-type routing). **Validates
our SigLIP** ("well aligned for image-text matching at single-frame level"). Beats AKS +1.98 on MLVU. No Focus cmp,
no SigLIP-vs-BLIP.

### Query-Conditioned Evidential (Apr 26) — similarity → EVIDENCE, and trained
Formalizes selection as max **I(S;O∣Q)** (conditional mutual info: keep frames that *change the MLLM answer*, not
ones that merely look like the query). Submodular → modular upper bound → frame-level scoring. Frozen CLIP-ViT-L +
**trained** ~10M evidence scorer (causal-window temporal aggregator, query gate, InfoNCE) on new **Seek-173K**
dataset. **This formalizes our blind-baseline / recall-vs-answer critique — before us.** Breaks frozen-encoder
purity. Limits: fails audio-centric + timestamp-grounded queries; evidence-coverage 50% @32f vs 33% uniform.

### Swift Sampling (May 26) — temporal surprise, QUERY-FREE
Scores frames by **higher-order temporal derivatives** via Taylor-series / finite differences = visual novelty.
No query, no training, no optical flow. = **our "distinctness" axis formalized** — but query-free, so finds
*change* not *relevance*; can't target a question-specific needle. Ceiling + our contrast.

### LDDR (May 26) — DPP diversity + dynamic resolution
**Linear-DPP** (determinant = anti-redundancy, probabilistic cousin of log-det) picks important+diverse frames,
then **dynamic resolution**: important frames hi-res, others lo-res/pruned. Query-aware, CLIP, training-free.
**New cost lever we haven't touched — vary resolution, not just frame count.** +2.5 budget-constrained, 3× faster
than DPP.

### AdaCodec (Jun 26) — predictive visual codec, closest to our thesis
Training-free **codec**: score each frame → retain full / compress / drop. Content-aware token compression, not
just frame subsampling. **1/7 token budget (32k vs 224k) beats the per-frame RGB baseline** on long-video; TTFT
9.26s→1.62s. **This is "compression buys cost" as a product** — our exact pitch, already built.

## Where we stand (honest, per project-scope rule)
Crowded, fast lane — 10 papers in 16 months, strong incumbent at every angle:
- coarse-to-fine → **Focus**
- diversity/coverage → **AdaRD, Adaptive Greedy, LDDR**
- evidence-not-similarity (our blind-baseline critique) → **Evidential** (formalized before us)
- distinctness (our peak/plateau axis) → **Swift** (query-free version)
- compression-buys-cost (our thesis headline) → **AdaCodec, LDDR** (dynamic resolution)
- our encoder (SigLIP) → **validated by Adaptive Greedy**

**Still genuinely ours:** (a) extreme-compression cost framing at *iso-accuracy* with the **decoupling proof**
(hit@k 0.24 yet acc 0.76 — answer ≠ gold frame); (b) the **query-aware** peak/plateau **distinctness diagnosis**
(Swift is query-free; nobody explains *why* cosine selection works/fails by moment structure). Both real but thin.

Benches recurring (all MCQA, none moment-span): **LongVideoBench, LVBench, MLVU, VideoMME, Q-Bench, EgoSchema** —
field stays on end-task MCQA, confirming our pivot away from frame-recall.
