# Manuscript notes (companion to `paper/arxiv.tex`)

`arxiv.tex` is the single source of truth as of 2026-08-21. Superseded versions
(`main.tex`, `main-old.tex`) moved to `paper/archive/`. This file is notes, not a twin —
its prose is now older than the LaTeX.

Spine: **cross-budget lead → T1 table → half-budget → k=16 reinvest.**
OMP framing (2026-08-21): stated once, positively, in the Introduction as a methodological
choice — no tuning degrees of freedom, and an informative bar. Cashed in as a *finding* in
the Discussion ("a 1993 algorithm is competitive with purpose-built selectors"). All the
old "we do not claim to propose a method" disclaimers are deleted; do not reintroduce them.

Your spoken notes stay in `paper.md`. Banked numbers/lab notes stay in `PAPER_DRAFT.md`.

<!--
PRE-SUBMISSION REVIEW NOTES (2026-08-16, reviewer-angle pass on main.tex) — before ARR Oct 12.

1. GPU-split (Limitations, main.tex ~L208/330/376/496-8): headline T1/A/B tables on one
   GPU family; allocation-shape (GD recon of LDDR stage-2) + reinvest-interaction control
   on a second, not merged. Root cause per HANDOFF_LVBENCH_CUDA.md: fresh RunPod stacks
   came up CUDA-dead (0 MiB, torch init hang) alongside the healthy :49810 2x RTX PRO 4500
   pod — MFS volume contention suspected. FIX: rerun those two control cells on the same
   healthy stack that made Tables 1-3, merge in directly, drop the caveat.

2. LVBench GPT-5-mini (main.tex L266/342/355): T1 selector ranking still summary-only
   (pod stopped before pull). Claim A/B per-item JSON was on Azure
   `slm-lab/results/gpt_mini/` — recovered 2026-08-16, n=1549 paired.
   McNemar: A +0.26 pp p=.84; B +2.39 pp p=.039. Written into T2/T3. No API rerun.

3. Venue fit: checked ACL Rolling Review CFP directly — negative-results / non-
   reproducibility papers are an explicitly named category, ACL main is NOT the wrong
   venue for a "not a new method" framing. Retracted earlier caution on this.
   Alt/parallel option: NeurIPS "Evaluations & Datasets" track (renamed from D&B, 2026)
   scope now explicitly covers "failure modes of existing benchmarks/evaluation
   practices" and "impact on scientific conclusions" — near-exact match for Sec 5.4
   (mechanism: rank-2 Gram, LongCLIP-specific geometry) and Appendix B (qtype/duration
   split). Consider promoting Sec 5.4 + App B to lead contribution for that track instead
   of leading with T1.

4. Other flags from the pass, not yet actioned:
   - references.bib: EFS authors fixed from arXiv 2603.00983 (2026-08-16). IDs
     2603.00983 (EFS) / 2607.00983 (QCA) both real; suffix collision is coincidence.
   - an2026adalloc cited as a project page, "arXiv ID incomplete" — chase real ID before
     camera-ready, it's load-bearing for the concurrent-work positioning para.
   - No code/data release statement anywhere in main.tex — undercuts the paper's own
     "fair frozen harness" pitch. Add a footnote/URL if the harness is releasable.
   - Three claims (T1/A/B) each get qualified until they nearly cancel each other's
     punch (Video-MME breaks T1's pattern, A misses +-2pp TOST, B's selector x budget
     interaction is ns) — no single one-sentence headline. Decide which claim leads
     before the next pass; Claim B (reinvest beats full-res, selector-agnostic) is the
     least-hedged and most novel result in the paper, currently buried as claim 3 of 3.
-->


**[Figure 1 — protocol schematic, ~page 2 / start of Method, not page 1.]** File: `figures/fig_protocol.pdf`. (a) Score once: 1 fps pool → LongCLIP stem → cached scores → all T1 rules. (b) Same tokens: 8 full-res vs 16 @ ~50%. Draw script: `slm-lab/scripts/make_protocol_figure.py`.

Existing plots (`video-understanding/figures/`):
- `fig_protocol` → **Method, Figure 1** (score once; 8 full vs 16 @50%).
- `fig2_long_video_effect` → **Results / T1**.
- `fig_claim_a` / `fig_claim_b` → **Claims A and B**. Combined leftover: `fig3_budget_allocation`.
- `fig1_residual_inert` → **Mechanism** (after Claim B). On-image "Fig. N" labels stripped so LaTeX owns numbering.
- `fig4_negative_sweep` → **Appendix: variant sweep**.

---

**Title.** A Controlled Study of Frame Selection and Visual Token Budget in Video MLLMs

## Abstract

Video MLLMs spend a fixed visual-token budget on which frames they keep and how finely each frame is encoded. Uniform sampling is still the default; adaptive selectors (top-k, AKS, FOCUS, LDDR) report large gains over it, but the tables are hard to compare because the scorer, the prompt, and the answerer all change from paper to paper. We freeze a LongCLIP stem-only scorer and one evaluation harness, and re-run the main training-free rules at k=8 on LongVideoBench, Video-MME, and LVBench.

Under that control, classical Orthogonal Matching Pursuit (OMP) is a strong selector: +5.7 / +5.9 / +11.8 pp over uniform on the three benchmarks, and a statistical tie with LDDR's stage-1 MinMax rule. It is a 1993 algorithm used here as a cheap reference, not a method we introduce. On Video-MME the gap over plain top-k is large (+5.2 pp); on LongVideoBench it is small.

Holding the same timestamps, cutting each frame to about half the visual tokens usually matches full-resolution accuracy (LongVideoBench / LVBench; Video-MME is slightly down, −0.44 pp). Reinvesting that budget into more frames at lower resolution (k=16 at ~50%) beats eight full-resolution frames by +2.4 pp on LongVideoBench's long bins and +3.0 pp on LVBench, including when the extra frames are uniform rather than OMP-selected. Primary numbers are Qwen3-VL-8B; InternVL3 and GPT-5-mini are used as checks. We report paired deltas, not leaderboard scores.

## 1. Introduction

A video MLLM does two things before it answers: it keeps a small set of frames, then encodes each at some resolution. Both decisions spend the same budget — visual tokens. The default is still uniform sampling. Adaptive selectors sit on top of that default — CLIP top-k, AKS, FOCUS, LDDR's stage-1 DPP, and a growing set of segment-then-diversify rules — and, separately, a line of work that prunes or reallocates tokens inside the frames that were kept. Those papers often report large gains over uniform sampling. The tables are hard to read against each other: each one brings its own scorer, prompt template, and answerer, so a gap on LongVideoBench in one paper is not the same experiment as a gap on Video-MME in another.

This paper freezes the comparison. We use one LongCLIP scorer, the question stem only (answer options never enter selection), one evaluation harness, and k=8 as the primary frame budget. We re-run uniform sampling, top-k, AKS, FOCUS (as a dense-score replay), classical Orthogonal Matching Pursuit (OMP), and LDDR's MinMax stage-1 selector on LongVideoBench, Video-MME, and LVBench. The primary answerer is Qwen3-VL-8B; InternVL3 and GPT-5-mini are checks, not a second leaderboard.

The first result is the matched table. Under that control, OMP is a strong selector: clearly above uniform on long video, and a statistical tie with LDDR's stage-1 rule. OMP is Pati et al.'s 1993 greedy pursuit, used here as a cheap reference row — not an algorithm we propose. The rest of the paper then holds timestamps fixed and varies only how many tokens each frame is allowed to cost. Cutting resolution to about half the tokens usually matches full-resolution accuracy. Spending the saved tokens on more frames at lower resolution (k=16 at ~50%) beats eight sharper frames at the same cost, including when those extra frames are uniform rather than OMP-selected. How large the original selection gain is still depends on which answerer reads the frames and which benchmark asks the question; we report that after the main tables, not as the lead.

**Contributions.**

- A matched selector comparison at k=8 under one LongCLIP stem scorer: OMP vs uniform, top-k, AKS, FOCUS replay, and LDDR stage-1, with paired deltas on three long-video benches.
- On fixed timestamps, about half the visual tokens usually preserve accuracy (LongVideoBench, LVBench); Video-MME is the exception (−0.44 pp).
- At matched token cost, more frames at lower resolution outperform fewer full-resolution frames, including on uniform picks — so the reinvest effect is a property of the budget, not of OMP.

**[Figure 1, start of Method ~page 2 — `figures/fig_protocol.pdf`.]** Same LongCLIP scores. 8 full-res vs 16 @ ~50%.

## 2. Related work

**Frame selection.** Uniform sampling is the default in most video-MLLM pipelines. Query-aware alternatives score frames against the question with a frozen dual encoder — CLIP, Long-CLIP, SigLIP, or BLIP-2 ITM — and keep the top k. Structure on top of those scores comes in a few recurring forms. AKS adds temporal coverage by recursively splitting the timeline. FOCUS treats short clips as arms in a combinatorial bandit. LDDR, AdaRD-Key, and Adaptive Greedy use DPP- or submodular-style diversity. A recent pattern is segment, then anchor, then diversify: QCA allocates a per-segment quota from query relevance and content deviation; EFS segments with DINO / PySceneDetect and refines with adaptive MMR.

Those papers report clear gains over uniform sampling. They also illustrate the comparison problem. Published tables mix scorers (BLIP-2 for QCA, CLIP for EFS, LongCLIP for LDDR's standardization, and whatever each cited baseline used), and they rarely report paired tests or an equal-token control. QCA and EFS moreover operate at much larger k than we do (QCA's headline is 64 frames), a regime where our own curve has already flattened. Under a matched LongCLIP scorer, coverage forcing and MMR are the two mechanisms that lose hardest against OMP in our ablation. That does not refute their stack results. It does mean those numbers cannot settle which *selection rule* is responsible.

We use Orthogonal Matching Pursuit as the oldest cheap member of the diversity family — not as a contribution. LDDR already uses a related residual / orthogonalization idea in its *allocation* stage; we use OMP only to pick timestamps.

**Token budgets and pruning.** The second lever is how many tokens each retained frame may cost. LDDR is the closest prior that couples both levers: a Group-DPP importance score drives resolution, giving more tokens to informative non-redundant frames. AdaCodec learns a retain/compress/drop code, but needs training. A parallel line prunes or reallocates tokens after frames are chosen — AdaptToken uses answerer entropy for group budgets; Vista-LLM and MoPrune prune query- or motion-aware tokens before the LLM; AdaAlloc (concurrent) splits a fixed budget between a low-resolution global overview and high-resolution local evidence. Those systems optimize *where* tokens go under a cap. Claims later in this paper ask a narrower question: holding timestamps fixed, how much does spatial fidelity matter, and does buying more frames with the saved budget help? We do not claim to outperform AdaAlloc-style allocators; a reconstruction of LDDR's stage-2 importance already beats our residual-proportional rule.

**Positioning.** On published absolute accuracy, LDDR remains ahead, and we do not challenge that — our LDDR row is stage-1 selection only, not their full dynamic-resolution pipeline. What we claim is a measurement under matched controls: which training-free rule wins when the scorer is shared, whether a token cut on fixed timestamps is free, and whether extra frames at lower resolution beat fewer sharp ones at the same cost.

## 3. Method

### 3.1 Scorer and frame pool

Videos are decoded at 1 fps. Each frame and the question *stem* are encoded with Long-CLIP; answer options never enter the scorer. That choice matches CLIP-family practice in AKS and avoids a documented failure mode: when the scorer reads the fused “question + options” string, ranking can anti-correlate with the stem and move about 40% of the picked frames on LongVideoBench-600 s. Embeddings are cached once. Every selection rule in this paper — including the published ones we re-implement — consumes that same score vector, so Table T1 isolates the *rule*, not the originating paper's encoder.

### 3.2 OMP as the reference selector

Let E be L2-normalized frame embeddings and q the stem embedding. Classical OMP picks greedily by residual correlation:

b_r = argmax_i  e_iᵀ q_{r-1},    q_r = q_{r-1} − Proj_span{e_{b1}…e_{br}}(q_{r-1})

for r = 1…k, and emits those timestamps. Training-free; O(kNd) with Gram–Schmidt.

The sparse-recovery reading — that successive picks reconstruct q from a dictionary of frames — does not hold in this frozen dual-encoder space, and we do not claim it. Direct traces show the residual norm falls by only ~3.3% over eight picks, and residual-correlation is numerically inert after pick ~5 (**plot:** `figures/fig1_residual_inert.pdf`, shown later with mechanism results). What the projection step does here is *decorrelation*: after pick r, frames collinear with an already-selected frame are suppressed. We keep the OMP formula because it is the cheapest exact way to obtain that suppression (one Gram–Schmidt step, no tuned diversity coefficient), not because the residual is being driven to zero.

### 3.3 Resolution on fixed timestamps

Selection is frozen to a timestamp set (OMP seconds unless we say otherwise). Per-frame resolution follows residual mass at pick time (resprop / D):

w_r = s_r / s_1,    frac_r = floor + (1−floor) w_r^γ

with γ chosen so the mean visual budget hits a target (0.53 or 0.50). Frames are resized with PIL; for Qwen3-VL we disable the processor's own resize. We also run a *flat* split at the same mean fraction and a reconstruction of LDDR's stage-2 importance (“GD”) on identical OMP-8 frames, so allocation *shape* is tested rather than assumed. GPT-5-mini compression is a pixel ratio after a max_side cap, not a Qwen token count; we report accuracy deltas only and never equate the two.

**[Later: fig3_budget_allocation.pdf — Claim A curve + Claim B bars. Recaption required.]**

### 3.4 Matched-scorer baselines

All of the following use the LongCLIP stem scores above, at k=8 unless noted.

- **Uniform:** k frames at the stock stride of the lmms-eval LongVideoBench loader.
- **Top-k:** the k highest cosine scores, no diversity step.
- **AKS:** the ADA coverage rule, ported onto our scores.
- **FOCUS★:** Algorithm 2 of FOCUS *replayed* on the dense LongCLIP vector. Test of the FOCUS *schedule*, not a reproduction of the paper's budgeted ITM scorer (Appendix).
- **LDDR-select:** MinMax (linear DPP) stage-1 only, not the full Group-DPP + 1024-tokens-per-frame pipeline.

Answerers and decoding: see §4.

## 4. Experimental setup

We evaluate on three long-video QA benchmarks: LongVideoBench (official val, n=1337, duration bins 15/60/600/3600 s), Video-MME (short/medium/long, n=2700), and LVBench (n=1549). The primary answerer is Qwen3-VL-8B-Instruct through lmms-eval, greedy decoding, temperature 0, subtitles off. InternVL3 at 2B and 8B is a second open family; GPT-5-mini is a closed check (LDDR Appendix F.3 prompt, effort=low, no system message, no timestamps in the prompt). All selector rows share the LongCLIP stem scores of §3; k=8 is the primary budget and k=16 is used only for the equal-token reinvest.

| | |
|--|--|
| Benches | LongVideoBench (1337) · Video-MME (2700) · LVBench (1549) |
| Primary answerer | Qwen3-VL-8B, lmms-eval, greedy, subs off |
| Other answerers | InternVL3 2B / 8B; GPT-5-mini |
| Scorer | LongCLIP, stem only |
| Primary k | 8 (selector table); 16 for the equal-token reinvest |

We report accuracy and paired McNemar tests on per-question correct/incorrect outcomes. Claim A (half tokens ≈ full resolution) is an equivalence claim, so we also report 90% intervals and two one-sided tests rather than treating a non-significant McNemar as proof of a null. Environment: headline selector and Claim A numbers are from one GPU family; the allocation-shape and some reinvest controls sit on a second machine and are not mixed into the same table (see Limitations).

## 5. Results

### 5.1 Matched selector comparison at k=8

Table T1 is the comparison the rest of the paper sits on. Uniform, top-k, AKS, FOCUS★, OMP, and LDDR-select all read the same LongCLIP stem scores. Under that control, OMP is a strong selector: +5.7 / +5.9 / +11.8 pp over uniform on LongVideoBench, Video-MME, and LVBench (McNemar p=5.9e-4 at LongVideoBench 3600 s; p=5.1e-10 and p=9.5e-17 on the other two benches). LDDR's MinMax stage-1 rule is a statistical tie with OMP wherever we tested it (p=.70 / p=.57). The gap over plain top-k is small on LongVideoBench (+2.0 pp, ns at most bins) and large on Video-MME (+5.2 pp, p=4.4e-11) and LVBench (+3.2 pp, p=.0075).

| Method | LongVideoBench | Video-MME | LVBench |
|--|--|--|--|
| Uniform | .5654 (−5.69) | .5637 (−5.85) | .3454 (−11.81) |
| Top-k | .6028 (−1.95) | .5704 (−5.18) | .4319 (−3.16) |
| **OMP (ref.)** | **.6223** | **.6222** | .4635 |
| AKS | .6021 (−2.02) | .5785 (−4.37) | .4280 (−3.55) |
| FOCUS★ | .5819 (−4.04) | .5578 (−6.44) | .3983 (−6.52) |
| LDDR-select | **.6320 (+0.97)** | .6193 (−0.29) | **.4693 (+0.58)** |

FOCUS★ = dense LongCLIP replay of the FOCUS schedule, not the paper's budgeted ITM scorer. LDDR-select = stage-1 MinMax only. Parentheses = Δ vs OMP in pp.

**[Figure — `figures/fig2_long_video_effect.pdf`.]** The selection gain on LongVideoBench is a long-video effect. At 15 s, uniform, top-k, and OMP are identical (.7249): there is no selection problem. The OMP–uniform gap is significant only at 600 s and 3600 s. LDDR-select tracks OMP.

At 15 s the three cheap rules coincide; the gain appears at 600 s (+7.8 pp, p=.0022) and 3600 s (+7.5 pp, p=5.9e-4). Selection is worth something where the 1 fps pool is much larger than k, not as a general “better frames” story. Slicing the same runs by LongVideoBench’s official question-type tags (T* vs the rest) does *not* isolate a temporal regime; the 600 s and 3600 s bins even reverse, and the interaction is non-significant (Appendix).

On Video-MME the picture is sharper and less convenient: top-k, AKS, and FOCUS★ are all non-significant vs uniform; only OMP and LDDR-select separate. “Any query-relevant rule beats uniform” is a LongVideoBench / LVBench result, not a Video-MME one. We do not have a mechanism for that split and do not average it away.

GPT-5-mini, fed the same picks, preserves the ranking: OMP .6200 / .6880 on LongVideoBench / Video-MME vs LDDR-select .6305 / .6874, again a tie, with AKS and FOCUS★ behind. LVBench GPT numbers exist only as a summary (the per-item file was not archived); we treat that ranking as qualitative. Absolutes differ across answerers; the claim is the matched delta.

### 5.2 Claim A: half the tokens on fixed timestamps

We now freeze the timestamps (OMP-8 unless noted) and change only per-frame resolution. Table T2 reports D@53% vs full resolution. On LongVideoBench and LVBench the cut is a wash (+0.30 / +0.39 pp). GPT-5-mini on LVBench is the same (+0.26 pp, p=.84, n=1549) under a pixel-ratio of about 0.70, which is *not* a Qwen token count. Video-MME is the exception: −0.44 pp. We do not average that cell away, and we do not yet have a paired McNemar on those Video-MME compression files.

| Answerer / bench | full | D@53% | Δ (pp) |
|--|--|--|--|
| Qwen / LongVideoBench | .6223 | .6253 | +0.30 |
| Qwen / Video-MME | .6222 | .6178 | −0.44 |
| Qwen / LVBench | .4635 | .4674 | +0.39 |
| GPT-5-mini / LVBench | .4900 | .4926 | +0.26 (p=.84) |

GPT LVBench is paired McNemar on recovered per-item files (n=1549).

A non-significant McNemar does not prove two arms are the same, so TOST is on the paired difference. Equivalence at α=.05 holds if the 90% CI sits inside a stated margin. On pooled LongVideoBench (n=976) the Qwen cut is equivalent at ±3 pp (p=.0022) and misses ±2 pp (p=.054). InternVL3's harsher 3× tile cut on the same videos clears ±2 pp (p=.022). Per-bin cells at n≤564 cannot support a tight margin; we do not claim one. The ±3 pp band was chosen after the runs, not pre-registered — we report ±2 / ±3 / ±4 so the reader can apply their own.

| Arm | n | Δ (pp) | 90% CI | ±3 pp |
|--|--|--|--|--|
| Qwen, LVB pooled, D@53 | 976 | +0.72 | [−0.60, +2.03] | equiv (p=.0022) |
| InternVL3-8B, 3× cut, pooled | 976 | −0.10 | [−1.66, +1.45] | ±2 also equiv |

**[Figure — `figures/fig_claim_a.pdf`.]** LongVideoBench 3600 s, fixed OMP-8 timestamps. Eight budget points from 32% to 100% sit inside the full-resolution 95% interval. The flat-split control at 50% lands on D@50: *our* residual-proportional rule does not beat a uniform half. That is not a claim that no allocation rule matters (LDDR stage-2 reconstruction does; later section).

The same half-budget cut on *uniform* timestamps is also a wash in a same-environment pair; that control currently lives on a second GPU stack and is not mixed into T2 (Limitations).

The defensible statement of Claim A is equivalence within ±3 pp on LongVideoBench and LVBench, not identity, and not “free on every benchmark.”

### 5.3 Claim B: spend the saved tokens on more frames

Table T3 spends the Claim A surplus on extra frames at lower resolution: k=16 at ~50% vs k=8 full. The two arms are matched on visual tokens (measured ratio 0.984–0.996); the reinvest arm is slightly *under* budget, so a gain is conservative. On LVBench the lift is +3.04 pp (p=9e-4). On LongVideoBench overall it is +2.24 pp; pooling the two long bins where selection exists (n=976) gives +2.36 pp (p=.0346), same sign in both bins. Video-MME is +1.56 pp. GPT-5-mini on LVBench tracks Qwen (+2.39 pp, p=.039).

| Answerer / bench | k=8 full | k=16 @50% | Δ (pp) |
|--|--|--|--|
| Qwen / LVBench | .4635 | .4939 | +3.04 (p=.0009) |
| Qwen / Video-MME | .6222 | .6378 | +1.56 |
| Qwen / LongVideoBench | .6223 | .6447 | +2.24 |
| GPT-5-mini / LVBench | .4900 | .5139 | +2.39 (p=.039) |

**[Figure — `figures/fig_claim_b.pdf`.]** LongVideoBench 600+3600 pooled. k=16 at ~50% vs k=8 full at token ratio ≤1. +2.36 pp, p=.0346.

On LVBench, sixteen *full*-resolution frames (twice the tokens) are .4861 vs .4635 for eight full — +2.26 pp — while sixteen at half resolution is .4939. The extra frames carry the gain, not extra pixels. A selector × budget interaction on LongVideoBench is +1.02 pp and non-significant (p=.4435); uniform picks also reinvest at +1.13 pp (ns). With 206 discordant pairs the test only rules out a *large* interaction, so this is absence of evidence that OMP is required, not proof that the selector never matters. Claim B is therefore stated as a property of the budget. Those interaction numbers currently sit on a second GPU stack (Limitations) and are not mixed into T1.

### 5.4 Why OMP does not reconstruct the query

If the Gram–Schmidt step were sparse recovery, successive picks would drive ‖q_r‖ down and residual correlation would keep ranking frames. Neither happens. On LongVideoBench-600 s the residual fraction ‖q_r‖/‖q_0‖ is 0.972 after pick 1 and 0.967 after pick 8 — about 3.3% of the query is ever removed, and pick 1 accounts for 85.6% of that sliver. Residual cosine with the chosen frame is already ~0 by pick 8. The dictionary is coherent (mean within-video pairwise cosine 0.714) and the query is nearly orthogonal to it (mean 0.175, sd 0.022; max 0.231): all of the ranking signal sits in a 0.06 band on top of a constant. That is a bound on this LongCLIP stem space, not a failure of OMP's formula.

**[Figure — `figures/fig1_residual_inert.pdf`.]** (a) Residual correlation dies; residual *norm* barely moves. Trace logged only at picks 1, 8, 16; line is piecewise linear; “inert by pick ~5” is from a separate denser check. (b) Summary statistics of the two cosine distributions (mean ± sd; max marked), not histograms.

Frame-frame geometry makes the same point for diversity terms. The raw Gram of the 1 fps pool has participation ratio 1.98; the top eigendirection carries 70.5% of the mass on *every* video in the bin — a shared cone of the encoder, not clip content. A log-det rule choosing eight frames from a rank-2 cloud is over-determined. Setting a DPP's query-weight β=0 (pure volume, query-blind) on the 600 s bin is indistinguishable from uniform (+0.97 pp, p=.75) and significantly worse than OMP (−6.80 pp, p=.0072). Any β≥0.5 jumps onto a plateau that is a statistical tie with OMP; none of the diversity knobs in the coverage-gated variant sweep (LDDR-select, α-orthogonalization, residual floors) significantly beat OMP in either long bin (appendix figure `fig4_negative_sweep.pdf`). OMP and greedy DPP-MAP are not the same algorithm — they coincide at the first pick only — so their empirical tie is a result: once residual ranking is flat, “relevant and mutually decorrelated” subsets look alike.

### 5.5 Same picks, different answerer

InternVL3 re-runs the *same* OMP vs uniform timestamps, the same 24-tile budget, and the same harness. A check that T1 is not a Qwen artifact, not a second leaderboard.

| Benchmark | n | IVL3-2B Δ | IVL3-8B Δ |
|--|--:|--|--|
| Video-MME short | 900 | +6.00 (p=1.2e-4) | +5.44 (p=1.9e-4) |
| Video-MME medium | 900 | +3.33 (p=.046) | +0.33 (p=.891) |
| Video-MME long | 900 | +4.22 (p=.0062) | +0.11 (p=1.000) |
| LVBench | 1549 | +9.04 (p=7.2e-11) | +10.14 (p=1.2e-13) |
| LongVideoBench val | 1337 | +4.71 (p=1.6e-4) | +6.51 (p<10^{-4}) |

The 2B model gains on every Video-MME duration bin. The 8B model in the same family gains only on short (+5.44 pp); medium and long are real nulls (+0.33 / +0.11 pp) with 213 and 191 discordant pairs — the arms saw different frames and still tied on accuracy. That is not a general “8B saturates” claim: the same 8B posts its largest selection gain anywhere on LVBench (+10.14 pp). The effect is an interaction of benchmark and capacity. Duration even flips sign: InternVL3-8B's LongVideoBench gain grows with duration, its Video-MME gain decays. A paper that stopped at InternVL3-8B / Video-MME long would have published a null; the same paper stopping at 2B would have published a positive. Claim A's 3× tile cut and Claim B's reinvest still replicate on this family.

## 6. Discussion

If the visual-token budget is fixed, the lever that moved accuracy was how many timestamps the answerer sees, not how sharp each frame is. On frozen OMP-8 picks, cutting each frame to about half the tokens matches full-resolution accuracy within a ±3 pp TOST band on LongVideoBench and LVBench — not identity, and not “free on every benchmark”: the same cut is −0.44 pp on Video-MME, and pooled Qwen misses ±2 pp. Spending the surplus on more frames at lower resolution (k=16 at ~50%) beats eight full-resolution frames at matched cost, including when the extra frames are uniform rather than OMP-selected. That is the systems move: a 50% cut is how you buy coverage of the timeline, not a claim that pixels never matter. We also do not claim a new selector, or that we beat LDDR's full pipeline. LDDR-select ties OMP at k=8; Claim B does not need OMP; the residual is not reconstructing the query.

Selection itself is a long-video problem. Figure 2 already shows the split: at 15 s, uniform, top-k, and OMP are the same accuracy; the gain lives in the 600 s and 3600 s bins, where k=8 is sparse. LongVideoBench's public taxonomy is useful as a stress test, not as a finding we own. Coverage-style papers often target temporally referred questions. Official T* tags looked, at 600 s, like those items do not benefit from selection; at 3600 s the sign reversed, and the interaction is non-significant in both long bins (appendix). Do not build a temporal-versus-scene router on one duration slice. Duration is the split we would trust; wording is not.

A single benchmark–answerer cell is not a portable conclusion either. The same OMP timestamps that lift InternVL3-2B on every Video-MME duration bin leave InternVL3-8B flat on medium and long, while that 8B gains +10.14 pp on LVBench. Selector ranking is bench-dependent in the same way: on Video-MME, top-k, AKS, and FOCUS★ do not separate from uniform; only OMP and LDDR-select do. The residual / rank-2 Gram account was measured on LongVideoBench, and we do not treat “any query-relevant rule beats uniform” as a universal statement. A paper that stopped at one cell would have published a positive or a null from identical frames. Report at least two answerer scales, and more than one long-video bench, before claiming that selection works — or that it does not.

## 7. Conclusion

Under one LongCLIP stem scorer and one harness, classical OMP is a strong k=8 row against uniform on long video, and a statistical tie with LDDR's stage-1 rule. It is Pati et al.'s 1993 algorithm used as a cheap reference, not a method we introduce. Holding timestamps fixed, about half the visual tokens usually match full-resolution accuracy on LongVideoBench and LVBench: a bounded equivalence, not a free lunch on every bench. Reinvesting that surplus into more frames at lower resolution beats fewer sharper frames at the same cost, including when the extra frames are uniform. If the visual budget is fixed, prefer a few decorrelated query-relevant frames and buy more of them by cutting resolution — and report at least two answerer scales, on more than one long-video bench, before claiming that selection works, or that it does not.

## 8. Limitations

*Comparison, not a full-stack bakeoff.* FOCUS★ replays Algorithm 2 of Zhu et al. on our dense LongCLIP scores. It is a test of that schedule, not a reproduction of their budgeted ITM scorer, and a FOCUS-paper underperformance claim would need their protocol. InternVL3 checks OMP against uniform only; OMP vs top-k / AKS / LDDR-select remains Qwen-only. The GD allocation arm reconstructs LDDR stage-2 importance from the paper, not their code: sign is likely robust, magnitude is not, and that result is not mixed into T1–T3.

*Claim A is a bound, and Video-MME is open.* TOST supports ±3 pp on pooled Qwen LongVideoBench (n=976); it misses ±2 pp (p=.054). Those margins were chosen after the runs. Video-MME compression is −0.44 pp and has no paired test. The 3600 s k=32 arm was discarded after a sharding bug and not re-run, so the long-bin budget curve stops at k=16.

*Mechanism is this encoder, mostly this duration.* Residual traces, Gram participation ratio ~2, and β=0 DPP are LongVideoBench-600 s in frozen LongCLIP stem space. The 3600 s bin has no cached LongCLIP frame embeddings, so the rank-2 measurement was not repeated where selection matters most. We do not claim the geometry for SigLIP, BLIP-ITM, or video-native scorers. Video-MME's selector ranking (top-k non-significant vs uniform) is already the counterexample inside this paper.

*Harness and accounting.* Headline T1 / Claim A / Claim B tables sit on one GPU family; allocation-shape and some reinvest controls sit on a second (different architecture and PyTorch build). Measured drift is a few tenths of a point. Those control cells stay out of T1–T3. GPT-5-mini compression is a pixel ratio after a max_side cap, not a Qwen token count. LVBench GPT Claim A/B pairs were recovered (n=1549); the T1 selector ranking for that answerer remains summary-archived. No MLVU. The scorer is vision-only with subtitles off, so subtitle-anchored questions are unreachable by every arm.

## 9. Ethical Considerations

All experiments run existing public video-QA benchmarks (LongVideoBench, Video-MME, LVBench) through frozen open or API models. We do not collect new human annotations, personal data, or videos. Frames used at inference are already in those evaluation sets; this paper does not redistribute video files. Selector scores are computed from question stems only (answer options never enter the scorer). No dual-use tooling is released beyond the measurement harness.

## Appendix A. FOCUS★ is a schedule replay

Every T1 row, including FOCUS★, consumes the same 1 fps LongCLIP stem-score vector. We mark the row with a star because that is not how Zhu et al. spend compute.

FOCUS selects frames by treating 16 s clips as arms in a combinatorial bandit (their Algorithm 2 / §2.4). The official story is *budgeted online scoring*: most frames are never scored; ITM pulls buy information. Our study already scores every frame, so an “arm pull” only *reveals* an already-computed cosine to the bandit. The schedule is theirs; the cost story is not, and the scorer is LongCLIP with a sigmoid map onto [0,1], not BLIP-2 ITM. Hyperparameters match their defaults: clip length 16 s, q=3 coarse pulls per clip, α=0.25, pull budget 0.5 of the pool, m≈k/4 surviving clips. Unobserved frames inside a clip remain eligible; we interpolate reward from the nearest observed frame rather than RBF (their public code is richer on that point). The RNG is seeded per item so the row is deterministic.

Dropping FOCUS from T1 would hide a negative under a matched scorer (−4.0 / −6.4 / −6.5 pp vs OMP). Presenting it without a star would overclaim fidelity. We do not claim the FOCUS paper fails under its ITM protocol.

---

*Closed 2026-08-15. arXiv manuscript: `paper/main.pdf` (preprint). No GPU re-run required for this version. Optional leftover (not blocking): torch 2.6 vs 2.11 on one L40S, same `modal_uniform16.py` image — do not H100/A40 “for quality,” do not write RunPod unless it ran there.*
