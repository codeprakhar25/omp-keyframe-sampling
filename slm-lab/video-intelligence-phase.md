# Video-intelligence phase — how we got here + what's next

**Updated:** Jul 3, 2026 · **Scope:** video only (image/GUI track dropped) · **Track:** Fork A first
**Companion docs:** `SPEC.md` (harness design), `FINDINGS.md` (S0–S2b results),
`research/video-intelligence.md` (landscape + prior art).

---

## 1. How we landed here (short narrative)

1. **Original idea:** an "Exa for visual" — give agents a way to pull info out of images/video.
   SLM optional; be honest; agent web-browse could plug in.
2. **Research reframe:** frontier models already understand single images; "Exa for video" already
   exists (TwelveLabs); small VLMs exist (SmolVLM/Moondream). The real unsolved, useful gap is
   **cost/volume** — agents can't afford to feed long video into context. So the thesis became
   **visual evidence compression**: a small model picks the few frames that matter.
3. **S0 harness built + validated** (conditions A=full-dump, C=selector; echo/OpenAI answerers,
   LLM-judge, recall@k). Needle stress test: uniform selection = 0.1667 acc at 6.5x fewer tokens →
   correct STOP; **accuracy == recall@k**, proving selection is the whole game.
4. **S1/S2 on harder/real-ish data:** stitched-GUI needles → S1 "both selectors fail (~0.5)" turned
   out to be a **weak-checkpoint artifact**; `siglip-so400m-384` lifted hit@k to **0.875**. Paid
   end-to-end (n=8): equal accuracy at **13.5x** fewer tokens. Also saw **compression beat full-dump**
   twice (context dilution).
5. **Scope decision:** focus **video only**, drop the image/GUI-screen track.
6. **Landscape + prior-art check:** our "skimmer" = keyframe selection (FOCUS/Q-Gate/VSI); the
   recursive-chunk idea = coarse-to-fine search (V*, T*, VideoTree) — known, works, so our
   contribution must be empirical/positioning, not "we invented selection." Biggest blind spot =
   audio/transcript. Real baselines are NOT full-dump.
7. **Decision:** pursue **Fork A first**, evaluate on **LVHaystack**, answerer **GPT-5.5**,
   transcript **deferred**.

## 2. How frame selection works (the "picking" — not random)

The core operation is **query-driven relevance ranking of frames**. Methods:

| Method | How it picks | Query-aware? | Role |
|---|---|---|---|
| **Uniform** | k evenly-spaced frames | no | deliberately-dumb baseline (scored 0.1667 on needle) |
| **Random** | k random frames | no | sanity floor only |
| **Embedding (SigLIP)** | embed question + each frame into a shared vector space; score by **cosine similarity**; take **top-k** | **yes** | our real selector (`siglip-so400m-384` best) |
| **Model-knob** | provider downsamples (fps<1 / low-res) | no | provider-side cheap baseline |
| **Generative (SmolVLM)** | small VLM judges each frame's relevance | yes | tested, lost to embedding, heavier |
| **Hierarchical (Fork B)** | score chunks → descend into best → zoom in | yes | coarse-to-fine (T*/VideoTree), future |

**Embedding selection = same idea as Exa's text search, but for images:** query → vector, each
frame → vector, rank by closeness, keep the top-k. That ranking quality is what decides whether the
needle frame survives (and we proved answer accuracy tracks it 1:1).

## 3. The two forks

**Fork A — Measurement/systems result (CHOSEN FIRST; lower risk, most original).**
> Study **"when does compression beat full-dump?"** — find the video-length crossover where sending
> a few well-selected frames matches or *beats* dumping everything into the frontier model, and where
> full-dump hits the token wall. Honest and publishable even if the answer is "compression only saves
> cost, not accuracy." Doesn't require out-engineering T*/VideoTree.

**Fork B — Method (later, higher risk).**
> Build the **transcript-first coarse-to-fine + high-res zoom + draft-verify/backtrack** selector and
> benchmark it against T*/VideoTree on LVHaystack. Worth it only if Phase 0 shows our cheap selector
> is already competitive. Note: "draft-verify/backtrack" borrows the *spirit* of speculative
> execution (small model proposes chunk, cheap verifier confirms, reject→backtrack) — NOT speculative
> *decoding* (that's a token-gen speedup, irrelevant to branch selection).

## 4. Chosen path: Phase 0 + Fork A on LVHaystack

**Benchmark: LVHaystack** — long videos with frame-level gold "needle" labels; the exact bench
T*/TStar report on → direct apples-to-apples with prior art; matches our needle framing.

**Answerer: GPT-5.5** (frames-as-images; OpenAI has no native-video input — that's confirmed, the
"native video" marketing is video *generation*). 1M context + `detail:low` (~85 tok/frame) → up to
~1,500 frames ≈ **~25 min at 1 fps in one request**. So full-dump is credible up to ~25 min; beyond
that it breaks — and that break is part of the finding.

**Transcript: deferred** — so Phase 0 full-dump = "all visual frames," not "everything" (audio lost).
Add transcript when we move to LongVideoBench.

### Phase 0 (shared): honest baselines on real data
- Pull a small LVHaystack subset (~40 questions), **binned by video length** (~2 / 8 / 20 / 40+ min).
- Run per item: **full-dump**, **uniform-k**, **model-knob** (fps<1 / low-res), **flat top-k (so400m)**.
- Metrics: **hit@k** (vs gold frames), **QA accuracy**, **input tokens**, **latency**.

### Fork A experiment
- Plot **accuracy(full-dump) vs accuracy(top-k)** across length bins → find the **crossover** and the
  **token wall**.
- Hypothesis: beyond length L*, `top-k ≥ full-dump` on accuracy AND far cheaper.

### Honest baselines reminder
Our earlier `results/` only beat **full-dump** (the weakest baseline). The real bars are: model-knob,
subtitle-based selection (later), TwelveLabs (optional Fork-A+ condition B), and T*/VideoTree numbers.

## 5. Cost / practical notes
- ~40 Q × full-dump (~127K tok) ≈ ~5M input tokens ≈ **~$25**; top-k conditions are pennies.
- Per-request frame ceiling is bounded by **API tier TPM** (old gpt-4o "39 images → 429" problem).
  Build with `detail:low` + retry/backoff + frame cap to stay robust.

## 6. Status / open items
| Item | Status |
|---|---|
| Scope = video only | decided |
| Fork A first, Fork B later | decided |
| Benchmark = LVHaystack | decided |
| Answerer = GPT-5.5, frames-as-images | decided |
| Transcript path | deferred to LongVideoBench phase |
| Phase 0 dev (LVHaystack subset + baselines + GPT-5.5 answerer + crossover output) | **not started — next** |
| Confirm API tier / TPM (frame ceiling per request) | open |
