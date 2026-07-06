# Video intelligence — landscape briefing

**Compiled:** Jul 3, 2026 · from web research (sources at bottom)
**Purpose:** understand how AI actually "decodes"/represents video, what the real systems do,
and what that means for our video-skimmer (evidence-compression) track. Written to be honest
about where our idea sits in a crowded field.

---

## 0. TL;DR

- **How video reaches a model:** sample frames (default **1 fps**) → vision encoder → visual
  tokens → projector → LLM; audio → tokens; timestamps interleaved. Token cost grows **linearly
  with length**, so a 1-hour video ≈ **~1M tokens** at default settings — compression isn't a
  nice-to-have, it's forced.
- **Two system families:** (1) **video-native embedding + retrieval** (TwelveLabs Marengo/Pegasus)
  = the real "Exa for video"; (2) **frame-based VLMs** (Gemini/GPT/Qwen-VL) that choke on length.
- **Our "video skimmer" = keyframe selection**, which is an **active, crowded research area**
  (FOCUS, Q-Gate, VSI, VideoAgent, Video-o3…). Selecting query-relevant frames is *not* a novel
  concept. We can't claim we invented it.
- **Biggest blind spot in our harness: audio/transcript.** Adding subtitles gives **+10.1%** on
  long-video QA (Gemini 1.5 Pro: 67.4% frames-only → 77.4% with subs). A frames-only skimmer
  fights one-handed.
- **Our only honest, defensible lane:** the cheap **compression/selector layer as an agent tool
  for videos you don't want to pre-index** (e.g. a one-off video an agent hits while browsing) —
  plus the "compression can *beat* full-dump" (context-dilution) thread, which is our most original
  observation and is corroborated by the literature's "context overload" finding.

---

## 1. How a model actually "decodes" video

The default pipeline across frontier models:

```
video → sample frames (≈1 fps) → vision encoder (ViT/SigLIP) → visual tokens
       → multimodal projector → interleave with [timestamp] + [audio] tokens → LLM
```

**Gemini's concrete numbers (a good reference for the whole industry):**

| Thing | Value |
|---|---|
| Default frame sampling | **1 fps** (configurable; <1 fps for static lectures, higher for fast motion) |
| Tokens / frame | ~**258** default, **66** at `media_resolution=low`, **70** ("medium", Gemini 3) |
| Audio | processed at 1 Kbps → **32 tokens/sec** |
| Total | ≈**300 tokens/sec** default, ≈**100/sec** at low res |
| Max | up to ~1 hr video, 1M-token context |

**The core problem = linear token growth.** At ~300 tok/s, a 1-hour video ≈ **1.08M tokens** →
blows the context window. This is *why* providers themselves recommend pre-transcoding to 1 fps +
480–720p and dropping `media_resolution` to low. Model-side research attacks the same wall:
- **STORM** (NVIDIA): Mamba temporal projector → up to **8× fewer visual tokens**.
- **GRT / DIVE**: motion-compensated tokenization skips static patches → **sub-linear** token
  growth with fps (borrows key-frame/P-frame ideas from video codecs).

Takeaway: everyone is fighting the token wall. Our external "select fewer frames" approach is one
valid attack on it — but so is the model's own `fps`/`media_resolution` knob, which is our real
baseline to beat, not naive full-dump.

## 2. The two families of "video intelligence"

### 2a. Video-native embedding + retrieval — TwelveLabs (the real "Exa for video")
- **Marengo** = multimodal embedding model. Treats video **holistically** (not as independent
  frames): default **6-second clips** (2–10s configurable), ~1 fps internal, ~256 patches/frame,
  **512–1024-dim multi-vector** embeddings across **visual + audio + text**. Handles up to 4 hr.
- **Any-to-any retrieval:** text / image / audio query → same vector space → returns the specific
  **timestamped clip(s)** with relevance scores. A 3-min video ≈ 46k tokens internally.
- **Pegasus** = video-language model for reasoning/generation: Marengo encoder → video-language
  **alignment** model → LLM decoder. So the "brain" (Pegasus) sits on Marengo's "eyes+ears."
- This is index → retrieve moment → (optionally) generate. It **already is** the product our
  original "Exa for visual" idea described. We will not out-index them.

### 2b. Frame-based VLMs — Gemini, GPT, Qwen-VL, SmolVLM
- Sample frames → tokens → reason. Simple, general, but hit the token wall on long video.
- This is the family our **answerer** (gpt-4.1) belongs to, and the family our skimmer feeds.

## 3. Keyframe / frame-selection research (this IS our track — and it's crowded)

The field has already moved past uniform sampling to **query-guided keyframe selection**. Directly
overlapping with what we're building:

| Work | What it does | Headline result |
|---|---|---|
| **FOCUS** (ICLR 2026) | training-free keyframe selection as a multi-armed-bandit combinatorial exploration; picks query-relevant frames under a token budget | processes **<2% of frames**; **+11.9%** on LongVideoBench for >20 min videos |
| **Q-Gate** (2026) | query-modulated *multimodal* keyframe selection; **single non-iterative pass** (avoids agentic latency); mutes irrelevant modalities | beats SOTA selection baselines on LongVideoBench + Video-MME |
| **VSI** (2025) | Visual–Subtitle Integration for keyframe selection | top-8 frames **+11.5%** over the subtitle baseline on LongVideoBench |
| **VideoAgent / Video-o3 / VideoTemp-o3 / LongVideoAgent** | agentic "localize → clip → answer": LLM iteratively seeks clues, crops segments (VideoCrop tool), stops when evidence is sufficient | mirrors our "agent gets more from video" framing; downside = multiple sequential LLM calls → latency |

**Implications for us (honest):**
1. "Small model picks the frames that matter" is a **well-populated idea**. Our contribution can't
   be the concept. At best it's (a) a *packaging*/tooling contribution, or (b) a specific empirical
   result.
2. The interesting sub-threads we already stumbled onto have names in the literature:
   - our "so400m-384 embedding selector" ≈ retrieval-style scoring (VSI/FOCUS pre-filter stage);
   - our "compression beat full-dump" ≈ the **context-overload** problem Video-MME documents.
3. **Agentic localize-clip-answer** (Video-o3) is the shape most aligned with the user's original
   "give the agent a tool to pull info from video" idea — worth knowing it's the frontier.

## 4. The audio/transcript blind spot (most important honest finding)

Our harness is **frames-only**. The literature is loud that this is the weaker setup:

| Setting | Effect (Video-MME, long videos) |
|---|---|
| frames only | baseline |
| **+ subtitles** | **+10.1%** (Gemini 1.5 Pro: 67.4% → **77.4%**); up to +16.7% multilingual |
| + audio | +12.5% |
| short videos, + subs | only +2.8% (so the win grows with length) |

- MLLMs are **text-centric**: converting audio→text (subtitles/ASR) usually beats feeding raw audio.
- Best-practice pattern (Google field notes): **audio-first transcription as a "temporal anchor,"**
  then visual reasoning grounded on that timeline.
- Caveat that keeps us honest the other way: good benchmarks (Video-MME) **discard questions
  answerable from the text prompt alone** ("blind" filter), so on curated sets the video is still
  required. But in the wild (tutorials, lectures, news), a large fraction of questions *are*
  answerable from transcript alone — which a frames-only system would miss cheaply.

**Action:** add at least a **transcript-only baseline** and ideally a **transcript-first selector**
(ASR → find candidate timestamps → visually confirm a few frames). Frames-only is a known ceiling.

## 5. Benchmarks / datasets to adopt for S1 (replace stitched-synthetic)

| Dataset | Why | Note |
|---|---|---|
| **LongVideoBench** | has **gold keyframe + subtitle annotations** and a subtitle baseline (GPT-4o+Sub ≈ 58%) | best for real `hit@k` (referable evidence), >20 min videos, "certificate length" metric |
| **Video-MME** | comprehensive, modality-ablated | **no** ground-truth frame annotations → can't measure keyframe hit@k on it |
| **MLVU** | multi-task long-video understanding | broad |
| **MME-VideoOCR** | subtitle/OCR + **multi-hop needle-in-a-haystack** | closest to our "needle" framing, but on real video |
| **EgoSchema** | egocentric long QA | shorter "certificate length" than Video-MME |

LongVideoBench is the natural home for our experiments: it gives real videos, real questions, gold
evidence for `hit@k`, and a subtitle baseline to compare our (soon multimodal) selector against.

## 6. Where this leaves our project (honest positioning)

1. **Don't claim to invent frame selection.** FOCUS/Q-Gate/VSI already do query-guided selection
   with strong numbers. Our value is empirical/engineering, not conceptual.
2. **Real baselines to beat are not "full-dump."** They are: (a) the model's own `fps`/
   `media_resolution` knobs, (b) TwelveLabs retrieval, (c) subtitle-based selection (VSI). Our
   `results/` so far only beats full-dump — the weakest baseline.
3. **Defensible lane:** an agent-facing **evidence-compression tool for un-indexed, one-off videos**
   — the case where you *can't* amortize a TwelveLabs index (agent encounters a random video while
   browsing). This ties directly back to the user's original "agent browses the web + video"
   afterthought, and is the one spot where "cheap, no-index, on-the-fly selection" has a real edge.
4. **Most original thread = "compression can beat full-dump" (context dilution).** We saw it twice
   (whispr n=1, paid item 004); Video-MME's context-overload result supports it. If we can show,
   with rigor and n, that top-k selection *systematically* beats feeding everything to a frontier
   model over some input-length threshold — that's a genuinely useful, publishable systems result,
   and it doesn't require out-competing TwelveLabs.
5. **Multimodal is table stakes.** Add transcript before drawing conclusions.

## 7. Suggested next experiments (revised by this research)

- Swap stitched-synthetic → **LongVideoBench** subset (real videos, gold evidence, subtitles).
- Add a **transcript-only** condition and a **frames+transcript** condition to the harness.
- Add **model-knob baselines** (fps<1, media_resolution=low via Gemini) as the honest cost baseline.
- Run the **"less is more" study**: accuracy(full-dump) vs accuracy(top-k) as a function of video
  length; find the crossover where compression starts *helping*.
- (Optional) a **TwelveLabs** condition B for a real retrieval comparison.

## 8. Prior art: coarse-to-fine / hierarchical search (recursive-chunk idea)

The "divide the video into intervals → descend only into the promising chunk → subdivide again"
idea (and its image analogy "find the dog → zoom into the bottom-right quadrant → repeat") is a
**known, published family**. Documented here so we don't reinvent it:

| Work | Domain | What it does |
|---|---|---|
| **V\*** / SEAL (CVPR 2024) | image | *"recursively divide the image into 4 equal patches, assign search-priority scores, recurse into the best until the target is found."* — the exact quadrant idea. Uses an MLLM to produce a "search-cue heatmap" (common-sense guidance) for where to look. |
| **T\*** (CVPR 2025, Stanford) | video | reframes **temporal** search as **spatial**: pack frames into a grid, score cells, discard low-signal regions, **zoom in temporally + spatially** on promising ones, repeat. **3× fewer FLOPs, 4× fewer frames**; plugs into GPT-4o. |
| **VideoTree** (CVPR 2025) | video | query-adaptive **hierarchical tree**; cluster → relevance-score → expand depth only into relevant branches (coarse-to-fine). Training-free; beats GPT-4V on Video-MME long. |

**The small model's role (answers "is a small model reading the coarse level being done?"): yes — that
is the standard design.** In all three, a cheap component does the coarse scan/scoring and the big
model is spent only at the end: V* uses an MLLM to emit search cues, T* uses lightweight detectors/
grounders, VideoTree uses embeddings + captions. "Small model reads each chunk/quadrant to decide
where to go" is the norm, not a gap.

**The failure mode = greedy wrong-branch pruning.** If the first descent picks the wrong chunk, the
answer is pruned and unrecoverable. Standard mitigations: keep **top-b branches (beam)**, allow
**backtracking**, or add a **reflection/verify** step (Video-o3 "reflection", VideoTemp-o3 "refine").

**Draft-verify ("speculative execution") framing.** Speculative *decoding* proper (small draft model
proposes tokens, big model verifies in parallel) is a *token-generation* speedup and does **not**
directly fix branch selection. But its *spirit* — **small model proposes, cheap check verifies,
reject → fall back** — maps cleanly onto the wrong-branch problem: small model drafts the descent
path (or several candidate chunks), a cheap verifier checks whether the chunk actually contains
sufficient evidence, and on reject it backtracks/expands siblings. This is effectively what
beam/reflection methods already approximate; an **explicit cheap draft-verify loop in the no-index,
on-the-fly agent-tool setting** is the less-explored variant.

**Our defensible slice (unchanged by this):** cheap + **no-index** + **transcript-aware**
coarse-to-fine, aimed at the **fine-detail case** — T*'s "zoom in for resolution" directly fixes the
failure we already hit (SmolVLM-500M downscaled frames → couldn't read UI text). Position against
T*/VideoTree as baselines, not against full-dump.

---

## Sources
- Gemini video understanding docs (1 fps, token math, `media_resolution`); Google Cloud "Notes from
  the field" (audio-first transcription as temporal anchor).
- TwelveLabs: "Introducing Pegasus-1" (three-part architecture), Marengo model docs (multimodal,
  512–1024d, 6s clips, 4hr), Marengo+Bedrock+Elasticsearch retrieval blog.
- Long-video / selection research: FOCUS (ICLR 2026, keyframe bandit, <2% frames, +11.9%),
  Q-Gate (query-modulated multimodal selection), VSI (visual+subtitle keyframe selection),
  VideoAgent / Video-o3 / VideoTemp-o3 / LongVideoAgent (agentic localize-clip-answer),
  STORM (Mamba projector, 8× token cut), GRT/DIVE (motion-gated tokenization), DATE (timestamp
  token injection).
- Benchmarks: Video-MME (CVPR 2025; modality ablation, "blind" filter), LongVideoBench (gold
  keyframes + subtitle baseline), MME-VideoOCR (subtitle/multi-hop needle), MLVU, EgoSchema.
- Coarse-to-fine / hierarchical search: V* + SEAL (CVPR 2024, recursive 4-quadrant guided visual
  search), T* (CVPR 2025, temporal-as-spatial grid zoom, 3× FLOPs / 4× fewer frames), VideoTree
  (CVPR 2025, query-adaptive coarse-to-fine tree).
