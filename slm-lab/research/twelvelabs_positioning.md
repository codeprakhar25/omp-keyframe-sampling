# TwelveLabs positioning — benchmarks, model classes, capability vs ours

**Compiled:** Jul 6, 2026 · from TwelveLabs' own docs + blog (sources at bottom).
**Purpose:** honest related-work / positioning for the paper. What TwelveLabs' commercial
video models actually claim, on which benchmarks, and how our cheap per-frame selector maps
against them — including where ours *could* extend. Written skeptically: separate marketing
from independently-verifiable, and separate "their benchmark" from "our task".

We ran their **current flagship, `marengo3.0`** (confirmed in the run logs: `index … (marengo3.0)`),
so the parity result below is against the newest model, not a stale version.

---

## 1. Two model classes (one foundation model, two heads)

| | **Marengo** (what we probed) | **Pegasus** |
|---|---|---|
| Class | Embedding / **retrieval-search** | **Generative** video-to-text |
| Job | any-to-any search (text/image/video/audio → clips), embeddings | summarize, caption, QA, segment into timestamped JSON |
| Current version | **3.0** | **1.5** (1.2 still available, analysis-only) |
| Max video length | **4 h** | **2 h** |
| Notable specs | 512-dim embeddings, 500-token queries, "fine-grained" (objects ≥10% of frame), counting | 261,120-token context window; ~**80B** params (orig Pegasus-1); URL/asset/base64, no pre-index |

Marengo is the retrieval engine — the correct analog to **our SigLIP selector**.
Pegasus is the answerer class — the analog to **our gpt-5.5 answerer**, not our selector.

---

## 2. Benchmarks they claim (and the catch)

**Marengo 2.7** (Dec 2024) headline: *"state-of-the-art across 60+ benchmark datasets, +15% over 2.6."*
Their self-run eval framework, broken out:

| category | Marengo 2.7 | dataset(s) | metric |
|---|---|---|---|
| text→visual | 74.9% | MSR-VTT (1000 clips) + COCO | recall |
| motion | 78.1% | Something-Something-v2 | recall |
| OCR | 77.0% | TextCaps, BLIP3-OCR | mAP |
| small-object | 52.7% | 3 **custom** datasets | recall |
| image→visual | 90.6% | obj365, etc. | recall |
| logo | 56.0% | OpenLogo, ads-logo, basketball-logo | mAP |
| audio | 57.7% | AudioCaps, Clotho, GTZAN | recall |

Baselines they report beating: **InternVideo2-1B, LanguageBind, Google Vertex Multimodal Embedding API**.

**The catch (honest read):** every one of those is **whole-clip retrieval on SHORT clips** (MSR-VTT
clips ~15s; COCO is images; SSv2 is short action snippets) — "does the returned clip match the query"
over a small pool. **None is fine-needle temporal localization in an hour-long video.** So the "SOTA
on 60 benchmarks" claim is real *and orthogonal* to what we test. Several categories also use
TwelveLabs' **own custom datasets**, self-run — marketing-grade, not third-party leaderboards.

**Pegasus** posts no VideoMME/MVBench. Pegasus-1 (2023): MSR-VTT captioning, **GPT-4-as-judge**, vs
Whisper+ChatGPT-3.5. Pegasus 1.5 **openly abandons academic benchmarks** (see §4).

---

## 3. Capability matrix — Marengo 3.0 vs Pegasus 1.5 vs **ours**

Ours = SigLIP so400m (400M) per-frame image-text selector + gpt-5.5 as the frontier answerer.
"Ours (built)" = what actually runs today. "Ours (feasible)" = honest near-term extension and its cost.

| capability | Marengo 3.0 | Pegasus 1.5 | Ours (built) | Ours (feasible extension) |
|---|---|---|---|---|
| **text → frame/clip search** | ✅ | — (generative) | ✅ per-frame image-text scoring | — |
| **image → visual search** | ✅ | — | ❌ | ✅ **near-free**: SigLIP is a dual image-text encoder → embed the query image, cosine vs frame embeds. No new model. |
| **video-clip → search** | ✅ | — | ❌ | ⚠️ possible: mean/max-pool frame embeds of the query clip; loses temporal order but works for gist. |
| **audio → search** | ✅ | — | ❌ | ⚠️ needs a second encoder (CLAP for sound, Whisper for speech). Real work, not free. |
| **any-to-any** | ✅ full | — | ❌ (text-only) | ⚠️ partial: text + image cheap; audio is the gap. |
| **temporal granularity** | ~**6 s** clips | timestamps in text | **1 frame @ 1 fps = 1 s** | finer w/ higher fps (cost = more forward passes). *This is our edge on the fine needle.* |
| **max duration** | 4 h | 2 h | frame-budget bound (tested to 1 h / ~3000 frames) | scales linearly w/ frames; adaptive-k keeps answerer cost flat. |
| **generative answer / summary** | ❌ | ✅ | ⚠️ via **external** frontier (gpt-5.5) on the k picked frames | swap in any VLM; the selector is model-agnostic. |
| **segmentation → timestamped JSON** | ❌ | ✅ | ❌ | ⚠️ selection already yields picked-frame timestamps; grouping into spans is a thin post-step. |
| **counting / OCR / logo (fine-grained)** | ✅ specialized | partial | ⚠️ generic (SigLIP has weak OCR/logo, no counting head) | not our thesis; would need specialist heads. |
| **deployment** | paid cloud API | paid cloud API | **local open-weights** (400M) + frontier only on k frames | fully self-host the selector; only the answerer is a paid call. |
| **cost shape** | per-minute indexing + query | per-request tokens | 1 SigLIP fwd/frame (cheap, local) + frontier on k≈6 | adaptive-k shrinks the frontier bill further at short length. |

**Reading the matrix:** ours is deliberately *narrow* — text→frame retrieval only — but on the two
axes that matter for the fine-needle task it is **at least as good**: finer granularity (1 s vs 6 s)
and open-weights/local cost. The honest gaps are **audio** and **native any-to-any** (Marengo's real
breadth). The cheap wins are **image-query search** (near-free with the same encoder) and **swappable
answerer** (not locked to one vendor). We are not a Marengo replacement; we are the right cheap tool
for one slice Marengo's benchmarks never measure.

---

## 4. The tell — TwelveLabs said the quiet part themselves

Their post *"Not everything worth solving fits a benchmark"* (Dan Kim):

> "Our own models sit at SOTA on academic benchmarks, and yet two models hitting the same number can
> have completely different [behavior]… Academic benchmarks test on edited footage. Production reality
> is hours of raw, uncut video… plenty of things never show up in benchmarks."

Candid — and also a hedge that lets them avoid head-to-head on exactly the tasks their benchmarks
don't cover. Ours is one of those tasks.

---

## 5. Tie-back to our experiment

We tested **precisely the un-benchmarked slice**: a 1–2 s visual needle in a 1-hour video, scored as
hit@k localization.

- **Right model class** — Marengo (retrieval) is the correct tool; 3.0 on paper is ideal (4 h, fine-grained, counting).
- **Wrong granularity for the needle** — Marengo returns ~6 s clips (retrieval-not-localization by design); on a 1–2 s single-frame needle it smears — same failure mode as X-CLIP in our scorer-swap.
- **Result matches the gap they admit:** Marengo `marengo3.0` scored **.30 lenient / .10 strict @3600 s**
  and **.20 lenient / .10 strict @600 s**, vs cheap SigLIP so400m **hit@6 .24 @3600 s / .30 @600 s**
  (same 10 videos). **Parity** — at 10-min the cheap selector even edges ahead. Across both lengths the
  commercial flagship never beats cheap per-frame scoring.

**Positioning claim (defensible):** the commercial SOTA's headline numbers live on short-clip retrieval;
the hour-scale fine visual needle is **un-benchmarked and unsolved both cheaply and by SOTA**. Our
parity finding is an empirical demonstration of the benchmark-vs-reality gap TwelveLabs itself describes,
not a claim of beating them at their own game.

---

## Sources
- Marengo 2.7 benchmarks — twelvelabs.io/blog/introducing-marengo-2-7 (Dec 4, 2024)
- Marengo model spec (v3.0, 4 h, 512-dim, fine-grained) — docs.twelvelabs.io/v1.3/docs/concepts/models/marengo
- Pegasus model spec (v1.5, 261k ctx, 2 h, segmentation) — docs.twelvelabs.io/v1.3/docs/concepts/models/pegasus
- Pegasus-1 (80B, MSR-VTT, GPT-4 judge) — twelvelabs.io/blog/introducing-pegasus-1
- "Not everything worth solving fits a benchmark" (Dan Kim) — twelvelabs.io/blog/we-re-solving-problems-that-aren-t-in-any-benchmark
- Model taxonomy — docs.twelvelabs.io/v1.3/docs/concepts/models
- Our numbers — results/marengo_ceiling.json, results/marengo_600s.json, results/forkB/recall_vs_k.json
