# HANDOFF — video-intelligence phase (visual evidence compression for agents)

> **⚠️ READ §10 FIRST. §10 (Decision Log, Jul 4) OVERRIDES §1–9 wherever they conflict.** §1–9 were
> written pre-execution and are now partly stale: benchmark is **LongVideoBench, not LVHaystack**;
> the so400m / `detail` / gpt-5.5 / hit@k / model-knob code changes are **already done**; Phase 0 /
> Fork A is **complete** (see FINDINGS "Phase 0 — Fork A"). Do not rebuild `build_lvhaystack.py`.

**Updated:** Jul 4, 2026 (was Jul 3)
**For:** an agent operating **directly on the RunPod** at `/workspace/slm-lab` (SSH, runs commands
on the pod). This is the authoritative handoff; it supersedes the earlier S0-only version.
**Companion docs (read for depth):** `SPEC.md` (harness design + gate), `FINDINGS.md` (S0–S2b
results), `research/video-intelligence.md` (landscape + prior art), `video-intelligence-phase.md`
(narrative + forks).

---

## 0. Mission in one paragraph

Agents can't afford to feed long video into a frontier model (a 1-hr video ≈ ~1M tokens at 1 fps).
We test whether a **small, cheap selector picks the few frames that matter** so a frontier answerer
answers **as accurately for a fraction of the tokens** — and whether, past some length, compression
actually **beats** full-dump (context dilution). Scope is **video only** (image/GUI track dropped).

## 1. How we landed here + pivotal decisions

1. Idea: "Exa for visual" — give agents info from images/video. Research reframed it: single-image
   understanding is solved; "Exa for video" exists (TwelveLabs); small VLMs exist. **Real gap =
   cost/volume of long video.** → thesis: **visual evidence compression**.
2. **S0 harness** built + validated (A=full-dump vs C=selector; echo/OpenAI answerers; LLM-judge;
   recall@k). Needle test: uniform = 0.1667 acc at 6.5× fewer tokens; **accuracy == recall@k**
   (answer correctness is fully explained by whether selection kept the evidence frame).
3. **S1/S2** on stitched-GUI needles: the "both selectors fail (~0.5)" result was a
   **weak-checkpoint artifact**; `siglip-so400m-patch14-384` lifted hit@k to **0.875**. Paid n=8:
   **equal accuracy at 13.5× fewer tokens**. Saw **compression beat full-dump twice** (context
   dilution).
4. **PIVOT — scope to video only.** Dropped image/GUI-screen track (it was the most artificial slice:
   stitched layout, near-identical screens, B-biased benchmark).
5. **Prior-art check (important):** our "skimmer" = **keyframe selection** (FOCUS, Q-Gate, VSI);
   the recursive-chunk idea = **coarse-to-fine search** (V*, T*, VideoTree). These are published and
   work. → **We cannot claim to have invented selection.** Contribution must be empirical/positioning.
6. **PIVOT — Fork A first.** Chose the measurement/systems result over method-building (lower risk,
   most original, doesn't require out-engineering T*/VideoTree).
7. **Decisions locked:** benchmark = **LVHaystack**; answerer = **GPT-5.5** (frames-as-images —
   OpenAI has **no native video input**, confirmed); transcript path **deferred**; API tier-2 (1M TPM).

## 2. Current code state (what already exists)

Package `harness/` (run from repo root as `python -m harness.run ...`):

| File | Contents |
|---|---|
| `media.py` | `Frame(index, image, seconds)`; `load_frames(item, dump_fps, max_frames)` — video via OpenCV at `dump_fps`, or images/dir; `frame_to_base64(frame, max_side, ...)`. |
| `selectors.py` | `FullDumpSelector` (cond A), `UniformSelector` (baseline), `EmbeddingSelector` (SigLIP image-text cosine top-k). **Default model_id is `google/siglip-base-patch16-224` — CHANGE to `google/siglip-so400m-patch14-384` (best in S2).** |
| `answerers.py` | `EchoAnswerer` (offline, token-estimated), `AnthropicAnswerer`, `OpenAIAnswerer` (**default `gpt-4o`, chat.completions, `image_url`, no `detail` param**), `build_answerer`, `make_text_judge`. |
| `metrics.py` | `exact_match`, `llm_judge`, `recall_at_k` (video: `gold_evidence_seconds` spans; images: `gold_evidence_frames`). |
| `run.py` | orchestrator; conditions `A`/`C`; `build_selector` (`uniform`,`embedding`); `summarize` incl. `token_reduction_vs_A`; prints the go/no-go gate. CLI: `--manifest --conditions --answerer --model --selector --k --dump-fps --max-dump-frames --max-side --judge --judge-provider --out`. |
| `scripts/make_synthetic.py`, `scripts/make_synthetic_hard.py` | synthetic generators (toy + needle-in-haystack). Superseded by real data now. |

**How selection works (so it's unambiguous):** the real selector is **query-driven similarity
search**, NOT random. Embed the question → vector; embed each frame → vector (SigLIP, same space);
score by cosine similarity; keep top-k. `uniform`/`random` are deliberately-dumb baselines only.

## 3. Pod environment / how to run

```bash
cd /workspace/slm-lab
source <the existing venv>/bin/activate      # S2 stack: torch 2.11+cu128, transformers 4.57.6
set -a && . ./.env && set +a                 # loads OPENAI_API_KEY, HF_HOME=/workspace/hf, etc.
# GPU: RTX 4050 present. Fork A / Phase 0 is CPU-friendly; SigLIP selector is light.
```
Sanity check before spending: `python3 -c "import torch,transformers,openai;print('ok',torch.__version__)"`
and confirm `OPENAI_API_KEY` is set (`env | grep -o '^OPENAI_API_KEY'`).

Tier-2 = **1M TPM**. A full-dump request at `detail:low` (~85 tok/frame) × ~1,500 frames ≈ ~127K
tokens → fine, ~7 such requests/min. Use retry/backoff on 429 anyway.

## 4. FORK 0 / PHASE 0 — honest baselines on real data (DO THIS FIRST)

**Goal:** stop trusting synthetic; measure where we actually stand on a real bench others use.

**Data — LVHaystack** (`LVHaystack/LongVideoHaystack` on HF): long videos with **frame-level gold
needle labels**; the exact bench **T\*/TStar** report on.
- Build `scripts/build_lvhaystack.py` → converts LVHaystack items to our manifest schema:
  ```json
  {"id": "...", "media_type": "video", "media_path": "data/videos/xxx.mp4",
   "question": "...", "gold_answer": "...", "gold_evidence_seconds": [[start,end], ...]}
  ```
  (gold_evidence_seconds = the needle timestamp(s); this drives hit@k via `recall_at_k`.)
- Start with a **~40-question subset, binned by video length**: ~2 / ~8 / ~20 / ~40+ min. Length is
  the independent variable for Fork A. Store videos under `data/videos/` (gitignored; heavy).

**Code changes needed (small):**
1. `EmbeddingSelector` default → `google/siglip-so400m-patch14-384`.
2. `OpenAIAnswerer`: allow `model="gpt-5.5"`, add a `detail` param (`"low"` for full-dump, `"high"`
   for the 6-frame top-k), and add **429 retry/backoff**. Keep frames-as-images.
3. Add a **`hit@k`** helper to `metrics.py` (= 1.0 if any selected frame falls in any gold span; we
   already have `recall_at_k` which is the coverage fraction — hit@k = `recall_at_k > 0`).
4. Add a **model-knob** baseline: full-dump but at `--dump-fps 0.5` and `detail:low` (the provider's
   own cheap path — this is a *real* baseline, unlike naive full-dump).
5. Add **per-length-bin aggregation** to `summarize` (accuracy/tokens/hit@k per bin), for the Fork A
   crossover plot.

**Baselines to run per item** (answerer = `gpt-5.5`):
| Condition | Selector / setting |
|---|---|
| A — full-dump | all frames @1fps up to cap, `detail:low` |
| knob — model-knob | full-dump @ fps<1, `detail:low` |
| U — uniform-k | `uniform`, k=6 |
| C — top-k | `embedding` (so400m), k=6 |

**Metrics:** `hit@k` (vs gold), QA accuracy (LLM judge), input tokens, latency.

**Example commands:**
```bash
# full-dump (cap ~1500 frames ≈ 25 min @1fps)
python3 -m harness.run --manifest data/manifest.lvhaystack.json --conditions A \
  --answerer openai --model gpt-5.5 --max-dump-frames 1500 --dump-fps 1 --judge --out results/p0_fulldump
# top-k so400m selector
python3 -m harness.run --manifest data/manifest.lvhaystack.json --conditions C \
  --answerer openai --model gpt-5.5 --selector embedding --k 6 --max-dump-frames 1500 --judge --out results/p0_topk
# free hit@k check (no API) to validate selection first:
python3 -m harness.run --manifest data/manifest.lvhaystack.json --conditions C \
  --answerer echo --selector embedding --k 6 --max-dump-frames 1500
```
**Cost:** ~40 Q full-dump (~127K tok each) ≈ ~5M input tok ≈ **~$25**; top-k is pennies.

**Phase 0 exit:** a table of hit@k + accuracy + tokens per condition per length bin. Now we know our
real standing vs the honest baselines (not just vs full-dump).

## 5. FORK A — "when does compression BEAT full-dump?" (the chosen result)

**Question:** across video length, plot **accuracy(full-dump) vs accuracy(top-k)** → find the
**crossover** (length past which top-k ≥ full-dump on accuracy) and the **token wall** (where
full-dump becomes infeasible: >~1500 frames / context).

**Hypothesis H:** beyond length L*, `top-k ≥ full-dump` on accuracy AND far cheaper (context dilution
hurts full-dump). We already saw this twice (whispr n=1; paid item 004).

**Deliverable:** the accuracy-vs-length curves for A vs C (+ knob, uniform), the crossover point (if
any), token/latency ratios, and hit@k as the selection diagnostic. **A clean negative ("full-dump
stays ≥ top-k") is still a valid, publishable finding** — it just means compression buys cost, not
accuracy.

**Analysis honesty:** report **hit@k as the primary metric** and treat accuracy as secondary until
answer-position bias is ruled out (see §7). Report **Wilson 95% CIs** — n is small.

## 6. FORK B — smarter selector — **CLOSED-NEGATIVE for SigLIP (Jul 4). See FINDINGS "Fork B".**

> **STATUS:** Closed as negative *for the SigLIP scorer*, NOT for cheap selection in general.
> - **RAN:** transcript-gate (fails hard), coarse-to-fine `hier` (fails). Both re-rank the SAME
>   SigLIP so400m scores at the SAME 384px.
> - **NOT RUN (deprioritized by the recall-vs-k diagnostic, not executed):** high-res **zoom** on
>   the winning window, **draft-verify/backtrack**, and — the real gap — a **video-native retrieval
>   scorer** (InternVideo2/LanguageBind/moment-retrieval). recall-vs-k shows 3600s is
>   *selector-limited*, so every lever downstream of the same scores inherits its ceiling — but a
>   *different* scorer is untested. **The scorer-swap is the real close-out test** → `scorer_swap_spec.md`.

Original design (kept for reference; zoom + draft-verify are the not-run parts above). Build a smarter
selector, benchmark **against T\*/VideoTree** (not full-dump). Components:
- **Transcript-first coarse locate:** ASR (Whisper) → candidate time regions from transcript. RAN — fails
  on frames-answerable questions (ASR describes speech, not the visual needle).
- **Coarse-to-fine:** score chunks → descend → **zoom at higher resolution** on the winning chunk.
  Coarse-to-fine RAN (`hier`, worse); **the zoom part was never built.**
- **Draft-verify / backtrack:** propose → verify → reject/backtrack. **Never built.**
- **Positioning / defensible lane:** cheap + **no pre-index** + transcript-aware, fine-detail case,
  one-off agent-browse videos (can't amortize a TwelveLabs index).

Baselines for Fork B: T*/TStar (public code, same LVHaystack numbers), VideoTree, VSI (subtitle+visual).

## 7. Pivotal caveats — things to care for (READ BEFORE TRUSTING ANY NUMBER)

1. **accuracy == recall@k** was the key S0 lesson → selection is the whole game, and **hit@k is a
   free proxy for accuracy** (validate selectors with `echo` before paying).
2. **Metric contamination / answer-position bias:** the old GUI-World benchmark was ~B-biased so
   accuracy overstated selection. **Check LVHaystack's answer format/bias**; prefer **hit@k** as
   primary; only trust accuracy once bias is ruled out.
3. **n is tiny** everywhere so far (n=8, n=1). Directional, not conclusive. Scale to ≥50 and report
   CIs. A single item = ±0.12 on n=8.
4. **Audio/transcript is a real blind spot** (deferred, not solved). Video-MME: subtitles add +10.1%
   on long video. Frames-only is a known ceiling — don't over-conclude from it.
5. **Real baselines are NOT full-dump.** They are: model-knob (fps/res), subtitle selection, T*/
   VideoTree, TwelveLabs. Our `results/` so far only beat full-dump (weakest baseline).
6. **Don't reinvent published work.** Coarse-to-fine (V*/T*/VideoTree) and keyframe selection
   (FOCUS/Q-Gate/VSI) exist. Cite + baseline against them.
7. **OpenAI has no native video input** (confirmed; "native video" marketing = video *generation*).
   We extract frames ourselves → we lose true temporal/audio info between frames unless transcript is
   added.
8. **Context-dilution ("compression beats full-dump") is our most original thread** — elevate it,
   study it rigorously in Fork A (it may be the headline).
9. **Leakage:** never let the selector see `gold_evidence_seconds` at inference; keep selection and
   evaluation strictly separate.
10. **Stitched ≠ real footage.** LVHaystack is real video — good — but earlier stitched results have
    synthetic *layout*; don't carry their absolute numbers forward.

## 8. Benchmarks reference
- **LVHaystack** (primary) — frame-level gold needles; direct T*/TStar comparison; hit@k.
- **LongVideoBench** — gold keyframes + subtitles + subtitle baseline (GPT-4o+Sub ≈58%); use when
  adding the transcript path.
- **Video-MME** — comprehensive, modality-ablated; **no gold frames** (can't do hit@k); use for final
  accuracy + modality study.
- **MME-VideoOCR** — subtitle/OCR multi-hop needle (fine-detail case). **EgoSchema/NExT-QA/MLVU** —
  VideoTree comparison points.

## 9. Immediate checklist for the pod agent
- [x] Confirm env: `OPENAI_API_KEY` set, `torch/transformers/openai` import OK.
- [x] Build subset → manifest + `data/videos/` (via LongVideoBench, not LVHaystack — see §10).
- [x] Code changes: SigLIP→so400m; `OpenAIAnswerer` gpt-5.5 + `detail` + 429 retry; `hit@k`; model-knob baseline; per-length-bin summary.
- [x] Run Phase 0 baselines (A / knob / uniform / top-k); saved to `results/p0_*` (n=25).
- [x] Fork A accuracy-vs-length table + crossover (hit@k primary, Wilson CIs) — see FINDINGS "Phase 0 — Fork A".
- [x] Fork B **CLOSED-NEGATIVE for SigLIP** — `hier` + transcript-gate ran (both fail); zoom +
      draft-verify NOT run (deprioritized by recall-vs-k). Real close-out = scorer-swap (`scorer_swap_spec.md`).

---

## 10. DECISION LOG + VERIFY (for the design agent to check, Jul 4 2026)

**Executed by the pod agent; please sanity-check these against your intent.**

### Decisions taken (deviations from this handoff)
1. **Benchmark: LongVideoBench, NOT LVHaystack.** Reason: LVHaystack's parquet is public but its
   videos are **gated Ego4D** (license + AWS + an unfinished video→clip script) → not fetchable
   autonomously. LongVideoBench is HF click-through gated (account clicked "Agree"), ships gold
   keyframes (`position`) + subtitles + native `duration_group` {15,60,600,3600}s = the exact length
   axis Fork A needs, and is what LVHaystack's own transform targets. **Cost:** we no longer share T\*'s
   exact eval set → Fork-B comparison to T\*/VideoTree is now *directional* (same task, different
   subset), not a same-numbers table. Flag if that breaks the positioning you intended.
2. **Frames pre-extracted to 1fps JPEG** (`scripts/extract_frames.py`) → manifest `media_type=images`;
   runtime reads JPEGs, not video (opencv full-decode of hour videos was the bottleneck). Gold
   `position` → `gold_evidence_seconds` self-validated per video from native fps (`gold_reliable`
   flag; 100/100 reliable).
3. **n=25/bin** (100 Q total, distinct videos). Wilson CI ±~0.18 — still overlapping; directional.
4. **Answerer gpt-5.5** verified: needs `max_completion_tokens` (rejects `max_tokens`),
   `reasoning_effort=low`, budget ≥2048, and a **500-image/request hard limit** → that cap IS
   full-dump's real token wall (`CAP_A=500`).

### Your pre-exp review — status of each item
- **[FIXED, our path] `_load_video_frames` truncates to first-N on long video.** Correct and important
  call. Note our pipeline runs `media_type=images` (`_load_image_dir`), which had the *same* bug — now
  fixed to **uniform-subsample-across-clip** to the cap. We also applied your exact fix to
  `_load_video_frames` for the video path. (Frames are zero-padded `f%06d.jpg` so ordering was already
  temporal — no scramble.)
- **[DONE differently] `build_lvhaystack.py` must set video_seconds/length_bin/gold_evidence_seconds.**
  → `build_longvideobench.py` does exactly this (length_bin from `duration_group`).
- **[DONE] cross-arm merge for the Fork A table.** → `scripts/analyze_forkA.py` reads N runs.jsonl,
  merges per (bin, arm), Wilson CI, crossover detect. (Your suggested `forkA_report.py`.)
- **[FIXED] `_create_with_backoff` retried a genuine 400 6×.** Now only retries 429 / 5xx / conn.
- **[OPEN, minor] `_image_part` doesn't shrink to 512px on `detail:low`** — still 768 max_side.
  Non-blocking; wasteful upload only.
- **[OPEN, minor] `make_text_judge` (openai) has no 429 backoff** — judge could rate-limit on big
  runs; didn't bite at n=100. Worth wiring the same backoff if we scale n.

### To VERIFY next (asks of the design agent)
- [ ] Does the LongVideoBench pivot + directional (not same-set) T\*/VideoTree comparison still satisfy
      the Fork-B positioning, or do we need the real LVHaystack/Ego4D set for the paper?
- [ ] Is "compression buys **cost** not accuracy, + context-dilution replicated" the headline you want
      elevated, given arm CIs overlap at n=25? (Scale to n≥50 to move from directional→significant?)
- [ ] Fork B success metric: we target lifting hit@k recall on 600/3600s (.36/.24 → ?). Confirm that's
      the right bar vs also requiring an **accuracy** win over knob (which currently ties/wins).

### Fork B closure + design-agent review (Jul 5 2026)
- **Fork B CLOSED-NEGATIVE for SigLIP**, per FINDINGS "Fork B". Design agent reviewed the closeout and
  gave one big honest pushback (ACCEPTED): "the bottleneck is the SCORER" was tested with **exactly one
  scorer (SigLIP so400m = image-text *matching*)**. Never tried a **video-native moment-retrieval**
  encoder — the on-thesis, cheap escape hatch. So earned conclusion is narrower: *cheap image-similarity
  scoring is localization-limited at hour scale*, NOT *cheap selection is dead*.
- **Two Fork-B levers never ran:** high-res **zoom** (`hier` re-ranks same 384px scores) and
  **draft-verify**. Docs now label them not-run, deprioritized by recall-vs-k (§6).
- **Agreed next steps (priority):** (1) **scorer-swap** = the real close-out — one cheap video-native
  encoder, same echo/hit@k harness, k=6, 4 bins (`scorer_swap_spec.md`); (2) characterize **adaptive-k**
  ≤600s (shippable sub-10-min operating point); (3) **blind question-only** baseline (guess floor,
  prereq for any accuracy claim).
- **Decision fork after (1):** works → cheap video-native selection is the product (scale n≥50 + write
  up). Fails too → honest pivot: ship adaptive-k as a <10-min tool (drop hour-scale claim) OR step to a
  stronger scorer and explicitly leave the "cheap" thesis.

### Scorer-swap RESULT + close (Jul 5 2026) — see FINDINGS "Scorer-swap"
- **RAN** (echo/hit@k, k=6, n=25): so400m `.96/.72/.36/.24` (reproduces exactly) · siglip2 `.88/.76/.24/.08`
  · videoret/X-CLIP `.44/.16/.12/.04`. **Both swaps LOSE.**
- **siglip2 lost cleanly** → "better image encoder ≠ better localization" (encoder-quality hypothesis dead).
- **X-CLIP lost by construction** (32-frame window pools/smears the 1–2 s needle) → not a clean video-native
  test, and any clip-pooled encoder (X-CLIP-p32, LanguageBind) fails by the same mechanism → **skipped as
  low-info** (design-agent concurred).
- **Principled negative banked:** fine needle @1fps = *per-frame* retrieval; SigLIP (per-frame image-text)
  is the right cheap tool; residual wall = *ranking capacity* at hour scale, not encoder (recall_vs_k). Cheap
  ceiling = SigLIP-in-hand; only a dense moment-retrieval head (Marengo) plausibly beats it = the paid stack.
- **Env-forced substitution:** LanguageBind_Video_FT (spec's named model) imports on torch 2.11 (needs a
  torchvision `functional_tensor` shim) but its decord/file-path processor can't consume pre-extracted JPEGs;
  X-CLIP was the transformers-native stand-in (wrong sub-family — action, not moment-retrieval). Flag if the
  writeup needs the exact named model.
- **Marengo ceiling RAN (Jul 6) — PARITY, not a win either way.** TwelveLabs Marengo 3.0, 3600s bin, n=10,
  k=6: strict **0.10** / lenient **0.30** vs cheap so400m **0.24**. At n=10 (Wilson ±0.28) that's a **tie in
  the noise** (Marengo nominally ahead on lenient) → **parity**: cheap SigLIP ≈ commercial SOTA on this slice.
  Evidence = the **cross-arm pattern** (uniform .00 / so400m .24 / siglip2 .08 / X-CLIP .04 / Marengo .10–.30
  → none escapes ~0.2–0.3 @1h), NOT this one cell. Robustness: Marengo ingests real video at its own denser
  sampling, still ~0.30 → not a 1-fps-starvation artifact. Scope: "hard" earned for the ADVERSARIAL slice
  (1–2s single-frame-answerable needle in 1h), not long video broadly. Runner `scripts/marengo_ceiling.py`
  (SDK 1.2.8, marengo3.0); `results/marengo_ceiling.json`.
- **Shippable positive:** adaptive-k for ≤10-min (60s→1.0, 600s→.60 @k=40) — validated on hit@k only.
- **Before any ACCURACY headline (both cheap, both UNRUN):** (a) end-to-end adaptive-k run (accuracy+cost)
  on ≤10-min bins; (b) blind question-only baseline (guess floor for the guessable MCQA).
- **Fork B fully closed** — on the hour-scale fine-needle slice cheap SigLIP ≈ SOTA; no further scorer work.
