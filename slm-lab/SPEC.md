# Visual Evidence Compression for Agents (working title)

**Status:** spec / pre-experiment (Jun 2026)
**One-line:** Can a *small* model select & compress visual evidence (frames / clips / screen regions) so a frontier agent answers visual & video questions at a fraction of the tokens/cost — without losing accuracy?

---

## 1. Why this, and not the obvious thing

The obvious framing — "build an SLM that understands images for agents" — is **already solved and over-served**, so we are explicitly *not* doing that:

- Frontier agents (Claude / GPT / Gemini) are **natively multimodal** for single images. "Let the agent understand an image" is mostly done.
- Small VLMs already exist and are strong: **SmolVLM** (256M / 500M / 2.2B, <1GB VRAM) and **Moondream** (0.5B / 2B / 3) do captioning, VQA, OCR, grounding. **SmolVLM2 already does video reasoning** at small scale.
- "Exa for video" already exists: **TwelveLabs** (Marengo embeddings) indexes video, searches by text/image, returns exact timestamps, ships an **MCP server + Claude Code plugin**.

The Exa lesson: their moat is **owning the index + crawl + compute**, not the embedding idea. Chasing a web-scale visual index = the same capital/crawl fight against funded incumbents (Exa $2.2B, TwelveLabs). We avoid that.

**The actual gap we target:** frontier agents can't cheaply consume *long* visual content (a 2h video, a 50-frame screen recording, a dense dashboard). Dumping it all in context is expensive and often impossible. The unclaimed, model-layer job is **cheap pre-filtering / compression**: pick the few frames/clips/regions that matter, hand the agent a compact, decision-relevant payload. This is the visual analogue of Exa's "highlights" trick (highlights = ~10x fewer tokens than full text).

So the SLM's job is **NOT** "understand the image better than a frontier model." It's **"decide what visual evidence is worth the agent's tokens, and compress it"** — a job frontier models are too expensive to do at scale.

## 2. Research question (measurable)

> Holding answer accuracy constant, how much token/cost reduction can a small selector model deliver versus feeding all visual content to a frontier agent?

Primary metric: **tokens (and $) to the frontier model at iso-accuracy.**
Target worth pursuing: **≥5x cost reduction at ≤2 pts accuracy drop.** If a tiny, untrained selector already hits this, that justifies training a custom SLM later. If it doesn't, the idea is dead and we saved months.

## 3. Hypotheses

- **H1 (compression win):** A small frame/clip selector keeps frontier-agent accuracy within 2 pts while cutting input tokens ≥5x vs. naive full-dump.
- **H2 (small ≈ retrieval):** An off-the-shelf small VLM selector (SmolVLM2 / Moondream) gets within a few pts of a dedicated video-retrieval system (TwelveLabs) on moment-finding, at lower cost / fully local.
- **H3 (training headroom):** A purpose-trained SLM selector beats the off-the-shelf small VLM selector on token-at-iso-accuracy by a meaningful margin. *(Only test if H1 looks promising.)*

## 4. Conditions to compare

For each task (video or long-visual + question):

| Cond | Visual handling | Answerer | Notes |
|---|---|---|---|
| A — Full dump | all frames / transcript / full screenshots | frontier model | upper bound on cost, ~upper bound on accuracy |
| B — Retrieval baseline | TwelveLabs / SmolVLM2 top-k moments | frontier model | "is there already a good enough tool?" |
| C — Small selector (untrained) | SmolVLM-500M / Moondream picks top-k frames/regions | frontier model | the cheap candidate |
| D — Trained SLM selector | our fine-tuned selector picks evidence | frontier model | only if A/C show a real gap to close |

Answerer (frontier model) is held **fixed** across conditions so we isolate the selector's effect.

## 5. Metrics

- **Accuracy** — task-appropriate (exact-match / LLM-judge / moment IoU for timestamp tasks).
- **Input tokens to frontier model** (the headline cost number).
- **$ cost** (selector compute + frontier tokens).
- **Wall-clock latency**.
- **Selector recall@k** — did the selected evidence contain the frames needed to answer? (diagnostic: separates "selector missed it" from "answerer failed").

Headline plot: **accuracy vs. tokens** across A/B/C/D. We want C/D to sit up-and-left of A.

## 6. Data

Start tiny and honest (~50 items), mix of:

- **Long video QA** — questions whose answer lives in a short span of a long video (use an existing video-QA set, or build from public talks/tutorials).
- **Screen / browsing captures** — multi-frame screen recordings or rendered pages where the DOM/text is useless (charts, dashboards, canvas apps). This ties to the agentic-browsing angle and is the most defensible slice.

Each item: `{ media, question, gold_answer, gold_evidence_span(s) }`. Gold evidence span lets us compute selector recall independently of the answerer.

Raw media stays **out of git** (per repo convention) — store under `data/`, reference by id.

## 7. Stages

1. **S0 — Harness + 10-item pilot.** Wire conditions A and C end-to-end on 10 items. Confirm the accuracy/token logging is trustworthy before scaling.
2. **S1 — Full A vs C on ~50 items.** Decide H1. Go/no-go gate.
3. **S2 — Add B (TwelveLabs).** Is an off-the-shelf tool already good enough? Decide H2.
4. **S3 — Train D (only if S1 passes).** Fine-tune a small selector; decide H3.

## 8. Go / No-Go gate (after S1)

```
PROCEED to training only if:
  Condition C accuracy >= Condition A accuracy - 2 pts
  AND   Condition C input tokens <= Condition A input tokens / 5
```

If C can't beat full-dump on cost at iso-accuracy with an *off-the-shelf* model, a custom SLM is unlikely to rescue it — stop and write up the negative result.

## 9. Honest risks / failure modes (flag before celebrating)

- **Selector misses the key frame → accuracy collapses.** Recall@k is the early-warning metric; watch it from S0.
- **Frontier models keep getting cheaper/longer-context**, eroding the cost win. The result must be expressed as a *ratio*, not absolute $, to stay meaningful.
- **TwelveLabs/SmolVLM2 may already clear the bar** → the contribution becomes "measured comparison + when to use which," not a new model. That's still a valid, publishable outcome.
- **Leakage:** if gold_evidence_span was used to build the question, don't let the selector see it at inference. Keep selection and evaluation strictly separated.
- **Tiny-N noise:** 50 items → wide error bars. Treat S1 as directional, not conclusive; report CIs.

## 10. Out of scope (explicitly)

- Building a web-scale visual/video **index or crawler** (that's the Exa/TwelveLabs capital game — not our fight).
- Training a general-purpose VLM from scratch.
- Beating SmolVLM/Moondream on standard VQA benchmarks.

## 11. Open decisions

- Which frontier model as the fixed answerer (cost vs. ceiling).
- Off-the-shelf selector: SmolVLM2-500M vs. Moondream — pick one for S0, note the choice.
- Video-QA dataset source vs. hand-built set.

## References

- Exa — neural index / next-link prediction / "highlights = 10x fewer tokens"; $250M Series C @ $2.2B (May 2026).
- TwelveLabs — Marengo video embeddings, MCP server + Claude Code plugin, search → timestamps.
- SmolVLM / SmolVLM2 (HF) — 256M/500M/2.2B, <1GB VRAM, video reasoning at small scale.
- Moondream — 0.5B/2B/3, grounding + pointing, edge deployment.
