# Plain-English summary — visual evidence compression for agents

*The simple version of what we did, what the numbers mean, and where we landed.*
*Technical detail lives in `WRITEUP.md` / `FINDINGS.md`; this is the no-jargon walkthrough.*

---

## First, the two numbers (there is no "sig@k")

- **hit@k** = "did the selector actually grab a frame that contains the answer?" (out of the k=6
  frames it picked). This measures **selection** — it's the honest metric.
- **accuracy** = "did the AI answer the question correctly?"
- **SigLIP ("sig", aka `so400m`) is NOT a metric** — it's the small model that *does the picking*.
  hit@k and accuracy are the scores; SigLIP is the tool being scored.

**Setup:** the AI that answers = **GPT-5.5**. The grader that checks answers = **GPT-4.1**. Benchmark =
**LongVideoBench** (real videos in 4 length buckets: 15s, 60s, 600s=10min, 3600s=1hr). For each video we
know the exact "needle" frame that holds the answer, so we can measure hit@k. Everything ran on
LongVideoBench (one benchmark, not several).

---

## The goal

An agent can't afford to feed a long video into a frontier model (a 1-hour video ≈ ~1M tokens). So:
**can a small, cheap model pick just the few frames that matter, so the big model answers just as well
for a fraction of the cost?**

---

## Part A — the frame selector (Fork A)

Instead of dumping *every* frame into GPT-5.5, the SigLIP selector reads the question, scores every
frame for relevance, and hands over only the **top 6**. We compared: **full-dump** (all frames),
**knob** (cheap "just lower the frame rate"), **uniform** (6 evenly-spaced frames = the dumb control),
and **SigLIP top-6** (the smart selector).

**hit@k — did selection keep the answer frame?**
| length | uniform (dumb) | SigLIP (smart) | full-dump |
|---|---|---|---|
| 15s | 1.0 | 0.96 | 1.0 |
| 60s | 0.56 | 0.72 | 1.0 |
| 600s (10m) | 0.04 | **0.36** | 1.0 |
| 3600s (1h) | 0.00 | **0.24** | 0.92 |

**Cost:** SigLIP stays ~2,500 tokens at every length; full-dump balloons to ~88,000 at 1hr →
**13–34× cheaper** on long video.

**What Fork A told us:**
1. **6 well-chosen frames answer as well as dumping everything** — same accuracy, ~30× cheaper.
   Compression buys **cost, not extra accuracy**. ✅
2. **The smart selector clearly beats the dumb one** at finding the needle (0.36 vs 0.04 at 10min) —
   so it's genuinely smart, not just cheap. ✅
3. **The problem:** the selector's hit@k **collapses as video gets longer** (0.96 → 0.24). At 1hr it
   finds the answer frame only 1 in 4 times. ❌
4. Accuracy on long video is "guessable" (4-option multiple choice), so it can look fine even when
   selection failed — that's *why* we trust **hit@k**, not accuracy, on long video.

So Fork A ended with one broken number to fix: **hit@k = 0.24 at 1 hour.**

---

## Part B — the coarse-to-fine idea (the "your idea" fork)

**The idea:** don't blindly score all 3,600 frames. **Split the hour into chunks, cheaply peek at
each, pick the promising chunk, then zoom in** — like finding the dog in a photo by checking quadrants
then zooming into the right one. Plus the **speculative-execution twist:** let the *small* model
propose which chunk to dive into, and the *big* model verify it (back up if wrong).

**What we actually built and ran:**
- **`hier` (coarse-to-fine):** split into time windows, cheaply probe each, keep the best, then score
  frames inside them.
- **`transcript-gate`:** use the subtitles/speech to guess where the answer is.

**What we did NOT build (honest):**
- The **high-res zoom** (re-look at the winning chunk in sharper detail).
- The **speculative-execution part** (small proposes / big verifies + backtrack).
- These stayed **ideas on paper — never coded** (why is below).

**Results (hit@k):**
| selector | 15s | 60s | 600s | 3600s |
|---|---|---|---|---|
| flat SigLIP top-6 (to beat) | 0.96 | 0.72 | **0.36** | **0.24** |
| `hier` coarse-to-fine | 0.96 | 0.76 | 0.32 | 0.20 |
| transcript-gate | 0.88 | 0.72 | 0.04 | 0.08 |

Both **worse**, not better. Why:
- **Coarse-to-fine failed** because it re-uses the *same* SigLIP scores, and its coarse peek was too
  sparse — it threw away the 1–2 second needle's chunk *before* the zoom step could run. You can't zoom
  into a chunk you already discarded.
- **Transcript failed hardest** because for *visual* questions the speech describes what's being *said*,
  not the *visual* thing asked about — so it pointed to the wrong moment.

**The experiment that explained it all — "recall vs budget":** give the *same* SigLIP scorer more
frames to keep (6 → 40) and watch hit@k:
| length | k=6 | k=40 | meaning |
|---|---|---|---|
| 60s | 0.72 | **1.0** | just needed more frames ("budget-limited") |
| 600s | 0.36 | 0.60 | partly |
| 3600s | 0.24 | **0.32** | barely moved → **the scorer itself can't rank the needle** |

At 1 hour, **6.7× more frames only nudged 0.24 → 0.32.** So the wall isn't "too few frames" or "need a
cleverer strategy" — it's that **SigLIP can't find the right frame among ~2,400.**

**Why we skipped zoom + speculative-execution:** every one of those tricks *still relies on the same
SigLIP scores* to decide where to look/zoom/propose. If the scorer can't rank the needle, then a big
model "verifying" the small model's proposals is just checking bad guesses — it can't rescue what was
never proposed. So the honest call: don't build speculative-execution when the diagnostic already shows
its input (the small model's scores) is the broken link. **Status of the speculative-execution idea:
designed and reasoned about, deliberately not built.**

---

## The close — SigLIP's fault, or is the task just hard?

**1. Swap the scorer (free):**
| scorer | 600s | 3600s | what happened |
|---|---|---|---|
| SigLIP so400m | 0.36 | 0.24 | baseline |
| SigLIP2 (newer image model) | 0.24 | 0.08 | **worse** → a better image encoder doesn't help |
| X-CLIP (a video model) | 0.12 | 0.04 | worse — but it blurs 32s together, smearing a 1–2s needle |

Lesson: finding a 1–2s needle in an hour at 1 frame/sec is fundamentally a **per-frame** job, and cheap
per-frame SigLIP is actually the *right* cheap tool for it.

**2. The ceiling check — TwelveLabs Marengo (paid, commercial best), 1-hour bin:**
| | Marengo 3.0 (paid) | cheap SigLIP |
|---|---|---|
| hit@k (fair/lenient) | 0.30 | 0.24 |
| hit@k (strict) | 0.10 | 0.24 |

Even the **commercial state of the art basically ties the cheap model** (both terrible at 1hr). So the
wall is **the task, not our tool** — and this answers "why not just use TwelveLabs?": *it doesn't solve
this either.*

---

## Where we landed (bottom line)

- **Cheap 6-frame SigLIP selection = same answers as full-dump, at ~13–34× lower cost.** ✅ solid win.
- **On long video, nobody (cheap or paid) reliably finds the exact needle.** ❌ not beatable with a
  cheap trick.
- **What ships:** cheap length-adaptive selection for **short video (≤ ~10 min)** — strong at ≤1 min
  (hit@k ~1.0), decent at 10 min (~0.60). The 1-hour extreme is a dead end for everyone and also the
  *least* common case for real agents.

**Fork B was a clean NEGATIVE — and that's a good outcome:** it cheaply killed a plausible idea and told
us exactly where the real wall is (the task, at hour scale), instead of burning effort chasing it.

**Honest health warning:** samples are small (25 per bucket, only 10 for the Marengo check), so every
number is **directional, not proven**. The story is consistent across many experiments, but no single
number is statistically locked down.
