# Path 3 — Multi-Facet Matching Pursuit (STARTED 2026-07-20)

## Thesis
The measured bottleneck is not the pick geometry — it is that **one 768-d text vector
cannot reach the image subspace** (residual trace: query keeps 96.7% of norm after 16
picks; cos_orig maxes .233). Ten selector variants tie because the objective is flat.

Path 3 attacks the **query** side, not the geometry: decompose the stem into m facets
q1..qm, giving m *different* small reachable subspaces instead of one averaged direction.
Budget k=8 is then allocated to satisfy each facet's residual.

## Why LVB specifically rewards this
LVB stems are compound by construction:
```
[scene description] + [subtitle/temporal anchor] + [the actual question]
```
Example (`VwZeSoYugZk_1`): green-lake/building/mountain-peaks description, THEN a quoted
subtitle anchor, THEN "what is the shape of the roof". Averaged into one vector the
answer-bearing facet ("shape of the roof") is diluted by a long scene description that
matches thousands of frames. Facets separate the locator from the interrogative.

Distinct from LDDR (frame-space volume, query as scalar magnitude) and from every variant
we have already run — all 10 kept q fixed and changed the pick rule. This changes q.

## Stages — each kills cheap, GPU only at the very end

**A. Decompose (API, ~976 stems, cents).** `scripts/gen_facets.py`.
Facets must be **caption-like visual search phrases**, not question fragments — the CLIP
text tower is trained on captions. Emit 1-4 facets + a facet type tag
(scene / subject / action / temporal-anchor / interrogative).

**B. Embed facets (CPU LongCLIP-L, pod).** Same tower and truncate=True path as
`dump_longclip_all.py` so facet vectors live in the identical space as `text_lc_*.npz`.

**C. Geometry diagnostic — FREE, decisive.** Before any picks:
1. facet-facet cosine — if facets collapse (cos > .9 to each other and to the stem),
   there is no new subspace and Path 3 is dead on arrival.
2. `max_frames cos(facet_i, f)` vs `max_frames cos(stem, f)` — does any facet reach the
   image subspace better than the averaged stem? (stem baseline: .233)
3. do facet argmax frames differ from the stem argmax frame?

**C2. Free evidence-recall oracle.** Manifest carries `gold_evidence_seconds`. Measure
hit@8 of facet-OMP picks vs stem-OMP picks against the gold window.
*Caveat, stated up front:* frame-recall is a **hostile proxy** (a near-duplicate frame one
second outside the window counts as a miss, though it carries the same evidence). Use as a
directional signal only — never as the accept/reject gate, and never as a paper number.

**D. Facet-OMP picks + pick-overlap kill-switch.** Allocation: round-robin over facets by
residual, k=8 total. Overlap vs current `picks_omp_lc` > 90% => picks did not move => null,
stop, zero GPU spend. (Reference scale: the query-bug fix moved 42% of picks and moved
accuracy.)

**E. Answerer — only if C and D live.** 600s first (decision bin, n=412, ~1 GPU-day).
Pre-registered bar: **facet-OMP vs omp-lc .6311, McNemar p<.05.**
Then 3600s (headline bin, base .5461) only on a positive.

## Inherited constraints
NSHARD=1, cov_gate, bs=1, lmms-eval for paper-facing numbers only, question-**stem** only
(options never touch the selector — facets derive from the stem, never from candidates:
letting options in re-creates the 2026-07-15 query bug). scp only on pods, kill by PID.

## Outcomes
- **Positive:** first method result that moves the measured bottleneck; the 10 negatives
  become the motivation section; method paper is live.
- **Negative:** query axis closes alongside the selection axis. Two axes shut with a
  measured mechanism => the analysis paper ("What Actually Matters in Training-Free Frame
  Selection") is the deliverable, and Lever 2 is the last swing.
