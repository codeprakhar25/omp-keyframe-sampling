# Lever 2 spec — VLM-as-scorer (PARKED 2026-07-19, decide ~2026-07-20)

Status: **PARKED, not started.** Big decision — user reviewing. No pod needed until go.

## What it is
Replace LongCLIP cosine relevance r with a cross-attention scorer (frame+question seen
together). Pick-math stays untouched (`replay_selectors.py` consumes new r as-is).

## Why it's the last live accuracy lever
- Selection axis k=8 FULLY CLOSED (pick-math 10 variants, routing NEG#8, hybrid closed on prior).
- Dominant failure = moment-missed (35/52 fails @3600s, ~20/41 @600s) = scoring failure, not picking.
- LDDR App B (their words): selector quality bounded by external scorer; Table 4b: LongCLIP already
  best CLIP -> within-CLIP maxed, only exit is non-CLIP r.
- Precedent: Focus Table 10 same-pipeline swap SigLIP->BLIP ITM = +2.6; Evidential trained scorer +10 LVBench.

## Staged plan (each stage can kill it cheap)

**Stage 0 — scorer:** BLIP-2 ITM head (field-proven, no prompt engineering, ~1B).
Qwen3-VL-2B yes-token-prob = arm B only if ITM shows life.

**Stage 1 — 600s bin, k=8 (decision bin; ~1 GPU-day, 24GB pod):**
1. Preflight: raw 1-fps frames on volume? (cached embeds are LongCLIP-only; ITM needs pixels;
   re-decode from videos if missing). Coverage gate on pools before scoring.
2. ITM-score full 1-fps pool: ~150-200k frame-question forwards, batched, ~2-4h.
3. CPU replay: r_itm -> topk + OMP picks.
4. **FREE KILL-SWITCH before answerer:** pick-overlap vs LongCLIP picks. Overlap >90% -> null,
   stop, zero answerer cost. (Reference scale: query-bug fix moved 42% of picks and moved accuracy.)
5. If picks move: lmms-eval, 2 arms (topk-itm, omp-itm) x 412, bs=1, NSHARD=1, cov_gate.
6. **Pre-registered bar: omp-itm vs omp-lc .6311, McNemar p<.05.** Expected if real: +2-3pt.

**Stage 2 — only if Stage 1 positive:** 3600s (headline bin, base .5461). Pool ~10x bigger ->
coarse-to-fine (score every 5th frame, refine +-2 around peaks), ~2M -> ~500k forwards. 1-2 GPU-days.

## Outcomes
- Positive: first non-CLIP scorer row; attacks proven dominant failure; lifts all r-based methods;
  lands on LDDR's own admitted ceiling.
- Negative at 600s: scorer axis closes; thesis pivots fully to "answerer forgives frames /
  knife-edge noise" + token-reduction claim (cross_budget_mcnemar: half tokens beats uniform, both
  long bins, p=.0011/.018).

## Hard constraints (inherited)
NO doc-sharding (NSHARD=1, one bin per GPU); cov_gate before trusting; bs=1; lmms-eval only for
paper-facing numbers.
