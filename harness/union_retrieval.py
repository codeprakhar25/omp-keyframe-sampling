"""Stage-1 retriever core (retrieve-then-ground, spec: research/retrieve_then_ground_spec.md).

PURE FUNCTIONS ONLY -- operate on plain score/time lists, no torch, no Frame objects, no I/O.
This is what makes `build_union_indices` unit-testable without a GPU (tests/test_union.py) and
replayable offline from a dumped score cache (scripts/union_ceiling.py). `harness/selectors.py`
wraps this in `TwoStageVLMSelector` to go from Frame objects -> scores -> union -> VLM prompt.

Algorithm (locked, see spec): adaptive noise floor -> local-maxima peaks above it -> greedy
temporal NMS -> expand surviving peaks to fixed-width windows -> union, frame-budgeted.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def _tau(scores: Sequence[float], k_std: float) -> float:
    """Adaptive per-video noise floor: mean + k_std * population-std of the score array."""
    n = len(scores)
    mu = sum(scores) / n
    var = sum((s - mu) ** 2 for s in scores) / n
    return mu + k_std * (var ** 0.5)


def _local_maxima(scores: Sequence[float]) -> List[int]:
    """Indices i where scores[i] >= both neighbors (non-strict; ends compare one side only).

    Non-strict so a flat plateau is a run of "local maxima" -- NMS (temporal, not score-strict)
    is what actually prunes redundant adjacent peaks down to one per neighborhood.
    """
    n = len(scores)
    out = []
    for i in range(n):
        left_ok = (i == 0) or (scores[i] >= scores[i - 1])
        right_ok = (i == n - 1) or (scores[i] >= scores[i + 1])
        if left_ok and right_ok:
            out.append(i)
    return out


def build_union_indices(
    scores: List[float],
    times: List[float],
    budget: int = 100,
    gmin: float = 10.0,
    pad: float = 2.0,
    k_std: float = 1.0,
) -> List[int]:
    """Frame-budgeted, multi-needle-safe union of candidate frame indices.

    1. tau = mean(scores) + k_std * std(scores)  -- adaptive noise floor, no fixed threshold.
    2. peaks = local maxima with score > tau.
    3. Greedy temporal NMS: sort peaks by score desc; accept a peak only if it is >= gmin
       seconds from EVERY already-accepted peak. This is what lets *disjoint* needles (e.g.
       2:20 and 8:10 apart) both survive -- NMS only suppresses near-duplicates of the SAME
       peak, not a second genuine needle far away.
    4. For each accepted peak, in score order, expand to [t-pad, t+pad] and pull in every
       frame whose time falls in that window (nearest-to-peak first, so a budget cutoff mid-
       window keeps the most relevant part). Accumulate into the union; stop once the union
       hits `budget` (a window can be partially added).
    5. Fallback: if no peak clears tau (e.g. all-flat scores), return the top-`budget` frames
       by raw score -- union is never empty as long as there are frames.

    Returns a sorted, de-duplicated list of frame indices. `scores` and `times` must align 1:1.
    """
    n = len(scores)
    if n == 0:
        return []
    if len(times) != n:
        raise ValueError(f"scores/times length mismatch: {n} vs {len(times)}")

    tau = _tau(scores, k_std)
    peaks = [i for i in _local_maxima(scores) if scores[i] > tau]

    if not peaks:  # nothing clears the noise floor -> flat top-budget by score
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        return sorted(order[:budget])

    # NMS: accept in score-desc order (stable tie-break by index), reject if too close to an
    # already-accepted peak.
    peaks.sort(key=lambda i: (-scores[i], i))
    accepted: List[int] = []
    for i in peaks:
        if all(abs(times[i] - times[j]) >= gmin for j in accepted):
            accepted.append(i)

    union: set = set()
    for i in accepted:  # still in score-desc order: highest-value peaks get first budget claim
        if len(union) >= budget:
            break
        lo, hi = times[i] - pad, times[i] + pad
        window = [j for j in range(n) if lo <= times[j] <= hi]
        window.sort(key=lambda j: abs(times[j] - times[i]))  # nearest-to-peak first
        for j in window:
            if len(union) >= budget:
                break
            union.add(j)

    return sorted(union)


def any_hit(union_times: Sequence[float], gold_spans: Optional[Sequence[Tuple[float, float]]],
            tol: float = 0.5) -> bool:
    """True if >=1 union timestamp lands inside ANY gold span (±tol, for 1fps quantization)."""
    if not gold_spans:
        return False
    return any(lo - tol <= t <= hi + tol for (lo, hi) in gold_spans for t in union_times)


def full_recall(union_times: Sequence[float], gold_spans: Optional[Sequence[Tuple[float, float]]],
                 tol: float = 0.5) -> bool:
    """True if EVERY gold span has >=1 union timestamp inside it (±tol). Multi-needle strict form."""
    if not gold_spans:
        return False
    return all(any(lo - tol <= t <= hi + tol for t in union_times) for (lo, hi) in gold_spans)
