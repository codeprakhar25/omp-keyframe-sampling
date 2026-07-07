"""Scoring: answer accuracy + selector recall@k.

recall@k is the early-warning metric from the SPEC: it separates "the selector
dropped the needed frame" from "the answerer saw it but still failed".
"""

from __future__ import annotations

import re
from math import comb
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .media import Frame


def normalize(s: Optional[str]) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, gold: str) -> float:
    """Lenient containment match on normalized text. 1.0 if either contains the other."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0


JUDGE_PROMPT = (
    "You are grading a model's answer against a reference.\n"
    "Question: {q}\n"
    "Reference answer: {gold}\n"
    "Model answer: {pred}\n"
    'Do they share the same key facts? Reply with only "yes" or "no".'
)


def llm_judge(question: str, pred: str, gold: str, judge_fn: Callable[[str], str]) -> float:
    verdict = judge_fn(JUDGE_PROMPT.format(q=question, gold=gold, pred=pred))
    return 1.0 if verdict.strip().lower().startswith("y") else 0.0


def recall_at_k(selected: List[Frame], item: dict) -> Optional[float]:
    """Fraction of gold evidence (time spans or frame indices) covered by selected frames.

    Returns None when the item carries no gold evidence annotation.
    """
    mtype = item.get("media_type", "video")
    if mtype == "video":
        spans = item.get("gold_evidence_seconds")
        if not spans:
            return None
        covered = sum(
            1
            for (lo, hi) in spans
            if any(f.seconds is not None and lo <= f.seconds <= hi for f in selected)
        )
        return covered / len(spans)

    gold_idx = item.get("gold_evidence_frames")
    if not gold_idx:
        return None
    sel_idx = {f.index for f in selected}
    covered = sum(1 for gi in gold_idx if gi in sel_idx)
    return covered / len(gold_idx)


def hit_at_k(selected: List[Frame], item: dict) -> Optional[float]:
    """1.0 if the selected frames contain AT LEAST ONE gold-evidence frame, else 0.0.

    The right diagnostic when the answer *persists* across a gold span (stitched needle):
    any single target frame suffices to answer, so coverage-fraction (recall_at_k) under-
    counts. hit@k asks the question that actually predicts accuracy: "is the answer in the
    payload at all?". Returns None when the item has no gold annotation.
    """
    mtype = item.get("media_type", "video")
    if mtype == "video":
        spans = item.get("gold_evidence_seconds")
        if not spans:
            return None
        hit = any(
            f.seconds is not None and lo <= f.seconds <= hi
            for (lo, hi) in spans
            for f in selected
        )
        return 1.0 if hit else 0.0
    gold_idx = item.get("gold_evidence_frames")
    if not gold_idx:
        return None
    sel_idx = {f.index for f in selected}
    return 1.0 if any(gi in sel_idx for gi in gold_idx) else 0.0


# --- region-recall gate (Fork B go/no-go) --------------------------------------------------
# "Can the cheap scorer find the right CHUNK, even when it misses the exact frame?"
# All pure functions over precomputed per-frame float scores + gold_evidence_seconds.


def _chunk_bounds(n: int, M: int) -> List[Tuple[int, int]]:
    """Contiguous, ~equal split of n frame indices into M chunks: list of [lo, hi) index pairs."""
    return [(i * n // M, (i + 1) * n // M) for i in range(M)]


def gold_chunks(frames: List[Frame], item: dict, M: int) -> Optional[set]:
    """Indices of chunks that contain >=1 frame inside a gold_evidence_seconds span.

    Returns None when the item has no usable gold. A gold span straddling a chunk boundary can
    mark two chunks (noted in the spec as a mild recall inflation for long spans).
    """
    spans = item.get("gold_evidence_seconds")
    if not spans:
        return None
    n = len(frames)
    if n == 0:
        return None
    gc = set()
    for ci, (lo, hi) in enumerate(_chunk_bounds(n, M)):
        for f in frames[lo:hi]:
            if f.seconds is not None and any(a <= f.seconds <= b for (a, b) in spans):
                gc.add(ci)
                break
    return gc or None


def _probe_indices(lo: int, hi: int, probe: Optional[int]) -> List[int]:
    """Evenly-spaced (deterministic) probe indices in [lo, hi). probe=None/>=size -> dense (all)."""
    size = hi - lo
    if probe is None or probe >= size:
        return list(range(lo, hi))
    if probe <= 1:
        return [lo + size // 2]
    step = (size - 1) / (probe - 1)
    return [lo + round(i * step) for i in range(probe)]


def _aggregate(vals: Sequence[float], agg: str) -> float:
    if not vals:
        return float("-inf")
    if agg == "max":
        return max(vals)
    mu = sum(vals) / len(vals)
    if agg == "mean":
        return mu
    if agg == "ucb":  # mean + 1.0*std, cheap optimism stand-in
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return mu + var ** 0.5
    raise ValueError(f"unknown agg {agg!r}")


def region_recall(
    scores: Sequence[float],
    frames: List[Frame],
    item: dict,
    M: int,
    agg: str = "max",
    probe: Optional[int] = None,
    ms: Sequence[int] = (1, 2, 3),
) -> Optional[Dict[int, float]]:
    """1.0 if a gold chunk lands in the top-m chunks (by aggregated frame score), else 0.0, per m.

    Returns None when the item has no usable gold. `scores` aligns 1:1 with `frames`.
    """
    gc = gold_chunks(frames, item, M)
    if gc is None:
        return None
    chunk_scores = [
        _aggregate([scores[i] for i in _probe_indices(lo, hi, probe)], agg)
        for (lo, hi) in _chunk_bounds(len(frames), M)
    ]
    order = sorted(range(M), key=lambda c: chunk_scores[c], reverse=True)
    return {m: (1.0 if set(order[:m]) & gc else 0.0) for m in ms}


def random_null(
    frames: List[Frame], item: dict, M: int, ms: Sequence[int] = (1, 2, 3)
) -> Optional[Dict[int, float]]:
    """Expected region_recall@m under random chunk ranking: 1 - C(M-G, m)/C(M, m), G = #gold chunks."""
    gc = gold_chunks(frames, item, M)
    if gc is None:
        return None
    G = len(gc)
    out = {}
    for m in ms:
        if m >= M or (M - G) < m:
            out[m] = 1.0
        else:
            out[m] = 1.0 - comb(M - G, m) / comb(M, m)
    return out


def wilson_ci(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for a binomial rate (k successes out of n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((center - half) / denom, (center + half) / denom)
