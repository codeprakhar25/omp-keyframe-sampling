"""Scoring: answer accuracy + selector recall@k.

recall@k is the early-warning metric from the SPEC: it separates "the selector
dropped the needed frame" from "the answerer saw it but still failed".
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

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
