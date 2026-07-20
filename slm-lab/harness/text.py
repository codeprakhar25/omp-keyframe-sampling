"""Query text for the retrieval/scoring stage.

The manifest's "question" field is the ANSWERER's prompt: stem + options + the
"Answer with the single option letter." instruction. Scorers must not see it --
they rank frames by image-text relevance, and the options are 3/4 distractors.

Reference: the official AKS implementation (CVPR 2025, feature_extract.py) passes
`data['question']` alone to its CLIP and BLIP branches. Only its SeViLA branch --
a QA-tuned ITM head explicitly asked "Is this a good frame can answer the
question?" -- appends candidates. SigLIP/LongCLIP are CLIP-family, so: stem only.

Derived, not stored: manifests predating 2026-07-15 have no "question_stem" key,
and rebuilding them to add one would reshuffle the subset away from the cached
embeds (which are keyed on id). Prefer the key, fall back to splitting.
"""
from __future__ import annotations

_MARKER = "\n\nOptions:"


def question_stem(item) -> str:
    """Scorer/retriever query: the bare question, no options, no answer instruction.

    Accepts a manifest item (dict) or a raw question string.
    """
    if isinstance(item, dict):
        stem = item.get("question_stem")
        if stem:
            return stem.strip()
        q = item.get("question", "")
    else:
        q = item or ""
    # "\n\nOptions:" is what build_longvideobench.py emits; bare "Options:" is the
    # loose fallback posterior_killtest.py used, kept for hand-built manifests.
    idx = q.find(_MARKER)
    if idx == -1:
        idx = q.find("Options:")
    return (q[:idx] if idx != -1 else q).strip()
