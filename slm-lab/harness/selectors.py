"""Evidence selectors: choose which frames get sent to the frontier answerer.

- FullDumpSelector  -> condition A (send everything; upper bound on cost & accuracy)
- UniformSelector   -> cheap non-semantic baseline (evenly spaced k frames)
- EmbeddingSelector -> condition C (semantic top-k via SigLIP image-text relevance)

The EmbeddingSelector is the concrete, runnable stand-in for the spec's
"small selector". Swapping in a SmolVLM2 / Moondream scorer is a drop-in: implement
`select()` and register it in run.build_selector.
"""

from __future__ import annotations

from typing import List

from .media import Frame


class Selector:
    name = "base"

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        raise NotImplementedError


class FullDumpSelector(Selector):
    """Condition A: pass through all candidate frames (already capped upstream)."""

    name = "full_dump"

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        return frames


class UniformSelector(Selector):
    """Non-semantic baseline: evenly spaced k frames. No deps, always runs."""

    name = "uniform"

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        if len(frames) <= k:
            return frames
        import numpy as np

        idxs = np.linspace(0, len(frames) - 1, k).round().astype(int)
        out: List[Frame] = []
        seen = set()
        for i in idxs:
            i = int(i)
            if i not in seen:
                seen.add(i)
                out.append(frames[i])
        return out


class EmbeddingSelector(Selector):
    """Condition C: rank frames by SigLIP image-text relevance to the question, take top-k.

    Lazy-imports torch/transformers so the rest of the harness runs without them.
    """

    name = "embedding"

    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        if len(frames) <= k:
            return frames
        torch = self.torch
        images = [f.image for f in frames]
        with torch.no_grad():
            inputs = self.processor(
                text=[question], images=images, return_tensors="pt", padding="max_length"
            ).to(self.device)
            out = self.model(**inputs)
            scores = out.logits_per_image.squeeze(-1)  # [n_images]
            topk = torch.topk(scores, k).indices.tolist()
        # preserve original temporal order of the picked frames
        return [frames[i] for i in sorted(topk)]
