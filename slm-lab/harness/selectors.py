"""Evidence selectors: choose which frames get sent to the frontier answerer.

- FullDumpSelector  -> condition A (send everything; upper bound on cost & accuracy)
- UniformSelector   -> cheap non-semantic baseline (evenly spaced k frames)
- EmbeddingSelector -> condition C (semantic top-k via SigLIP image-text relevance)

The EmbeddingSelector is the concrete, runnable stand-in for the spec's
"small selector". Swapping in a SmolVLM2 / Moondream scorer is a drop-in: implement
`select()` and register it in run.build_selector.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .media import Frame, frame_to_base64
from .union_retrieval import build_union_indices


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

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.name = "embedding:" + model_id.split("/")[-1]

    def _score(self, images: list, question: str, batch_size: int = 64):
        """SigLIP image-text relevance for a list of PIL images -> 1D score tensor.

        Batched: a whole long video at once (up to 1500 frames) OOMs a 400M encoder.
        SigLIP text encoder caps at 64 positions; long VQA prompts overflow -> truncate
        (selection needs question semantics, not options; the answerer sees full text).
        """
        torch = self.torch
        scores = []
        # bf16 autocast: 2-3x on Ampere+; cosine top-k ranking is insensitive to the
        # precision loss (scores used for ordering, never absolute thresholds)
        use_ac = self.device == "cuda"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                inputs = self.processor(
                    text=[question], images=chunk, return_tensors="pt",
                    padding="max_length", max_length=64, truncation=True,
                ).to(self.device)
                out = self.model(**inputs)
                scores.append(out.logits_per_image.squeeze(-1).float())  # [chunk]
        return torch.cat(scores)

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        if len(frames) <= k:
            return frames
        scores = self._score([f.image for f in frames], question, batch_size)
        topk = self.torch.topk(scores, k).indices.tolist()
        # preserve original temporal order of the picked frames
        return [frames[i] for i in sorted(topk)]


class PESelector(Selector):
    """Scorer-swap: Meta Perception Encoder (PE-Core) as the per-frame image-text scorer.

    Same job as EmbeddingSelector (rank frames by image-text relevance, take top-k) but with
    PE-Core instead of SigLIP. PE is not a plain HF AutoModel; it ships in Meta's
    `perception_models` repo (`core.vision_encoder.pe`). Point PE_REPO at the checkout (default
    /workspace/perception_models) so `import core...` resolves.

    Caveat baked in: PE-Core's text context is only 32 tokens (SigLIP so400m=64, Marengo=500),
    so question-style queries truncate harder here; a null result may partly reflect that, not
    encoder quality. Interface matches EmbeddingSelector (._score + .torch) so recall_vs_k.py
    can drive it unchanged.
    """

    name = "pe"

    def __init__(self, config: str = "PE-Core-L14-336", device: str | None = None,
                 repo: str | None = None):
        import os
        import sys

        repo = repo or os.environ.get("PE_REPO", "/workspace/perception_models")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pe.CLIP.from_config(config, pretrained=True).to(self.device).eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        self.name = "pe:" + config

    def _score(self, images: list, question: str, batch_size: int = 64):
        """PE-Core image-text relevance for a list of PIL images -> 1D score tensor.

        text tokenizer truncates to the model's 32-token context. Score = logit_scale *
        (image_features @ text_features.T), the same ranking signal SigLIP's logits_per_image gives.
        """
        torch = self.torch
        text = self.tokenizer([question]).to(self.device)
        scores = []
        with torch.no_grad(), torch.autocast(self.device if self.device != "cpu" else "cpu"):
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                px = torch.stack([self.preprocess(im) for im in chunk]).to(self.device)
                img_f, txt_f, logit_scale = self.model(px, text)
                s = (logit_scale * img_f @ txt_f.T).squeeze(-1)  # [chunk]
                scores.append(s.float())
        return torch.cat(scores)

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        if len(frames) <= k:
            return frames
        scores = self._score([f.image for f in frames], question, batch_size)
        topk = self.torch.topk(scores, k).indices.tolist()
        return [frames[i] for i in sorted(topk)]


class HierarchicalSelector(EmbeddingSelector):
    """Fork B: coarse-to-fine SigLIP selection (T*/VideoTree family).

    Flat top-k (EmbeddingSelector) collapses on long video: the one gold frame competes
    against thousands of distractors and many spuriously outscore it (Fork A: hit@k 1.0 ->
    0.24 as length grows). Coarse-to-fine narrows the haystack first:
      1. split the clip into `n_chunks` temporal windows,
      2. cheaply probe a few frames/window and score each window (max over probes),
      3. keep the top `top_chunks` windows,
      4. densely re-score every frame inside the kept windows, take the global top-k.
    The gold frame now only has to win *within* its window + its window must survive the
    coarse cut -> far fewer distractors than a flat 2000-way argmax.
    """

    name = "hier"

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str | None = None,
                 n_chunks: int = 32, coarse_probe: int = 4, top_chunks: int = 4):
        super().__init__(model_id, device)
        self.n_chunks = n_chunks
        self.coarse_probe = coarse_probe
        self.top_chunks = top_chunks
        self.name = f"hier:{model_id.split('/')[-1]}(nc{n_chunks},p{coarse_probe},tc{top_chunks})"

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        import numpy as np
        torch = self.torch
        n = len(frames)
        if n <= k:
            return frames

        # 1. temporal windows
        nch = int(min(self.n_chunks, max(1, n // max(self.coarse_probe, 1))))
        bounds = np.linspace(0, n, nch + 1).round().astype(int)
        chunks = [list(range(bounds[c], bounds[c + 1])) for c in range(nch) if bounds[c + 1] > bounds[c]]

        # 2. coarse probe: a few frames per window, one batched score pass
        probe_idx: List[int] = []
        probe_owner: List[int] = []
        for ci, ch in enumerate(chunks):
            sel = np.linspace(0, len(ch) - 1, min(self.coarse_probe, len(ch))).round().astype(int)
            for s in sel:
                probe_idx.append(ch[int(s)])
                probe_owner.append(ci)
        pscores = self._score([frames[i].image for i in probe_idx], question, batch_size)
        cscore = [-1e30] * len(chunks)
        for j, ci in enumerate(probe_owner):
            cscore[ci] = max(cscore[ci], float(pscores[j]))

        # 3. keep top windows (enough frames to hold k)
        order = sorted(range(len(chunks)), key=lambda c: cscore[c], reverse=True)
        kept, cand = [], []
        for c in order:
            kept.append(c)
            cand = [i for c2 in sorted(kept) for i in chunks[c2]]
            if len(kept) >= self.top_chunks and len(cand) >= k:
                break

        # 4. fine: dense re-score inside kept windows, global top-k
        fscores = self._score([frames[i].image for i in cand], question, batch_size)
        top = torch.topk(fscores, min(k, len(cand))).indices.tolist()
        picked = sorted(cand[t] for t in top)
        return [frames[i] for i in picked]


class BeamCoarseToFineSelector(EmbeddingSelector):
    """Fork B, redesigned: beam coarse-to-fine descent, validated by the region-recall gate (Jul 7).

    The old HierarchicalSelector failed because it asked the coarse pass to find the exact *frame*
    (sparse probe + greedy single-window hard-prune -> dropped the 1-2s needle). The region-recall
    gate reframed the coarse job as "find the right *chunk*" and measured it: with SigLIP, M=4,
    max-aggregation, the gold chunk lands in the **top-2 of 4** ~84-88% of the time at 600s/3600s
    (vs frame-level hit@6 of only .36/.24, and clear of the random-chunk null). That result locks
    the design:
      - split each surviving window into `M` sub-windows (M=4 is the sweet spot; finer decays to null),
      - window score = **max** frame score inside it (peak, not mean),
      - keep the top `beam` windows (beam=2, NOT greedy-1: region@1 .56-.64 is too low to compound
        over levels, region@2 .84-.88 survives two levels ~.77),
      - recurse until windows are small, then take the global top-k *within the surviving windows*.

    Crucially the whole video is scored **once, densely** (probe-sparsity hurt: rr@2 .52->.84 as
    probe 1->all). That costs seconds for SigLIP; the payoff is the *answerer* seeing k frames not
    500 -- the compression win was never in the selector. Beam descent then just shrinks the
    distractor pool the final top-k competes in.

    Next lever (NOT yet built/validated -> intentionally absent): temporal *zoom* -- re-decode the
    surviving window from the SOURCE video at >1fps so a 1-2s needle becomes many frames. That needs
    the source mp4 (we run on pre-extracted 1fps JPEGs), so it's a separate code path gated on
    `self._item`; adding it here untested would overstate what's measured.
    """

    name = "beam"

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str | None = None,
                 M: int = 4, beam: int = 2, max_levels: int = 8, stop_frames: int | None = None):
        super().__init__(model_id, device)
        self.M = M
        self.beam = beam
        self.max_levels = max_levels
        self.stop_frames = stop_frames  # stop descending once a window has <= this many frames
        self.name = f"beam:{model_id.split('/')[-1]}(M{M},b{beam})"

    def _descend(self, scores, n: int, k: int) -> List[int]:
        """Beam coarse-to-fine over precomputed per-frame `scores`. Returns candidate frame indices."""
        stop = self.stop_frames or max(4 * k, self.M)
        windows = [(0, n)]  # (lo, hi) half-open, in frame-index space
        for _ in range(self.max_levels):
            if all((hi - lo) <= stop or (hi - lo) < self.M for (lo, hi) in windows):
                break
            subs = []  # (score, lo, hi)
            for (lo, hi) in windows:
                size = hi - lo
                if size < self.M:  # too small to split -> carry forward intact
                    subs.append((max(float(scores[i]) for i in range(lo, hi)), lo, hi))
                    continue
                for c in range(self.M):
                    slo = lo + c * size // self.M
                    shi = lo + (c + 1) * size // self.M
                    if shi > slo:
                        subs.append((max(float(scores[i]) for i in range(slo, shi)), slo, shi))
            subs.sort(key=lambda t: t[0], reverse=True)
            windows = [(lo, hi) for (_, lo, hi) in subs[: self.beam]]
        return [i for (lo, hi) in sorted(windows) for i in range(lo, hi)]

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        n = len(frames)
        if n <= k:
            return frames
        scores = self._score([f.image for f in frames], question, batch_size)  # dense, once
        cand = self._descend(scores, n, k)
        # global top-k WITHIN the surviving windows, by the same dense scores
        cand_scores = self.torch.tensor([float(scores[i]) for i in cand])
        top = self.torch.topk(cand_scores, min(k, len(cand))).indices.tolist()
        picked = sorted(cand[t] for t in top)
        return [frames[i] for i in picked]


class TwoStageVLMSelector(EmbeddingSelector):
    """Retrieve-then-ground, Fork B reopened (research/retrieve_then_ground_spec.md, Jul 8).

    Beam coarse-to-fine (above) failed because both stages re-ranked the SAME SigLIP cosine,
    which goes flat inside one scene (region@2 ~.84 but hit@6 stuck at .34/.20 -- "neighborhood
    findable, exact frame not"). This selector swaps stage 2 for a different *kind* of model:

      Stage 1 (search, cosine):  dense SigLIP score over every frame (same `._score` as beam) ->
        `build_union_indices` (peak-NMS, harness/union_retrieval.py) turns the score array into a
        frame-budgeted union of candidate windows. Multi-needle-safe by construction: disjoint
        peaks far apart (e.g. 2:20 and 8:10) both survive NMS.
      Stage 2 (localize, generative): the union frames, in temporal order and each tagged
        `[t=<sec>s]`, go into ONE multi-image prompt to a reasoning VLM (gpt-5.5), which reads
        them jointly and names the evidence timestamp(s) -- the thing a per-frame cosine cannot
        do within a single scene.

    Robust by construction: if the VLM stage returns nothing parseable, falls back to SigLIP
    top-k *within the union* -- so twostage can never score worse than flat-in-union (same
    fallback pattern as TranscriptGatedSelector / beam's own top-k-within-survivors).

    The Stage-2 API call is isolated in `_vlm_localize` (only place that touches the network) so
    tests can monkeypatch/mock it without torch or a network connection. Phase 1 does NOT call
    the real API; `vlm_model` / `_client` are lazily created on first real use only.
    """

    name = "twostage"

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str | None = None,
                 vlm_model: str = "gpt-5.5", budget: int = 100, gmin: float = 10.0, pad: float = 2.0,
                 k_std: float = 1.0, reasoning_effort: str = "low"):
        super().__init__(model_id, device)
        self.vlm_model = vlm_model
        self.budget = budget
        self.gmin = gmin
        self.pad = pad
        self.k_std = k_std
        self.reasoning_effort = reasoning_effort
        self._client = None  # lazy OpenAI client -- never touched in Phase 1 / tests
        self.name = f"twostage:{model_id.split('/')[-1]}+{vlm_model}"

    def _vlm_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _vlm_localize(self, union_frames: List[Frame], question: str) -> Optional[List[float]]:
        """Stage-2 API call, isolated so it's the only method a test needs to mock.

        Builds one multi-image prompt: union frames in temporal order, each preceded by its
        `[t=Xs]` tag, and asks for the evidence timestamp(s). Returns parsed floats, or None on
        any failure (bad response, network error, unparseable text) -- caller must fall back.
        """
        content: list = [{
            "type": "text",
            "text": (
                f"Question: {question}\n"
                "The images below are frames from a video, in time order, each preceded by its "
                "timestamp tag. Reply with ONLY the timestamp(s) in seconds (comma-separated if "
                "more than one) of the frame(s) showing the visual evidence needed to answer."
            ),
        }]
        for f in union_frames:
            content.append({"type": "text", "text": f"[t={f.seconds:.1f}s]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_to_base64(f)}"},
            })
        try:
            resp = self._vlm_client().chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=128,
                reasoning_effort=self.reasoning_effort,
            )
            text = resp.choices[0].message.content or ""
        except Exception:
            return None
        return self._parse_timestamps(text)

    @staticmethod
    def _parse_timestamps(text: str) -> Optional[List[float]]:
        import re

        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return [float(n) for n in nums] if nums else None

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        n = len(frames)
        if n <= k:
            return frames

        # Stage 1: dense SigLIP score, once, then peak-NMS union (pure function, no torch inside)
        scores = self._score([f.image for f in frames], question, batch_size)
        times = [f.seconds if f.seconds is not None else float(i) for i, f in enumerate(frames)]
        union_idx = build_union_indices(
            [float(s) for s in scores], times,
            budget=self.budget, gmin=self.gmin, pad=self.pad, k_std=self.k_std,
        )
        union_frames = [frames[i] for i in union_idx]  # already time-ordered (indices are)

        # Stage 2: VLM localizes within the union
        vlm_ts = self._vlm_localize(union_frames, question)
        if vlm_ts:
            picked_idx: List[int] = []
            seen = set()
            for t in vlm_ts:
                j = min(range(len(union_idx)), key=lambda j: abs(times[union_idx[j]] - t))
                gi = union_idx[j]
                if gi not in seen:
                    seen.add(gi)
                    picked_idx.append(gi)
                if len(picked_idx) >= k:
                    break
            if picked_idx:
                return [frames[i] for i in sorted(picked_idx)]

        # Fallback: VLM parse failed / returned nothing usable -> SigLIP top-k WITHIN the union
        # (never worse than beam-in-union; the union is already the frame-budgeted candidate set)
        union_scores = self.torch.tensor([float(scores[i]) for i in union_idx])
        top = self.torch.topk(union_scores, min(k, len(union_idx))).indices.tolist()
        picked = sorted(union_idx[t] for t in top)
        return [frames[i] for i in picked]


def _subs_to_seconds(ts: str) -> float:
    """'HH:MM:SS.mmm' -> seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_JUNK_LINES = {"", "foreign", "[music]", "[applause]", "[laughter]", "(music)", "..."}


class TranscriptGatedSelector(EmbeddingSelector):
    """Fork B lever 2: use the subtitle track to *localize* candidate time regions, then run
    SigLIP top-k only inside them.

    Motivation (recall_vs_k diagnostic): at 3600s the collapse is a *scoring* wall — SigLIP
    can't rank the needle among ~2400 frames even at k=40 (recall .24->.32). A different modality
    (audio/transcript) can localize the region cheaply; restricting the visual search to a few
    transcript-matched windows shrinks the distractor set by ~100x so SigLIP only has to win locally.

    Pipeline: score each subtitle segment's text vs the question (text-text retriever) -> keep the
    top windows (+pad) -> candidate = frames inside those windows -> SigLIP top-k within candidates.
    Falls back to flat top-k when the video has no usable transcript. The manifest item (with the
    subtitle path) is passed by run.run_condition via `self._item`.
    """

    name = "transcript"

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str | None = None,
                 subtitles_dir: str = "data/subtitles",
                 text_model: str = "BAAI/bge-small-en-v1.5",
                 top_segments: int = 4, pad_sec: float = 6.0):
        super().__init__(model_id, device)
        from sentence_transformers import SentenceTransformer

        self.subtitles_dir = subtitles_dir
        self.top_segments = top_segments
        self.pad_sec = pad_sec
        self.txt = SentenceTransformer(text_model, device=self.device)
        self._item = None
        self.name = f"transcript:{text_model.split('/')[-1]}(ts{top_segments},pad{int(pad_sec)})"

    def _load_segments(self, item: dict):
        import os
        sf = (item or {}).get("subtitle_file")
        if not sf:
            return []
        path = os.path.join(self.subtitles_dir, sf)
        if not os.path.exists(path):
            return []
        try:
            raw = json.load(open(path))
        except Exception:
            return []
        segs = []
        for e in raw:
            line = str(e.get("line", "")).strip()
            if line.lower() in _JUNK_LINES or len(line) < 2:
                continue
            try:
                s = _subs_to_seconds(e["start"]); en = _subs_to_seconds(e["end"])
            except Exception:
                continue
            segs.append((s, en, line))
        return segs

    def select(self, frames: List[Frame], question: str, k: int, batch_size: int = 64) -> List[Frame]:
        torch = self.torch
        n = len(frames)
        if n <= k:
            return frames

        segs = self._load_segments(getattr(self, "_item", None))
        if not segs:  # no usable transcript -> flat top-k (Fork-A behaviour)
            return super().select(frames, question, k, batch_size)

        # 1. text-text retrieval: question vs each subtitle line
        import numpy as np
        qv = self.txt.encode([question], normalize_embeddings=True)
        sv = self.txt.encode([t for _, _, t in segs], normalize_embeddings=True, batch_size=128)
        sims = (sv @ qv[0])  # cosine, normalized

        # 2. expand top segments into padded time windows until we have >=k candidate frames
        order = list(np.argsort(sims)[::-1])
        cand: List[int] = []
        used = 0
        for rank in order:
            s, en, _ = segs[int(rank)]
            lo, hi = s - self.pad_sec, en + self.pad_sec
            cand = sorted({j for j, f in enumerate(frames)
                           if f.seconds is not None and lo <= f.seconds <= hi} | set(cand))
            used += 1
            if used >= self.top_segments and len(cand) >= k:
                break
        if len(cand) < k:  # transcript windows too thin -> flat fallback
            return super().select(frames, question, k, batch_size)

        # 3. SigLIP top-k *within* the transcript-gated candidates
        fscores = self._score([frames[j].image for j in cand], question, batch_size)
        top = torch.topk(fscores, min(k, len(cand))).indices.tolist()
        picked = sorted(cand[t] for t in top)
        return [frames[i] for i in picked]


class SmolVLMSelector(Selector):
    """Condition C2: a small *generative* VLM scores each frame by reasoning about whether
    it answers the question (P("yes")), instead of embedding-matching like SigLIP.

    Hypothesis (from S1): embedding selection fails on near-identical screens because it
    matches appearance, not meaning. A generative VLM that *reads* the frame should separate
    them. Cost: ~100x SigLIP's compute, still tiny vs the frontier answerer.

    Lazy-imports torch/transformers. ~1GB weights (SmolVLM2-500M).
    """

    name = "smolvlm"

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        device: str | None = None,
        longest_edge: int | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        proc_kwargs = {}
        if longest_edge:
            # raise the processor's input resolution so fine UI text survives; the
            # image splitter then tiles the larger image into more sub-crops
            proc_kwargs["size"] = {"longest_edge": longest_edge}
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        eff = getattr(getattr(self.processor, "image_processor", None), "size", None)
        print(f"[smolvlm] model={model_id} effective size={eff}")
        self.name = "smolvlm:" + model_id.split("/")[-1] + (f"@{longest_edge}" if longest_edge else "")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        tok = self.processor.tokenizer
        # token ids for yes/no (capitalised + lowercase variants), for a P(yes) relevance score
        self.yes_ids = {tok(v, add_special_tokens=False).input_ids[0] for v in (" Yes", "Yes", " yes", "yes")}
        self.no_ids = {tok(v, add_special_tokens=False).input_ids[0] for v in (" No", "No", " no", "no")}

    def _score(self, image, question: str) -> float:
        torch = self.torch
        prompt = (
            f"Question: {question}\n"
            "Does THIS image contain the answer or the key visual evidence for the question? "
            "Answer only Yes or No."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)  # no dtype cast: input_ids must stay int; model handles pixel dtype
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        yes = float(sum(probs[i] for i in self.yes_ids))
        no = float(sum(probs[i] for i in self.no_ids))
        return yes / (yes + no + 1e-9)  # normalised P(yes) so frames are comparable

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        if len(frames) <= k:
            return frames
        scored = [(self._score(f.image, question), i) for i, f in enumerate(frames)]
        topk = [i for _, i in sorted(scored, reverse=True)[:k]]
        return [frames[i] for i in sorted(topk)]

class VideoRetSelector(Selector):
    """Fork B close-out: video-native temporal scorer (X-CLIP), the on-thesis swap.

    SigLIP (EmbeddingSelector) is an image-text *matching* model — it scores single stills
    and cannot use temporal context. The Fork-B wall (hit@k .36/.24 at 600/3600s) might be
    SigLIP's, not cheap-selection's. This scorer swaps ONLY the model for a video-text
    encoder built for temporal retrieval (X-CLIP has cross-frame attention), everything else
    fixed: same manifest, same k, hit@k via echo. See research/scorer_swap_spec.md.

    Clip-window protocol: tile the clip into `window_frames`-length windows (stride = window,
    non-overlap), score each window vs the question with the video tower, assign the score to
    the window's centre frame, take the top-k centre frames. The scorer may look at more frames
    internally (cheap), but returns exactly k frames -> equal downstream budget vs SigLIP.

    NOTE: the spec's first-choice model was LanguageBind_Video_FT; it imports on torch 2.11
    (with a torchvision functional_tensor shim) but its processor is decord/file-path only, so
    feeding pre-extracted JPEG windows would need a temp video per window (~2400/clip). X-CLIP's
    processor takes PIL frame lists natively and is the same class of test (temporal video-text).
    """

    name = "videoret"

    def __init__(self, model_id: str = "microsoft/xclip-base-patch16-zero-shot",
                 device: str | None = None, window_frames: int | None = None, stride: int | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        cfg = self.model.config
        self.num_frames = int(getattr(cfg, "num_frames", None)
                              or getattr(getattr(cfg, "vision_config", cfg), "num_frames", 8))
        self.window_frames = window_frames or self.num_frames
        self.stride = stride or self.window_frames
        self.name = "videoret:" + model_id.split("/")[-1]

    def _window_clip(self, frames: List[Frame], start: int):
        """Return (num_frames PIL images spanning [start, start+window), centre_index)."""
        import numpy as np
        n = len(frames)
        end = min(start + self.window_frames, n)
        span = list(range(start, end))
        # sample exactly num_frames from the span (pad by repeating last if span shorter)
        pick = np.linspace(0, len(span) - 1, self.num_frames).round().astype(int)
        clip = [frames[span[p]].image for p in pick]
        centre = span[len(span) // 2]
        return clip, centre

    def _score_windows(self, clips, question: str, batch_size: int = 16):
        # The combined XCLIPProcessor(videos=...) kwarg silently drops pixel_values in this
        # transformers version; call image_processor + tokenizer separately (version-proof).
        import numpy as np
        torch = self.torch
        ip = self.processor.image_processor
        tok = self.processor.tokenizer
        text = tok([question], return_tensors="pt", padding=True, truncation=True).to(self.device)
        scores = []
        with torch.no_grad():
            for i in range(0, len(clips), batch_size):
                chunk = clips[i:i + batch_size]            # list of clips; each = list of PIL frames
                vids = [[np.asarray(im) for im in clip] for clip in chunk]
                pv = ip(vids, return_tensors="pt")["pixel_values"].to(self.device)  # [B, num_frames, 3, H, W]
                b = pv.shape[0]
                out = self.model(
                    pixel_values=pv,
                    input_ids=text["input_ids"].expand(b, -1),
                    attention_mask=text["attention_mask"].expand(b, -1),
                )
                # logits_per_video: [n_videos, n_text]; single question -> column 0
                scores.append(out.logits_per_video[:, 0].float())
        return torch.cat(scores)

    def select(self, frames: List[Frame], question: str, k: int) -> List[Frame]:
        if len(frames) <= k:
            return frames
        starts = list(range(0, len(frames), self.stride))
        clips, centres = [], []
        for s in starts:
            clip, c = self._window_clip(frames, s)
            clips.append(clip)
            centres.append(c)
        scores = self._score_windows(clips, question)
        kk = min(k, len(centres))
        top = self.torch.topk(scores, kk).indices.tolist()
        picked = sorted({centres[i] for i in top})
        return [frames[i] for i in picked]
