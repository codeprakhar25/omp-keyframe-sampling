"""Single source of truth for resolving a cached per-frame image embedding.

There are two caches with different layouts, for historical reasons:
  results/embeds/<qid>.npz     keys: times, siglip (N,1152), longclip (N,768), dinov2
  results/embeds_lc/<qid>.npz  keys: times, emb (N,768)   <- the 600s LongCLIP fill

So "the LongCLIP embedding for qid" lives under key 'longclip' for 15s/60s and under
key 'emb' in a different directory for 600s. Every consumer needs that rule, and the
2026-07-15 retro is explicit about why it lives in exactly one place: the question-stem
fix already existed twice (posterior_killtest.py, _build_hunt_packs.py) and never
propagated, so the bug survived a week. Copy #3 of a resolution rule is how that repeats.

Both caches are CLEAN: dump_embeds.py only ever calls get_image_features /
encode_image / dino. No text ever touched them.
"""
from __future__ import annotations

import os

import numpy as np

KEY = {"sig": "siglip", "lc": "longclip"}


def load_image_embed(qid: str, scorer: str, emb_dir: str = "results/embeds",
                     emb_lc_dir: str = "results/embeds_lc"):
    """-> (times[N], emb[N,D]) or (None, None) if not cached.

    scorer: "sig" | "lc". Prefers emb_dir; for "lc" falls back to emb_lc_dir/'emb'.
    Returns raw (unnormalized) embeddings — normalize at the call site.

        Notes on np.load / np.asarray:
        - np.load(p) on a .npz file returns an NpzFile (like a dict) where you can access
            stored arrays by name (e.g. z['times'], z['emb']). It does not copy data until
            you convert it to an array.
        - np.asarray(x, dtype=...) converts the input to a numpy ndarray with the
            requested dtype. It's used here to ensure a consistent float32 output.

        Example:
            # suppose results/embeds/foo.npz contains arrays 'times' and 'longclip'
            times, emb = load_image_embed('foo', 'lc')
            # times is an (N,) float32 array, emb is an (N,D) float32 array
            emb_norm = l2(emb)  # normalize per-row
    """
    key = KEY[scorer]
    p = os.path.join(emb_dir, qid + ".npz")
    if os.path.exists(p):
        # np.load returns an NpzFile (mapping of arrays). Accessing z[key]
        # gives an array-like object; np.asarray ensures a concrete ndarray
        # with dtype float32 before returning.
        z = np.load(p)
        if key in z.files:
            return np.asarray(z["times"], dtype=np.float32), np.asarray(z[key])
    if scorer == "lc":
        p = os.path.join(emb_lc_dir, qid + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            if "emb" in z.files:
                return np.asarray(z["times"], dtype=np.float32), np.asarray(z["emb"])
    return None, None


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
