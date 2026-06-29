"""Harder synthetic set: needle-in-haystack to pressure-test the selector + gate.

Each item is a long sequence of frames where most are plausible distractors
(gray "REF: #####") and exactly one is the needle (red border + ALERT banner +
"CODE: ####"). The question asks for the ALERT code.

Why this shape:
  - condition A (all frames) should answer correctly -> high accuracy ceiling.
  - condition C with `uniform` usually MISSES the single needle -> recall@k and
    accuracy drop. This is the failure the gate must catch.
  - condition C with `embedding` (SigLIP) can query "red ALERT" and recover the
    needle -> the contrast that tells us a semantic selector is worth it.

recall@k does NOT depend on the answerer, so run `--answerer echo` first to inspect
selection quality for free before spending API calls.

    python scripts/make_synthetic_hard.py --n-items 12 --frames 40 --seed 0

Then (free recall check):
    python -m harness.run --manifest data/manifest.synthetic_hard.json \
        --conditions A C --answerer echo --selector uniform --k 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_distractor(idx: int, ref: int) -> Image.Image:
    img = Image.new("RGB", (384, 288), (28, 28, 36))
    d = ImageDraw.Draw(img)
    d.text((10, 8), f"frame {idx:02d}", fill=(120, 120, 130), font=_font(20))
    d.text((70, 120), f"REF: {ref:05d}", fill=(150, 150, 160), font=_font(40))
    return img


def _draw_needle(idx: int, code: int) -> Image.Image:
    img = Image.new("RGB", (384, 288), (28, 28, 36))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, 379, 283], outline=(220, 40, 40), width=10)
    d.text((10, 8), f"frame {idx:02d}", fill=(120, 120, 130), font=_font(20))
    d.text((110, 50), "ALERT", fill=(230, 60, 60), font=_font(56))
    d.text((70, 140), f"CODE: {code:04d}", fill=(245, 230, 230), font=_font(48))
    return img


def _uniform_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    import numpy as np

    return sorted({int(i) for i in np.linspace(0, n - 1, k).round().astype(int)})


def main() -> None:
    ap = argparse.ArgumentParser(description="generate needle-in-haystack synthetic items")
    ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="k used only to report expected uniform recall")
    ap.add_argument("--out-root", default="data/synthetic_hard")
    ap.add_argument("--manifest", default="data/manifest.synthetic_hard.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_root, exist_ok=True)

    manifest = []
    uniform_hits = 0
    for n in range(args.n_items):
        item_id = f"needle-{n:03d}"
        item_dir = os.path.join(args.out_root, item_id)
        os.makedirs(item_dir, exist_ok=True)

        target = rng.randrange(args.frames)
        code = rng.randint(1000, 9999)
        used_refs = set()
        for i in range(args.frames):
            if i == target:
                img = _draw_needle(i, code)
            else:
                ref = rng.randint(10000, 99999)
                while ref in used_refs:
                    ref = rng.randint(10000, 99999)
                used_refs.add(ref)
                img = _draw_distractor(i, ref)
            img.save(os.path.join(item_dir, f"frame_{i:02d}.png"))

        if target in _uniform_indices(args.frames, args.k):
            uniform_hits += 1

        manifest.append(
            {
                "id": item_id,
                "media_type": "images",
                "media_path": item_dir,
                "question": (
                    "One frame is a red ALERT frame showing 'CODE: NNNN'. "
                    "What is that ALERT code? Answer with the 4 digits only."
                ),
                "gold_answer": f"{code:04d}",
                "gold_evidence_frames": [target],
            }
        )

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {args.n_items} items x {args.frames} frames to {args.out_root}/")
    print(f"wrote manifest: {args.manifest}")
    print(
        f"expected uniform recall@{args.k}: {uniform_hits}/{args.n_items} "
        f"= {uniform_hits / args.n_items:.2f}  (lower = harder needle; this is what `uniform` will roughly score)"
    )


if __name__ == "__main__":
    main()
