"""Generate a tiny synthetic 'images' item so the harness runs end-to-end offline.

Creates N frames; one frame is marked TARGET with a red border. The question asks
which frame is the target, with gold evidence pointing at that frame index -- so
recall@k is meaningful even though EchoAnswerer can't actually answer.

    python scripts/make_synthetic.py
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw


def main() -> None:
    out_dir = os.path.join("data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    n = 12
    target = 7

    for i in range(n):
        img = Image.new("RGB", (320, 240), (30, 30, 40))
        d = ImageDraw.Draw(img)
        d.text((150, 110), str(i), fill=(230, 230, 230))
        if i == target:
            d.rectangle([3, 3, 316, 236], outline=(220, 40, 40), width=6)
            d.text((120, 150), "TARGET", fill=(220, 40, 40))
        img.save(os.path.join(out_dir, f"frame_{i:02d}.png"))

    manifest = [
        {
            "id": "synthetic-001",
            "media_type": "images",
            "media_path": out_dir,
            "question": "Which frame number is marked TARGET with a red border?",
            "gold_answer": str(target),
            "gold_evidence_frames": [target],
        }
    ]
    manifest_path = os.path.join("data", "manifest.synthetic.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {n} frames to {out_dir}/ and {manifest_path}")


if __name__ == "__main__":
    main()
