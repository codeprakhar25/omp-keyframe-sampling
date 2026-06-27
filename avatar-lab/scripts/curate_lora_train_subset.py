#!/usr/bin/env python3
"""Pick a training subset from data/video_lora/clips — reduce session redundancy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "video_lora" / "clips"
OUT = ROOT / "data" / "video_lora" / "clips_train"
MANIFEST = ROOT / "data" / "video_lora" / "manifest.json"

MAX_PER_SOURCE = 2
MAX_PER_PRIORITY_SOURCE = 3
# Prefer spreading across sources; always keep diversity tags
PRIORITY_TAGS = {
    "outdoor_bright",
    "outdoor_setting",
    "low_light",
    "reference_talking",
}


def pick_windows(items: list, limit: int) -> list:
    items = sorted(items, key=lambda x: x["start_sec"])
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    if limit == 2:
        return [items[0], items[len(items) // 2]]
    # limit >= 3: start, middle, end
    return [items[0], items[len(items) // 2], items[-1]]


def main() -> None:
    data = json.loads(MANIFEST.read_text())
    clips = data["clips"]

    by_source: dict[str, list] = {}
    for c in clips:
        by_source.setdefault(c["source"], []).append(c)

    selected: list[dict] = []
    for source, items in sorted(by_source.items()):
        has_priority = any(i["tag"] in PRIORITY_TAGS for i in items)
        limit = MAX_PER_PRIORITY_SOURCE if has_priority else MAX_PER_SOURCE
        selected.extend(pick_windows(items, limit))

    # Drop exact duplicate ids while preserving order
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for c in selected:
        if c["id"] in seen_ids:
            continue
        seen_ids.add(c["id"])
        deduped.append(c)
    selected = deduped

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    train_manifest = []
    for i, c in enumerate(sorted(selected, key=lambda x: x["id"]), start=1):
        src_mp4 = CLIPS / f"{c['id']}.mp4"
        src_txt = CLIPS / f"{c['id']}.txt"
        new_stem = f"clip_{i:03d}"
        shutil.copy2(src_mp4, OUT / f"{new_stem}.mp4")
        shutil.copy2(src_txt, OUT / f"{new_stem}.txt")
        train_manifest.append({**c, "train_id": new_stem})

    out_json = ROOT / "data" / "video_lora" / "manifest_train.json"
    out_json.write_text(
        json.dumps(
            {
                "note": "Curated subset for LoRA training — max 2 windows per source file",
                "clip_count": len(train_manifest),
                "max_per_source": MAX_PER_SOURCE,
                "output_dir": "data/video_lora/clips_train",
                "clips": train_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(train_manifest)} clips -> {OUT}")


if __name__ == "__main__":
    main()
