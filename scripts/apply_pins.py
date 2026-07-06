#!/usr/bin/env python3
"""Apply a PIN PASS: rewrite gold_evidence_seconds in the manifest from human-marked
answer-frame indices, turning broad keyframe gold into a true single-needle span.

pins.json schema:  { "<item_id>": <frame_index>  OR  [start_idx, end_idx]  OR null }
  null  -> leave that item's gold as-is (or drop it; see --drop-unpinned)

Usage:
  python3 scripts/apply_pins.py --manifest data/manifest.s1.json \
      --pins guiworld_sheets/pins.json --pad 0.6
"""
from __future__ import annotations

import argparse
import json

from harness.media import load_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.s1.json")
    ap.add_argument("--pins", required=True)
    ap.add_argument("--pad", type=float, default=0.6, help="seconds padding around the pinned frame")
    ap.add_argument("--drop-unpinned", action="store_true", help="remove items with null pin")
    ap.add_argument("--out", default=None, help="default: overwrite --manifest")
    args = ap.parse_args()

    items = json.load(open(args.manifest))
    pins = json.load(open(args.pins))
    out, dropped, pinned = [], 0, 0
    for it in items:
        pin = pins.get(it["id"])
        if pin is None:
            if args.drop_unpinned:
                dropped += 1
                continue
            out.append(it)
            continue
        frames = load_frames(it, dump_fps=1.0, max_frames=64)
        by_idx = {f.index: f.seconds for f in frames}
        idxs = pin if isinstance(pin, list) else [pin]
        secs = [by_idx[i] for i in idxs if i in by_idx and by_idx[i] is not None]
        if not secs:
            print(f"WARN {it['id']}: pin {pin} not in frame range; left unchanged")
            out.append(it)
            continue
        it["gold_evidence_seconds"] = [[min(secs) - args.pad, max(secs) + args.pad]]
        pinned += 1
        out.append(it)

    json.dump(out, open(args.out or args.manifest, "w"), indent=1)
    print(f"pinned {pinned} items, dropped {dropped}, total {len(out)} -> {args.out or args.manifest}")


if __name__ == "__main__":
    main()
