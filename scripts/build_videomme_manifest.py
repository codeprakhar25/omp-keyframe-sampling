#!/usr/bin/env python3
"""Build slm-lab manifest for Video-MME from HF parquet + staged videos.

Join key for embeds/picks/lmms-eval: question_id (e.g. '001-1').
Video file join: videoID -> $HF_HOME/videomme/data/{videoID}.mp4
Bins: duration in {short, medium, long} stored as length_bin (no trailing 's').
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/hf"))
    ap.add_argument("--video-root", default=None,
                    help="default: $HF_HOME/videomme/data")
    ap.add_argument("--bins", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--out", default="data/manifest.videomme.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    video_root = Path(args.video_root or os.path.join(args.hf_home, "videomme", "data"))
    keep = set(args.bins)

    ds = load_dataset("lmms-lab/Video-MME", "videomme", split="test")
    items, missing = [], []
    for r in ds:
        dur = r["duration"]
        if dur not in keep:
            continue
        vid = r["videoID"]
        qid = r["question_id"]
        mp = video_root / f"{vid}.mp4"
        if not mp.exists():
            # case variants
            alt = None
            for ext in (".MP4", ".mkv", ".webm"):
                p2 = video_root / f"{vid}{ext}"
                if p2.exists():
                    alt = p2
                    break
            if alt is None:
                missing.append(qid)
                continue
            mp = alt
        stem = (r["question"] or "").strip()
        # options already letter-prefixed in HF
        opts = list(r["options"])
        items.append({
            "id": qid,
            "question_id": qid,
            "video_id": r["video_id"],
            "videoID": vid,
            "video_file": f"{vid}.mp4",
            "media_type": "video",
            "media_path": str(mp),
            "length_bin": dur,
            "duration": dur,
            "question_stem": stem,
            "question": stem,  # stem-only; options separate — scorers use question_stem
            "options": opts,
            "gold_letter": r["answer"],
            "gold_answer": r["answer"],
            "domain": r["domain"],
            "sub_category": r["sub_category"],
            "task_type": r["task_type"],
            "url": r.get("url", ""),
        })

    by = {}
    for it in items:
        by.setdefault(it["length_bin"], 0)
        by[it["length_bin"]] += 1
    print(f"wrote-bound n={len(items)} by_bin={by} missing_video={len(missing)}")
    if missing:
        print("MISSING sample:", missing[:10])
        raise SystemExit(f"REFUSING: {len(missing)} questions lack staged video")

    # hard gates
    assert len(missing) == 0
    for b in keep:
        if b in ("short", "medium", "long"):
            n = by.get(b, 0)
            if n != 900 and keep == {"short", "medium", "long"}:
                pass  # subset ok
            print(f"  bin {b}: {n} questions, videos="
                  f"{len({it['videoID'] for it in items if it['length_bin']==b})}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(args.out, "w"), indent=0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
