#!/usr/bin/env python3
"""Build a length-binned LongVideoBench subset -> our manifest schema (Fork A / Phase 0).

Why LongVideoBench (not LVHaystack/Ego4D): LVHaystack's videos are gated Ego4D with an
unfinished video->clip pipeline. LongVideoBench is HF click-through gated, ships gold
keyframes (`position`) + subtitles + natural duration groups {15,60,600,3600}s -> the exact
length axis Fork A needs, and it's what LVHaystack's own transform_longvideobench.py targets.

This writes the manifest with `gold_position` (raw). Gold seconds are resolved at
extract time (fetch_lvb_subset.py) from each video's native fps, because LongVideoBench's
`position` units are ambiguous from the metadata alone -> we self-validate per video.

Primary Fork-A metric on this set = MCQA accuracy (real randomized `correct_choice`, no
GUI-World B-bias); hit@k is the best-effort selection diagnostic.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from collections import defaultdict

LETTERS = ["A", "B", "C", "D", "E"]


def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--per-bin", type=int, default=10, help="questions per duration group")
    ap.add_argument("--bins", default=None,
                    help="comma list of duration groups to keep, e.g. '15,60' (default: all)")
    ap.add_argument("--keep-text-qs", action="store_true",
                    help="keep subtitle-dependent categories ('T' in question_category); "
                         "pair with gpt_mcqa.py --subtitles-dir so the answerer sees them")
    ap.add_argument("--distinct-videos", action="store_true",
                    help="one question per video (fewer videos to extract/decode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/manifest.lvb.json")
    ap.add_argument("--videos-dir", default="data/videos")
    args = ap.parse_args()
    random.seed(args.seed)

    if os.path.exists(args.split):
        p = args.split  # local copy (curl with HF_TOKEN works without huggingface_hub)
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("longvideobench/LongVideoBench", args.split, repo_type="dataset")
    data = json.load(open(p))

    bins = set(args.bins.split(",")) if args.bins else None
    # default: frames-answerable only (drop 'T' categories); --keep-text-qs keeps them
    pool = defaultdict(list)
    seen_videos = set()
    for d in data:
        qcat = str(d.get("question_category", ""))
        if "T" in qcat and not args.keep_text_qs:
            continue
        pos = parse_list(d.get("position"))
        if not pos and not args.keep_text_qs:
            # frames-only protocol needs a visual keyframe; text-Qs may lack one
            continue
        cand = parse_list(d.get("candidates"))
        cc = d.get("correct_choice")
        try:
            cc = int(cc)
        except Exception:
            continue
        if cc >= len(cand):
            continue
        dg = str(d.get("duration_group"))
        if bins and dg not in bins:
            continue
        pool[dg].append(d)

    manifest = []
    for dg in sorted(pool, key=lambda x: float(x)):
        items = pool[dg][:]
        random.shuffle(items)
        picked = 0
        for d in items:
            if picked >= args.per_bin:
                break
            vid = d["video_id"]
            if args.distinct_videos and vid in seen_videos:
                continue
            cand = parse_list(d.get("candidates"))
            cc = int(d["correct_choice"])
            pos = [int(x) for x in parse_list(d.get("position"))]
            opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(cand))
            question = (
                f"{d.get('question','').strip()}\n\nOptions:\n{opts}\n\n"
                "Answer with the single option letter."
            )
            manifest.append({
                "id": d["id"],
                "media_type": "video",
                "media_path": f"{args.videos_dir}/{d['video_path']}",
                "video_file": d["video_path"],
                "question": question,          # answerer prompt: stem + options + instruction
                "question_stem": d.get("question", "").strip(),  # scorer query: stem only (see harness/text.py)
                "gold_answer": f"{LETTERS[cc]}) {cand[cc]}",
                "gold_letter": LETTERS[cc],
                "candidates": cand,
                "gold_position": pos,           # raw LongVideoBench keyframe index/indices
                "video_seconds": float(d.get("duration", 0)),
                "length_bin": f"{dg}s",
                "question_category": d.get("question_category"),
                "subtitle_file": d.get("subtitle_path"),
                # gold_evidence_seconds filled by fetch_lvb_subset.py from native fps
            })
            seen_videos.add(vid)
            picked += 1
        print(f"bin {dg}s: picked {picked} (pool {len(pool[dg])})")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    vids = sorted({m["video_file"] for m in manifest})
    print(f"\nwrote {args.out}: {len(manifest)} questions across {len(vids)} distinct videos")
    print("distinct video files:", len(vids))


if __name__ == "__main__":
    main()
