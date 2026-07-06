#!/usr/bin/env python3
"""Build long needle-in-haystack items from short GUI-World clips, WITHOUT re-encoding
video (ffmpeg-free). Concatenate the 1fps frames of M real clips into one ordered PNG
sequence; the question targets ONE clip, so its frame-range is the gold needle and the
other M-1 clips are real, same-app distractors.

Why: raw GUI-World single-task clips have a *persistent* answer (visible in ~every frame)
-> no needle -> recall@k meaningless. Stitching restores the property the thesis needs:
the answer lives in a SHORT span of LONG content, among hard distractors.

Output per item:
  data/s1_stitch/<id>/0000.png .. NNNN.png        (concatenated frames)
  manifest item: {media_type:"images", media_path, question, gold_answer,
                  gold_evidence_frames:[i.. j]}   # the target clip's frames

recall_at_k (harness/metrics.py) reads gold_evidence_frames for image items -> exact.

LEAKAGE GUARD: distractors are picked from clips whose `goal` differs from the target's,
so the target's specific answer-state is unlikely to recur in a distractor. Residual risk
remains (same app) -> spot check a couple of sheets.

Usage:
  PYTHONPATH=. python3 scripts/stitch_needle.py --ann <jsonl> --video-root guiworld_media \
      --n-items 8 --clips-per-item 6 --seed 0 --out-dir data/s1_stitch --manifest data/manifest.s1_stitch.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

from harness.media import _load_video_frames


def load_rows(ann, video_root):
    raw = open(ann).read().strip()
    rows = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    out = []
    for r in rows:
        vp = r.get("video_path") or r.get("old_video_path") or ""
        mc = r.get("MCQA")
        if isinstance(mc, list):
            mc = mc[0] if mc else None
        if not vp.endswith((".mp4", ".mov")) or not isinstance(mc, dict) or not mc.get("Question"):
            continue
        path = os.path.join(video_root, os.path.basename(vp))
        if os.path.exists(path):
            out.append({"path": path, "goal": r.get("goal", ""), "mcqa": mc})
    return out


def clean_answer(mcqa):
    import re
    ans = str(mcqa.get("Correct Answer", "")).strip()
    m = re.search(r"\[\[\s*([A-Da-d])\s*\]\]\s*(.*)", ans)
    if m:
        return f"{m.group(1).upper()}. {m.group(2).strip()}".rstrip(". ")
    return ans


def question_text(mcqa):
    q = mcqa["Question"]
    opts = mcqa.get("Options")
    if isinstance(opts, (list, dict)):
        q = f"{q}\nOptions: {json.dumps(opts)}\nAnswer with the option letter only."
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--video-root", default="guiworld_media")
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--clips-per-item", type=int, default=6)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames-per-clip", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/s1_stitch")
    ap.add_argument("--manifest", default="data/manifest.s1_stitch.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(args.ann, args.video_root)
    if len(rows) < args.clips_per_item:
        raise SystemExit(f"need >= {args.clips_per_item} clips, have {len(rows)}")

    manifest = []
    for it in range(args.n_items):
        target = rng.choice(rows)
        # distractors: different goal than target where possible
        pool = [r for r in rows if r["goal"] != target["goal"] and r is not target] or [r for r in rows if r is not target]
        distractors = rng.sample(pool, min(args.clips_per_item - 1, len(pool)))
        clips = distractors + [target]
        rng.shuffle(clips)

        item_id = f"stitch-{it:03d}"
        out_dir = os.path.join(args.out_dir, item_id)
        os.makedirs(out_dir, exist_ok=True)

        seq_idx = 0
        gold = []
        for clip in clips:
            frames = _load_video_frames(clip["path"], args.fps, args.max_frames_per_clip)
            is_target = clip is target
            for fr in frames:
                fr.image.save(os.path.join(out_dir, f"{seq_idx:04d}.png"))
                if is_target:
                    gold.append(seq_idx)
                seq_idx += 1
        manifest.append({
            "id": item_id,
            "media_type": "images",
            "media_path": out_dir,
            "question": question_text(target["mcqa"]),
            "gold_answer": clean_answer(target["mcqa"]),
            "gold_evidence_frames": gold,
            "n_frames": seq_idx,
        })
        print(f"{item_id}: {seq_idx} frames, target span {gold[0]}..{gold[-1]} ({len(gold)} gold)")

    json.dump(manifest, open(args.manifest, "w"), indent=1)
    print(f"\nwrote {len(manifest)} stitched items -> {args.manifest}")
    print("note: gold is AUTOMATIC (target clip range) — no manual pin pass needed for stitched items.")


if __name__ == "__main__":
    main()
