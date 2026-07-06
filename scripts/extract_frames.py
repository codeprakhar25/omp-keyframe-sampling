#!/usr/bin/env python3
"""Pre-extract 1fps JPEG frames per video to disk ONCE, so every Phase-0 arm loads
frames instantly instead of re-decoding the mp4 (decoding a 1-hr video grabs ~108k
source frames -> the harness bottleneck when 4 arms each re-decode 40 videos).

Writes data/frames/<id>/f<sec>.jpg (one per second, downscaled to --max-side) and
rewrites the manifest to media_type=images pointing at each frame dir. Frame filename
encodes the second, so `seconds` alignment (and gold_evidence_seconds) is exact.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
from PIL import Image


def extract(video_path: str, out_dir: str, dump_fps: float, max_frames: int, max_side: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native / max(dump_fps, 1e-6))), 1)
    src_idx = 0
    n = 0
    while True:
        if not cap.grab():
            break
        if src_idx % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            if max_side and max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            sec = int(round(src_idx / native))
            img.save(os.path.join(out_dir, f"f{sec:06d}.jpg"), quality=88)
            n += 1
            if n >= max_frames:
                break
        src_idx += 1
    cap.release()
    return n


def _worker(job):
    m, videos_dir, frames_dir, dump_fps, max_frames, max_side = job
    out_dir = os.path.join(frames_dir, m["id"])
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return m["id"], len(os.listdir(out_dir))
    vid = os.path.join(videos_dir, m["video_file"])
    return m["id"], extract(vid, out_dir, dump_fps, max_frames, max_side)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.json")
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--out-manifest", default="data/manifest.lvb.frames.json")
    ap.add_argument("--dump-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=3600, help="cap frames per video (full 1hr @1fps)")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    data = json.load(open(args.manifest))

    # decode is single-threaded per video but we have many cores -> parallelize across videos
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(m, args.videos_dir, args.frames_dir, args.dump_fps, args.max_frames, args.max_side)
            for m in data]
    counts = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(data))) as ex:
        for vid_id, n in ex.map(_worker, jobs):
            counts[vid_id] = n
            print(f"  {vid_id} -> {n} frames", flush=True)

    for i, m in enumerate(data):
        out_dir = os.path.join(args.frames_dir, m["id"])
        n = counts.get(m["id"], 0)
        m["media_type"] = "images"
        m["media_path"] = out_dir
        m["n_frames_extracted"] = n
        # images-mode metrics key on gold_evidence_frames; at 1fps frame index == second,
        # so map each gold second-span to the integer frame indices it covers.
        gef = []
        for span in (m.get("gold_evidence_seconds") or []):
            lo, hi = int(span[0]), int(round(span[1]))
            gef.extend(range(max(0, lo), hi + 1))
        m["gold_evidence_frames"] = sorted(set(gef)) or None
        print(f"[{i+1}/{len(data)}] {m['id']} ({m['length_bin']}) -> {n} frames, gold_frames={m['gold_evidence_frames']}")

    json.dump(data, open(args.out_manifest, "w"), indent=2)
    print(f"\nwrote {args.out_manifest}")


if __name__ == "__main__":
    main()
