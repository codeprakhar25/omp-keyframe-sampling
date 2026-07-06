#!/usr/bin/env python3
"""Stream the LongVideoBench split-tar and extract ONLY the subset's videos.

The videos live in a single `videos.tar` split into 31 parts (aa..be, ~162GB total,
no per-file index). We can't selectively fetch a member, but we *can* stream the parts
in order through one `tar -x` that extracts only the named members -> disk holds just the
subset (a few GB), so a 100GB volume is plenty (no volume increase needed).

After extraction, probe each video's native fps with OpenCV and resolve
`gold_evidence_seconds` from the raw `gold_position` (self-validating: keep the mapping
that lands inside [0, duration]).

Env: HF_TOKEN must be set (gated repo).
"""
from __future__ import annotations

import argparse
import json
import os
import string
import subprocess
import sys

BASE = "https://huggingface.co/datasets/longvideobench/LongVideoBench/resolve/main"


def part_names(n: int = 31):
    # split default suffixes: aa, ab, ... (2-letter, base-26)
    letters = string.ascii_lowercase
    out = []
    for i in range(n):
        out.append("videos.tar.part." + letters[i // 26] + letters[i % 26])
    return out


def stream_extract(members, out_dir, n_parts, token):
    os.makedirs(out_dir, exist_ok=True)
    members_file = os.path.join(out_dir, "_members.txt")
    with open(members_file, "w") as f:
        for m in members:
            f.write(f"videos/{m}\n")
    parts = " ".join(part_names(n_parts))
    # sequential curl of each part -> single tar; --occurrence + explicit member list lets
    # GNU tar exit once the last named member is extracted (may finish before EOF).
    cmd = f"""
set -o pipefail
( for p in {parts}; do
    curl -sL -H "Authorization: Bearer $HF_TOKEN" "{BASE}/$p" || exit 1
  done ) | tar -x -C {out_dir} --strip-components=1 --occurrence -T {members_file} 2>{out_dir}/_tar.err
echo "TAR_RC=${{PIPESTATUS[1]}}"
"""
    env = dict(os.environ, HF_TOKEN=token)
    print(f"streaming {n_parts} parts, extracting {len(members)} members -> {out_dir} ...")
    r = subprocess.run(["bash", "-c", cmd], env=env)
    return r.returncode


def resolve_gold(manifest_path, out_dir):
    import cv2

    data = json.load(open(manifest_path))
    ok = miss = 0
    for m in data:
        path = os.path.join(out_dir, m["video_file"])
        if not os.path.exists(path):
            m["gold_evidence_seconds"] = None
            m["_video_present"] = False
            miss += 1
            continue
        m["_video_present"] = True
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        dur = m.get("video_seconds") or (nframes / fps if fps else 0)
        pos = m.get("gold_position") or []
        spans, reliable = [], True
        for pidx in pos:
            # hypothesis: position is a native-frame index -> seconds = pos / fps
            sec = pidx / fps if fps else None
            if sec is None or not (0 <= sec <= dur * 1.05):
                reliable = False
                continue
            spans.append([max(0.0, sec - 1.0), sec + 1.0])  # +-1s span for hit@k at 1fps
        m["gold_evidence_seconds"] = spans or None
        m["gold_reliable"] = reliable and bool(spans)
        m["native_fps"] = round(fps, 3)
        ok += 1
    json.dump(data, open(manifest_path, "w"), indent=2)
    print(f"resolved gold for {ok} present videos ({miss} missing); "
          f"reliable={sum(1 for m in data if m.get('gold_reliable'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.json")
    ap.add_argument("--out-dir", default="data/videos")
    ap.add_argument("--n-parts", type=int, default=31)
    ap.add_argument("--skip-download", action="store_true", help="only resolve gold from already-extracted videos")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")

    data = json.load(open(args.manifest))
    members = sorted({m["video_file"] for m in data})

    if not args.skip_download:
        rc = stream_extract(members, args.out_dir, args.n_parts, token)
        print(f"stream_extract rc={rc}")

    present = [m for m in members if os.path.exists(os.path.join(args.out_dir, m))]
    print(f"videos present: {len(present)}/{len(members)}")
    resolve_gold(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
