#!/usr/bin/env python3
"""Build data/manifest.s1.json from a GUI-World subset (S1 screen slice).

GUI-World ships exactly the fields S1 needs:
  - MCQA: {Question, Options, Correct Answer}  -> question + gold_answer
  - keyframes: native-fps frame numbers + sub_goal -> gold evidence span
  - video_path: the MOV recording

We convert each item to the harness manifest schema (see harness/media.py):
  {id, media_type:"video", media_path, question, gold_answer, gold_evidence_seconds:[[t0,t1]]}

recall_at_k (harness/metrics.py) reads gold_evidence_seconds for video items.

HONEST CAVEAT (must verify per item before trusting numbers):
  GUI-World keyframes mark *interaction* sub-goals, not necessarily the frame that
  answers a given MCQA. For the pilot we take the keyframe span as gold, BUT you must
  spot-check that the MCQA answer is actually visible in that span — else recall@k is
  measuring the wrong thing (SPEC ��9 leakage/eval-separation risk).

Usage:
  python3 scripts/build_s1_from_guiworld.py \
      --ann <guiworld_annotations.json> --video-root <dir with MOVs> \
      --n 10 --out data/manifest.s1.json
"""
from __future__ import annotations

import argparse
import json
import os

import cv2  # opencv-python-headless; needs ffmpeg for MOV decode


def native_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def keyframe_span_seconds(keyframes, fps: float, pad_s: float = 1.0):
    """Turn GUI-World keyframe frame-numbers into [[t0,t1]] second spans (padded)."""
    nums = []
    for kf in keyframes or []:
        # keyframe entries vary: int, or dict with a frame-number field
        if isinstance(kf, int):
            nums.append(kf)
        elif isinstance(kf, dict):
            for key in ("frame", "frame_number", "index", "id"):
                if key in kf and isinstance(kf[key], (int, float)):
                    nums.append(int(kf[key]))
                    break
    if not nums:
        return None
    return [[max(0.0, n / fps - pad_s), n / fps + pad_s] for n in nums]


def mcqa_to_qa(mcqa: dict):
    """Flatten an MCQA dict to (question_text, gold_answer)."""
    q = mcqa.get("Question") or mcqa.get("question")
    opts = mcqa.get("Options") or mcqa.get("options")
    ans = mcqa.get("Correct Answer") or mcqa.get("answer")
    if not q or ans is None:
        return None
    import re
    m = re.search(r"\[\[\s*([A-Da-d])\s*\]\]\s*(.*)", str(ans))  # "[[B]] foo" -> ("B","foo")
    if m:
        letter, rest = m.group(1).upper(), m.group(2).strip()
        ans = f"{letter}. {rest}" if rest else letter
    else:
        ans = str(ans).strip()
    if isinstance(opts, (list, dict)):
        q = f"{q}\nOptions: {json.dumps(opts)}\nAnswer with the option letter only."
    return q, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="GUI-World annotation json/jsonl")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="data/manifest.s1.json")
    args = ap.parse_args()

    raw = open(args.ann).read().strip()
    rows = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]

    out, skipped = [], 0
    for i, r in enumerate(rows):
        if len(out) >= args.n:
            break
        vp = r.get("video_path") or r.get("old_video_path")
        mcqa = r.get("MCQA") or {}
        if isinstance(mcqa, list):  # some rows store a list of MCQA dicts
            mcqa = mcqa[0] if mcqa else {}
        qa = mcqa_to_qa(mcqa) if mcqa else None
        if not vp or not qa:
            skipped += 1
            continue
        media_path = os.path.join(args.video_root, os.path.basename(vp))
        if not os.path.exists(media_path):
            skipped += 1
            continue
        fps = native_fps(media_path)
        span = keyframe_span_seconds(r.get("keyframes"), fps)
        item = {
            "id": f"guiworld-{i:03d}",
            "media_type": "video",
            "media_path": media_path,
            "question": qa[0],
            "gold_answer": qa[1],
        }
        if span:
            item["gold_evidence_seconds"] = span
        out.append(item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {len(out)} items -> {args.out} (skipped {skipped})")
    n_gold = sum(1 for x in out if "gold_evidence_seconds" in x)
    print(f"items with gold_evidence_seconds: {n_gold}/{len(out)}  "
          f"(items without -> recall@k=None, accuracy still scored)")


if __name__ == "__main__":
    main()
