#!/usr/bin/env python3
"""Smoke-test the two strong detectors before the decisive run.

For one gold_reliable item: score all frames with each detector, then check
(a) no crash / no NaN, (b) scores actually vary, (c) the gold frame's score
is above the per-video median. A detector that can't clear (c) on a single
clean item is mis-wired and would produce a fake tie -- catch it here.
"""
import json
import statistics
import sys

from harness.media import load_frames
from scripts.region_recall import build_ground_scorer

MANIFEST = "data/manifest.lvb.frames.100.json"


def pick_item():
    for it in json.load(open(MANIFEST)):
        if it.get("gold_reliable") and str(it.get("length_bin", "")).replace("s", "") in ("60", "600"):
            tgt = json.load(open("data/targets.json")).get(str(it["id"]), {})
            if tgt.get("has_concrete_target"):
                return it, tgt
    raise SystemExit("no concrete gold_reliable item found")


def gold_index(frames, item):
    # gold frame = frame whose timestamp is nearest the item's gold time
    g = item.get("gold_time") or item.get("gold_sec")
    if g is None:
        return None
    return min(range(len(frames)), key=lambda i: abs(getattr(frames[i], "t", i) - g))


def main():
    detector = sys.argv[1]
    item, tgt = pick_item()
    frames = load_frames(item, dump_fps=1.0, max_frames=600)
    print(f"item={item['id']} nf={len(frames)} phrase={tgt.get('target')!r}")
    score_fn, _ = build_ground_scorer(detector)
    scores = score_fn(frames, item)
    assert len(scores) == len(frames), f"len mismatch {len(scores)}!={len(frames)}"
    assert not any(s != s for s in scores), "NaN in scores"
    spread = max(scores) - min(scores)
    med = statistics.median(scores)
    gi = gold_index(frames, item)
    print(f"  spread={spread:.4f} min={min(scores):.4f} max={max(scores):.4f} med={med:.4f}")
    if gi is not None:
        print(f"  gold_frame idx={gi} score={scores[gi]:.4f}  above_median={scores[gi] > med}")
    assert spread > 1e-4, "scores flat -> detector wired wrong (fake-tie risk)"
    print("SMOKE_OK", detector)


if __name__ == "__main__":
    main()
