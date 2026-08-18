#!/usr/bin/env python3
"""Empirical vision-token audit for the resolution-compression arms.

Replicates _extract_frames_restier's sizing EXACTLY (smart_resize, factor=32,
min_pixels=256*28*28, max_pixels=1605632) but reads only the video's W/H via
cv2 properties -- no frame decode. 1 vision token = 32x32 px (patch 16, merge 2).

Compares a compressed arm against a full-res reference arm on the SAME qids, so
"equal token budget" claims can be checked instead of assumed.

Usage:
  token_audit.py --picks-a A.json --picks-b B.json --label-a k16@50 --label-b k8-full
    (picks values: [[sec, frac], ...]  OR  [sec, ...] -- bare seconds treated as frac 1.0)
"""
import argparse, json, os, statistics, sys
import cv2
from qwen_vl_utils import smart_resize

FACTOR, MINP, MAXP = 32, 256 * 28 * 28, 1605632
TOKPX = 32 * 32


def load_picks(p):
    d = json.load(open(p))
    out = {}
    for qid, v in d.items():
        ents = []
        for e in v:
            if isinstance(e, (list, tuple)):
                ents.append((float(e[0]), float(e[1])))
            else:
                ents.append((float(e), 1.0))
        out[qid] = ents
    return out


def video_wh(path, cache={}):
    if path in cache:
        return cache[path]
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cache[path] = (W, H)
    return W, H


def tokens_for(W, H, entries):
    bh, bw = smart_resize(H, W, factor=FACTOR, min_pixels=MINP, max_pixels=MAXP)
    tot = 0
    for _, frac in entries:
        if frac >= 0.999:
            th, tw = bh, bw
        else:
            cap = max(MINP, int(frac * bh * bw))
            th, tw = smart_resize(H, W, factor=FACTOR, min_pixels=MINP, max_pixels=cap)
        tot += (th * tw) // TOKPX
    return tot


ap = argparse.ArgumentParser()
ap.add_argument("--picks-a", required=True)
ap.add_argument("--picks-b", required=True)
ap.add_argument("--label-a", default="A")
ap.add_argument("--label-b", default="B")
ap.add_argument("--video-map", required=True, help="json {qid: video_path}")
a = ap.parse_args()

A, B = load_picks(a.picks_a), load_picks(a.picks_b)
vmap = json.load(open(a.video_map))
qids = sorted(set(A) & set(B) & set(vmap))
if not qids:
    sys.exit("FATAL: no overlapping qids")

ta, tb, missing = [], [], 0
for q in qids:
    vp = vmap[q]
    if not os.path.exists(vp):
        missing += 1
        continue
    W, H = video_wh(vp)
    if W <= 0 or H <= 0:
        missing += 1
        continue
    ta.append(tokens_for(W, H, A[q]))
    tb.append(tokens_for(W, H, B[q]))

n = len(ta)
print(f"\n=== vision-token audit  (n={n} qids, {missing} skipped) ===")
for lab, t, pk in ((a.label_a, ta, A), (a.label_b, tb, B)):
    fr = statistics.mean(len(pk[q]) for q in qids)
    print(f"  {lab:>12}: frames/item {fr:.1f}   tokens/item mean {statistics.mean(t):8.1f}  "
          f"median {statistics.median(t):8.1f}  min {min(t)}  max {max(t)}")
ratio = statistics.mean(ta) / statistics.mean(tb)
per = [x / y for x, y in zip(ta, tb)]
print(f"\n  {a.label_a} / {a.label_b}  = {ratio:.4f}  "
      f"({(ratio-1)*100:+.1f}% tokens)   per-item ratio median {statistics.median(per):.4f}")
print(f"  {'EQUAL-BUDGET OK (within 5%)' if abs(ratio-1) <= 0.05 else 'NOT equal budget -- footnote or re-tune'}\n")
