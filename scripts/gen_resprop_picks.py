#!/usr/bin/env python3
"""Strategy D: residual-PROPORTIONAL resolution. Per-frame tokens follow the OMP residual curve.

vs restier (fixed 1/1/.5/.5/.5/.25/.25/.25 by rank position): here each frame's resolution
fraction is a CONTINUOUS function of how much query mass its OMP pick actually explained
(the greedy score s_r = q_res . f_r at selection). Videos whose residual is flat get a more
uniform allocation; videos that saturate fast concentrate tokens on picks 1-2. The fixed-tier
schedule ignores the per-video curve; this doesn't.

  w_r  = s_r / s_1              in [0,1], w_1 = 1 (greedy scores are non-increasing)
  frac = FLOOR + (1-FLOOR) * w_r**gamma
  gamma auto-bisected so the dataset-average budget == --target-budget (default .47, matched
  to restier for a clean 3-way: full-res .5461 / restier .5426 / D at the SAME ~47% tokens).

--target-budget is the ONE knob strategy B sweeps (0.25/0.35/0.47/0.60 -> accuracy-token curve).

Frames sourced from the baseline OMP picks (identical selection); only per-frame frac differs.
Emits {qid:[[sec,frac]]} consumed by the existing restier task (doc_to_visual resizes to
frac*baseline). CPU, seconds. Usage:
  PYTHONPATH=. python3 scripts/gen_resprop_picks.py --bin 3600 --k 8 --target-budget 0.47 \
    --baseline results/picks_lmmseval/picks_omp_lc_k8.json \
    --out results/picks_lmmseval/picks_resprop_omp_lc_3600_k8_b47.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed

FLOOR = 200704.0 / (1196.0 * 1024.0)   # min_pixels token floor as a fraction of ~baseline frame


def omp_scored(query, E, k):
    """OMP; returns [(idx, greedy_score)] in pick order. score = q_res . f_pick at selection."""
    q = query.astype(np.float32).copy()
    basis, chosen, out = [], np.zeros(E.shape[0], dtype=bool), []
    for _ in range(min(k, E.shape[0])):
        s = E @ q
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        out.append((b, max(0.0, float(s[b]))))
        chosen[b] = True
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            q = q - (q @ v) * v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--target-budget", type=float, default=0.47)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--match-tol", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(f"{args.text_dir}/text_lc_{args.bin}.npz")
    qmap = dict(zip(z["ids"].tolist(), z["text"]))
    base = json.load(open(args.baseline))

    # collect per-frame weights w_r (matched to baseline frames) once; gamma is tuned after.
    recs = {}          # qid -> list of (sec, w) in baseline time order
    fb_frames = fb_qids = miss = 0
    for qid in qmap:
        if qid not in base:
            miss += 1
            continue
        bsecs = [float(x) for x in base[qid]]
        times, emb = load_image_embed(qid, "lc", args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < 1:
            continue
        E = l2(emb)
        q = l2(np.asarray(qmap[qid], dtype=np.float32))
        tarr = np.asarray(times, dtype=np.float32)
        scored = omp_scored(q, E, min(args.k, E.shape[0]))
        s1 = scored[0][1] if scored and scored[0][1] > 0 else 1.0
        w_of_time = {round(float(tarr[i]), 1): max(0.0, sc) / s1 for i, sc in scored}
        cos_of_time = {round(float(tarr[i]), 1): float(E[i] @ q) for i, _ in scored}
        used, per, fb_here = set(), [], 0
        for s in bsecs:
            j = int(np.argmin(np.abs(tarr - s)))
            key = round(float(tarr[j]), 1)
            w = w_of_time.get(key)
            if w is None or key in used:
                fb_here += 1
                # unmatched (bf16 near-tie): weight from cosine rank instead
                w = max(0.0, cos_of_time.get(key, float(E[j] @ q)))
                w = w / (s1 if s1 > 0 else 1.0)
            else:
                used.add(key)
            per.append((s, float(np.clip(w, 0.0, 1.0))))
        if fb_here:
            fb_qids += 1
            fb_frames += fb_here
        recs[qid] = per

    allw = np.array([w for per in recs.values() for _, w in per], dtype=np.float64)

    def budget(gamma):
        frac = args.floor + (1.0 - args.floor) * (allw ** gamma)
        return float(np.mean(np.clip(frac, args.floor, 1.0)))

    # bisect gamma so mean budget == target (budget decreases as gamma grows)
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if budget(mid) > args.target_budget:
            lo = mid
        else:
            hi = mid
    gamma = (lo + hi) / 2
    ach = budget(gamma)

    picks = {}
    for qid, per in recs.items():
        picks[qid] = [[round(s, 3),
                       float(np.clip(args.floor + (1.0 - args.floor) * (w ** gamma),
                                     args.floor, 1.0))]
                      for s, w in sorted(per, key=lambda t: t[0])]
    json.dump(picks, open(args.out, "w"))

    fr = np.array([f for p in picks.values() for _, f in p])
    print(f"bin={args.bin} k={args.k} target={args.target_budget:.2f}  floor={args.floor:.3f}")
    print(f"gamma={gamma:.4f}  achieved avg budget={ach:.4f}  n={len(picks)}  "
          f"(absent from baseline: {miss}, cosine-fallback frames: {fb_frames})")
    print(f"frac dist: min={fr.min():.3f} p25={np.percentile(fr,25):.3f} "
          f"med={np.median(fr):.3f} p75={np.percentile(fr,75):.3f} max={fr.max():.3f} "
          f"mean={fr.mean():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
