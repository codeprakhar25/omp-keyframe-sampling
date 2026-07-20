#!/usr/bin/env python3
"""Residual-plateau early-stop OMP picks (adaptive-k by residual saturation).

Runs OMP step-by-step; watches ||q_res||. When the residual stops moving
(delta < eps for `patience` consecutive steps) the query is "explained" -> stop.

  variant1 (plateau)      : k = the plateau step (whatever it is)
  variant2 (plateau+min8) : k = max(plateau_step, min_frames)   [budget floor]

Emits lmms-eval picks ({qid: [secs]}) for both, prints the k distribution, and
-- because the trace says the residual is flat by ~pick 5 -- checks whether
variant2 reproduces the existing fixed-OMP@8 picks (if so, its accuracy is the
already-measured k8 number and no GPU run is needed).

CPU, seconds.  Usage:
  PYTHONPATH=. python3 scripts/gen_plateau_picks.py --bin 3600 \
      --eps 1e-3 --patience 2 --min-frames 8 \
      --k8-ref results/picks_lmmseval/picks_omp_lc_k8.json \
      --out-v1 results/picks_lmmseval/picks_omp_lc_3600_plateau.json \
      --out-v2 results/picks_lmmseval/picks_omp_lc_3600_plateau_min8.json
"""
import argparse
import json
from collections import Counter

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_order(query, emb, kmax):
    """OMP to kmax; return (picked_idx_in_order, resid_frac_after_each_step)."""
    q0 = query.astype(np.float32).copy()
    q0n = float(np.linalg.norm(q0)) or 1.0
    q = q0.copy()
    E = emb.astype(np.float32)
    n = E.shape[0]
    basis, chosen = [], np.zeros(n, dtype=bool)
    order, resid = [], []
    for _ in range(min(kmax, n)):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        chosen[best] = True
        order.append(best)
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            q = q - (q @ v) * v
        resid.append(float(np.linalg.norm(q) / q0n))
    return order, resid


def plateau_k(resid, eps, patience):
    """First step index (1-based) at which residual has been flat for `patience`
    consecutive deltas. Falls back to len(resid) if never triggers."""
    run = 0
    for t in range(1, len(resid)):
        if abs(resid[t - 1] - resid[t]) < eps:
            run += 1
            if run >= patience:
                return t + 1          # frames picked so far (1-based count)
        else:
            run = 0
    return len(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--scorer", default="lc")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--k8-ref", default=None, help="fixed-OMP@8 picks json to compare variant2 against")
    ap.add_argument("--out-v1", default=None)
    ap.add_argument("--out-v2", default=None)
    args = ap.parse_args()

    t = np.load(f"results/embeds_text/text_lc_{args.bin}.npz")
    qmap = dict(zip(t["ids"].tolist(), t["text"]))
    k8ref = json.load(open(args.k8_ref)) if args.k8_ref else {}

    v1, v2 = {}, {}
    kdist_v1, kdist_v2 = Counter(), Counter()
    v2_eq_k8, v2_cmp = 0, 0
    used = 0
    for qid, qv in qmap.items():
        times, emb = load_image_embed(qid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < 1:
            continue
        E = l2(emb)
        q = l2(np.asarray(qv, dtype=np.float32))
        order, resid = omp_order(q, E, args.kmax)
        used += 1
        kp = plateau_k(resid, args.eps, args.patience)
        k1 = min(kp, len(order))
        k2 = min(max(kp, args.min_frames), len(order))
        kdist_v1[k1] += 1
        kdist_v2[k2] += 1
        secs1 = sorted(round(float(times[i]), 1) for i in order[:k1])
        secs2 = sorted(round(float(times[i]), 1) for i in order[:k2])
        v1[qid] = secs1
        v2[qid] = secs2
        if qid in k8ref:
            v2_cmp += 1
            if sorted(k8ref[qid]) == secs2:
                v2_eq_k8 += 1

    def dist_str(c):
        tot = sum(c.values())
        return "  ".join(f"k{k}:{c[k]}({100*c[k]/tot:.0f}%)" for k in sorted(c))

    ks1 = [k for k, n in kdist_v1.items() for _ in range(n)]
    ks2 = [k for k, n in kdist_v2.items() for _ in range(n)]
    print(f"\nbin={args.bin}  n={used}  eps={args.eps}  patience={args.patience}  min_frames={args.min_frames}\n")
    print(f"variant1 (plateau)      mean_k={np.mean(ks1):.2f}  {dist_str(kdist_v1)}")
    print(f"variant2 (plateau+min8) mean_k={np.mean(ks2):.2f}  {dist_str(kdist_v2)}")
    tot_frames_v1 = sum(ks1); tot_frames_v2 = sum(ks2); fixed8 = 8 * used
    print(f"\ntoken/frame budget vs fixed k8 ({fixed8} frames):")
    print(f"  variant1: {tot_frames_v1} frames = {100*tot_frames_v1/fixed8:.1f}% of k8 budget")
    print(f"  variant2: {tot_frames_v2} frames = {100*tot_frames_v2/fixed8:.1f}% of k8 budget")
    if args.k8_ref:
        print(f"\nvariant2 vs fixed-OMP@8 picks: {v2_eq_k8}/{v2_cmp} identical "
              f"({100*v2_eq_k8/max(v2_cmp,1):.1f}%)")
        if v2_eq_k8 == v2_cmp and v2_cmp:
            print("  => variant2 == k8 exactly. Accuracy = the already-measured k8 number. No GPU needed.")

    if args.out_v1:
        json.dump(v1, open(args.out_v1, "w")); print(f"\nwrote {args.out_v1}")
    if args.out_v2:
        json.dump(v2, open(args.out_v2, "w")); print(f"wrote {args.out_v2}")


if __name__ == "__main__":
    main()
