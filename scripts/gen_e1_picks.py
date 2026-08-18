#!/usr/bin/env python3
"""Strategy E1: halve-on-drop resolution. Budget EMERGENT (no target), driven by the residual.

The OMP greedy score s_r (= q_res . f_r) barely moves -- the CLIP modality gap keeps it in a
tiny band (~[0.95,1] once normalized to s_1). So absolute magnitude is useless; what matters
is a DROP between consecutive picks. Rule:

  res_1 = 1.0
  for r>1:  if (w_{r-1} - w_r) >= delta:  res_r = res_{r-1} * 0.5     (a real drop -> step down)
            else:                          res_r = res_{r-1}          (flat -> keep)
  floored at FLOOR (min_pixels).  w_r = s_r / s_1.

Flat residual -> few/no steps -> stays sharp -> ~no compression (every frame is evidence).
Steep residual -> many steps -> heavy compression. The per-video budget is a RESULT, and the
dataset-average cut is the honest headline (reported, not imposed). Contrast restier / D which
force a fixed 53% on every video regardless of the curve.

Frames sourced from baseline OMP picks (identical selection); only per-frame frac differs.
Emits {qid:[[sec,frac]]} for the existing restier task. CPU. Usage:
  PYTHONPATH=. python3 scripts/gen_e1_picks.py --bin 3600 --k 8 --delta 0.005 \
    --baseline results/picks_lmmseval/picks_omp_lc_k8.json \
    --out results/picks_lmmseval/picks_e1_omp_lc_3600_k8.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed

FLOOR = 200704.0 / (1196.0 * 1024.0)


def omp_scored(query, E, k):
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


def e1_fracs(w, delta, floor, factor=0.5):
    """w: greedy scores in pick order (rank), normalised outside. Returns frac per rank."""
    res = [1.0]
    for r in range(1, len(w)):
        if w[r - 1] - w[r] >= delta:
            res.append(max(floor, res[-1] * factor))
        else:
            res.append(res[-1])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(f"{args.text_dir}/text_lc_{args.bin}.npz")
    qmap = dict(zip(z["ids"].tolist(), z["text"]))
    base = json.load(open(args.baseline))

    picks = {}
    miss = fb_frames = fb_qids = 0
    budgets, nsteps = [], []
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
        # rank order of scores; map to baseline frames by nearest time
        rank_time = [round(float(tarr[i]), 1) for i, _ in scored]
        w_rank = [max(0.0, sc) / s1 for _, sc in scored]
        cos_of_time = {round(float(tarr[i]), 1): float(E[i] @ q) for i, _ in scored}

        fr = e1_fracs(w_rank, args.delta, args.floor, args.factor)      # frac per RANK
        frac_of_time = {rank_time[r]: fr[r] for r in range(len(fr))}
        nsteps.append(sum(1 for r in range(1, len(w_rank))
                          if w_rank[r - 1] - w_rank[r] >= args.delta))

        out_frames, fb_here = [], 0
        for s in bsecs:
            j = int(np.argmin(np.abs(tarr - s)))
            key = round(float(tarr[j]), 1)
            f = frac_of_time.get(key)
            if f is None:
                fb_here += 1
                f = args.floor                       # unmatched bf16 near-tie: conservative floor
            out_frames.append([round(s, 3), float(f)])
        if fb_here:
            fb_qids += 1
            fb_frames += fb_here
        out_frames.sort(key=lambda t: t[0])
        picks[qid] = out_frames
        budgets.append(np.mean([f for _, f in out_frames]))

    json.dump(picks, open(args.out, "w"))
    b = np.array(budgets)
    print(f"bin={args.bin} k={args.k} delta={args.delta} factor={args.factor} "
          f"floor={args.floor:.3f}")
    print(f"n={len(picks)}  (absent {miss}, cosine/floor-fallback frames {fb_frames})")
    print(f"EMERGENT budget: mean={b.mean():.3f}  (=~{100*b.mean():.0f}% tokens, "
          f"{100-100*b.mean():.0f}% cut)  p10={np.percentile(b,10):.3f} "
          f"med={np.median(b):.3f} p90={np.percentile(b,90):.3f}")
    print(f"steps/video: mean={np.mean(nsteps):.2f}  "
          f"videos with 0 steps (no compression): {int(np.sum(np.array(nsteps)==0))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
