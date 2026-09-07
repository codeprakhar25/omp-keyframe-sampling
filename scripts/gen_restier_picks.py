#!/usr/bin/env python3
"""Residual-tiered resolution picks: the EXACT baseline OMP-8 frames, a resolution FRACTION
per OMP rank.

Compression experiment (2026-07-21): the answerer gets the identical OMP-selected 8 frames
of the banked baseline (.5461, 3600s k8), but each frame's token budget scales with its OMP
RANK (marginal query-residual coverage): rank 1-2 full res, 3-5 half pixels, 6-8 quarter.
Claim under test: token cut at equal accuracy.

Frames are SOURCED FROM THE BASELINE picks file (exact seconds) so the answerer provably sees
the same frames as the .5461 run -> its accuracy is a valid paired control and any delta is
purely resolution. The baseline file is time-sorted (OMP rank lost), so rank is recovered by
recomputing OMP on the pool and matching each baseline second to its pick order; the few
frames that don't match (bf16-autocast near-ties in the original scorer vs this fp32 recompute)
fall back to cosine order. Frames themselves are never changed -- only the tier label is.

Output {qid: [[sec, frac], ...]} time-sorted, frac in {1.0, 0.5, 0.25}.
CPU, seconds. Usage:
  PYTHONPATH=. python3 scripts/gen_restier_picks.py --bin 3600 --k 8 \
    --baseline results/picks_lmmseval/picks_omp_lc_k8.json \
    --out results/picks_lmmseval/picks_restier_omp_lc_3600_k8.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_rank(query, E, k):
    q = query.astype(np.float32).copy()
    basis, chosen, sel = [], np.zeros(E.shape[0], dtype=bool), []
    for _ in range(min(k, E.shape[0])):
        s = E @ q
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        chosen[b] = True
        sel.append(b)
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            q = q - (q @ v) * v
    return sel


def tier_fraction(pos, k):
    """pos = 0-indexed priority rank. 1-2 -> 1.0, 3-5 -> 0.5, 6-8 -> 0.25 (k=8)."""
    if pos < 2:
        return 1.0
    if pos < 5:
        return 0.5
    return 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
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

    picks = {}
    n = fb_qids = fb_frames = miss = 0
    fr_tot = {1.0: 0, 0.5: 0, 0.25: 0}
    for qid in qmap:
        if qid not in base:
            miss += 1
            continue
        bsecs = [float(x) for x in base[qid]]                    # EXACT baseline frames
        times, emb = load_image_embed(qid, "lc", args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < 1:
            continue
        E = l2(emb)
        q = l2(np.asarray(qmap[qid], dtype=np.float32))
        tarr = np.asarray(times, dtype=np.float32)

        order = omp_rank(q, E, min(args.k, E.shape[0]))          # pool indices, pick order
        rank_of_time = {}                                        # sec(rounded) -> omp rank
        for r, idx in enumerate(order):
            rank_of_time[round(float(tarr[idx]), 1)] = r

        # priority key per baseline frame: omp rank if matched, else after-matched by -cosine
        used_ranks = set()
        keyed = []
        fb_here = 0
        for s in bsecs:
            j = int(np.argmin(np.abs(tarr - s)))                 # nearest pool frame
            cos = float(E[j] @ q)
            rk = rank_of_time.get(round(float(tarr[j]), 1))
            if rk is None or rk in used_ranks:
                rk = None
            if rk is None:
                fb_here += 1
                keyed.append((args.k + (1.0 - cos), s))          # unmatched: sort by cosine
            else:
                used_ranks.add(rk)
                keyed.append((float(rk), s))
        if fb_here:
            fb_qids += 1
            fb_frames += fb_here

        keyed.sort(key=lambda t: t[0])                           # ascending priority
        frac_of_sec = {s: tier_fraction(pos, len(keyed)) for pos, (_, s) in enumerate(keyed)}
        for f in frac_of_sec.values():
            fr_tot[f] = fr_tot.get(f, 0) + 1

        picks[qid] = [[round(s, 3), frac_of_sec[s]] for s in sorted(bsecs)]
        n += 1

    json.dump(picks, open(args.out, "w"))
    nom = 1196.0
    tok = 2 * nom + 3 * nom * 0.5 + 3 * nom * 0.25
    print(f"bin={args.bin} k={args.k}  n={n}  (qids absent from baseline: {miss})")
    print(f"cosine-fallback: {fb_qids} qids, {fb_frames} frames "
          f"({100*fb_frames/max(1,8*n):.1f}% of frames) -- FRAMES UNCHANGED, only tier label")
    print(f"tier frames: full={fr_tot.get(1.0,0)} half={fr_tot.get(0.5,0)} "
          f"quarter={fr_tot.get(0.25,0)}")
    print(f"est tokens/item: baseline {8*nom:.0f}  restier {tok:.0f}  "
          f"= {100*tok/(8*nom):.0f}% ({100-100*tok/(8*nom):.0f}% cut)")
    print(f"wrote {args.out}  ({len(picks)} qids)")


if __name__ == "__main__":
    main()
