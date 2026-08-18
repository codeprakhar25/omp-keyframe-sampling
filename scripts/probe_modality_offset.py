#!/usr/bin/env python3
"""Is the CLIP modality gap a TRANSLATION or STRUCTURAL? Decides if cheap coverage can work.

Motivation (2026-07-21): pooled-cosine, pooled learned-map, pooled DPP all hit the same
wall (cos ~.20, ~5 reachable dims). Before investing in a token-level MaxSim re-dump, find
out WHICH gap we have on the pooled embeds we already hold:

  - If the gap is largely a GLOBAL OFFSET  mu = mean(image) - mean(text)  (a translation),
    then adding alpha*mu to the query should pull it into the image cloud and LIFT
    gold-recall. Cheap fixes (offset removal, MaxSim, whitening) then have room.
  - If removing the offset is FLAT, the gap is structural: the query's information simply
    has no image correlate, and no cheap re-scoring escapes it.

mu is computed on TRAIN videos, applied to HELD-OUT videos (5-fold), so a lifted number is
not the offset memorising the eval set. alpha swept; alpha=0 is the raw baseline.

CPU, minutes, pooled embeds only -- no re-dump. Usage:
  PYTHONPATH=. python3 scripts/probe_modality_offset.py --bins 600 3600 --k 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", nargs="+", default=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--manifest", default="data/manifest.lvb.long976.json")
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--out", default="results/facets/probe_modality_offset.json")
    args = ap.parse_args()

    gold = {}
    for it in json.load(open(args.manifest, encoding="utf-8")):
        g = it.get("gold_evidence_seconds")
        if isinstance(g, str):
            g = json.loads(g)
        if g:
            gold[it["id"]] = g

    # Load pooled query + frame pool for every gold video. img_mean = per-video image-cloud
    # centre (mean of that video's frames); q = L2 text query.
    qids, Q, IMG_MEAN, pools = [], [], [], {}
    for b in args.bins:
        z = np.load(f"{args.text_dir}/text_lc_{b}.npz")
        qmap = dict(zip(z["ids"].tolist(), z["text"]))
        for qid, qv in qmap.items():
            if qid not in gold:
                continue
            times, emb = load_image_embed(qid, "lc", args.embeds_dir, args.embeds_lc_dir)
            if times is None or emb.shape[0] < args.k:
                continue
            E = l2(emb)
            win = gold[qid]
            if not any(any(w[0] - args.tol <= float(t) <= w[1] + args.tol for w in win)
                       for t in times):
                continue                                   # no gold frame -> can't score recall
            qids.append(qid)
            Q.append(l2(np.asarray(qv, dtype=np.float32)))
            IMG_MEAN.append(E.mean(0))
            pools[qid] = (np.asarray(times, dtype=np.float32), E, win)

    Q = np.asarray(Q, dtype=np.float32)
    IMG_MEAN = np.asarray(IMG_MEAN, dtype=np.float32)
    n = len(qids)
    print(f"\ngold set: n={n}  bins={args.bins}  k={args.k}")
    if n < 2 * args.folds:
        print("too few gold videos."); return

    def inwin(qid, idx):
        times, _, win = pools[qid]
        return any(any(w[0] - args.tol <= float(times[i]) <= w[1] + args.tol for w in win)
                   for i in idx)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, args.folds)

    # global offset magnitude report (all data, descriptive only)
    mu_all = IMG_MEAN.mean(0) - Q.mean(0)
    print(f"global offset |mu| = {np.linalg.norm(mu_all):.4f}   "
          f"|img_mean_center|={np.linalg.norm(IMG_MEAN.mean(0)):.4f}  "
          f"|txt_mean|={np.linalg.norm(Q.mean(0)):.4f}")

    print(f"\n{'alpha':>6} {'recall@k':>9} {'mean_cos_gold':>13}")
    results = []
    for a in args.alphas:
        rec = cosg = 0.0
        for f in range(args.folds):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
            mu = IMG_MEAN[tr].mean(0) - Q[tr].mean(0)      # offset from TRAIN only
            for i in te:
                qid = qids[i]
                times, E, win = pools[qid]
                q = Q[i] + a * mu
                nq = float(np.linalg.norm(q))
                if nq > 1e-8:
                    q = q / nq
                sim = E @ q
                rec += inwin(qid, np.argsort(-sim)[:args.k])
                gmask = np.array([any(w[0]-args.tol <= float(t) <= w[1]+args.tol for w in win)
                                  for t in times])
                cosg += float(sim[gmask].mean()) if gmask.any() else 0.0
        rec /= n; cosg /= n
        print(f"{a:>6g} {rec:>9.4f} {cosg:>13.4f}")
        results.append({"alpha": a, "recall": rec, "cos_gold": cosg})

    raw = next(r for r in results if r["alpha"] == 0.0)["recall"]
    best = max(results, key=lambda r: r["recall"])
    lift = best["recall"] - raw
    print(f"\nraw(alpha=0) recall={raw:.4f}   best alpha={best['alpha']:g} "
          f"recall={best['recall']:.4f}   lift={lift:+.4f}")
    verdict = ("TRANSLATION -- offset removal lifts recall; cheap coverage (MaxSim/whitening) "
               "has room" if lift > 0.02 else
               "STRUCTURAL -- offset removal flat; no cheap re-scoring escapes the gap")
    print(f"VERDICT: {verdict}")
    json.dump({"n": n, "bins": args.bins, "k": args.k, "offset_norm": float(np.linalg.norm(mu_all)),
               "raw_recall": raw, "best": best, "lift": lift, "curve": results,
               "verdict": verdict}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
