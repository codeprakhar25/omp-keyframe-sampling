#!/usr/bin/env python3
"""Idea-B: diversity-penalized fill on top of OMP anchors (the redundancy fix).

Late OMP picks (4-8) are 79-83rd-pct relevant but 40-49% are cos>0.8 near-duplicates of an
earlier pick (2026-07-17 finding). Hypothesis: keep the first `--anchors` OMP picks (they
find the moment), then fill the remaining budget by MMR — maximize relevance MINUS redundancy
to already-picked frames — so the tail stops re-picking the same moment.

  score(f) = lambda*rel[f] - (1-lambda)*max_{s in chosen} cos(f, s)
  rel[f]   = E[f] . q0   (original query relevance, same vector OMP/top-k rank on)

Writes lmms-eval picks {qid: sorted([secs])} AND prints the triage: how many frames per item
MMR moves vs full OMP (if it moves ~nothing, no GPU arm is warranted). CPU, seconds.
"""
import argparse
import json
import os

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_order(q0, E, k):
    """OMP pick ORDER (not sorted) — faithful to replay_selectors.omp_indices."""
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    q = q0.astype(np.float32).copy()
    basis, order, chosen = [], [], np.zeros(n, bool)
    for _ in range(k):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        order.append(best)
        chosen[best] = True
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-6:
            v /= nrm
            basis.append(v)
            q = q - (q @ v) * v
    return order


def mmr_fill(q0, E, k, anchors, lam):
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    rel = E @ q0
    chosen = list(omp_order(q0, E, k)[:anchors])
    chosenm = np.zeros(n, bool)
    chosenm[chosen] = True
    # running max cosine to any chosen frame
    maxcos = E @ E[chosen].T
    maxcos = maxcos.max(1) if maxcos.ndim == 2 else maxcos
    while len(chosen) < k:
        score = lam * rel - (1 - lam) * maxcos
        score[chosenm] = -np.inf
        best = int(np.argmax(score))
        chosen.append(best)
        chosenm[best] = True
        maxcos = np.maximum(maxcos, E @ E[best])
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scorer", default="lc", choices=["sig", "lc"])
    ap.add_argument("--text-embeds", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t = np.load(a.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))
    rows = {}
    for line in open(a.scores):
        if line.strip():
            r = json.loads(line)
            if str(r["length_bin"]).rstrip("s") == str(a.bin).rstrip("s"):
                rows[r["id"]] = r

    picks, moved, skip = {}, [], 0
    for rid, r in rows.items():
        if rid not in qmap:
            skip += 1
            continue
        times, emb = load_image_embed(rid, a.scorer)
        if times is None:
            skip += 1
            continue
        E = l2(emb)
        if len(r["scores"]) != E.shape[0] or E.shape[0] <= a.k:
            skip += 1
            continue
        q0 = l2(qmap[rid].astype(np.float32))
        omp = set(omp_order(q0, E, a.k))
        mmr = mmr_fill(q0, E, a.k, a.anchors, a.lam)
        moved.append(a.k - len(omp.intersection(mmr)))
        picks[rid] = sorted(round(float(times[i]), 1) for i in mmr)

    if skip:
        raise SystemExit(f"REFUSING: {skip} unusable items in bin {a.bin} — fix before GPU.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(picks, open(a.out, "w"))
    moved = np.array(moved)
    print(f"bin={a.bin} k={a.k} anchors={a.anchors} lam={a.lam}  n={len(picks)}")
    print(f"frames MMR moves vs OMP: mean={moved.mean():.2f}/{a.k}  "
          f"median={int(np.median(moved))}  items moving 0 frames={int((moved==0).sum())} "
          f"({(moved==0).mean():.1%})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
