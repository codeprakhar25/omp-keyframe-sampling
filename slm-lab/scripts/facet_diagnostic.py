#!/usr/bin/env python3
"""Path 3 stage C: does facet decomposition actually open new query-reachable subspace?

This is the FREE kill-switch, run before any picks and long before any GPU.

The residual trace established the bottleneck: one stem vector reaches the image subspace
with cos_orig maxing at .233, and OMP ever explains only ~3.3% of the query norm. Path 3
is worth running iff facets reach FURTHER or reach ELSEWHERE than the averaged stem.

Reports, over the full pool of every video in a bin:

  1. distinctness   mean pairwise cos between facets of the same question, and
                    cos(facet, stem). If facets collapse onto the stem (~.9+), the
                    decomposition is cosmetic -> DEAD.
  2. reach          max_frames cos(facet_i, f)  vs  max_frames cos(stem, f).
                    "best_facet_reach" = max over facets. If it does not exceed the stem
                    baseline (.233), facets do not see the image subspace any better.
  3. displacement   does any facet's argmax frame differ from the stem's argmax frame,
                    and by how many seconds? Frames moving is necessary (not sufficient)
                    for accuracy to move.
  4. union reach    fraction of query mass explained when OMP is run against the facet SET
                    (m directions) instead of the single stem. The stem ceiling is ~3.3%.

Grounding filter: facets with grounding < --min-grounding are dropped and counted.
Ungrounded facets (the hallucination failure caught in the 2026-07-20 smoke test) DO move
picks, so they would pass a naive pick-overlap gate while being noise -- they must never
reach the selector.

CPU, cached embeds only. Usage:
  PYTHONPATH=. python3 scripts/facet_diagnostic.py --bin 3600 \
      --facet-embeds results/facets/facet_embeds.npz --out results/facets/diag_3600.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_explained(q, E, k):
    """Fraction of ||q|| explained after k OMP steps against dictionary E."""
    q0n = float(np.linalg.norm(q)) or 1.0
    r = q.astype(np.float32).copy()
    basis, chosen = [], np.zeros(E.shape[0], dtype=bool)
    for _ in range(min(k, E.shape[0])):
        s = E @ r
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        chosen[b] = True
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv <= 1e-6:
            break
        v /= nv
        basis.append(v)
        r = r - (r @ v) * v
    return 1.0 - float(np.linalg.norm(r)) / q0n


def multi_omp_explained(qs, E, k):
    """Round-robin OMP over m facet queries; fraction of TOTAL facet mass explained.

    Each step advances the facet whose residual is currently largest, so a facet that is
    already satisfied stops consuming budget. Comparable to omp_explained when m == 1.
    """
    R = [q.astype(np.float32).copy() for q in qs]
    n0 = [float(np.linalg.norm(q)) or 1.0 for q in qs]
    basis, chosen = [], np.zeros(E.shape[0], dtype=bool)
    for _ in range(min(k, E.shape[0])):
        j = int(np.argmax([np.linalg.norm(r) / n for r, n in zip(R, n0)]))
        s = E @ R[j]
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        chosen[b] = True
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv <= 1e-6:
            break
        v /= nv
        basis.append(v)
        R = [r - (r @ v) * v for r in R]          # every facet sees every pick
    return 1.0 - float(np.sum([np.linalg.norm(r) for r in R]) / np.sum(n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--facet-embeds", default="results/facets/facet_embeds.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-grounding", type=float, default=0.5)
    ap.add_argument("--scorer", default="lc")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # facet_embeds.npz holds BOTH bins; without this filter every --bin reported the
    # pooled 976 under a per-bin label (caught 2026-07-20 by both bins printing identical
    # numbers). text_lc_{bin}.npz is the same id set the rest of the pipeline bins on.
    binids = set(np.load(f"{args.text_dir}/text_lc_{args.bin}.npz")["ids"].tolist())

    z = np.load(args.facet_embeds, allow_pickle=True)
    ids, slots, vec = z["ids"], z["slots"], z["vec"].astype(np.float32)
    gnd, types = z["grounding"], z["types"]

    stem_of, facets_of = {}, defaultdict(list)
    inbin = np.array([i in binids for i in ids])
    for i in range(len(ids)):
        if not inbin[i]:
            continue
        if slots[i] == -1:
            stem_of[ids[i]] = vec[i]
        elif gnd[i] >= args.min_grounding:
            facets_of[ids[i]].append((vec[i], str(types[i]), float(gnd[i])))
    dropped = int(np.sum(inbin & (slots != -1) & (gnd < args.min_grounding)))

    S = defaultdict(list)
    nfac = defaultdict(int)
    used = 0
    for qid, q in stem_of.items():
        fl = facets_of.get(qid, [])
        if not fl:
            continue
        times, emb = load_image_embed(qid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < args.k:
            continue
        E = l2(emb)
        qs = l2(np.asarray(q, dtype=np.float32))
        F = [l2(np.asarray(f, dtype=np.float32)) for f, _, _ in fl]
        used += 1
        nfac[len(F)] += 1

        # 1. distinctness
        if len(F) > 1:
            pw = [float(F[a] @ F[b]) for a in range(len(F)) for b in range(a + 1, len(F))]
            S["facet_facet_cos"].append(float(np.mean(pw)))
        S["facet_stem_cos"].append(float(np.mean([f @ qs for f in F])))

        # 2. reach into the image subspace
        sim_stem = E @ qs
        stem_reach = float(sim_stem.max())
        reaches = [float((E @ f).max()) for f in F]
        S["stem_reach"].append(stem_reach)
        S["best_facet_reach"].append(max(reaches))
        S["mean_facet_reach"].append(float(np.mean(reaches)))
        S["reach_gain"].append(max(reaches) - stem_reach)

        # 3. displacement of the top-1 frame
        top_stem = int(np.argmax(sim_stem))
        tops = [int(np.argmax(E @ f)) for f in F]
        S["top1_moved"].append(float(any(t != top_stem for t in tops)))
        S["top1_shift_sec"].append(float(max(abs(times[t] - times[top_stem]) for t in tops)))

        # 4. explained mass, stem-OMP vs facet-set-OMP
        S["explained_stem"].append(omp_explained(qs, E, args.k))
        S["explained_facets"].append(multi_omp_explained(F, E, args.k))

    def m(key):
        return float(np.mean(S[key])) if S[key] else float("nan")

    print(f"\nbin={args.bin}s  n={used}  k={args.k}  "
          f"facets dropped (grounding<{args.min_grounding}): {dropped}")
    print(f"facet count dist: " + "  ".join(f"m{k}:{nfac[k]}" for k in sorted(nfac)))

    print("\n1. DISTINCTNESS (want LOW -- facets pointing different ways)")
    print(f"   mean facet-facet cos : {m('facet_facet_cos'):.4f}")
    print(f"   mean facet-stem  cos : {m('facet_stem_cos'):.4f}")

    print("\n2. REACH into image subspace (stem baseline from residual trace ~.233)")
    print(f"   stem   max cos       : {m('stem_reach'):.4f}")
    print(f"   best facet max cos   : {m('best_facet_reach'):.4f}")
    print(f"   mean facet max cos   : {m('mean_facet_reach'):.4f}")
    print(f"   REACH GAIN           : {m('reach_gain'):+.4f}")

    print("\n3. DISPLACEMENT of top-1 frame")
    print(f"   top1 moved           : {100*m('top1_moved'):.1f}% of videos")
    print(f"   max shift            : {m('top1_shift_sec'):.1f}s mean")

    print(f"\n4. QUERY MASS EXPLAINED at k={args.k} (stem ceiling ~3.3%)")
    print(f"   stem-OMP             : {100*m('explained_stem'):.2f}%")
    print(f"   facet-set OMP        : {100*m('explained_facets'):.2f}%")
    print(f"   GAIN                 : {100*(m('explained_facets')-m('explained_stem')):+.2f}pt")

    print("\nVERDICT GUIDE: facet-facet cos > .9 => cosmetic split, dead. "
          "reach gain <= 0 => facets see no more of the image subspace than the stem "
          "(kill unless displacement is large). Neither of these proves accuracy moves -- "
          "they only decide whether stage D is worth running.")

    if args.out:
        json.dump({"bin": args.bin, "n": used, "k": args.k, "dropped": dropped,
                   "facet_dist": {str(k): v for k, v in nfac.items()},
                   **{k: m(k) for k in S}}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
