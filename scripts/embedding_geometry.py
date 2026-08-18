#!/usr/bin/env python3
"""Why does OMP's residual never drop below ~0.96?

Tests the geometric explanation: LongCLIP frame embeddings of one video form a
near-maximally COHERENT dictionary (all share a large mean direction mu), while
the text query is nearly orthogonal to the image cone (modality gap). OMP's
first pick spends itself removing mu; after that the residual is orthogonal to
the cone every remaining frame lives in, so there is nothing left to correlate.

Measures per video:
  ||mu||                    norm of mean frame embedding (cone tightness)
  mean pairwise cos(e_i,e_j)  dictionary coherence
  cos(q, e) max/mean/std    cross-modal similarity and its spread
  cos(q, mu_hat)            how much query similarity is the shared component
  resid_frac after OMP      raw vs mean-centered dictionary

The decisive comparison is RAW vs CENTERED: if removing mu lets OMP drain far
more query norm, the coherence explanation is confirmed.

Usage:
  python3 embedding_geometry.py --emb-dir DIR --text NPZ [--k 8] [--out JSON]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def l2n(x, axis=-1, eps=1e-8):
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), eps, None)


def omp_trace(q, E, k):
    """Textbook OMP. Returns (picks, resid_frac_per_step, cos_resid_per_step)."""
    n = E.shape[0]
    k = min(k, n)
    q0 = q.copy()
    q_res = q.copy()
    basis, picks, rf, cr = [], [], [], []
    chosen = np.zeros(n, dtype=bool)
    n0 = np.linalg.norm(q0)
    for _ in range(k):
        s = E @ q_res
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        picks.append(b)
        chosen[b] = True
        # cos between residual query and the frame just picked
        rn = np.linalg.norm(q_res)
        cr.append(float(s[b] / max(rn, 1e-8)))
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            q_res = q_res - (q_res @ v) * v
        rf.append(float(np.linalg.norm(q_res) / max(n0, 1e-8)))
    return picks, rf, cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True, help="dir of <id>.npz with key 'emb'")
    ap.add_argument("--text", required=True, help="npz with keys 'ids','text'")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tz = np.load(args.text, allow_pickle=True)
    ids = [str(x) for x in tz["ids"]]
    T = l2n(np.asarray(tz["text"], dtype=np.float32))
    tmap = {i: T[j] for j, i in enumerate(ids)}

    rows = []
    missing = []
    all_frame_means = []

    for i in ids:
        p = os.path.join(args.emb_dir, i + ".npz")
        if not os.path.exists(p):
            missing.append(i)
            continue
        E = np.asarray(np.load(p, allow_pickle=True)["emb"], dtype=np.float32)
        if E.ndim != 2 or E.shape[0] < 2:
            missing.append(i)
            continue
        E = l2n(E)
        q = tmap[i]
        n = E.shape[0]

        # --- dictionary geometry ---
        mu = E.mean(axis=0)
        mu_norm = float(np.linalg.norm(mu))
        all_frame_means.append(mu)
        # mean pairwise cosine over i != j, via ||sum e||^2 = n + sum_{i!=j} e_i.e_j
        ssum = E.sum(axis=0)
        pair_mean = float((ssum @ ssum - n) / (n * (n - 1)))

        # --- cross-modal similarity ---
        s = E @ q
        mu_hat = mu / max(mu_norm, 1e-8)
        cos_q_mu = float(q @ mu_hat)

        # --- OMP raw vs mean-centered dictionary ---
        _, rf_raw, cr_raw = omp_trace(q, E, args.k)
        Ec = l2n(E - mu[None, :])          # remove shared cone direction, renormalize
        _, rf_cen, cr_cen = omp_trace(q, Ec, args.k)

        rows.append(dict(
            id=i, n_frames=n,
            mu_norm=mu_norm, pair_cos_mean=pair_mean,
            cos_q_max=float(s.max()), cos_q_mean=float(s.mean()),
            cos_q_std=float(s.std()), cos_q_mu=cos_q_mu,
            resid_raw_end=rf_raw[-1], resid_cen_end=rf_cen[-1],
            cos_resid_raw_1=cr_raw[0], cos_resid_raw_end=cr_raw[-1],
            cos_resid_cen_1=cr_cen[0], cos_resid_cen_end=cr_cen[-1],
        ))

    def agg(key):
        return float(np.mean([r[key] for r in rows]))

    G = l2n(np.mean(np.stack(all_frame_means), axis=0))
    # how aligned are per-video cone centers with the GLOBAL image mean?
    cone_align = float(np.mean([
        float(l2n(m) @ G) for m in all_frame_means
    ]))
    q_vs_global = float(np.mean([tmap[r["id"]] @ G for r in rows]))

    summary = dict(
        n_videos=len(rows), n_missing=len(missing), k=args.k,
        dictionary=dict(
            mu_norm=agg("mu_norm"),
            pair_cos_mean=agg("pair_cos_mean"),
            cone_align_to_global_mean=cone_align,
        ),
        cross_modal=dict(
            cos_q_max=agg("cos_q_max"),
            cos_q_mean=agg("cos_q_mean"),
            cos_q_std=agg("cos_q_std"),
            cos_q_to_video_cone=agg("cos_q_mu"),
            cos_q_to_global_image_mean=q_vs_global,
        ),
        omp_raw=dict(
            resid_frac_end=agg("resid_raw_end"),
            cos_resid_pick1=agg("cos_resid_raw_1"),
            cos_resid_pick_last=agg("cos_resid_raw_end"),
            query_mass_removed=1.0 - agg("resid_raw_end") ** 2,
        ),
        omp_centered=dict(
            resid_frac_end=agg("resid_cen_end"),
            cos_resid_pick1=agg("cos_resid_cen_1"),
            cos_resid_pick_last=agg("cos_resid_cen_end"),
            query_mass_removed=1.0 - agg("resid_cen_end") ** 2,
        ),
    )

    print(json.dumps(summary, indent=2))
    if missing:
        print(f"\n[warn] {len(missing)} ids missing embeddings: {missing[:5]}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(summary=summary, per_video=rows, missing=missing), f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
