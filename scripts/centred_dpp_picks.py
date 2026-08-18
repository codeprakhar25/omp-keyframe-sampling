#!/usr/bin/env python3
"""Diversity in the centred frame space instead of the raw one.

Motivation (scripts/gram_spectrum.py): the raw frame-frame Gram of a LongCLIP
video has median effective rank 1.97, with 70.9% of spectral mass in a single
direction shared by every frame. Any volume-based rule reading that matrix is
choosing inside a ~2-dimensional cloud, which is why DPP / MMR / LDDR-MinMax all
tie with OMP. Removing the per-video mean frame lifts effective rank to 9.70 --
above the k=8 budget -- so the structure exists but is masked.

This builds greedy log-det MAP picks on the CENTRED Gram, keeping query relevance
computed on the ORIGINAL embeddings (only the diversity geometry changes), and
reports how far the picks move from the existing OMP / DPP / LDDR picks.

If the picks barely move, the idea is dead without spending a GPU.
"""
import argparse
import glob
import json
import os

import numpy as np


def unit(X, axis=-1):
    return X / (np.linalg.norm(X, axis=axis, keepdims=True) + 1e-12)


def greedy_logdet(r, S, k):
    """Greedy log-det MAP for L = diag(r) S diag(r), via incremental Cholesky.
    r: (n,) nonneg relevance weights. S: (n,n) similarity. Returns k indices."""
    n = len(r)
    d2 = (r ** 2) * np.diag(S)
    picked, C = [], np.zeros((k, n))
    for it in range(min(k, n)):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        picked.append(j)
        if it + 1 == k:
            break
        L_j = (r * r[j]) * S[j]                      # column j of L
        e = (L_j - C[:it].T @ C[:it, j]) / np.sqrt(d2[j])
        C[it] = e
        d2 = np.maximum(d2 - e ** 2, 0.0)
        d2[picked] = -1.0
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--text", required=True, help="text_lc_<bin>.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="existing picks json to diff against")
    a = ap.parse_args()

    tz = np.load(a.text, allow_pickle=True)
    qmap = {str(i): v for i, v in zip(tz["ids"], tz["text"])}

    files = {os.path.basename(f)[:-4]: f
             for f in glob.glob(os.path.join(a.emb_dir, "**", "*.npz"), recursive=True)}
    ids = [i for i in qmap if i in files]
    print(f"{len(ids)} of {len(qmap)} query ids have embeddings locally")

    out_raw, out_cen = {}, {}
    rank_raw, rank_cen = [], []
    for qid in ids:
        z = np.load(files[qid])
        E = unit(z["emb"].astype(np.float64))
        t = z["times"].astype(float)
        q = unit(qmap[qid].astype(np.float64))

        r = E @ q                                     # relevance, ORIGINAL space
        r = np.clip(r, 1e-6, None)                    # DPP weights must be positive

        S_raw = E @ E.T
        Ec = unit(E - E.mean(0, keepdims=True))       # drop the shared cone
        S_cen = Ec @ Ec.T

        for S, store, rk in ((S_raw, out_raw, rank_raw), (S_cen, out_cen, rank_cen)):
            idx = greedy_logdet(r, S, a.k)
            store[qid] = [round(float(t[i]), 1) for i in idx]
            lam = np.linalg.eigvalsh(S)[::-1]
            lam = lam[lam > 1e-12]
            rk.append((lam.sum() ** 2) / (lam ** 2).sum())

    json.dump(out_cen, open(a.out, "w"))
    print(f"wrote {len(out_cen)} centred picks -> {a.out}")
    print(f"effective rank  raw {np.median(rank_raw):.2f}   centred {np.median(rank_cen):.2f}")

    def overlap(A, B):
        common = [i for i in A if i in B]
        if not common:
            return None, 0
        fr = [len(set(np.round(A[i], 1)) & set(np.round(B[i], 1))) / len(A[i]) for i in common]
        return float(np.mean(fr)), len(common)

    print(f"\npick overlap (fraction of the {a.k} timestamps shared):")
    o, n = overlap(out_cen, out_raw)
    print(f"  centred-DPP vs raw-DPP (ours, same code)   {o:.3f}   n={n}")
    for p in a.compare:
        if not os.path.exists(p):
            continue
        B = json.load(open(p))
        o, n = overlap(out_cen, B)
        o2, _ = overlap(out_raw, B)
        print(f"  centred-DPP vs {os.path.basename(p):28s} {o:.3f}   "
              f"(raw-DPP vs same: {o2:.3f})   n={n}")


if __name__ == "__main__":
    main()
