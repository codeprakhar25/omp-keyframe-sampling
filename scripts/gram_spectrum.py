#!/usr/bin/env python3
"""Is there any frame-frame signal for a diversity rule to exploit?

Every selection rule we test -- query-side (OMP, alpha-orth, rfloor) and
frame-frame (DPP log-det, LDDR MinMax, MMR) -- reads the same object: the Gram
matrix of L2-normalised LongCLIP frame embeddings for one video. This script
measures that matrix's spectrum directly, which bounds what ANY rule built on it
can do, independently of the rule.

Reported per video, then aggregated:
  lam1_frac      fraction of spectral mass in the top eigendirection
  eff_rank       participation ratio (sum L)^2 / sum(L^2) -- "how many directions
                 the cloud really occupies"
  r90            number of eigendirections needed for 90% of the mass
  centred_*      the same after removing the per-video mean frame (the shared
                 cone), i.e. the signal a diversity rule can actually use

If eff_rank is far below the frame budget k, then a k-subset chosen to maximise
volume is over-determined: many different subsets achieve near-identical log-det,
so all volume-based rules must tie. That is a property of the representation,
not of the rules.
"""
import argparse
import glob
import json
import os

import numpy as np


def spectrum(E):
    """E: (n_frames, d) float. Returns eigenvalues of the frame-frame Gram, desc."""
    E = E.astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    # eigenvalues of E E^T == singular values squared; use the cheaper side
    s = np.linalg.svd(E, compute_uv=False)
    return np.sort(s ** 2)[::-1]


def stats(lam):
    lam = lam[lam > 1e-12]
    tot = lam.sum()
    eff = (tot ** 2) / (lam ** 2).sum()
    csum = np.cumsum(lam) / tot
    r90 = int(np.searchsorted(csum, 0.90) + 1)
    return lam[0] / tot, eff, r90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, default=8, help="frame budget to compare against")
    ap.add_argument("--out", default=None, help="optional json dump")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "**", "*.npz"), recursive=True))
    print(f"{len(files)} videos\n")

    rows = []
    for f in files:
        z = np.load(f)
        E = z["emb"]
        if E.shape[0] < 4:
            continue
        lam = spectrum(E)
        l1, eff, r90 = stats(lam)

        Ec = E.astype(np.float64)
        Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12
        Ec = Ec - Ec.mean(0, keepdims=True)          # drop the shared cone
        lam_c = spectrum(Ec)
        l1c, effc, r90c = stats(lam_c)

        rows.append(dict(video=os.path.basename(f), n=int(E.shape[0]),
                         lam1_frac=l1, eff_rank=eff, r90=r90,
                         c_lam1_frac=l1c, c_eff_rank=effc, c_r90=r90c))

    def col(k):
        return np.array([r[k] for r in rows])

    n = col("n")
    print(f"frames per video: median {np.median(n):.0f}  "
          f"range {n.min()}-{n.max()}\n")

    print("RAW Gram (what a log-det / volume rule sees)")
    print(f"  top eigendirection holds  median {np.median(col('lam1_frac')):.3f} "
          f"of all spectral mass   (mean {col('lam1_frac').mean():.3f})")
    print(f"  effective rank            median {np.median(col('eff_rank')):.2f}   "
          f"(mean {col('eff_rank').mean():.2f})")
    print(f"  dirs for 90% of mass      median {np.median(col('r90')):.0f}   "
          f"(mean {col('r90').mean():.1f})")

    print("\nCENTRED Gram (after removing the per-video mean frame = the cone)")
    print(f"  top eigendirection holds  median {np.median(col('c_lam1_frac')):.3f} "
          f"of residual mass   (mean {col('c_lam1_frac').mean():.3f})")
    print(f"  effective rank            median {np.median(col('c_eff_rank')):.2f}   "
          f"(mean {col('c_eff_rank').mean():.2f})")
    print(f"  dirs for 90% of mass      median {np.median(col('c_r90')):.0f}   "
          f"(mean {col('c_r90').mean():.1f})")

    er = col("eff_rank")
    print(f"\nvs frame budget k={a.k}:")
    print(f"  videos with raw effective rank < k : "
          f"{100*(er < a.k).mean():.1f}%")
    print(f"  videos with raw effective rank < 2 : "
          f"{100*(er < 2).mean():.1f}%")
    erc = col("c_eff_rank")
    print(f"  videos with centred eff rank  < k : {100*(erc < a.k).mean():.1f}%")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nper-video dump -> {a.out}")


if __name__ == "__main__":
    main()
