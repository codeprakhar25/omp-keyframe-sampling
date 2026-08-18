#!/usr/bin/env python3
"""Lever-1 Arm A finalizer: MinMax-DPP vs banked OMP + banked z-DPP, both long bins."""
import json, glob
from scipy.stats import binomtest


def load(pat):
    fs = glob.glob(pat, recursive=True)
    assert len(fs) == 1, f"{pat} -> {len(fs)} files"
    d = {}
    for line in open(fs[0]):
        if not line.strip():
            continue
        r = json.loads(line)
        la = r["lvb_acc"]
        d[r["doc_id"]] = int(la["answer"] == la["parsed_pred"])
    return d


def mc(base, new):
    ks = set(base) & set(new)
    nb = sum(1 for k in ks if base[k] == 1 and new[k] == 0)  # base better
    nn = sum(1 for k in ks if base[k] == 0 and new[k] == 1)  # new better
    p = binomtest(min(nb, nn), nb + nn).pvalue if nb + nn else 1.0
    return nn, nb, p


def acc(d):
    return sum(d.values()) / len(d)


for BIN in ["600", "3600"]:
    mm = load(f"results/dppmm_ablation/{BIN}_sh0/**/*picks_lc_{BIN}s_k8.jsonl")
    omp = load(f"results/lmmseval_matrix_clean/k8_{BIN}/**/*picks_omp_lc_{BIN}s_k8.jsonl")
    z1 = load(f"results/dpp_ablation/{BIN}_sh0/**/*picks_lc_{BIN}s_k8.jsonl")
    z2 = load(f"results/dpp_ablation/{BIN}_sh0/**/*picks_sig_{BIN}s_k8.jsonl")
    z4 = load(f"results/dpp_ablation/{BIN}_sh0/**/*picks_omp_lc_{BIN}s_k8.jsonl")
    print(f"=== {BIN}s  n={len(mm)} ===")
    print(f"  MinMax-DPP = {acc(mm):.4f}")
    print(f"  banked OMP = {acc(omp):.4f}")
    print(f"  z-DPP b1/b2/b4 = {acc(z1):.4f}/{acc(z2):.4f}/{acc(z4):.4f}")
    nn, nb, p = mc(omp, mm)
    print(f"  MinMax vs OMP:  mm>OMP {nn}, OMP>mm {nb}, p={p:.3f}, diff={100*(acc(mm)-acc(omp)):+.2f}pt")
    zbest = max([(acc(z1), "b1"), (acc(z2), "b2"), (acc(z4), "b4")])
    nn2, nb2, p2 = mc(z1 if zbest[1] == "b1" else z2 if zbest[1] == "b2" else z4, mm)
    print(f"  best z-DPP = {zbest[0]:.4f} ({zbest[1]}); MinMax vs bestZ diff={100*(acc(mm)-zbest[0]):+.2f}pt p={p2:.3f}")
