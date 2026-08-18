#!/usr/bin/env python3
"""Beta curve: how much of DPP's accuracy is query relevance vs frame-frame diversity?

Banked DPP used beta = 1/2/4 only. This adds beta = 0 (pure diversity, query-blind) and
beta = 0.5, each in raw and per-video-CENTRED frame geometry, on the LVB 600s bin.

Two questions, one run:
  (1) beta curve -- does accuracy fall off as the kernel stops using the query?
  (2) centring -- the raw frame Gram has median effective rank 1.98 (70.5% of spectral mass
      in one shared direction); centring lifts it to 11.17. Does exposing that structure
      help, at the betas where it actually changes the picks?

All arms are k=8 full-res, so the visual token budget is identical (6,984 tok/item) and this
is a pure selection comparison.

Usage: analyze_dpp_beta.py --results-dir DIR --baselines-dir DIR
"""
import argparse
import glob
import json
import os
from math import comb

ARMS = {
    "b00r": "beta=0.0 raw",
    "b00c": "beta=0.0 centred",
    "b05r": "beta=0.5 raw",
    "b05c": "beta=0.5 centred",
}


def load_samples(path_glob):
    """-> {qid: correct}. qid at lvb_acc.id (NOT doc_id: doc_id is shard-local)."""
    d = {}
    for p in glob.glob(path_glob, recursive=True):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            a = json.loads(line).get("lvb_acc")
            if isinstance(a, dict) and "id" in a:
                d[a["id"]] = int(a["answer"] == a["parsed_pred"])
    return d


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    b01 = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    b10 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0, len(ids)
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), len(ids)


def acc(d):
    return sum(d.values()) / len(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="holds dpp_<arm>_600/**")
    ap.add_argument("--baselines-dir", required=True, help="holds banked omp/uniform/dpp samples")
    a = ap.parse_args()

    arms = {}
    for k, lab in ARMS.items():
        d = load_samples(os.path.join(a.results_dir, f"dpp_{k}_600", "**", "*samples*.jsonl"))
        print(f"{lab:20s} n={len(d):4d} acc={acc(d):.4f}"
              f"{'   COVERAGE FAIL' if len(d) != 412 else ''}")
        arms[lab] = d

    base = {}
    for name, pat in [("uniform", "*val_i_600s_k8*.jsonl"),
                      ("top-k", "*picks_lc_600s_k8*.jsonl"),
                      ("OMP", "*picks_omp_lc_600s_k8*.jsonl")]:
        d = load_samples(os.path.join(a.baselines_dir, "**", pat))
        if d:
            base[name] = d
            print(f"{name:20s} n={len(d):4d} acc={acc(d):.4f}  (banked)")

    ref = base.get("OMP")
    if not ref:
        print("\n!! no banked OMP found -- paired tests skipped")
        return

    print("\nvs OMP (McNemar, exact two-sided binomial):")
    for lab, d in arms.items():
        b01, b10, p, n = mcnemar(ref, d)
        star = "SIG" if p < .05 else ("~" if p < .10 else "ns")
        print(f"  {lab:20s} d={100*(acc(d)-acc(ref)):+6.2f} pp  disc {b10}v{b01}  "
              f"p={p:.4g}  {star}  n={n}")

    if "uniform" in base:
        print("\nvs uniform:")
        for lab, d in arms.items():
            b01, b10, p, n = mcnemar(base["uniform"], d)
            star = "SIG" if p < .05 else ("~" if p < .10 else "ns")
            print(f"  {lab:20s} d={100*(acc(d)-acc(base['uniform'])):+6.2f} pp  "
                  f"disc {b10}v{b01}  p={p:.4g}  {star}")

    print("\ncentring effect at matched beta (the actual question):")
    for b in ("0.0", "0.5"):
        r, c = arms.get(f"beta={b} raw"), arms.get(f"beta={b} centred")
        if r and c:
            b01, b10, p, n = mcnemar(r, c)
            star = "SIG" if p < .05 else ("~" if p < .10 else "ns")
            print(f"  beta={b}: centred − raw = {100*(acc(c)-acc(r)):+6.2f} pp  "
                  f"disc {b10}v{b01}  p={p:.4g}  {star}")

    print("\nbeta curve (accuracy vs relevance weight; banked 1/2/4 for context):")
    print("  0.0 raw     %.4f" % acc(arms["beta=0.0 raw"]))
    print("  0.5 raw     %.4f" % acc(arms["beta=0.5 raw"]))
    print("  1.0 raw     .6456  (banked)")
    print("  2.0 raw     .6505  (banked)")
    print("  4.0 raw     .6408  (banked)")
    print("  OMP         %.4f  (query-side reference)" % acc(ref))


if __name__ == "__main__":
    main()
