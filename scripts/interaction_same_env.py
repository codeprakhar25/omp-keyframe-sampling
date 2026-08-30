#!/usr/bin/env python3
"""Recompute the selector x budget interaction using only same-environment arms.

The banked ``interaction_selector_x_budget.json`` mixed environments: it took the
eight-frame arms from the primary RTX PRO 4500 bank and the sixteen-frame arms from
the secondary L40S run. Because that run also replicated both full-resolution
baselines (``repl_unif8full_*`` for uniform, ``fracone_*`` for OMP), the whole 2x2
design is available inside one environment and the interaction can be computed
without any cross-stack comparison.

Arms expected under --arms-dir, as lmms-eval ``samples_*.jsonl``:

    repl_unif8full_{600,3600}   uniform,  k=8,  full resolution
    uni16half_{600,3600}        uniform,  k=16, ~46% per-frame budget
    fracone_{600,3600}          OMP,      k=8,  full resolution
    omp16half_{600,3600}        OMP,      k=16, ~46% per-frame budget

Usage:
    python scripts/interaction_same_env.py --arms-dir results/uniform_half_2026-07-28/arms
"""
import argparse
import json
import os
from math import comb

ARMS = ("repl_unif8full", "uni16half", "fracone", "omp16half")
BINS = ("600", "3600")


def load_arm(arms_dir, arm, dur):
    """Map question id -> 1 if the answerer was correct, else 0."""
    path = os.path.join(arms_dir, f"{arm}_{dur}.jsonl")
    correct = {}
    with open(path) as fh:
        for line in fh:
            acc = json.loads(line)["lvb_acc"]
            correct[acc["id"]] = int(acc["answer"] == acc["parsed_pred"])
    return correct


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    args = ap.parse_args()

    out = {"per_bin": {}}
    pooled = {"n": 0, "gain_u": 0, "gain_o": 0, "omp_more": 0, "unif_more": 0}

    for dur in BINS:
        arms = {a: load_arm(args.arms_dir, a, dur) for a in ARMS}
        ids = sorted(set.intersection(*(set(v) for v in arms.values())))

        gain_u = [arms["uni16half"][i] - arms["repl_unif8full"][i] for i in ids]
        gain_o = [arms["omp16half"][i] - arms["fracone"][i] for i in ids]

        omp_more = sum(1 for o, u in zip(gain_o, gain_u) if u < o)
        unif_more = sum(1 for o, u in zip(gain_o, gain_u) if o < u)

        n = len(ids)
        gu, go = sum(gain_u) / n * 100, sum(gain_o) / n * 100
        out["per_bin"][dur] = {
            "n": n,
            "acc": {a: sum(arms[a][i] for i in ids) / n for a in ARMS},
            "gain_uniform_pt": round(gu, 4),
            "gain_omp_pt": round(go, 4),
            "interaction_pt": round(go - gu, 4),
            "omp_gained_more": omp_more,
            "uniform_gained_more": unif_more,
            "p_interaction": round(mcnemar_exact(omp_more, unif_more), 6),
        }

        pooled["n"] += n
        pooled["gain_u"] += sum(gain_u)
        pooled["gain_o"] += sum(gain_o)
        pooled["omp_more"] += omp_more
        pooled["unif_more"] += unif_more

    n = pooled["n"]
    gu, go = pooled["gain_u"] / n * 100, pooled["gain_o"] / n * 100
    out["pooled"] = {
        "n": n,
        "gain_uniform_pt": round(gu, 4),
        "gain_omp_pt": round(go, 4),
        "interaction_pt": round(go - gu, 4),
        "omp_gained_more": pooled["omp_more"],
        "uniform_gained_more": pooled["unif_more"],
        "p_interaction": round(mcnemar_exact(pooled["omp_more"], pooled["unif_more"]), 6),
    }
    out["note"] = (
        "All four arms come from the same compute environment; no value is compared "
        "against the primary RTX PRO 4500 bank."
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
