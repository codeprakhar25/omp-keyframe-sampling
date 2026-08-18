#!/usr/bin/env python3
"""Where does OMP's win live -- by QUESTION TYPE? Zero GPU, existing lmms-eval logs.

Hypothesis (2026-07-21): cosine frame-selection assumes ONE evidence frame. Point-evidence
questions ("what color is X") fit that; sequence/relational ones ("which order", "what
happens before X") do NOT -- the answer is in the RELATION between frames. So OMP's gain over
uniform should be CONCENTRATED in point-evidence categories, and may VANISH or INVERT on
temporal/sequence categories where coverage beats a peak.

Each lmms-eval sample line carries lvb_acc = {id, question_category, answer, parsed_pred},
so correctness per LVB category is read straight off the logs. Arms parsed from filename:
  _i_        -> uniform
  _picks_lc_ -> topk (cosine)
  _picks_omp_lc_ -> omp
Dedup by id within each (arm,bin,k). Reports per-category acc per arm + omp-minus-uniform,
grouped by first-letter and by a temporal(T*)-vs-rest split.

Usage: python3 scripts/qtype_analysis.py --root results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict


ARM = [("picks_omp_lc", "omp"), ("picks_lc", "topk"), ("i", "uniform")]  # order: longest first


def parse_name(fn):
    b = os.path.basename(fn)
    m = re.search(r"_val_(picks_omp_lc|picks_lc|i)_(\d+)s_k(\d+)\.jsonl", b)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))    # armtag, bin, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/facets/qtype_analysis.json")
    args = ap.parse_args()

    # cell = (bin,k) -> arm -> {id: correct};  cat lookup id->category
    cell = defaultdict(lambda: defaultdict(dict))
    cat_of = {}
    for fn in glob.glob(f"{args.root}/**/*samples_longvideobench_val_*.jsonl", recursive=True):
        p = parse_name(fn)
        if not p:
            continue
        armtag, b, k = p
        arm = dict((a, n) for a, n in ARM)[armtag]
        for line in open(fn, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            la = d.get("lvb_acc")
            if not la:
                continue
            qid = la["id"]
            cat_of[qid] = la["question_category"]
            cell[(b, k)][arm][qid] = int(str(la["answer"]).strip() == str(la["parsed_pred"]).strip())

    out = {}
    for (b, k) in sorted(cell, key=lambda x: (int(x[0]), x[1])):
        arms = cell[(b, k)]
        if not all(a in arms for a in ("uniform", "topk", "omp")):
            continue
        ids = set(arms["uniform"]) & set(arms["topk"]) & set(arms["omp"])
        if len(ids) < 20:
            continue
        by = defaultdict(lambda: {"n": 0, "uniform": 0, "topk": 0, "omp": 0})
        for qid in ids:
            c = cat_of[qid]
            by[c]["n"] += 1
            for a in ("uniform", "topk", "omp"):
                by[c][a] += arms[a][qid]

        print(f"\n{'='*74}\nBIN {b}s  k={k}  (n={len(ids)} common ids, all 3 arms)")
        print(f"{'cat':>5} {'n':>4}  {'unif':>6} {'topk':>6} {'omp':>6}  "
              f"{'omp-unif':>8} {'topk-unif':>9}")
        rows = []
        for c in sorted(by, key=lambda c: by[c]["omp"]/by[c]["n"] - by[c]["uniform"]/by[c]["n"]):
            r = by[c]; n = r["n"]
            u, t, o = r["uniform"]/n, r["topk"]/n, r["omp"]/n
            print(f"{c:>5} {n:>4}  {u:>6.3f} {t:>6.3f} {o:>6.3f}  {o-u:>+8.3f} {t-u:>+9.3f}")
            rows.append({"cat": c, "n": n, "uniform": u, "topk": t, "omp": o})

        # first-letter groups + temporal(T*) split
        def agg(pred):
            g = {"n": 0, "uniform": 0, "topk": 0, "omp": 0}
            for qid in ids:
                if pred(cat_of[qid]):
                    g["n"] += 1
                    for a in ("uniform", "topk", "omp"):
                        g[a] += arms[a][qid]
            return g
        print("  -- first-letter groups --")
        for L in sorted(set(c[0] for c in by)):
            g = agg(lambda c, L=L: c[0] == L); n = g["n"]
            print(f"   {L}*   n={n:>4}  unif {g['uniform']/n:.3f}  topk {g['topk']/n:.3f}  "
                  f"omp {g['omp']/n:.3f}   omp-unif {g['omp']/n-g['uniform']/n:+.3f}")
        print("  -- temporal(T*) vs rest --")
        for name, pred in (("T*  ", lambda c: c[0] == "T"), ("rest", lambda c: c[0] != "T")):
            g = agg(pred); n = g["n"]
            if n:
                print(f"   {name} n={n:>4}  unif {g['uniform']/n:.3f}  topk {g['topk']/n:.3f}  "
                      f"omp {g['omp']/n:.3f}   omp-unif {g['omp']/n-g['uniform']/n:+.3f}  "
                      f"topk-unif {g['topk']/n-g['uniform']/n:+.3f}")
        out[f"{b}_{k}"] = {"n": len(ids), "rows": rows}

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
