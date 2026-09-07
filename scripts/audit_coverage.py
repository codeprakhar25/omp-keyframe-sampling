#!/usr/bin/env python3
"""Exhaustive unique-id coverage audit of EVERY banked result (2026-07-19).
One row per (run-dir, bin, k). Reports min unique-doc-id across arms vs official count.
Flags any run where the sharding bug (duplicated / missing docs) left <full coverage."""
import json, glob, re
from collections import defaultdict

EXP = {"15": 189, "60": 172, "600": 412, "3600": 564}


def cov(f):
    ids = [json.loads(l)["doc_id"] for l in open(f) if l.strip()]
    return len(ids), len(set(ids))


agg = defaultdict(lambda: {"u": 10**9, "n": 0, "arms": set()})
for f in glob.glob("results/**/*.jsonl", recursive=True):
    if "picks_lmmseval" in f:
        continue
    m = re.search(r"_(\d+)s_k(\d+)\.jsonl$", f)
    if not m:
        continue
    b, k = m.group(1), m.group(2)
    if b not in EXP:
        continue
    top = f.split("results/")[1].split("/")[0]
    am = re.search(r"val_(i|picks_lc|picks_sig|picks_omp_lc)_", f)
    arm = am.group(1) if am else "?"
    n, u = cov(f)
    a = agg[(top, b, k)]
    a["u"] = min(a["u"], u)
    a["n"] = max(a["n"], n)
    a["arms"].add(arm)

hdr = "{:<26}{:>6}{:>4}{:>6}{:>6}{:>5}  STATUS".format("RUN DIR", "bin", "k", "exp", "uniq", "arms")
print(hdr)
print("-" * 78)
corrupt = 0
for (top, b, k), v in sorted(agg.items()):
    e = EXP[b]
    ok = v["u"] == e
    corrupt += 0 if ok else 1
    st = "OK" if ok else "CORRUPT  ({}/{} unique)".format(v["u"], e)
    print("{:<26}{:>6}{:>4}{:>6}{:>6}{:>5}  {}".format(top, b, k, e, v["u"], len(v["arms"]), st))
print("-" * 78)
print("TOTAL (dir,bin,k) cells: {} | CORRUPT: {}".format(len(agg), corrupt))
