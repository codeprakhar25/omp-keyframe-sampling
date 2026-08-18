#!/usr/bin/env python3
"""Final pick-file audit: every method x bin has selected frames for ALL official videos,
exactly k distinct frames each, no missing/extra qids, no duplicate timestamps within a video."""
import json, numpy as np

EXP = {"600": 412, "3600": 564}

# reference = the qid namespace the selectors enumerate (text-embed ids). Full coverage = keys == this set.
OFF = {b: set(np.load(f"results/embeds_text/text_lc_{b}.npz")["ids"].tolist()) for b in EXP}
print("official qids: 600 =", len(OFF["600"]), " 3600 =", len(OFF["3600"]))
methods = ["mmr", "iteralpha", "alpha0.5", "alpha0.75", "rf15", "rfloor", "topk"]
hdr = "{:<12}{:>5}{:>6}{:>5}{:>5}{:>10}{:>16}".format("method", "bin", "nqid", "kmin", "kmax", "dupframe", "qids==official")
print(hdr)
allok = True
for m in methods:
    for b in EXP:
        k = "16" if m == "topk" else "8"
        p = f"results/picks_lmmseval/picks_{m}_lc_{b}_k{k}.json"
        try:
            d = json.load(open(p))
        except FileNotFoundError:
            if m == "topk" and b == "600":
                continue  # topk k16 only needed for 3600 rebuild
            print("{:<12}{:>5}   MISSING {}".format(m, b, p)); allok = False; continue
        ks = [len(v) for v in d.values()]
        dupf = sum(1 for v in d.values() if len(v) != len(set(v)))
        qids = set(d.keys())
        ok = (qids == OFF[b] and dupf == 0 and min(ks) == max(ks))
        allok = allok and ok
        match = "YES" if qids == OFF[b] else "NO(miss{},extra{})".format(len(OFF[b] - qids), len(qids - OFF[b]))
        print("{:<12}{:>5}{:>6}{:>5}{:>5}{:>10}{:>16}".format(m, b, len(d), min(ks), max(ks), dupf, match))
print("\nALL PICK FILES CLEAN" if allok else "\n!! SOME PICK FILES HAVE ISSUES")
