#!/usr/bin/env python3
"""Coverage gate for the SigLIP scorer ablation.

Reports which cached embeds actually carry a ``siglip`` tower, broken down by
LongVideoBench duration bin. The cache is mixed: it was filled in passes, so some
items have siglip+dinov2 and others only longclip.

This exists because "the directory is non-empty" is not coverage. The join that
matters is over every id the run will consume; one missing id has previously killed
a multi-hour GPU run partway through. Run this before renting anything.

    python scripts/check_siglip_coverage.py --embeds-dir results/embeds_azure/flat
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np


def tower_keys(path):
    try:
        return set(np.load(path).files) - {"times"}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds-dir", required=True)
    ap.add_argument("--manifest", default=None,
                    help="optional lvb_val.json to resolve duration_group per id")
    ap.add_argument("--bin", default=None, help="report pass/fail for this duration bin")
    ap.add_argument("--write-ids", default=None, help="write siglip-covered ids to this json")
    args = ap.parse_args()

    combos = collections.Counter()
    sig, lc, corrupt = set(), set(), []

    for path in sorted(glob.glob(os.path.join(args.embeds_dir, "*.npz"))):
        qid = os.path.basename(path)[:-4]
        keys = tower_keys(path)
        if keys is None:
            corrupt.append(qid)
            continue
        combos[tuple(sorted(keys))] += 1
        if "siglip" in keys:
            sig.add(qid)
        if "longclip" in keys:
            lc.add(qid)

    total = sum(combos.values())
    print(f"embeds dir : {args.embeds_dir}")
    print(f"files      : {total}  (corrupt: {len(corrupt)})")
    print("tower combinations:")
    for combo, n in combos.most_common():
        print(f"   {'+'.join(combo):28s} {n}")
    print(f"\nsiglip covered : {len(sig)}")
    print(f"longclip covered: {len(lc)}")
    print(f"BOTH towers     : {len(sig & lc)}   <- usable for a same-item scorer swap")

    if args.manifest and os.path.exists(args.manifest):
        groups = {}
        for row in json.load(open(args.manifest)):
            groups[row["id"]] = str(row.get("duration_group"))
        per_bin = collections.defaultdict(lambda: [0, 0, 0])
        for qid, grp in groups.items():
            per_bin[grp][0] += 1
            if qid in sig:
                per_bin[grp][1] += 1
            if qid in sig and qid in lc:
                per_bin[grp][2] += 1
        print("\nper duration bin:  total / siglip / both")
        for grp in sorted(per_bin, key=lambda g: int(g)):
            t, s, b = per_bin[grp]
            print(f"   {grp:>6s}s   {t:5d} / {s:5d} / {b:5d}   siglip {s/t:6.1%}")
        if args.bin:
            t, s, b = per_bin.get(args.bin, (0, 0, 0))
            ok = t > 0 and s == t
            print(f"\nGATE bin={args.bin}: {'PASS' if ok else 'FAIL'} "
                  f"({s}/{t} siglip). {'' if ok else 'Re-encode the missing ids before renting a GPU.'}")
    else:
        print("\n(no --manifest given; per-bin breakdown skipped)")

    if args.write_ids:
        json.dump(sorted(sig), open(args.write_ids, "w"))
        print(f"\nwrote {len(sig)} siglip-covered ids -> {args.write_ids}")


if __name__ == "__main__":
    main()
