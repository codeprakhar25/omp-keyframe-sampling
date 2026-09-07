#!/usr/bin/env python3
"""Refuse a run whose picks do not cover EVERY qid its tasks will evaluate.

Why this exists: on 2026-07-15 the k=8 matrix died at 1498/2491 (60%, ~70 min of GPU)
because ONE qid (3hyPwjkdHEA_1) was absent from the picks file. picks_utils fails loud
on a missing qid -- correctly -- but it does so INSIDE lmms-eval, after the model is
loaded and most of the run is spent. The old preflight only checked that picks files
were non-empty. Non-empty is not coverage.

    python3 scripts/preflight_picks.py <bin> <picks.json> [picks.json ...]

Exit 1 (and name the missing ids) if any qid in the bin is absent from any picks file.
"""
import json
import os
import sys

MANIFESTS = ["data/manifest.lvb.full1560.json", "data/manifest.lvb.long976.json"]


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: preflight_picks.py <bin> <picks.json> [...]")
    want_bin = sys.argv[1].rstrip("s")
    picks_files = sys.argv[2:]

    ids = set()
    for mp in MANIFESTS:
        if not os.path.exists(mp):
            continue
        for x in json.load(open(mp)):
            if str(x.get("length_bin", "")).rstrip("s") == want_bin:
                ids.add(x["id"])
    if not ids:
        raise SystemExit(f"REFUSING: no manifest items for bin {want_bin}")

    bad = False
    for pf in picks_files:
        if not os.path.exists(pf):
            print(f"  MISSING FILE {pf}")
            bad = True
            continue
        have = set(json.load(open(pf)))
        miss = ids - have
        tag = "OK" if not miss else f"MISSING {len(miss)}"
        print(f"  {os.path.basename(pf):28s} covers {len(ids & have):4d}/{len(ids):<4d} {tag}"
              + (f"  e.g. {sorted(miss)[:3]}" if miss else ""))
        if miss:
            bad = True

    if bad:
        raise SystemExit(
            f"REFUSING to launch: picks do not cover the whole {want_bin}s bin.\n"
            f"  lmms-eval would load the model, run for ~an hour, then KeyError and lose "
            f"every arm. Regenerate picks (scripts/make_picks_all.sh) first."
        )
    print(f"  preflight OK: bin {want_bin}s fully covered by {len(picks_files)} picks file(s)")


if __name__ == "__main__":
    main()
