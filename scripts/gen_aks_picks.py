#!/usr/bin/env python3
"""AKS ADA (meanstd) picks from cached LongCLIP scores.jsonl.

Faithful port of ncTimTang/AKS frame_select.py::meanstd via harness.replay_selectors.aks_indices.
CPU only. Emits lmms-eval injection format {qid: [seconds]}.
"""
import argparse, json, os
from harness.replay_selectors import aks_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--bin", default=None, help="length_bin filter (LVB 15/60/600/3600); omit for all rows")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--t1", type=float, default=0.8)
    ap.add_argument("--t2", type=float, default=-100.0)
    ap.add_argument("--all-depth", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-skips", action="store_true")
    a = ap.parse_args()

    rows = []
    with open(a.scores) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if a.bin is not None:
                b = str(r.get("length_bin", "")).rstrip("s")
                if b and b != str(a.bin).rstrip("s"):
                    continue
            rows.append(r)
    print(f"scores rows in scope: {len(rows)} (bin={a.bin or 'ALL'}) AKS t1={a.t1} t2={a.t2} depth={a.all_depth}")

    picks, bad = {}, []
    for r in rows:
        rid = r["id"]
        scores = r["scores"]
        times = r.get("times") or r.get("secs")
        if times is None:
            # assume 1fps indexed
            times = list(range(len(scores)))
        if len(times) != len(scores):
            bad.append(rid); continue
        idx = aks_indices(scores, times, a.k, t1=a.t1, t2=a.t2, all_depth=a.all_depth)
        picks[rid] = sorted(round(float(times[i]), 1) for i in idx)

    if bad:
        print(f"!! {len(bad)} frame/time mismatch e.g. {bad[:3]}")
        if not a.allow_skips:
            raise SystemExit("REFUSING incomplete picks")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(picks, open(a.out, "w"))
    print(f"wrote {a.out}: {len(picks)} items")


if __name__ == "__main__":
    main()
