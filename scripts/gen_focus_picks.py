#!/usr/bin/env python3
"""FOCUS replay picks from cached LongCLIP scores.jsonl.

Uses harness.replay_selectors.focus_indices (Alg 2 + Sec 2.4 of arXiv 2510.27280)
on *precomputed* dense scores — NOT the paper's budgeted online scorer. Disclose
in writeups: fidelity is algorithmic replay on full score vectors, not budgeted ITM.
"""
import argparse, json, os
from harness.replay_selectors import focus_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--clip-len", type=float, default=16.0)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--pull-budget", type=float, default=0.5)
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
    print(f"scores rows in scope: {len(rows)} (bin={a.bin or 'ALL'}) FOCUS replay")

    picks, bad = {}, []
    for r in rows:
        rid = r["id"]
        scores = r["scores"]
        times = r.get("times") or r.get("secs")
        if times is None:
            times = list(range(len(scores)))
        if len(times) != len(scores):
            bad.append(rid); continue
        idx = focus_indices(scores, times, a.k, clip_len=a.clip_len, q=a.q,
                            alpha=a.alpha, pull_budget=a.pull_budget)
        picks[rid] = sorted(round(float(times[i]), 1) for i in idx)

    if bad:
        print(f"!! {len(bad)} mismatch e.g. {bad[:3]}")
        if not a.allow_skips:
            raise SystemExit("REFUSING incomplete picks")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(picks, open(a.out, "w"))
    print(f"wrote {a.out}: {len(picks)} items")


if __name__ == "__main__":
    main()
