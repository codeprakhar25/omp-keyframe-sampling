#!/usr/bin/env python3
"""Export our frame picks to an lmms-eval injection file.

Output: JSON mapping LVB question id -> temporally-sorted list of frame
timestamps (seconds). Our ids == LVB qids (e.g. "q2bpkhjNxf0_0"), so the
join into lmms-eval is direct, zero remap.

Two sources:
  --from-scores results/scores/scores.jsonl --method topk --k 6 [--bin 600]
      derive SigLIP top-k picks: argsort per-frame scores desc, take k, map
      to times, sort ascending.
  --from-picks results/picks/omp_600_k6.json
      pass through an existing {id: {idx, secs}} picks file.

The consumer (lmms-eval task/adapter patch) reads $LVB_PICKS and, for each
qid, extracts exactly these timestamps from the video instead of uniform
sampling. Timestamps are seconds from clip start.
"""
import argparse
import json
import os
import sys


def load_scores(path, bin_filter=None):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if bin_filter is not None and str(d.get("length_bin")) != str(bin_filter):
                continue
            rows[d["id"]] = d
    return rows


def topk_picks(scores_rows, k):
    """id -> sorted list of timestamps for the k highest-scoring frames."""
    out = {}
    for qid, d in scores_rows.items():
        times, scores = d["times"], d["scores"]
        if not times or not scores:
            continue
        n = min(len(times), len(scores))
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        secs = sorted(float(times[i]) for i in order)
        out[qid] = secs
    return out


def from_picks_file(path):
    """Accept {id:{idx,secs}}, {id:[secs]}, or a wrapper
    {method,bin,k,picks:{...}}; emit {id: sorted secs}."""
    d = json.load(open(path))
    if isinstance(d, dict) and "picks" in d and isinstance(d["picks"], dict):
        d = d["picks"]
    out = {}
    for qid, v in d.items():
        if isinstance(v, dict):
            secs = v.get("secs")
            if secs is None and "idx" in v and "times" in v:
                secs = [v["times"][i] for i in v["idx"]]
        else:
            secs = v
        if secs is None:
            continue
        out[qid] = sorted(float(s) for s in secs)
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-scores", help="scores.jsonl with per-frame times+scores")
    src.add_argument("--from-picks", help="existing {id:{idx,secs}} picks json")
    ap.add_argument("--method", default="topk", choices=["topk"],
                    help="pick rule for --from-scores")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--bin", default=None, help="filter scores to one length_bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.from_scores:
        rows = load_scores(args.from_scores, args.bin)
        if not rows:
            sys.exit(f"no rows in {args.from_scores} for bin={args.bin}")
        picks = topk_picks(rows, args.k)
    else:
        picks = from_picks_file(args.from_picks)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(picks, f)
    kv = list(picks.items())
    nfr = [len(v) for v in picks.values()]
    print(f"wrote {args.out}: {len(picks)} qids, "
          f"frames/q min={min(nfr)} max={max(nfr)} "
          f"(e.g. {kv[0][0]} -> {kv[0][1][:4]}{'...' if len(kv[0][1])>4 else ''})")


if __name__ == "__main__":
    main()
