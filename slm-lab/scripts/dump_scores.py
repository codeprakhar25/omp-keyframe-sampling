#!/usr/bin/env python3
"""Score cache for the retrieve-then-ground Stage-1 gate (research/retrieve_then_ground_spec.md).

Scores every candidate frame of each manifest item ONCE with SigLIP and dumps per-frame scores +
timestamps to a JSONL cache. GPU-only to RUN (needs torch/transformers); import-safe offline --
model deps are lazy-imported inside build_scorer, so `import scripts.dump_scores` (or running
scripts/union_ceiling.py, which reads this script's OUTPUT, not the module) never needs torch.

Downstream: scripts/union_ceiling.py replays these scores with zero GPU to compare union-building
methods (peak-NMS vs quadrant vs flat-top-B) at equal frame budget -- the free decision gate before
any GPT-5.5 spend.

Mirrors scripts/region_recall.py's load/score loop and output style.
"""
from __future__ import annotations

import argparse
import json
import os

from harness.media import load_frames


def build_scorer(model_id: str):
    """Return score_fn(frames, question) -> list[float], one SigLIP score per frame.

    Lazy: transformers/torch only imported here, on first real call -- not at module import.
    """
    from harness.selectors import EmbeddingSelector

    sel = EmbeddingSelector(model_id=model_id)

    def score_fn(frames, question):
        return [float(x) for x in sel._score([f.image for f in frames], question)]

    return score_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--dump-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=3600, help="cap so long-bin items aren't truncated")
    ap.add_argument("--bins", nargs="+", default=None,
                    help="restrict to these length_bin values (e.g. 600 3600); default = all")
    ap.add_argument("--gold-reliable-only", action="store_true", default=True)
    ap.add_argument("--out", default="results/scores/scores.jsonl")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    bins = set(args.bins) if args.bins else None
    score_fn = build_scorer(args.model)  # first torch import happens here

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for item in manifest:
            b = str(item.get("length_bin", "all")).replace("s", "")
            if bins and b not in bins:
                continue
            if args.gold_reliable_only and not item.get("gold_reliable"):
                continue
            frames = load_frames(item, dump_fps=args.dump_fps, max_frames=args.max_frames)
            if not frames:
                continue
            scores = score_fn(frames, item["question"])
            row = {
                "id": item.get("id"),
                "length_bin": b,
                "times": [f.seconds for f in frames],
                "scores": scores,
                "gold_evidence_seconds": item.get("gold_evidence_seconds"),
            }
            f.write(json.dumps(row) + "\n")
            n += 1
            print(f"[{n}] {item.get('id')} bin={b} nf={len(frames)}")

    print(f"\nwrote {args.out} ({n} items)")


if __name__ == "__main__":
    main()
