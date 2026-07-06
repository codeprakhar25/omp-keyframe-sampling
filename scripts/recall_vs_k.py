#!/usr/bin/env python3
"""Free diagnostic: does long-video recall recover with a bigger frame budget k?

Scores every candidate frame ONCE per video with SigLIP (the expensive step), then reads off
hit@k for several k without re-embedding. Tells us whether the Fork-A recall collapse (hit@k
1.0->0.24 at k=6) is a *budget* problem (fixable by adaptive-k) or a *selector* problem (SigLIP
can't rank the needle no matter how deep we look).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from harness.media import load_frames
from harness.metrics import hit_at_k
from harness.selectors import EmbeddingSelector, PESelector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.local.json")
    ap.add_argument("--ks", default="6,12,20,40")
    ap.add_argument("--max-dump-frames", type=int, default=3600)
    ap.add_argument("--out", default="results/recall_vs_k.json")
    ap.add_argument("--model", default=None, help="scorer HF id (e.g. google/siglip2-so400m-patch14-384)")
    ap.add_argument("--pe-config", default=None,
                    help="Meta PE-Core config (e.g. PE-Core-L14-336); uses PESelector instead of SigLIP")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    manifest = json.load(open(args.manifest))
    if args.pe_config:
        sel = PESelector(config=args.pe_config)
    elif args.model:
        sel = EmbeddingSelector(model_id=args.model)
    else:
        sel = EmbeddingSelector()
    torch = sel.torch

    # per (bin, k): list of hit@k (0/1); also n gold-bearing items per bin
    agg = defaultdict(lambda: defaultdict(list))
    per_item = []
    for i, item in enumerate(manifest):
        b = item.get("length_bin", "all")
        frames = load_frames(item, dump_fps=1.0, max_frames=args.max_dump_frames)
        scores = sel._score([f.image for f in frames], item["question"])
        row = {"id": item["id"], "bin": b, "n_frames": len(frames)}
        for k in ks:
            kk = min(k, len(frames))
            top = torch.topk(scores, kk).indices.tolist()
            selected = [frames[j] for j in sorted(top)]
            h = hit_at_k(selected, item)
            row[f"hit@{k}"] = h
            if h is not None:
                agg[b][k].append(h)
        per_item.append(row)
        print(f"[{i+1}/{len(manifest)}] {item['id']} {b} nf={len(frames)} "
              + " ".join(f"h@{k}={row.get(f'hit@{k}')}" for k in ks))

    table = {}
    for b in sorted(agg, key=lambda x: float(x[:-1]) if x[:-1].replace('.', '').isdigit() else 1e9):
        table[b] = {f"hit@{k}": (round(sum(v) / len(v), 3) if v else None) for k, v in
                    ((k, agg[b][k]) for k in ks)}
        table[b]["n"] = len(agg[b][ks[0]])
    out = {"ks": ks, "table": table, "per_item": per_item}
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n=== hit@k vs budget k (by length bin) ===")
    hdr = "bin    " + "".join(f"k={k}".rjust(8) for k in ks)
    print(hdr)
    for b, r in table.items():
        print(b.ljust(7) + "".join((f"{r[f'hit@{k}']:.2f}" if r[f'hit@{k}'] is not None else "  -  ").rjust(8) for k in ks))


if __name__ == "__main__":
    main()
