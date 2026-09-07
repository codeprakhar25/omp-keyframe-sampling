#!/usr/bin/env python3
"""Generate OMP (query-residual) picks from cached embeddings -> picks json.

Consumes cached image embeds (harness.embeds resolves siglip / longclip / embeds_lc)
plus a STEM-ONLY text-embed npz from score_from_embeds.py --text-embeds-out, and runs
harness.replay_selectors.omp_indices per item. CPU, seconds.

Emits the lmms-eval injection format directly ({qid: [seconds]}) alongside the legacy
{idx,secs} shape, so it can drive $LVB_PICKS_OMP_{SC}_K{k} with no extra hop.

Two things this file used to get wrong, both fixed 2026-07-15:
  * `E = d["siglip"]` was hardcoded -> LongCLIP OMP was impossible, and LongCLIP is
    the encoder LDDR standardizes every baseline to. Now goes through harness.embeds.
  * missing embeds / frame-count mismatches did `skipped += 1` and still exited 0, so
    an incomplete picks file looked like success and only blew up inside lmms-eval
    (KeyError) after a model load. Now it refuses unless --allow-skips.

The text embeds MUST be stem-only. Passing the answerer's fused prompt here is the
2026-07-15 bug: OMP leans on the query embedding harder than top-k does, because each
pick is chosen to explain a residual component of that vector.
"""
import argparse
import json
import os

import numpy as np

from harness.embeds import l2, load_image_embed
from harness.replay_selectors import omp_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.jsonl (frame-count check + id filter)")
    ap.add_argument("--scorer", required=True, choices=["sig", "lc"],
                    help="which image tower the text embeds live in — must match --text-embeds")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--text-embeds", required=True,
                    help="STEM-ONLY text embeds from score_from_embeds.py --text-embeds-out")
    ap.add_argument("--bin", default=None, help="length_bin filter on scores rows")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-skips", action="store_true",
                    help="proceed despite unusable items (default: refuse)")
    args = ap.parse_args()

    t = np.load(args.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))

    rows = {}
    with open(args.scores) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.bin and str(r["length_bin"]).rstrip("s") != str(args.bin).rstrip("s"):
                continue
            rows[r["id"]] = r
    print(f"scores rows in scope: {len(rows)}  (bin={args.bin or 'ALL'})")

    picks, reasons = {}, {"no_text_embed": [], "no_image_embed": [], "frame_mismatch": []}
    for rid, r in rows.items():
        if rid not in qmap:
            reasons["no_text_embed"].append(rid)
            continue
        times, emb = load_image_embed(rid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None:
            reasons["no_image_embed"].append(rid)
            continue
        E = l2(emb)
        if len(r["scores"]) != E.shape[0]:
            reasons["frame_mismatch"].append(rid)
            continue
        idx = omp_indices(l2(qmap[rid].astype(np.float32)), E, args.k)
        picks[rid] = {"idx": [int(i) for i in idx],
                      "secs": [round(float(times[i]), 1) for i in idx]}

    skipped = sum(len(v) for v in reasons.values())
    if skipped:
        print(f"!! {skipped} items unusable:")
        for why, ids in reasons.items():
            if ids:
                print(f"   {why:16s} {len(ids):4d}  e.g. {ids[:3]}")
        if not args.allow_skips:
            raise SystemExit(
                "REFUSING: an incomplete picks file fails inside lmms-eval (KeyError) only "
                "AFTER a model load. Fix the inputs, or pass --allow-skips deliberately."
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # lmms-eval injection format: qid -> temporally sorted seconds
    json.dump({q: sorted(v["secs"]) for q, v in picks.items()}, open(args.out, "w"))
    side = args.out.replace(".json", ".meta.json")
    json.dump({"method": f"omp{args.k}", "scorer": args.scorer, "bin": args.bin,
               "k": args.k, "n": len(picks), "picks": picks}, open(side, "w"))
    print(f"wrote {args.out}: {len(picks)} items ({skipped} skipped)  [+ {side}]")


if __name__ == "__main__":
    main()
