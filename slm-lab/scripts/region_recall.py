#!/usr/bin/env python3
"""Region-recall gate (Fork B go/no-go). Free: local GPU scoring + echo, no frontier answerer.

Question: split a video into M chunks, score each chunk by its best frame -- does the GOLD chunk
land in the top-m chunks? And is that region-recall meaningfully above (a) the random-chunk null and
(b) the frame-level hit@k? If yes for some scorer -> coarse-to-fine is worth building.

Scores each video ONCE, then sweeps M / agg / probe / m in-memory. Filters to gold_reliable.
See research/region_recall_spec.md.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from harness.media import load_frames
from harness.metrics import hit_at_k, random_null, region_recall, wilson_ci


def build_scorer(name: str, model: str | None, pe_config: str | None, detector: str):
    """Return (score_fn, meta). score_fn(frames, item) -> list[float] per-frame relevance."""
    if name == "ground":
        return build_ground_scorer(detector)
    from harness.selectors import EmbeddingSelector, PESelector

    if name == "pe":
        sel = PESelector(config=pe_config or "PE-Core-L14-336")
    else:  # siglip
        sel = EmbeddingSelector(model_id=model or "google/siglip-so400m-patch14-384")

    def score_fn(frames, item):
        return [float(x) for x in sel._score([f.image for f in frames], item["question"])]

    return score_fn, {"scorer": name}


def build_ground_scorer(detector_id: str):
    """Open-vocab detection scorer: per-frame = max detection confidence for a cached target phrase.

    Targets come from data/targets.json ({id: {"target": str, "has_concrete_target": bool}}); if an
    item is missing, we fall back to the raw question text and flag has_concrete_target=False.
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    targets_path = "data/targets.json"
    targets = json.load(open(targets_path)) if os.path.exists(targets_path) else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    proc = AutoProcessor.from_pretrained(detector_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(dev).eval()
    batch = int(os.environ.get("GROUND_BATCH", "32"))

    def score_fn(frames, item):
        rec = targets.get(str(item["id"]), {})
        phrase = (rec.get("target") or "").strip()
        if not phrase:  # fallback: nothing concrete to detect
            phrase = item.get("question", "").split("\n")[0].strip()
        text = phrase if phrase.endswith(".") else phrase + "."
        scores = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=use_bf16):
            for i in range(0, len(frames), batch):
                imgs = [f.image for f in frames[i : i + batch]]
                enc = proc(images=imgs, text=[text] * len(imgs), return_tensors="pt").to(dev)
                out = model(**enc)
                # max box logit per image as the "is the target here" confidence
                logits = out.logits.float().sigmoid().amax(dim=(1, 2))  # (B,)
                scores.extend(float(x) for x in logits)
        return scores

    def has_concrete(item):
        return bool(targets.get(str(item["id"]), {}).get("has_concrete_target", False))

    return score_fn, {"scorer": "ground", "detector": detector_id, "has_concrete": has_concrete}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.local.json")
    ap.add_argument("--scorer", choices=["siglip", "pe", "ground"], default="siglip")
    ap.add_argument("--model", default=None)
    ap.add_argument("--pe-config", default=None)
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--bins", nargs="+", default=["60", "600", "3600"])
    ap.add_argument("--M", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--agg", nargs="+", default=["max", "mean", "ucb"])
    ap.add_argument("--probe", nargs="+", default=["1", "2", "4", "all"])
    ap.add_argument("--ms", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--hit-k", type=int, default=6, help="frame-level hit@k reference")
    ap.add_argument("--max-frames", type=int, default=3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch  # for topk on hit@k reference

    bins = set(args.bins)
    probes = [None if p == "all" else int(p) for p in args.probe]
    score_fn, meta = build_scorer(args.scorer, args.model, args.pe_config, args.detector)
    has_concrete = meta.get("has_concrete")

    manifest = json.load(open(args.manifest))
    # region_recall / null / hit@k accumulators keyed by the full config tuple
    acc = defaultdict(lambda: defaultdict(list))  # key -> m -> [0/1]
    null_acc = defaultdict(lambda: defaultdict(list))
    hit_acc = defaultdict(list)  # bin(,seg) -> [0/1]
    n_reliable = defaultdict(int)
    n_concrete = defaultdict(int)

    used = 0
    for item in manifest:
        b = str(item.get("length_bin", "all")).replace("s", "")
        if b not in bins:
            continue
        if not item.get("gold_reliable"):
            continue
        frames = load_frames(item, dump_fps=1.0, max_frames=args.max_frames)
        if not frames:
            continue
        scores = score_fn(frames, item)
        if len(scores) != len(frames):
            print(f"  !! score/frame mismatch {item['id']} {len(scores)}!={len(frames)}, skip")
            continue

        segs = [b]
        if args.scorer == "ground" and has_concrete is not None:
            hc = has_concrete(item)
            n_concrete[b] += int(hc)
            segs.append(f"{b}|concrete={hc}")
        n_reliable[b] += 1
        used += 1

        # frame-level hit@k reference (same score vector)
        st = torch.tensor(scores)
        top = torch.topk(st, min(args.hit_k, len(scores))).indices.tolist()
        h = hit_at_k([frames[j] for j in sorted(top)], item)
        for seg in segs:
            if h is not None:
                hit_acc[seg].append(h)

        for M in args.M:
            null = random_null(frames, item, M, args.ms)
            for seg in segs:
                if null:
                    for m in args.ms:
                        null_acc[(seg, M)][m].append(null[m])
            for agg in args.agg:
                for probe, plabel in zip(probes, args.probe):
                    rr = region_recall(scores, frames, item, M, agg=agg, probe=probe, ms=args.ms)
                    if rr is None:
                        continue
                    for seg in segs:
                        for m in args.ms:
                            acc[(seg, M, agg, plabel)][m].append(rr[m])
        print(f"[{used}] {item['id']} bin={b} nf={len(frames)} hit@{args.hit_k}={h}")

    def rate(vals):
        n = len(vals)
        k = sum(vals)
        lo, hi = wilson_ci(k, n)
        return {"rate": (k / n if n else None), "n": n, "ci": [round(lo, 3), round(hi, 3)]}

    result = {
        "meta": {k: v for k, v in meta.items() if k != "has_concrete"},
        "args": {"bins": sorted(bins), "M": args.M, "agg": args.agg, "probe": args.probe, "ms": args.ms},
        "n_reliable": dict(n_reliable),
        "n_concrete": dict(n_concrete),
        "frame_hit_at_k": {seg: rate(v) for seg, v in hit_acc.items()},
        "random_null": {
            f"{seg}|M={M}": {m: round(sum(v) / len(v), 3) for m, v in md.items()}
            for (seg, M), md in null_acc.items()
        },
        "region_recall": {
            f"{seg}|M={M}|agg={agg}|probe={p}": {m: rate(v) for m, v in md.items()}
            for (seg, M, agg, p), md in acc.items()
        },
    }
    out = args.out or f"results/region_recall_{args.scorer}.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nwrote {out}  ({used} items scored)")

    # headline: region_recall@1 at M=4, agg=max, dense, per bin, vs null + hit@k
    print("\n== headline: region_recall@1, M=4, agg=max, probe=all ==")
    for b in sorted(bins):
        key = f"{b}|M=4|agg=max|probe=all"
        rr = result["region_recall"].get(key, {}).get(1)
        nl = result["random_null"].get(f"{b}|M=4", {}).get(1)
        hk = result["frame_hit_at_k"].get(b, {}).get("rate")
        if rr:
            print(f"  bin {b}: rr@1={rr['rate']} ci={rr['ci']} (n={rr['n']}) | null@1={nl} | hit@{args.hit_k}={hk}")


if __name__ == "__main__":
    main()
