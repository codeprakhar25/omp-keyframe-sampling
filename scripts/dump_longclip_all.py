#!/usr/bin/env python3
"""One-pass LongCLIP dump: per-frame scores + image embeddings + question text embeds.

Encodes every frame ONCE and emits everything the three selectors need:
  - scalar cosine score  -> scores jsonl  (top-k)                     [dump_scores schema]
  - per-frame 768-d image vector -> results/embeds_lc/{id}.npz         (AdaRD Gram diversity)
  - per-question 768-d text vector -> results/embeds_lc/text_embeds.npz (OMP query residual)

LongCLIP-L, 248-token text window (truncate=True). Run with vLLM DOWN so it gets the
full card (big batches, ~5-10x faster than sharing the 2.8G left under a warm server).
Needs ./Long-CLIP repo + HF BeichenZhang/LongCLIP-L checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from harness.media import load_frames
from harness.manifest import check_gold_reliable
from harness.text import question_stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--repo-dir", default="Long-CLIP")
    ap.add_argument("--ckpt-repo", default="BeichenZhang/LongCLIP-L")
    ap.add_argument("--ckpt-file", default="longclip-L.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dump-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=3600)
    ap.add_argument("--bins", nargs="+", default=None)
    ap.add_argument("--gold-reliable-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--scores-out", default="results/scores/scores_longclip.jsonl")
    ap.add_argument("--emb-dir", default="results/embeds_lc")
    ap.add_argument("--text-out", default="results/embeds_lc/text_embeds.npz")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    import torch
    from huggingface_hub import hf_hub_download
    from model import longclip

    ck = hf_hub_download(args.ckpt_repo, args.ckpt_file)
    model, prep = longclip.load(ck, device=args.device)
    model = model.float().eval()  # fp32 everywhere (CLIP LayerNorm stays fp32)
    dev = args.device

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    # refuse a filter that would silently drop every item (see harness/manifest.py)
    check_gold_reliable(manifest, args.gold_reliable_only, args.manifest)
    bins = set(args.bins) if args.bins else None
    os.makedirs(args.emb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.scores_out) or ".", exist_ok=True)

    done_ids = set()
    if args.resume and os.path.exists(args.scores_out):
        done_ids = {json.loads(l)["id"] for l in open(args.scores_out) if l.strip()}
        print(f"resume: {len(done_ids)} ids already scored")

    text_ids, text_vecs = [], []
    n = 0
    sf = open(args.scores_out, "a" if args.resume else "w", encoding="utf-8")
    for item in manifest:
        b = str(item.get("length_bin", "all")).replace("s", "")
        if bins and b not in bins:
            continue
        if args.gold_reliable_only and not item.get("gold_reliable"):
            continue
        rid = item.get("id")
        if args.resume and rid in done_ids:
            continue
        frames = load_frames(item, dump_fps=args.dump_fps, max_frames=args.max_frames)
        if not frames:
            continue
        tok = longclip.tokenize([question_stem(item)], truncate=True).to(dev)
        with torch.no_grad():
            ft = model.encode_text(tok).float()
            ft = ft / ft.norm(dim=-1, keepdim=True)
            embs = []
            for i in range(0, len(frames), args.batch):
                px = torch.stack([prep(f.image) for f in frames[i:i + args.batch]]).to(dev)
                fi = model.encode_image(px).float()
                fi = fi / fi.norm(dim=-1, keepdim=True)
                embs.append(fi.cpu())
            E = torch.cat(embs)                       # (N,768) normed
            scores = (E @ ft.cpu().T).squeeze(1)      # (N,) cosine
        times = np.array([f.seconds for f in frames], dtype=np.float32)
        np.savez_compressed(os.path.join(args.emb_dir, f"{rid}.npz"),
                            emb=E.numpy().astype(np.float16), times=times)
        text_ids.append(rid)
        text_vecs.append(ft.cpu().numpy().astype(np.float16)[0])
        sf.write(json.dumps({
            "id": rid, "length_bin": b, "times": times.tolist(),
            "scores": [float(x) for x in scores.tolist()],
            "gold_evidence_seconds": item.get("gold_evidence_seconds"),
        }) + "\n")
        sf.flush()
        n += 1
        print(f"[{n}] {rid} bin={b} nf={len(frames)}", flush=True)
    sf.close()

    # merge text embeds (keep any prior ones on resume)
    if text_ids:
        if args.resume and os.path.exists(args.text_out):
            prev = np.load(args.text_out, allow_pickle=True)
            pid, pv = prev["ids"].tolist(), list(prev["text"])
            keep = [i for i, x in enumerate(pid) if x not in set(text_ids)]
            text_ids = [pid[i] for i in keep] + text_ids
            text_vecs = [pv[i] for i in keep] + text_vecs
        np.savez_compressed(args.text_out,
                            ids=np.array(text_ids), text=np.stack(text_vecs))
    print(f"\nwrote {n} items: scores->{args.scores_out} embeds->{args.emb_dir}/ text->{args.text_out}")


if __name__ == "__main__":
    main()
