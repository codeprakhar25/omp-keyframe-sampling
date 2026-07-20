#!/usr/bin/env python3
"""Path 3 stage B: LongCLIP-L text embeddings for the facets from gen_facets.py.

Encoding path is byte-identical to dump_longclip_all.py's question path
(longclip.tokenize(truncate=True) -> encode_text -> L2 normalize -> fp16) so facet
vectors live in exactly the same space as results/embeds_text/text_lc_*.npz and can be
compared against the cached image embeds without any rescaling.

Text tower only -- CPU, ~2-3k short strings, a couple of minutes.

Also re-encodes each STEM in the same batch. The stem vector is already in text_lc_*.npz,
but re-encoding it here means the stage-C diagnostic compares facets against a stem that
went through the identical code path in the identical process -- no cross-run drift.
The recomputed stem is asserted against the cached one (cos > 0.999) so a mismatch is
caught loudly instead of silently biasing the comparison.

Usage (pod):
  cd /workspace/slm-lab
  PYTHONPATH=. HF_HOME=/workspace/hf python3 scripts/dump_facet_embeds.py \
      --facets results/facets/facets_long976.jsonl \
      --out results/facets/facet_embeds.npz --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", default="results/facets/facets_long976.jsonl")
    ap.add_argument("--repo-dir", default="Long-CLIP")
    ap.add_argument("--ckpt-repo", default="BeichenZhang/LongCLIP-L")
    ap.add_argument("--ckpt-file", default="longclip-L.pt")
    ap.add_argument("--ckpt-path", default=None,
                    help="direct path to longclip-L.pt; skips hf_hub_download (and its "
                         "httpx dependency) when the checkpoint is already on the volume")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--text-dir", default="results/embeds_text",
                    help="for the cached-stem consistency assert")
    ap.add_argument("--out", default="results/facets/facet_embeds.npz")
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    import torch
    from model import longclip

    rows = [json.loads(l) for l in open(args.facets, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} questions, {sum(len(r['facets']) for r in rows)} facets")

    if args.ckpt_path:
        ck = args.ckpt_path
        if not os.path.exists(ck):
            sys.exit(f"!! --ckpt-path not found: {ck}")
    else:
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download(args.ckpt_repo, args.ckpt_file)
    model, _ = longclip.load(ck, device=args.device)
    model = model.float().eval()

    # flat list of every string to encode, tagged back to (qid, slot)
    texts, tags = [], []
    for r in rows:
        texts.append(r["stem"]); tags.append((r["id"], -1))
        for j, f in enumerate(r["facets"]):
            texts.append(f["text"]); tags.append((r["id"], j))

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            tok = longclip.tokenize(texts[i:i + args.batch], truncate=True).to(args.device)
            ft = model.encode_text(tok).float()
            ft = ft / ft.norm(dim=-1, keepdim=True)
            vecs.append(ft.cpu().numpy())
            if (i // args.batch) % 10 == 0:
                print(f"  {i}/{len(texts)}")
    V = np.concatenate(vecs).astype(np.float16)

    ids = np.array([t[0] for t in tags])
    slots = np.array([t[1] for t in tags], dtype=np.int16)
    types = []
    ftext = []
    gnd = []
    for r in rows:
        types.append("__stem__"); ftext.append(r["stem"]); gnd.append(1.0)
        for f in r["facets"]:
            types.append(f.get("type", "")); ftext.append(f["text"])
            gnd.append(float(f.get("grounding", 1.0)))

    # consistency assert vs the cached stem embeds the whole pipeline already uses
    worst, checked = 1.0, 0
    cache = {}
    for b in ("600", "3600"):
        p = os.path.join(args.text_dir, f"text_lc_{b}.npz")
        if os.path.exists(p):
            z = np.load(p)
            cache.update(dict(zip(z["ids"].tolist(), z["text"].astype(np.float32))))
    for k in range(len(ids)):
        if slots[k] != -1 or ids[k] not in cache:
            continue
        a = V[k].astype(np.float32); b = cache[ids[k]]
        a = a / (np.linalg.norm(a) or 1.0); b = b / (np.linalg.norm(b) or 1.0)
        worst = min(worst, float(a @ b)); checked += 1
    print(f"stem consistency vs cached text_lc: checked={checked} worst_cos={worst:.5f}")
    if checked and worst < 0.999:
        sys.exit(f"!! stem embeds diverge from cache (worst cos {worst:.5f}) -- "
                 "encoding path differs, stage C would compare apples to oranges")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, ids=ids, slots=slots, vec=V,
                        types=np.array(types), text=np.array(ftext, dtype=object),
                        grounding=np.array(gnd, dtype=np.float32))
    print(f"wrote {args.out}: {V.shape}")


if __name__ == "__main__":
    main()
