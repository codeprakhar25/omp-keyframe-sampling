#!/usr/bin/env python3
"""QVHighlights peak/plateau confirmation gate — does frozen cosine separate the moment?

The question (reframed for QVH's broad moments, median 30 s / 15 clips / 0.20 of video): a similarity
selector won't show a sharp peak, it should show an ELEVATED PLATEAU across the gold window vs
background. We measure separability, not sharp-peak detection. No training, frozen encoder.

Per query: score each 2-s clip by cosine(query_emb, clip_emb), then vs relevant_clip_ids:
  clip_AP        average precision ranking gold clips by cosine (THE number)
  plateau_gap    mean(cos|gold) - mean(cos|background)  (raw separability)
  recall@k       |topk ∩ gold| / |gold|          (fat target -> high; report, not decisive)
  anyhit@k       >=1 gold clip in top-k
  span_R1@0.5    top contiguous run as predicted window, tIoU vs relevant_windows >= 0.5
Baseline = shuffled scores (random) -> AP floor ~ |gold|/n_clips. Selector must beat that.

Embeddings: pluggable. --feat-dir holds per-vid clip embeddings; --backend picks the loader.
  moment_detr : {vid}.npz with key 'features' [n_clips, D] (CLIP ViT-B/32, 2-s clips) +
                query embeddings from --qfeat-dir/{qid}.npz key 'last_hidden_state' mean-pooled,
                OR pass --clip-text to re-embed queries with open_clip (needs the pkg).
  npy         : {vid}.npy [n_clips, D] video, {qid}.npy [D] query under --qfeat-dir.
Wire the loader to whatever we dump (SigLIP so400m on frames is the eventual apples-to-apples backend;
CLIP-B/32 features are the cheap first pass). Metric code is backend-agnostic.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np


def cosine_scores(q: np.ndarray, V: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-8)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    return V @ q


def average_precision(scores: np.ndarray, gold_mask: np.ndarray) -> float:
    """AP: rank clips by score desc, integrate precision at each gold hit."""
    order = np.argsort(-scores)
    g = gold_mask[order]
    if g.sum() == 0:
        return float("nan")
    tp = np.cumsum(g)
    prec = tp / (np.arange(len(g)) + 1)
    return float((prec * g).sum() / g.sum())


def top_contiguous_window(scores: np.ndarray, width: int) -> tuple:
    """Best-scoring contiguous run of `width` clips (cheap span proxy, no boundary model)."""
    if width >= len(scores):
        return 0, len(scores)
    csum = np.concatenate([[0], np.cumsum(scores)])
    best_s, best_i = -1e9, 0
    for i in range(len(scores) - width + 1):
        s = csum[i + width] - csum[i]
        if s > best_s:
            best_s, best_i = s, i
    return best_i, best_i + width


def tiou(a: tuple, b: tuple) -> float:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_feats(backend: str, feat_dir: str, qfeat_dir: str, vid: str, qid: int):
    if backend == "npy":
        V = np.load(os.path.join(feat_dir, f"{vid}.npy"))
        q = np.load(os.path.join(qfeat_dir, f"{qid}.npy"))
        return V, q
    if backend == "moment_detr":
        V = np.load(os.path.join(feat_dir, f"{vid}.npz"))["features"]
        qd = np.load(os.path.join(qfeat_dir, f"{qid}.npz"))
        key = "last_hidden_state" if "last_hidden_state" in qd else qd.files[0]
        q = qd[key]
        if q.ndim == 2:
            q = q.mean(0)
        return V, q
    raise ValueError(backend)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/qvh/highlight_val_release.jsonl")
    ap.add_argument("--feat-dir", required=True, help="per-vid clip embeddings")
    ap.add_argument("--qfeat-dir", required=True, help="per-qid query embeddings")
    ap.add_argument("--backend", default="moment_detr", choices=["moment_detr", "npy"])
    ap.add_argument("--clip-sec", type=float, default=2.0)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--n", type=int, default=0, help="0=all")
    ap.add_argument("--out", default="results/qvh_peak_check.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    rows = [json.loads(l) for l in open(args.ann) if l.strip()]
    if args.n:
        rows = rows[: args.n]

    agg = {"clip_AP": [], "clip_AP_rand": [], "plateau_gap": [],
           **{f"recall@{k}": [] for k in ks}, **{f"anyhit@{k}": [] for k in ks},
           "span_R1@0.5": []}
    done = skip = 0
    rng = np.random.default_rng(0)
    for r in rows:
        vid, qid = r["vid"], r["qid"]
        try:
            V, q = load_feats(args.backend, args.feat_dir, args.qfeat_dir, vid, qid)
        except (FileNotFoundError, KeyError):
            skip += 1
            continue
        n = len(V)
        scores = cosine_scores(q, V)
        gold = np.zeros(n, dtype=float)
        for cid in r["relevant_clip_ids"]:
            if cid < n:
                gold[cid] = 1.0
        if gold.sum() == 0:
            skip += 1
            continue
        done += 1
        agg["clip_AP"].append(average_precision(scores, gold))
        agg["clip_AP_rand"].append(average_precision(rng.random(n), gold))
        agg["plateau_gap"].append(float(scores[gold == 1].mean() - scores[gold == 0].mean())
                                  if (gold == 0).any() else float("nan"))
        order = np.argsort(-scores)
        for k in ks:
            topk = set(order[:k].tolist())
            gset = set(np.where(gold == 1)[0].tolist())
            agg[f"recall@{k}"].append(len(topk & gset) / len(gset))
            agg[f"anyhit@{k}"].append(1.0 if topk & gset else 0.0)
        gw = int(round(np.mean([e - b for b, e in r["relevant_windows"]]) / args.clip_sec))
        gw = max(1, gw)
        b, e = top_contiguous_window(scores, gw)
        pred = (b * args.clip_sec, e * args.clip_sec)
        best = max(tiou(pred, (wb, we)) for wb, we in r["relevant_windows"])
        agg["span_R1@0.5"].append(1.0 if best >= 0.5 else 0.0)

    def m(x):
        x = [v for v in x if v == v]
        return round(float(np.mean(x)), 4) if x else None
    result = {"n": done, "skipped": skip, "backend": args.backend,
              "metrics": {k: m(v) for k, v in agg.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nclip_AP {result['metrics']['clip_AP']} vs random {result['metrics']['clip_AP_rand']} "
          f"| plateau_gap {result['metrics']['plateau_gap']} | span_R1@0.5 {result['metrics']['span_R1@0.5']}")


if __name__ == "__main__":
    main()
