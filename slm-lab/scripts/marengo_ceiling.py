#!/usr/bin/env python3
"""Marengo (TwelveLabs) ceiling + positioning probe — Fork B optional diagnostic.

WHY (not part of the cheap-selector verdict — see FINDINGS "Scorer-swap"):
  The scorer-swap banked a *principled* negative: fine-needle localization at 1 fps is a
  per-frame retrieval task; cheap per-frame image-text scoring (SigLIP) is the right tool and
  its residual wall is *ranking capacity* at hour scale, not encoder choice. The one untested
  assumption is "a DENSE moment-retrieval head would localize a 1-2 s needle in an hour." This
  script tests exactly that with the purpose-built commercial retriever (Marengo), answering
  two things at once:
    1. CEILING  — can even a dedicated dense retriever hit the needle at 600 s / 3600 s?
    2. POSITIONING — the recurring "why not just use TwelveLabs?" question, answered once.
  It does NOT change the cheap-thesis verdict either way.

COST/CAVEAT: paid API. Indexing hour-videos runs minutes each. Default subset = 10 videos of
  the 3600 s bin (~600 min ≈ ~$30 depending on plan). Marengo is ~6 s-clip granular, so it is a
  better-but-not-perfect per-frame test; noted.

RUNS ANYWHERE (no GPU) — TwelveLabs does the compute in the cloud. Run it LOCAL so the pod can
  stop: scp the subset mp4s first (see --print-subset), then:
    export TWELVELABS_API_KEY=...          # never commit / print this
    pip install twelvelabs
    python scripts/marengo_ceiling.py --manifest data/manifest.lvb.25.json \
        --videos-dir data/videos --bins 3600s --n 10 --out results/marengo_ceiling.json

hit@k is made comparable to the SigLIP arms two ways:
  - strict  : segment MIDPOINT in a gold span (treats a returned clip like one selected frame)
  - lenient : returned segment [s,e] OVERLAPS a gold span (gives the dense retriever full credit)
We report both; lenient is the fair headline for a span-returning retriever.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict


def load_items(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("items", data)
    return data


def pick_subset(items, bins, n):
    want = set(bins)
    out = defaultdict(list)
    for it in items:
        b = it.get("length_bin")
        if b in want and len(out[b]) < n:
            # need a real video file + gold to score
            if it.get("video_file") and it.get("gold_evidence_seconds"):
                out[b].append(it)
    return [it for b in bins for it in out.get(b, [])]


def overlaps(seg, span):
    (s, e), (lo, hi) = seg, span
    return s <= hi and lo <= e


def midpoint_in(seg, span):
    m = (seg[0] + seg[1]) / 2.0
    return span[0] <= m <= span[1]


def hit_from_segments(segments_topk, gold_spans):
    """Return (lenient_hit, strict_hit) in {0.0,1.0} over the top-k returned segments."""
    lenient = any(overlaps(seg, sp) for seg in segments_topk for sp in gold_spans)
    strict = any(midpoint_in(seg, sp) for seg in segments_topk for sp in gold_spans)
    return (1.0 if lenient else 0.0), (1.0 if strict else 0.0)


# ---- TwelveLabs SDK glue (validated against SDK v1.2.8) ----

def _load_env_file(path=".env"):
    """Minimal .env loader (no dependency). Never prints values."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client():
    _load_env_file()
    # accept either name; the repo .env uses TWELVELABS_API
    key = os.environ.get("TWELVELABS_API_KEY") or os.environ.get("TWELVELABS_API")
    if not key:
        sys.exit("TWELVELABS_API(_KEY) not set (checked env + .env). Never commit/print it.")
    try:
        from twelvelabs import TwelveLabs
    except ImportError:
        sys.exit("pip install twelvelabs")
    import twelvelabs
    print(f"twelvelabs SDK {getattr(twelvelabs, '__version__', '?')}")
    return TwelveLabs(api_key=key)


def ensure_index(client, name, model):
    """Create (or reuse) a Marengo index with visual+audio search. Returns index id."""
    try:  # idempotent: reuse an index already named `name`
        for idx in client.indexes.list():
            if getattr(idx, "index_name", getattr(idx, "name", None)) == name:
                return idx.id
    except Exception:
        pass
    idx = client.indexes.create(
        index_name=name,
        models=[{"model_name": model, "model_options": ["visual", "audio"]}],
    )
    return idx.id


def index_video(client, index_id, path):
    """Upload + block until indexed; return the TwelveLabs video id."""
    # must pass a (filename, filehandle, mimetype) tuple; a bare path 400s ('video_file_broken')
    with open(path, "rb") as f:
        task = client.tasks.create(
            index_id=index_id,
            video_file=(os.path.basename(path), f, "video/mp4"),
        )
    done = client.tasks.wait_for_done(task_id=task.id, sleep_interval=5)
    if getattr(done, "status", "") != "ready":
        raise RuntimeError(f"indexing status={getattr(done,'status','?')} for {path}")
    return done.video_id


def search_video(client, index_id, video_id, query, k, scan_cap=200):
    """Text search the index; return THIS video's top-k [start,end] clips in rank order."""
    pager = client.search.query(
        index_id=index_id, search_options=["visual"], query_text=query, group_by="clip",
    )
    segs = []
    for n, item in enumerate(pager):
        if n >= scan_cap:
            break
        if getattr(item, "video_id", None) != video_id:
            continue
        s = float(getattr(item, "start", 0.0) or 0.0)
        e = float(getattr(item, "end", s) or s)
        segs.append((s, e))              # pager already yields best-rank first
        if len(segs) >= k:
            break
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.local.json")
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--bins", nargs="+", default=["3600s"])
    ap.add_argument("--n", type=int, default=10, help="videos per bin")
    ap.add_argument("--k", type=int, default=6, help="top-k segments (match SigLIP k)")
    ap.add_argument("--model", default="marengo3.0")  # valid: marengo3.0 (retrieval), pegasus1.2
    ap.add_argument("--index-name", default="slmlab-marengo-ceiling")
    ap.add_argument("--out", default="results/marengo_ceiling.json")
    ap.add_argument("--print-subset", action="store_true",
                    help="just print the chosen video files (to scp locally) and exit")
    args = ap.parse_args()

    items = load_items(args.manifest)
    subset = pick_subset(items, args.bins, args.n)
    if not subset:
        sys.exit(f"no items matched bins={args.bins} with video_file+gold")

    if args.print_subset:
        print(f"# {len(subset)} videos for bins={args.bins} n={args.n}:")
        for it in subset:
            print(os.path.join(args.videos_dir, it["video_file"]))
        return

    client = make_client()
    index_id = ensure_index(client, args.index_name, args.model)
    print(f"index {index_id} ({args.model})")

    agg = defaultdict(lambda: {"lenient": [], "strict": []})
    rows = []
    for i, it in enumerate(subset):
        vpath = os.path.join(args.videos_dir, it["video_file"])
        if not os.path.exists(vpath):
            print(f"[skip] missing {vpath}")
            continue
        b = it["length_bin"]
        try:
            vid = index_video(client, index_id, vpath)
            segs = search_video(client, index_id, vid, it["question"], args.k)
        except Exception as e:
            print(f"[err] {it['id']}: {type(e).__name__} {str(e)[:120]}")
            continue
        len_hit, str_hit = hit_from_segments(segs, it["gold_evidence_seconds"])
        agg[b]["lenient"].append(len_hit)
        agg[b]["strict"].append(str_hit)
        rows.append({"id": it["id"], "bin": b, "n_segments": len(segs),
                     "top_segments": segs, "gold": it["gold_evidence_seconds"],
                     "hit_lenient": len_hit, "hit_strict": str_hit})
        print(f"[{i+1}/{len(subset)}] {it['id']} {b} k={len(segs)} "
              f"lenient={len_hit} strict={str_hit}")

    table = {b: {"n": len(v["lenient"]),
                 "hit@k_lenient": round(sum(v["lenient"]) / len(v["lenient"]), 3) if v["lenient"] else None,
                 "hit@k_strict": round(sum(v["strict"]) / len(v["strict"]), 3) if v["strict"] else None}
             for b, v in agg.items()}
    out = {"k": args.k, "model": args.model, "bins": args.bins, "n_per_bin": args.n,
           "table": table, "per_item": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n=== Marengo ceiling hit@k (compare to so400m: 600s .36 / 3600s .24) ===")
    for b, r in table.items():
        print(f"{b:6} n={r['n']} lenient={r['hit@k_lenient']} strict={r['hit@k_strict']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
