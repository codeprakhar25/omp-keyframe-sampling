#!/usr/bin/env python3
"""Stage-2 probe (retrieve-then-ground): does GPT-5.5 recover the rank headroom?

Stage-1 = plain flat top-K SigLIP union (peak-NMS was dominated, see FINDINGS). We hand GPT-5.5
the K timestamped union frames JOINTLY and ask for the 6 seconds most likely to hold the answer
evidence. Metric = GPT hit@6 (>=1 pick inside a gold span), read against:
  - FLOOR  = SigLIP flat top-6 direct (0.33 @600s) -- what Stage-1 alone gets.
  - CEIL   = gold-in-union @ flat top-K (0.66 @K=50) -- the most GPT could recover.
If GPT hit@6 > floor, the joint read is buying rank that cosine can't. If ~floor, thesis dead.

Reads results/scores/scores.jsonl (Stage-1 cache) + the manifest (for frame images). Frames are
loaded IDENTICALLY to dump_scores.py so score index i aligns to frame i. No GPU needed.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from typing import List

from harness.media import load_frames


def encode(frame, max_side=512, quality=85) -> str:
    img = frame.image
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def flat_top_k(scores: List[float], k: int) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


def any_hit(picks: List[float], spans, tol=0.5) -> bool:
    if not spans:
        return False
    return any(lo - tol <= t <= hi + tol for (lo, hi) in spans for t in picks)


def ask_gpt(client, model, question, sel_frames, effort, max_side):
    parts = [{"type": "text", "text":
              f"You are shown {len(sel_frames)} still frames sampled from ONE video, each tagged "
              f"with its timestamp in seconds. Question about the video:\n\n{question}\n\n"
              "Study the frames jointly. Return ONLY the timestamps (in seconds) of the 6 frames "
              "most likely to contain the visual evidence needed to answer, as a comma-separated "
              "list of numbers, highest-confidence first. No other text."}]
    for fr in sel_frames:
        parts.append({"type": "text", "text": f"[t={fr.seconds:.1f}s]"})
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{encode(fr, max_side)}"}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        max_completion_tokens=2048,
        reasoning_effort=effort,
    )
    return resp.choices[0].message.content or ""


def parse_ts(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--bin", default="600")
    ap.add_argument("--k", type=int, default=50, help="flat top-K union size = images/call")
    ap.add_argument("--n", type=int, default=30, help="subsample count")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--out", default="results/gpt_probe_600.json")
    args = ap.parse_args()

    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (source .env)")
    client = OpenAI()

    manifest = {it["id"]: it for it in json.load(open(args.manifest))}
    rows = [json.loads(l) for l in open(args.scores) if l.strip()]
    rows = [r for r in rows if str(r["length_bin"]) == args.bin and r.get("gold_evidence_seconds")]
    rows = rows[: args.n]

    floor_hits = gpt_hits = ceil_hits = done = 0
    records = []
    for r in rows:
        item = manifest[r["id"]]
        scores, times = r["scores"], r["times"]
        spans = r["gold_evidence_seconds"]
        idx = flat_top_k(scores, args.k)
        union_times = [times[i] for i in idx]
        floor_idx = flat_top_k(scores, 6)
        floor_hit = any_hit([times[i] for i in floor_idx], spans)
        ceil_hit = any_hit(union_times, spans)

        frames = load_frames(item, dump_fps=1.0, max_frames=3600)
        if len(frames) != len(scores):
            print(f"  !! frame/score mismatch {r['id']} {len(frames)}!={len(scores)}, skip")
            continue
        sel = [frames[i] for i in idx]
        try:
            text = ask_gpt(client, args.model, item["question"], sel, args.effort, args.max_side)
        except Exception as e:
            print(f"  !! GPT error {r['id']}: {e}")
            continue
        picks = parse_ts(text)[:6]
        # snap each pick to nearest union timestamp (GPT may echo a rounded value)
        snapped = [min(union_times, key=lambda t: abs(t - p)) for p in picks] if picks else []
        gpt_hit = any_hit(snapped, spans)

        done += 1
        floor_hits += floor_hit
        ceil_hits += ceil_hit
        gpt_hits += gpt_hit
        records.append({"id": r["id"], "nf": len(scores), "picks": picks,
                        "gpt_hit": gpt_hit, "floor_hit": floor_hit, "ceil_hit": ceil_hit,
                        "raw": text[:200]})
        print(f"[{done}] {r['id']} nf={len(scores)} picks={picks} "
              f"gpt={'H' if gpt_hit else '.'} floor={'H' if floor_hit else '.'} "
              f"ceil={'H' if ceil_hit else '.'}")

    result = {
        "args": vars(args), "n": done,
        "gpt_hit_at_6": round(gpt_hits / done, 3) if done else None,
        "floor_top6": round(floor_hits / done, 3) if done else None,
        "ceiling_union": round(ceil_hits / done, 3) if done else None,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\n== GPT-5.5 probe @{args.bin}s, K={args.k}, n={done} ==")
    print(f"  floor (SigLIP top-6):   {result['floor_top6']}")
    print(f"  GPT-5.5 hit@6:          {result['gpt_hit_at_6']}")
    print(f"  ceiling (gold-in-union):{result['ceiling_union']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
