#!/usr/bin/env python3
"""LVB-native task: MCQA accuracy under frame compression (the real thesis, not frame-recall).

LVB is scored on the ANSWER letter, never the frame. Our earlier frame-recall metric was a
hostile proxy: it counted a near-duplicate frame at cosine rank 55 as a MISS even though it
carries the same visual evidence the answerer needs. Here we measure the thing LVB actually
grades -- did GPT pick the correct option -- under four frame budgets:

  blind   : question + options only, NO frames        -> text-prior floor (D)
  uniform : k frames sampled uniformly, no selector   -> control (C)
  topk    : k frames = SigLIP flat top-k (the product)-> compressed (B)
  full    : F uniform frames (big budget)             -> ceiling (A)

Win: topk ~= full at k<<F (iso-accuracy under compression = COST win) AND topk > uniform
(selector earns its keep) AND full >> blind (task is genuinely visual). blind high => whole
visual thesis moot on LVB. Scores vs manifest gold_letter. Reuses scores.jsonl (Stage-1 cache)
and the same frame loader as gpt_probe.py so indices align.

Local answerer (Qwen3-VL via vLLM): the vLLM server speaks the OpenAI chat API, so point
this script at it -- set OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy,
pass --model Qwen/Qwen3-VL-8B-Instruct --effort none (skips reasoning_effort, greedy
temperature=0) and --workers 8 so vLLM can batch concurrent requests. See
scripts/pod_probe_k32.sh for the pod-side driver.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import string
import sys
from typing import List

LETTERS = string.ascii_uppercase


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


def uniform_k(n: int, k: int) -> List[int]:
    if k >= n:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def build_prompt(question: str, candidates: List[str], has_frames: bool) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(candidates))
    src = (f"You are shown still frames sampled from ONE video, each tagged with its timestamp. "
           if has_frames else
           "Answer from the question and options alone. ")
    return (f"{src}Question:\n\n{question}\n\nOptions:\n{opts}\n\n"
            "Answer with ONLY the single letter of the correct option. No other text.")


def ask(client, model, question, candidates, sel_frames, effort, max_side):
    prompt = build_prompt(question, candidates, bool(sel_frames))
    parts = [{"type": "text", "text": prompt}]
    for fr in sel_frames:
        parts.append({"type": "text", "text": f"[t={fr.seconds:.1f}s]"})
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{encode(fr, max_side)}"}})
    kwargs = dict(model=model,
                  messages=[{"role": "user", "content": parts}],
                  max_completion_tokens=2048)
    if effort == "none":
        # non-reasoning answerer (e.g. Qwen3-VL on vLLM): greedy decode, short answer
        kwargs["temperature"] = 0
        kwargs["max_completion_tokens"] = 16
    else:
        kwargs["reasoning_effort"] = effort
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def parse_letter(text: str, ncand: int) -> str:
    valid = set(LETTERS[:ncand])
    m = re.search(r"\b([A-Z])\b", text.strip().upper())
    if m and m.group(1) in valid:
        return m.group(1)
    for ch in text.strip().upper():
        if ch in valid:
            return ch
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--bin", default="600")
    ap.add_argument("--cond", required=True, choices=["blind", "uniform", "topk", "full", "pins"])
    ap.add_argument("--pins", default="results/posterior_killtest.json",
                    help="cond=pins: JSON from posterior_killtest.py; each question's "
                         "greedy_files (greedy-KL selected frame filenames) become the frames")
    ap.add_argument("--k", type=int, default=8, help="frames for uniform/topk")
    ap.add_argument("--full", type=int, default=32, help="frames for the full/ceiling budget")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low", choices=["none", "low", "medium", "high"],
                    help="none = non-reasoning local model (greedy, temperature=0)")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 sends concurrent requests; lets a vLLM server batch them")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (source .env; for vLLM use OPENAI_API_KEY=dummy)")
    client = OpenAI()

    manifest = {it["id"]: it for it in json.load(open(args.manifest))}
    rows = [json.loads(l) for l in open(args.scores) if l.strip()]
    rows = [r for r in rows if str(r["length_bin"]) == args.bin]
    rows = rows[: args.n]

    need_frames = args.cond != "blind"
    if need_frames:
        from harness.media import load_frames

    pins_by_id = {}
    if args.cond == "pins":
        pins = json.load(open(args.pins))
        pins_by_id = {rec["id"]: rec["greedy_files"] for rec in pins["records"]}

    def run_one(r):
        item = manifest[r["id"]]
        cands = item["candidates"]
        gold = item["gold_letter"]
        sel, idx, secs = [], [], []
        if need_frames:
            frames = load_frames(item, dump_fps=1.0, max_frames=3600)
            if len(frames) != len(r["scores"]):
                print(f"  !! frame/score mismatch {r['id']} {len(frames)}!={len(r['scores'])}, skip")
                return None
            if args.cond == "topk":
                idx = flat_top_k(r["scores"], args.k)
            elif args.cond == "uniform":
                idx = uniform_k(len(frames), args.k)
            elif args.cond == "pins":
                # greedy-KL pins: match by wall-clock second parsed from the pinned
                # filename (f<sec>.jpg) -- robust to any frame-list reordering, and the
                # extracted-images path sets Frame.seconds from the same filename.
                if r["id"] not in pins_by_id:
                    print(f"  !! no pins for {r['id']}, skip")
                    return None
                want_secs = {int(os.path.splitext(f)[0][1:]) for f in pins_by_id[r["id"]]}
                idx = [i for i, fr in enumerate(frames) if int(fr.seconds) in want_secs]
                if len(idx) != len(want_secs):
                    print(f"  !! pin mismatch {r['id']}: matched {len(idx)}/{len(want_secs)}, skip")
                    return None
            else:  # full
                idx = uniform_k(len(frames), args.full)
            sel = [frames[i] for i in idx]
            secs = [round(float(fr.seconds), 1) for fr in sel]
        try:
            text = ask(client, args.model, item["question"], cands, sel, args.effort, args.max_side)
        except Exception as e:
            print(f"  !! answerer error {r['id']}: {e}")
            return None
        pred = parse_letter(text, len(cands))
        return {"id": r["id"], "pred": pred, "gold": gold, "ok": pred == gold,
                "idx": idx, "secs": secs, "raw": text[:80]}

    records = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_one, rows))
    else:
        results = [run_one(r) for r in rows]
    for rec in results:
        if rec is None:
            continue
        records.append(rec)
        print(f"[{len(records)}] {rec['id']} pred={rec['pred']} gold={rec['gold']} "
              f"{'OK' if rec['ok'] else 'x'}")
    done = len(records)
    correct = sum(rec["ok"] for rec in records)

    acc = round(correct / done, 3) if done else None
    result = {"args": vars(args), "n": done, "accuracy": acc, "records": records}
    out = args.out or f"results/mcqa_{args.bin}_{args.cond}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print(f"\n== MCQA {args.cond} @{args.bin}s k={args.k} ==  accuracy={acc}  n={done}")
    print(f"wrote {out}")
    print(f"RESULT cond {args.cond} acc {acc} n {done}")


if __name__ == "__main__":
    main()
