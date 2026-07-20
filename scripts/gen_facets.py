#!/usr/bin/env python3
"""Path 3 stage A: decompose LVB question stems into visual-search facets.

The residual trace says one 768-d text vector reaches only ~3.3% of the query norm into
the image subspace (cos_orig maxes .233). LVB stems are compound -- scene description +
subtitle/temporal anchor + the actual interrogative -- so that single vector is an average
of several search intents, and the answer-bearing one is diluted.

This emits 1-4 facets per stem, each phrased as a CLIP-style *caption of what to look for*
rather than a question fragment (the LongCLIP text tower is trained on captions, so
"a small building with a triangular roof by a lake" retrieves; "what is the shape of the
roof" does not).

Reads the stem via harness.text.question_stem -- options NEVER enter (letting them in is
the 2026-07-15 query bug). Resumable: appends jsonl, skips ids already present.

Usage:
  set -a; . ./.env; set +a
  PYTHONPATH=. python3 scripts/gen_facets.py \
      --manifest data/manifest.lvb.long976.json --bins 600 3600 \
      --out results/facets/facets_long976.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from harness.text import question_stem

SYSTEM = """You decompose a video-QA question into visual search facets.

The question will be answered by looking at frames of a long video. A CLIP-style image-text
model must FIND the right frames using your facets, so each facet must read like a CAPTION
describing what is visible on screen -- not like a question.

Split the question into its distinct visual search intents. Typical structure:
  - scene      : the setting / background being described
  - subject    : the specific object or person the question is about
  - action     : an event or motion referenced
  - temporal   : a spoken line, caption, or moment used as an anchor
  - interrogative : the thing actually being asked about (what the answer depends on)

Rules:
- 1 to 4 facets. Use 1 only if the question truly has a single visual intent.
- Each facet is a short caption phrase, 3-15 words, describing visible content.
- Never phrase a facet as a question. No "what", "which", "how many" openers.
- The interrogative facet must describe the thing being asked about as a visible object,
  e.g. "the roof of the small lakeside building" -- NOT "the shape of the roof".
- Never invent an answer or guess. Never include answer options.
- GROUNDING (critical): every facet must be built from content that is actually stated in
  the question. Never invent scenes, objects, people, or details the question does not
  mention. If the question contains no visual description at all -- e.g. "Which of the
  following sequences of scenes is correct?" -- return EXACTLY ONE facet of type
  "interrogative" restating the question as a caption. Inventing plausible-sounding scenes
  is the worst possible failure here; an ungrounded facet points the retriever at random
  frames.
- Facets should be as DISSIMILAR to each other as the question allows -- they exist to
  point at different frames.

Return strict JSON only:
{"facets": [{"type": "scene|subject|action|temporal|interrogative", "text": "..."}]}"""


_STOP = set("a an the of in on at by with and or to is are was were there some this that "
            "it its his her their as for from what which how many when who whom whose "
            "video scene shown appears appearing moment".split())


def grounding(text, stem):
    """Fraction of a facet's content words that actually occur in the stem.

    Guards the failure seen in the 2026-07-20 smoke test: for a stem with no visual
    content ("Which of the following sequences of scenes is correct?") the model invented
    "a bustling city street", "a quiet park", "a busy cafe". Ungrounded facets point the
    retriever at random frames -- and because they DO move the picks, they would sail
    through the pick-overlap kill-switch while being pure noise. Scored here, gated in
    stage C.
    """
    sl = stem.lower()
    words = [w.strip(".,'\"?!():;") for w in text.lower().split()]
    words = [w for w in words if w and w not in _STOP and len(w) > 2]
    if not words:
        return 1.0
    return sum(w in sl for w in words) / len(words)


def build_client(base_url=None):
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set -- run: set -a; . ./.env; set +a")
    return OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))


def decompose(client, model, stem, retries=3):
    last = None
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": stem}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            facets = json.loads(r.choices[0].message.content).get("facets", [])
            facets = [f for f in facets
                      if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip()]
            if facets:
                return facets[:4], None
            last = "empty facets"
        except Exception as e:                      # noqa: BLE001 -- retry any API/parse error
            last = f"{type(e).__name__}: {e}"
    return None, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.long976.json")
    ap.add_argument("--bins", nargs="+", default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on first N stems")
    ap.add_argument("--out", default="results/facets/facets_long976.jsonl")
    args = ap.parse_args()

    items = json.load(open(args.manifest, encoding="utf-8"))
    if args.bins:
        keep = {str(b).rstrip("s") for b in args.bins}
        items = [it for it in items
                 if str(it.get("length_bin", it.get("duration_group", ""))).rstrip("s") in keep]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["id"])
        print(f"resume: {len(done)} already done")
    items = [it for it in items if it["id"] not in done]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} stems to decompose, model={args.model}")

    client = build_client(args.base_url)
    fh = open(args.out, "a", encoding="utf-8")
    fails = ungrounded = 0

    def work(it):
        stem = question_stem(it)
        facets, err = decompose(client, args.model, stem)
        return it, stem, facets, err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (it, stem, facets, err) in enumerate(ex.map(work, items), 1):
            if facets is None:
                fails += 1
                print(f"!! {it['id']}: {err}")
                continue
            for f in facets:
                f["grounding"] = round(grounding(f["text"], stem), 3)
            lo = min(f["grounding"] for f in facets)
            ungrounded += lo < 0.5
            fh.write(json.dumps({"id": it["id"], "bin": it.get("length_bin"),
                                 "stem": stem, "facets": facets,
                                 "min_grounding": lo}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                fh.flush()
                print(f"  {n}/{len(items)}  fails={fails}")
    fh.close()
    print(f"done. wrote {args.out}  fails={fails}  ungrounded(min_grounding<0.5)={ungrounded}")


if __name__ == "__main__":
    main()
