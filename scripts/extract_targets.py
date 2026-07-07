#!/usr/bin/env python3
"""Decompose each LongVideoBench question into a GroundingDINO target phrase.

Output: data/targets.json = {id: {"target": <short noun phrase | "">,
                                  "has_concrete_target": bool,
                                  "question_category": <lvb code>}}

The `target` is the single most boxable visual anchor the question hinges on
(a person/object with visible attributes) — what an open-vocab detector could
localize to point at the evidence frame. `has_concrete_target=False` marks
questions with no stable boxable object (pure text-slide, audio/narration,
abstract sequence-ordering, scene-only) — grounding cannot help those, and
segmenting hit@6 by this flag IS the finding.
"""
import argparse
import json
import os
import re
import sys

from openai import OpenAI

SYS = """You convert a video multiple-choice QUESTION STEM into a target phrase for an \
open-vocabulary object detector (GroundingDINO), used to find WHICH FRAME of a long \
video holds the answer.

Return STRICT JSON: {"target": string, "has_concrete_target": boolean}

Rules:
- target = the SINGLE most boxable visual anchor the question hinges on: a person or \
physical object, with its distinguishing visible attributes if given. Keep it a short \
noun phrase a detector can localize, e.g. "man in a gray coat", "blue hat", "bowl of \
white powder", "airplane". Lowercase, no trailing period, <= 8 words.
- has_concrete_target=false (and target="") when the question has NO stable boxable \
physical object to detect: it is about on-screen TEXT/slides, audio/narration/speech, \
abstract SEQUENCE ORDERING with no single consistent object, or a pure SCENE/setting \
with no focal object. When in doubt about whether a detector could box it, choose false.
- Pick the anchor that best LOCALIZES the moment, even if the question then asks about \
an action/attribute of that anchor (the anchor is still boxable -> true).
- Ignore the answer options; use only the stem.

Output JSON only."""


def stem(q: str) -> str:
    # drop the options block
    s = re.split(r"\bOptions:\s*", q, maxsplit=1)[0]
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--out", default="data/targets.json")
    ap.add_argument("--model", default="gpt-5-mini")
    args = ap.parse_args()

    items = json.load(open(args.manifest))
    client = OpenAI()
    out = {}
    n = len(items)
    for idx, it in enumerate(items):
        qid = str(it["id"])
        s = stem(it.get("question", ""))
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": s}],
                response_format={"type": "json_object"},
            )
            rec = json.loads(resp.choices[0].message.content)
            target = (rec.get("target") or "").strip().lower().rstrip(".")
            hc = bool(rec.get("has_concrete_target")) and bool(target)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {qid} {type(e).__name__}: {e}", flush=True)
            target, hc = "", False
        out[qid] = {"target": target, "has_concrete_target": hc,
                    "question_category": it.get("question_category")}
        if idx % 25 == 0 or idx == n - 1:
            nc = sum(v["has_concrete_target"] for v in out.values())
            print(f"  [{idx+1}/{n}] concrete={nc}", flush=True)
    json.dump(out, open(args.out, "w"), indent=0)
    nc = sum(v["has_concrete_target"] for v in out.values())
    print(f"DONE {len(out)} targets, {nc} concrete ({nc/len(out):.0%}) -> {args.out}")


if __name__ == "__main__":
    main()
