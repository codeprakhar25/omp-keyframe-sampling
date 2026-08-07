#!/usr/bin/env python3
"""Score smoke tags after human fill. Reads data/smoke/clips.json.

  .venv/bin/python scripts/score_smoke.py
  .venv/bin/python scripts/score_smoke.py --pass2-only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "smoke" / "clips.json"
META = ROOT / "data" / "smoke" / "meta.json"

# native Rasa style → frozen tag (for ear-match cross-check only)
RASA_MAP = {
    "HAPPY": "excited",
    "ANGER": "angry",
    "SAD": "sad",
    "SURPRISE": "surprise",
    "FEAR": "fear",
    "DISGUST": "disgust",
    "CONV": "none",  # often neutral-ish; not forced
    "BOOK": "none",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass2-only", action="store_true")
    args = ap.parse_args()

    data = json.loads(CLIPS.read_text(encoding="utf-8"))
    clips = data["clips"]
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    go = meta.get("go_nogo", "")

    tagged = [c for c in clips if (c.get("tag1") or "").strip()]
    print(f"clips={len(clips)} tagged={len(tagged)} done_flag={sum(1 for c in clips if c.get('done'))}")
    print(f"by source tagged: {Counter(c['source'] for c in tagged)}")
    print(f"tag1 hist: {Counter((c.get('tag1') or '').strip() or '∅' for c in clips)}")
    print(f"confidence: {Counter((c.get('confidence') or '').strip() or '∅' for c in tagged)}")

    # pass2 self-agree
    p2 = [
        c
        for c in clips
        if (c.get("tag1") or "").strip() and (c.get("pass2_tag1") or "").strip()
    ]
    if p2:
        agree = sum(1 for c in p2 if c["tag1"].strip() == c["pass2_tag1"].strip())
        rate = agree / len(p2)
        flag = "GO" if rate >= 0.70 else "NO-GO"
        print(f"pass2 self-agree: {agree}/{len(p2)} = {rate:.0%}  [{flag} ≥70%]")
    else:
        print("pass2 self-agree: no pass2_tag1 yet (do ~15 blind re-tags)")

    if args.pass2_only:
        return

    # Rasa ear-match field
    rasa = [c for c in clips if c.get("source") == "rasa" and (c.get("ear_match_native") or "").strip()]
    if rasa:
        y = sum(1 for c in rasa if c["ear_match_native"].strip().lower() == "y")
        n = sum(1 for c in rasa if c["ear_match_native"].strip().lower() == "n")
        na = sum(1 for c in rasa if c["ear_match_native"].strip().lower() == "na")
        denom = y + n
        rate = y / denom if denom else 0.0
        flag = "GO" if denom and rate >= 0.80 else ("NO-GO" if denom else "—")
        print(f"Rasa ear_match y/n: {y}/{denom} = {rate:.0%} (na={na})  [{flag} ≥80% on clear]")
    else:
        print("Rasa ear_match: empty")

    # optional: tag1 vs mapped native (emotions only)
    emo = [
        c
        for c in clips
        if c.get("source") == "rasa"
        and c.get("native_label") in RASA_MAP
        and RASA_MAP[c["native_label"]] != "none"
        and (c.get("tag1") or "").strip()
    ]
    if emo:
        hit = sum(1 for c in emo if c["tag1"].strip() == RASA_MAP[c["native_label"]])
        print(
            f"Rasa tag1 vs native map: {hit}/{len(emo)} = {hit/len(emo):.0%}  "
            f"(map={RASA_MAP})"
        )
        misses = [
            (c["id"], c["native_label"], c["tag1"], c.get("confidence"))
            for c in emo
            if c["tag1"].strip() != RASA_MAP[c["native_label"]]
        ]
        if misses:
            print("  misses:", misses)

    # IV event: native bucket vs tag1
    iv = [c for c in clips if c.get("source") == "iv" and (c.get("tag1") or "").strip()]
    if iv:
        print("IV tag1 vs native bucket:")
        for c in iv:
            bucket = (c.get("native_label") or "").split("|")[-1]
            print(f"  {c['id']} native={bucket:7} tag1={c['tag1']!r} conf={c.get('confidence')}")

    low = [c["id"] for c in tagged if (c.get("confidence") or "").upper() == "L"]
    if low:
        print(f"low-confidence ids ({len(low)}): {low}")

    print(f"\ngo_nogo rule: {go}")


if __name__ == "__main__":
    main()
