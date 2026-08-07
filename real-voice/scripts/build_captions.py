#!/usr/bin/env python3
"""Build captions.jsonl from reviewed scale drafts (+ auto_accepted H).

  .venv/bin/python scripts/build_captions.py
  → data/scale/captions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "data" / "scale" / "drafts.json"
OUT = ROOT / "data" / "scale" / "captions.jsonl"


def redact(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def make_caption(tag1: str, tag2: str, text: str) -> str:
    parts = []
    if tag1 and tag1 != "none":
        parts.append(f"[{tag1}]")
    if tag2 and tag2 not in ("", "none", tag1):
        parts.append(f"[{tag2}]")
    body = redact(text)
    return (("".join(parts) + " " + body).strip() if parts else body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=Path, default=DRAFTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--include-auto", action="store_true", default=True)
    args = ap.parse_args()

    data = json.loads(args.drafts.read_text(encoding="utf-8"))
    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in data.get("clips", []):
            if c.get("reviewed") and c.get("human_tag1"):
                t1, t2 = c["human_tag1"], c.get("human_tag2") or ""
            elif c.get("status") == "auto_accepted" and args.include_auto:
                t1, t2 = c.get("tag1") or "none", c.get("tag2") or ""
            elif c.get("final_tag1"):
                t1, t2 = c["final_tag1"], c.get("final_tag2") or ""
            else:
                continue
            cap = make_caption(t1, t2, c.get("text") or "")
            if not cap:
                continue
            row = {
                "id": c.get("id"),
                "source": c.get("source"),
                "audio": c.get("file"),
                "tag1": t1,
                "tag2": t2,
                "text": redact(c.get("text") or ""),
                "caption": cap,
                "native_label": c.get("native_label"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} captions → {args.out}")


if __name__ == "__main__":
    main()
