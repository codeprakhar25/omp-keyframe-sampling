#!/usr/bin/env python3
"""Count freeze tags in scale train pack vs DATA_PIPELINE targets.

  .venv/bin/python scripts/tag_inventory.py
  .venv/bin/python scripts/tag_inventory.py --jsonl data/scale/train/train.jsonl
  .venv/bin/python scripts/tag_inventory.py --out data/scale/tag_inventory.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE = ROOT / "data" / "scale"
DEFAULT_JSONL = SCALE / "train" / "train.jsonl"

# From TAG_FREEZE.md / DATA_PIPELINE.md
CORE_EMO = ["angry", "sad", "excited", "fear", "surprise", "disgust"]
CORE_EVENT = [
    "pause",
    "thinking",
    "inhaling",
    "breathing",
    "sigh",
    "whispering",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
]
BONUS = ["laugh"]
ALL_TAGS = CORE_EMO + CORE_EVENT + BONUS + ["none"]

# Concrete fill targets (clips where tag appears as tag1 or tag2)
TARGET_EMO = 40
TARGET_EVENT = 25
TARGET_BONUS = 10  # optional


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def count_tags(rows: list[dict]) -> tuple[Counter, Counter, Counter]:
    """Return (any-slot counts, tag1-only, co-occurrence as frozenset counts)."""
    any_c: Counter = Counter()
    t1_c: Counter = Counter()
    pairs: Counter = Counter()
    for r in rows:
        t1 = (r.get("tag1") or "none").strip() or "none"
        t2 = (r.get("tag2") or "").strip()
        t1_c[t1] += 1
        tags = [t1]
        if t2:
            tags.append(t2)
        for t in tags:
            any_c[t] += 1
        if t2:
            pairs[tuple(sorted([t1, t2]))] += 1
        else:
            pairs[(t1,)] += 1
    return any_c, t1_c, pairs


def gap(have: int, need: int) -> int:
    return max(0, need - have)


def report(rows: list[dict], src: Path) -> dict:
    any_c, t1_c, pairs = count_tags(rows)
    n = len(rows)

    emo_rows = []
    for tag in CORE_EMO:
        h = any_c.get(tag, 0)
        emo_rows.append(
            {
                "tag": tag,
                "n": h,
                "target": TARGET_EMO,
                "gap": gap(h, TARGET_EMO),
                "ok": h >= TARGET_EMO,
            }
        )
    event_rows = []
    for tag in CORE_EVENT:
        h = any_c.get(tag, 0)
        event_rows.append(
            {
                "tag": tag,
                "n": h,
                "target": TARGET_EVENT,
                "gap": gap(h, TARGET_EVENT),
                "ok": h >= TARGET_EVENT,
            }
        )

    sources = Counter((r.get("source") or "?") for r in rows)
    unknown = sorted(t for t in any_c if t not in ALL_TAGS)

    doc = {
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "n_rows": n,
        "targets": {"emotion": TARGET_EMO, "event": TARGET_EVENT, "bonus_laugh": TARGET_BONUS},
        "emotion": emo_rows,
        "event": event_rows,
        "bonus": [
            {
                "tag": "laugh",
                "n": any_c.get("laugh", 0),
                "target": TARGET_BONUS,
                "gap": gap(any_c.get("laugh", 0), TARGET_BONUS),
                "ok": any_c.get("laugh", 0) >= TARGET_BONUS,
            }
        ],
        "none": any_c.get("none", 0),
        "tag1": dict(t1_c.most_common()),
        "any_slot": dict(any_c.most_common()),
        "top_combos": [
            {"tags": list(k), "n": v} for k, v in pairs.most_common(15)
        ],
        "by_source": dict(sources.most_common()),
        "unknown_tags": unknown,
        "summary": {
            "emo_ok": sum(1 for r in emo_rows if r["ok"]),
            "emo_total": len(CORE_EMO),
            "event_ok": sum(1 for r in event_rows if r["ok"]),
            "event_total": len(CORE_EVENT),
            "emo_gap_total": sum(r["gap"] for r in emo_rows),
            "event_gap_total": sum(r["gap"] for r in event_rows),
        },
    }
    return doc


def print_table(doc: dict) -> None:
    print(f"source: {doc['source']}  n={doc['n_rows']}")
    print(f"by_source: {doc['by_source']}")
    print()
    print(f"{'tag':<12} {'n':>5} {'target':>7} {'gap':>5} {'ok':>4}")
    print("-" * 40)
    print("## emotion")
    for r in doc["emotion"]:
        mark = "✓" if r["ok"] else "·"
        print(f"{r['tag']:<12} {r['n']:>5} {r['target']:>7} {r['gap']:>5} {mark:>4}")
    print("## event")
    for r in doc["event"]:
        mark = "✓" if r["ok"] else "·"
        print(f"{r['tag']:<12} {r['n']:>5} {r['target']:>7} {r['gap']:>5} {mark:>4}")
    print("## bonus")
    for r in doc["bonus"]:
        mark = "✓" if r["ok"] else "·"
        print(f"{r['tag']:<12} {r['n']:>5} {r['target']:>7} {r['gap']:>5} {mark:>4}")
    print(f"{'none':<12} {doc['none']:>5}")
    s = doc["summary"]
    print()
    print(
        f"fill: emotion {s['emo_ok']}/{s['emo_total']} tags at target "
        f"(need +{s['emo_gap_total']} clips); "
        f"event {s['event_ok']}/{s['event_total']} (need +{s['event_gap_total']})"
    )
    if doc["unknown_tags"]:
        print("UNKNOWN tags:", doc["unknown_tags"])
    print("\ntop combos:")
    for c in doc["top_combos"][:8]:
        print(f"  {c['n']:>4}  {'+'.join(c['tags'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path, default=SCALE / "tag_inventory.json")
    args = ap.parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"missing {args.jsonl}")
    rows = load_rows(args.jsonl)
    doc = report(rows, args.jsonl)
    print_table(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
