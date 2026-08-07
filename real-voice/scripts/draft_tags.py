#!/usr/bin/env python3
"""Rule-based tag drafts (+ optional OpenAI audio) for scale annotation.

Reads smoke clips or a scale catalog; writes data/scale/drafts.json.

  # rules only from smoke table
  .venv/bin/python scripts/draft_tags.py --from-smoke

  # rules + OpenAI audio on M/L or --audio-all
  .venv/bin/python scripts/draft_tags.py --from-smoke --audio

  # merge: accept H when rule+audio agree; else queue review
  .venv/bin/python scripts/draft_tags.py --from-smoke --audio --merge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CLIPS = ROOT / "data" / "smoke" / "clips.json"
SCALE = ROOT / "data" / "scale"
DRAFTS = SCALE / "drafts.json"
SAMPLES = ROOT / "data" / "samples"

RASA_MAP = {
    "HAPPY": "excited",
    "ANGER": "angry",
    "SAD": "sad",
    "SURPRISE": "surprise",
    "FEAR": "fear",
    "DISGUST": "disgust",
}

# IV event tag → frozen tag (priority order when scanning)
EVENT_PATTERNS = [
    (re.compile(r"<whispering>", re.I), "whispering"),
    (re.compile(r"<laughter>", re.I), "laugh"),
    (re.compile(r"<sigh>", re.I), "sigh"),
    (re.compile(r"<inhaling>", re.I), "inhaling"),
    (re.compile(r"<breathing>", re.I), "inhaling"),
    (re.compile(r"<(umm|uhh|hmm|uh-huh)>", re.I), "thinking"),
]

BREATH = {"inhaling", "sigh"}
CORE_EMOTION = set(RASA_MAP.values())


def caption(tag1: str, tag2: str, text: str) -> str:
    tags = [t for t in (tag1, tag2) if t and t != "none"]
    prefix = "".join(f"[{t}]" for t in tags[:2])
    # strip IV angle events from caption text
    clean = re.sub(r"<[^>]+>", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return f"{prefix} {clean}".strip() if prefix else clean


def rule_draft(row: dict) -> dict:
    """Draft from native_label / transcript events. Returns tag1, tag2, conf, reason."""
    src = row.get("source") or ""
    native = str(row.get("native_label") or "")
    text = str(row.get("text") or "")
    tag1, tag2, conf, reason = "none", "", "L", "fallback"

    if src == "rasa":
        style = native.split("|")[0] if native else native
        mapped = RASA_MAP.get(style)
        if mapped:
            return {
                "tag1": mapped,
                "tag2": "",
                "confidence": "H",
                "reason": f"rasa style={style}",
                "needs_review": False,
            }
        return {
            "tag1": "none",
            "tag2": "",
            "confidence": "M",
            "reason": f"rasa unknown style={style}",
            "needs_review": True,
        }

    if src == "iv":
        bucket = native.split("|")[-1].lower() if "|" in native else ""
        events = []
        for rx, tag in EVENT_PATTERNS:
            if rx.search(text) and tag not in events:
                events.append(tag)
        # prefer bucket if informative
        if bucket in ("whispering", "sigh", "inhaling", "thinking", "laughter"):
            bmap = {"laughter": "laugh"}.get(bucket, bucket)
            if bmap == "breathing":
                bmap = "inhaling"
            if bmap not in events:
                events.insert(0, bmap)
        if bucket == "breathing" and "inhaling" not in events:
            events.insert(0, "inhaling")

        if events:
            tag1 = events[0]
            tag2 = events[1] if len(events) > 1 else ""
            # event in transcript = H for primary; still review whisper/laugh
            conf = "H" if tag1 in ("inhaling", "sigh", "thinking") else "M"
            if tag1 in ("whispering", "laugh"):
                conf = "M"
            return {
                "tag1": tag1,
                "tag2": tag2 if tag2 != tag1 else "",
                "confidence": conf,
                "reason": f"iv events={events} bucket={bucket}",
                "needs_review": conf != "H",
            }
        if bucket in ("conv", "talk"):
            return {
                "tag1": "none",
                "tag2": "",
                "confidence": "M",
                "reason": "iv conversation — no event tag",
                "needs_review": True,
            }
        return {
            "tag1": "none",
            "tag2": "",
            "confidence": "L",
            "reason": "iv no events",
            "needs_review": True,
        }

    if src == "ivr":
        return {
            "tag1": "none",
            "tag2": "",
            "confidence": "M",
            "reason": "ivr extempore prior",
            "needs_review": True,
        }

    if src == "hiacc":
        return {
            "tag1": "none",
            "tag2": "",
            "confidence": "L",
            "reason": "hiacc — human label",
            "needs_review": True,
        }

    return {
        "tag1": tag1,
        "tag2": tag2,
        "confidence": conf,
        "reason": reason,
        "needs_review": True,
    }


def wav_path_for(row: dict) -> Path | None:
    f = row.get("file") or ""
    if not f:
        return None
    # smoke/foo.wav under samples/
    p = SAMPLES / f
    if p.exists():
        return p
    p2 = SAMPLES / "smoke" / Path(f).name
    if p2.exists():
        return p2
    return None


def agree(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    if a in BREATH and b in BREATH:
        return True
    return False


def merge_rule_audio(rule: dict, audio: dict | None) -> dict:
    if not audio:
        out = dict(rule)
        out["draft_source"] = "rule"
        out["needs_review"] = bool(rule.get("needs_review") or rule.get("confidence") != "H")
        return out

    r1, a1 = rule.get("tag1", "none"), audio.get("tag1", "none")
    if agree(r1, a1):
        tag1 = r1 if r1 != "none" else a1
        tag2 = rule.get("tag2") or audio.get("tag2") or ""
        return {
            "tag1": tag1,
            "tag2": tag2 if tag2 != tag1 else "",
            "confidence": "H",
            "reason": f"agree rule={r1} audio={a1}",
            "needs_review": False,
            "draft_source": "rule+audio",
            "audio": audio,
        }
    # disagree → keep rule primary, force review
    return {
        "tag1": r1,
        "tag2": rule.get("tag2") or "",
        "confidence": "M",
        "reason": f"DISAGREE rule={r1}/{rule.get('tag2')} audio={a1}/{audio.get('tag2')}",
        "needs_review": True,
        "draft_source": "conflict",
        "audio": audio,
        "audio_tag1": a1,
        "audio_tag2": audio.get("tag2") or "",
    }


def load_rows(args) -> list[dict]:
    if args.from_smoke:
        data = json.loads(SMOKE_CLIPS.read_text(encoding="utf-8"))
        return list(data["clips"])
    if args.catalog:
        data = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
        return list(data.get("clips") or data)
    raise SystemExit("need --from-smoke or --catalog")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft scale tags from rules (+ optional audio)")
    ap.add_argument("--from-smoke", action="store_true")
    ap.add_argument("--catalog", type=Path, help="JSON with clips[]")
    ap.add_argument("--audio", action="store_true", help="call OpenAI audio on review candidates")
    ap.add_argument("--audio-all", action="store_true", help="audio draft every clip (costly)")
    ap.add_argument("--merge", action="store_true", help="merge rule+audio confidences")
    ap.add_argument("--append", action="store_true", help="keep existing drafts.json rows; append new ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DRAFTS)
    ap.add_argument("--provider", default="openai", choices=["auto", "openai", "gemini"])
    args = ap.parse_args()

    rows = load_rows(args)
    if args.limit:
        rows = rows[: args.limit]

    existing = []
    existing_ids = set()
    if args.append and args.out.exists():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        existing = list(prev.get("clips") or [])
        existing_ids = {c.get("id") for c in existing}
        rows = [r for r in rows if r.get("id") not in existing_ids]
        print(f"append: keep {len(existing)} existing; draft {len(rows)} new", flush=True)

    audio_mod = None
    if args.audio or args.audio_all:
        sys.path.insert(0, str(ROOT / "scripts"))
        import audio_draft  # noqa: E402

        audio_mod = audio_draft
        audio_mod.load_env()
        print("audio providers:", audio_mod.available_providers() or ["(none)"], flush=True)

    drafts = []
    stats = Counter()
    for row in rows:
        rule = rule_draft(row)
        audio_out = None
        want_audio = args.audio_all or (
            args.audio and (rule.get("needs_review") or rule.get("confidence") != "H")
        )
        if want_audio and audio_mod:
            wp = wav_path_for(row)
            if wp:
                try:
                    audio_out = audio_mod.draft_audio(
                        wp, text=row.get("text") or "", provider=args.provider
                    )
                    stats["audio_ok"] += 1
                    print(f"  audio {row.get('id')} → {audio_out.get('tag1')}", flush=True)
                except Exception as e:
                    stats["audio_fail"] += 1
                    print(f"  audio FAIL {row.get('id')}: {e}", flush=True)
            else:
                stats["audio_no_wav"] += 1

        if args.merge or audio_out:
            merged = merge_rule_audio(rule, audio_out)
        else:
            merged = dict(rule)
            merged["draft_source"] = "rule"
            merged["needs_review"] = bool(
                rule.get("needs_review") or rule.get("confidence") != "H"
            )

        text = row.get("text") or ""
        entry = {
            "id": row.get("id"),
            "source": row.get("source"),
            "dataset": row.get("dataset"),
            "file": row.get("file"),
            "hf_or_path": row.get("hf_or_path") or (f"samples/{row['file']}" if row.get("file") else ""),
            "text": text,
            "native_label": row.get("native_label"),
            "duration_s": row.get("duration_s"),
            "tag1": merged["tag1"],
            "tag2": merged.get("tag2") or "",
            "confidence": merged["confidence"],
            "needs_review": merged.get("needs_review", True),
            "draft_source": merged.get("draft_source"),
            "reason": merged.get("reason"),
            "caption": caption(merged["tag1"], merged.get("tag2") or "", text),
            "status": "pending_review" if merged.get("needs_review") else "auto_accepted",
            "human_tag1": "",
            "human_tag2": "",
            "reviewed": False,
        }
        if merged.get("audio"):
            entry["audio_draft"] = {
                k: merged["audio"].get(k)
                for k in ("tag1", "tag2", "confidence", "notes", "provider", "model")
            }
        if merged.get("audio_tag1"):
            entry["audio_tag1"] = merged["audio_tag1"]
            entry["audio_tag2"] = merged.get("audio_tag2")
        drafts.append(entry)
        stats[entry["status"]] += 1
        stats[f"conf_{entry['confidence']}"] += 1

    SCALE.mkdir(parents=True, exist_ok=True)
    all_clips = existing + drafts if args.append else drafts
    # refresh aggregate stats
    agg = Counter()
    for d in all_clips:
        agg[d.get("status") or "?"] += 1
        agg[f"conf_{d.get('confidence') or '?'}"] += 1
        if d.get("needs_review") and not d.get("reviewed"):
            agg["queue"] += 1
    out = {
        "version": 1,
        "freeze": "TAG_FREEZE.md",
        "n": len(all_clips),
        "stats": dict(agg),
        "new_batch_stats": dict(stats),
        "clips": all_clips,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out, "n=", len(all_clips), "new_stats=", dict(stats), flush=True)
    review_n = sum(1 for d in all_clips if d.get("needs_review") and not d.get("reviewed"))
    print(f"review queue: {review_n}  total: {len(all_clips)}", flush=True)


if __name__ == "__main__":
    main()
