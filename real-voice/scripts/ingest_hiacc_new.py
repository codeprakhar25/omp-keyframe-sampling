#!/usr/bin/env python3
"""Ingest hiacc-new adult+child CS clips → scale drafts (GPT tags, all queued).

  .venv/bin/python scripts/ingest_hiacc_new.py
  .venv/bin/python scripts/ingest_hiacc_new.py --min-each 10 --no-audio   # dry path

Matches transcript by wav basename against:
  hiacc-new/code_switched_labels.json          (adult: audio)
  hiacc-new/hiacc-ch/code_switched_labels.json  (child: audio_filepath)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HIACC = ROOT / "hiacc-new"
AD_JSON = HIACC / "code_switched_labels.json"
CH_JSON = HIACC / "hiacc-ch" / "code_switched_labels.json"
SAMPLES = ROOT / "data" / "samples" / "scale"
DRAFTS = ROOT / "data" / "scale" / "drafts.json"


def load_map(path: Path, key: str) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for r in rows:
        raw = r.get(key) or ""
        name = Path(raw).name
        if name:
            out[name] = r
    return out


def duration_s(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def next_id(clips: list[dict]) -> str:
    n = 0
    for c in clips:
        cid = c.get("id") or ""
        if cid.startswith("sc") and cid[2:].isdigit():
            n = max(n, int(cid[2:]))
    return f"sc{n + 1:03d}" if n >= 100 else f"sc{n + 1}"


def pick_wavs(dir_path: Path, pattern: str, n: int) -> list[Path]:
    wavs = sorted(dir_path.glob(pattern))
    if len(wavs) < n:
        raise SystemExit(f"need ≥{n} {pattern} in {dir_path}, found {len(wavs)}")
    return wavs[: max(n, len(wavs))]  # take all if more than min


def make_caption(tag1: str, tag2: str, text: str) -> str:
    parts = []
    if tag1 and tag1 != "none":
        parts.append(f"[{tag1}]")
    if tag2 and tag2 not in ("", "none", tag1):
        parts.append(f"[{tag2}]")
    body = (text or "").strip()
    return (("".join(parts) + " " + body).strip() if parts else body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-each", type=int, default=10)
    ap.add_argument("--no-audio", action="store_true", help="skip GPT; tag1=none L")
    ap.add_argument("--provider", default="openai", choices=["auto", "openai", "gemini"])
    ap.add_argument("--out", type=Path, default=DRAFTS)
    args = ap.parse_args()

    ad_map = load_map(AD_JSON, "audio")
    ch_map = load_map(CH_JSON, "audio_filepath")

    adults = pick_wavs(HIACC, "AD*.wav", args.min_each)
    children = pick_wavs(HIACC / "hiacc-ch", "CH*.wav", args.min_each)
    print(f"adult={len(adults)} child={len(children)} (min_each={args.min_each})")

    if not args.no_audio:
        import audio_draft

        audio_draft.load_env()
        print("providers:", audio_draft.available_providers())

    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"clips": []}
    clips = list(data.get("clips") or [])
    existing_files = {c.get("file") for c in clips}
    existing_basenames = {Path(c.get("file") or "").name for c in clips}

    SAMPLES.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []

    batches = [
        ("adult", "hiacc_ad", adults, ad_map),
        ("child", "hiacc_ch", children, ch_map),
    ]

    for cohort, prefix, wavs, tmap in batches:
        for i, src in enumerate(wavs):
            meta = tmap.get(src.name)
            if not meta:
                print(f"WARN skip {src.name}: no transcript in json")
                continue
            text = (meta.get("transcription") or "").strip()
            dest_name = f"{prefix}_{src.stem}.wav"
            rel = f"scale/{dest_name}"
            if rel in existing_files or dest_name in existing_basenames:
                print(f"skip already in drafts: {rel}")
                continue

            dest = SAMPLES / dest_name
            shutil.copy2(src, dest)

            if args.no_audio:
                draft = {
                    "tag1": "none",
                    "tag2": "",
                    "confidence": "L",
                    "notes": "no-audio",
                    "provider": "none",
                    "model": "",
                }
            else:
                print(f"GPT {cohort} {src.name} …", flush=True)
                draft = audio_draft.draft_audio(dest, text, provider=args.provider)

            cid = next_id(clips + new_rows)
            t1, t2 = draft["tag1"], draft.get("tag2") or ""
            entry = {
                "id": cid,
                "source": "hiacc",
                "cohort": cohort,
                "dataset": "HiACC-new CS",
                "file": rel,
                "hf_or_path": f"samples/{rel}",
                "text": text,
                "native_label": meta.get("label") or "",
                "stem": src.stem,
                "duration_s": round(duration_s(dest), 3),
                "tag1": t1,
                "tag2": t2,
                "confidence": draft.get("confidence") or "M",
                "needs_review": True,
                "draft_source": "audio" if not args.no_audio else "placeholder",
                "reason": f"hiacc-new {cohort} GPT | {draft.get('notes') or ''}".strip(),
                "caption": make_caption(t1, t2, text),
                "status": "pending_review",
                "reviewed": False,
                "audio_draft": {
                    "tag1": t1,
                    "tag2": t2,
                    "confidence": draft.get("confidence"),
                    "notes": draft.get("notes"),
                    "provider": draft.get("provider"),
                    "model": draft.get("model"),
                },
            }
            new_rows.append(entry)
            print(
                f"  + {cid} {cohort} {src.name} → {t1}"
                + (f"+{t2}" if t2 else "")
                + f" conf={entry['confidence']}"
            )

    if not new_rows:
        print("nothing new to append")
        return

    all_clips = clips + new_rows
    queue = sum(1 for c in all_clips if c.get("needs_review") and not c.get("reviewed"))
    data["clips"] = all_clips
    data["n"] = len(all_clips)
    data.setdefault("hiacc_new_batch", {})
    data["hiacc_new_batch"] = {
        "added": len(new_rows),
        "adult": sum(1 for c in new_rows if c.get("cohort") == "adult"),
        "child": sum(1 for c in new_rows if c.get("cohort") == "child"),
        "queue_after": queue,
    }
    data.setdefault("stats", {})
    data["stats"]["queue"] = queue
    data["stats"]["pending_review"] = queue
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"appended {len(new_rows)} → {args.out}  queue={queue} total={len(all_clips)}")


if __name__ == "__main__":
    main()
