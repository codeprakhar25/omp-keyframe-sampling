#!/usr/bin/env python3
"""Pack reviewed captions + wavs into a flat train folder.

  .venv/bin/python scripts/pack_train.py
  → data/scale/train/{wavs,manifest.jsonl,README.md}
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPS = ROOT / "data" / "scale" / "captions.jsonl"
OUT = ROOT / "data" / "scale" / "train"


def resolve_wav(audio: str) -> Path | None:
    """audio is often 'smoke/...' or 'scale/...' relative to data/samples or data."""
    candidates = [
        ROOT / "data" / "samples" / audio,
        ROOT / "data" / audio,
        ROOT / "data" / "samples" / Path(audio).name,
        ROOT / "data" / "smoke" / Path(audio).name,
        ROOT / "data" / "samples" / "scale" / Path(audio).name,
        ROOT / "data" / "samples" / "smoke" / Path(audio).name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=Path, default=CAPS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--copy", action="store_true", help="copy wavs (default: symlink)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.captions.read_text(encoding="utf-8").splitlines() if l.strip()]
    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    man_path = args.out / "manifest.jsonl"

    missing = []
    written = 0
    with man_path.open("w", encoding="utf-8") as mf:
        for r in rows:
            src = resolve_wav(r["audio"])
            if src is None:
                missing.append(r["id"] + ":" + r["audio"])
                continue
            dest_name = f"{r['id']}.wav"
            dest = wav_dir / dest_name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if args.copy:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src.resolve())
            out = {
                "id": r["id"],
                "audio": f"wavs/{dest_name}",
                "text": r.get("text") or "",
                "caption": r["caption"],
                "tag1": r.get("tag1"),
                "tag2": r.get("tag2") or "",
                "source": r.get("source"),
                "native_label": r.get("native_label"),
            }
            mf.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    tags = Counter()
    for r in rows:
        tags[r.get("tag1") or "none"] += 1
    readme = args.out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Train pack (expressive Hindi TTS)",
                "",
                f"- rows: {written}",
                f"- missing wav: {len(missing)}",
                f"- tag1: {dict(tags.most_common())}",
                "",
                "Fields in `manifest.jsonl`:",
                "- `caption` — tagged text for conditioning (`[angry] ...`)",
                "- `text` — plain transcript (no tags)",
                "- `audio` — relative path under this folder",
                "",
                "**Not a Fireworks SFT dataset.** Fireworks managed fine-tune is chat-LLM only.",
                "Use this pack on a TTS trainer (RunPod / local / Modal).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {written} → {args.out}")
    if missing:
        print(f"MISSING {len(missing)}: {missing[:8]}")


if __name__ == "__main__":
    main()
