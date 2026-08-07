#!/usr/bin/env python3
"""Pull ONE HF dataset parquet shard → N short wavs + catalog row.

Examples:
  .venv/bin/python scripts/pull_shard.py \\
      --repo ai4bharat/Rasa --path Hindi/train-00000-of-00025.parquet --n 5

  .venv/bin/python scripts/pull_shard.py \\
      --repo ai4bharat/IndicVoices --path hindi/train-00000-of-00082.parquet \\
      --n 3 --tag iv_hi --stats-only

Token: HF_TOKEN in env or real-voice/.env
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "samples"


def load_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip().strip('"')
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("No HF_TOKEN in env or .env")


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s.strip())
    return s.strip("_")[:40] or "clip"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sip one HF parquet shard → wav samples")
    ap.add_argument("--repo", required=True, help="e.g. ai4bharat/Rasa")
    ap.add_argument("--path", required=True, help="parquet path inside repo")
    ap.add_argument("--n", type=int, default=5, help="max wavs to write")
    ap.add_argument("--tag", default="", help="filename prefix (default from repo)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--stats-only", action="store_true", help="print duration/style counts, no wavs")
    ap.add_argument("--list-files", action="store_true", help="list parquet paths matching --prefix")
    ap.add_argument("--prefix", default="", help="with --list-files, filter paths")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    token = load_token()

    if args.list_files:
        files = list_repo_files(args.repo, repo_type="dataset", token=token)
        pref = args.prefix or args.path
        for f in sorted(files):
            if f.endswith(".parquet") and (not pref or f.startswith(pref) or pref in f):
                print(f)
        return

    print(f"download {args.repo} :: {args.path}", flush=True)
    local = hf_hub_download(args.repo, args.path, repo_type="dataset", token=token)
    pf = pq.ParquetFile(local)
    names = pf.schema_arrow.names
    print("schema:", names, flush=True)

    meta_cols = [c for c in ("duration", "style", "scenario", "emotion", "gender", "text", "transcript", "verbatim") if c in names]
    styles: Counter[str] = Counter()
    total_dur = 0.0
    nutt = 0
    if meta_cols:
        t = pf.read(columns=meta_cols)
        for i in range(len(t)):
            if "duration" in meta_cols:
                d = t.column("duration")[i].as_py()
                if d is not None:
                    d = float(d)
                    if d > 1000:
                        d /= 16000.0
                    total_dur += d
                    nutt += 1
            for key in ("style", "scenario", "emotion"):
                if key in meta_cols:
                    styles[str(t.column(key)[i].as_py())] += 1
                    break
    print(f"shard utt={nutt} hours={total_dur/3600:.3f}", flush=True)
    if styles:
        print("top labels:", styles.most_common(15), flush=True)

    if args.stats_only:
        return

    # Rasa/IV-R: "audio"; IndicVoices: "audio_filepath" (HF audio struct)
    audio_col = next((c for c in ("audio", "audio_filepath") if c in names), None)
    if not audio_col:
        raise SystemExit("no audio / audio_filepath column in this parquet")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or slug(args.repo.split("/")[-1])
    catalog_path = args.out / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else []
    catalog = [c for c in catalog if not str(c.get("file", "")).startswith(f"{tag}_")]

    # prefer diverse styles across first row groups
    written = 0
    seen_styles: set[str] = set()
    text_keys = (
        "unsanitized_verbatim",
        "text",
        "transcript",
        "verbatim",
        "normalized",
        "sentence",
    )

    for rg in range(pf.num_row_groups):
        if written >= args.n:
            break
        table = pf.read_row_group(rg)
        for i in range(len(table)):
            if written >= args.n:
                break
            row = {c: table.column(c)[i].as_py() for c in table.column_names}
            audio = row.get(audio_col)
            if not isinstance(audio, dict) or not audio.get("bytes"):
                continue
            style = str(row.get("style") or row.get("scenario") or row.get("emotion") or "")
            # diversify by style only when real labels exist
            if style and style in seen_styles and len(seen_styles) < min(args.n, 6):
                continue
            text = ""
            for k in text_keys:
                if row.get(k):
                    text = str(row[k])
                    break
            wav, sr = sf.read(io.BytesIO(audio["bytes"]))
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(axis=1)
            max_len = int(sr * args.max_sec)
            wav = wav[:max_len]
            dur = len(wav) / float(sr)
            if dur < 0.4:
                continue
            fname = f"{tag}_{written:02d}.wav"
            sf.write(args.out / fname, wav, sr)
            entry = {
                "file": fname,
                "dataset": args.repo,
                "shard": args.path,
                "style": style or "na",
                "text": text[:400],
                "duration_s": round(dur, 2),
                "sr": int(sr),
            }
            for k in ("gender", "speaker_id", "speaker", "ratio_english_words", "ratio_hindi_words"):
                if row.get(k) is not None:
                    entry[k] = str(row[k])[:80]
            catalog.append(entry)
            if style:
                seen_styles.add(style)
            written += 1
            print(f"wrote {fname} [{entry['style']}] {text[:70]}", flush=True)

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    print(f"done: {written} wavs → {args.out} (catalog.json updated)", flush=True)


if __name__ == "__main__":
    main()
