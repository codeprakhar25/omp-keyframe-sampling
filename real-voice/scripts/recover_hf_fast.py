#!/usr/bin/env python3
"""Fast HF recover: local cache only + write-through after each hit.

  .venv/bin/python scripts/recover_hf_fast.py --sources rasa,iv
  # skips IVR (use ASR for the 14 early clips)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import scan_cache_dir

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples" / "scale"
DRAFTS = ROOT / "data" / "scale" / "drafts.json"
CATALOG = ROOT / "data" / "scale" / "pull_catalog.json"


def pcm_hash(wav) -> str:
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    return hashlib.sha1((np.clip(wav, -1, 1) * 32767.0).astype(np.int16).tobytes()).hexdigest()


def wav_info(path: Path) -> tuple[str, float]:
    wav, sr = sf.read(str(path), always_2d=False)
    return pcm_hash(wav), float(len(wav) / sr)


def bytes_info(ab: bytes) -> tuple[str, float]:
    wav, sr = sf.read(io.BytesIO(ab), always_2d=False)
    return pcm_hash(wav), float(len(wav) / (sr or 1))


def audio_bytes(obj) -> bytes | None:
    return obj.get("bytes") if isinstance(obj, dict) else None


def cached_parquets(repo_id: str, subglob: str) -> list[Path]:
    """Find parquet files already on disk for a dataset repo."""
    out: list[Path] = []
    try:
        cache = scan_cache_dir()
    except Exception:
        cache = None
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    # datasets--org--name
    slug = "datasets--" + repo_id.replace("/", "--")
    root = hub / slug
    if root.exists():
        out = sorted(root.rglob(subglob))
    # prefer snapshot files
    snaps = [p for p in out if "/snapshots/" in str(p)]
    return snaps or out


def resolve_wav(file_field: str) -> Path | None:
    name = Path(file_field).name
    for p in (SAMPLES / name, ROOT / "data" / "samples" / file_field):
        if p.exists():
            return p
    return None


def save_drafts(drafts: dict) -> None:
    DRAFTS.write_text(json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_hit(clip: dict, path: Path, text: str, drafts: dict) -> None:
    clip["text"] = text
    clip["text_source"] = "hf_cache"
    side = Path(str(path) + ".json")
    meta = json.loads(side.read_text()) if side.exists() else {"file": clip.get("file")}
    meta["text"] = text
    meta["text_source"] = "hf_cache"
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_drafts(drafts)


def recover_rasa(targets: list[dict], drafts: dict) -> int:
    need = {t["hash"]: t for t in targets}
    filled = 0
    files = cached_parquets("ai4bharat/Rasa", "Hindi/train-*.parquet")
    print(f"Rasa cache parquets={len(files)} need={len(need)}", flush=True)
    styles = {t["style"] for t in targets}
    for fpath in files:
        if not need:
            break
        print(f"  {fpath.name} need={len(need)}", flush=True)
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.num_row_groups):
            if not need:
                break
            table = pf.read_row_group(rg, columns=["text", "audio", "style", "duration"])
            for i in range(len(table)):
                style = str(table.column("style")[i].as_py() or "")
                if style not in styles:
                    continue
                dur = float(table.column("duration")[i].as_py() or 0)
                if not any(
                    t["style"] == style and abs(t["dur"] - dur) < 0.08 for t in need.values()
                ):
                    continue
                ab = audio_bytes(table.column("audio")[i].as_py())
                if not ab:
                    continue
                try:
                    h, _ = bytes_info(ab)
                except Exception:
                    continue
                if h not in need:
                    continue
                text = str(table.column("text")[i].as_py() or "")
                if not text:
                    continue
                t = need.pop(h)
                write_hit(t["clip"], t["path"], text, drafts)
                filled += 1
                print(f"    + {t['clip']['id']}", flush=True)
    return filled


def recover_iv(targets: list[dict], drafts: dict) -> int:
    need = {t["hash"]: t for t in targets}
    filled = 0
    files = cached_parquets("ai4bharat/IndicVoices", "**/hindi/train-*.parquet")
    if not files:
        files = cached_parquets("ai4bharat/IndicVoices", "train-*.parquet")
        files = [p for p in files if "hindi" in str(p).lower()]
    print(f"IV cache parquets={len(files)} need={len(need)}", flush=True)
    for fpath in files:
        if not need:
            break
        print(f"  {fpath.name} need={len(need)}", flush=True)
        pf = pq.ParquetFile(fpath)
        names = set(pf.schema_arrow.names)
        if "unsanitized_verbatim" not in names:
            continue
        meta = pf.read(columns=["unsanitized_verbatim", "duration"])
        hit_idx = set()
        for i in range(len(meta)):
            dur = float(meta.column("duration")[i].as_py() or 0)
            if dur > 1000:
                dur /= 16000.0
            if any(abs(t["dur"] - dur) < 0.08 for t in need.values()):
                hit_idx.add(i)
        if not hit_idx:
            continue
        offset = 0
        audio_col = "audio_filepath" if "audio_filepath" in names else ("audio" if "audio" in names else None)
        if not audio_col:
            continue
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["unsanitized_verbatim", audio_col])
            for j in range(len(table)):
                i = offset + j
                if i not in hit_idx:
                    continue
                text = str(table.column("unsanitized_verbatim")[j].as_py() or "")
                ab = audio_bytes(table.column(audio_col)[j].as_py())
                if not ab or not text:
                    continue
                try:
                    h, _ = bytes_info(ab)
                except Exception:
                    continue
                if h not in need:
                    continue
                t = need.pop(h)
                write_hit(t["clip"], t["path"], text, drafts)
                filled += 1
                print(f"    + {t['clip']['id']}", flush=True)
                if not need:
                    break
            offset += len(table)
            if not need:
                break
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="rasa,iv", help="comma: rasa,iv,ivr")
    args = ap.parse_args()
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))
    clips = drafts["clips"]

    # pull text from existing sidecars first
    side_n = 0
    for c in clips:
        if (c.get("text") or "").strip():
            continue
        p = resolve_wav(c["file"])
        if not p:
            continue
        side = Path(str(p) + ".json")
        if side.exists():
            t = (json.loads(side.read_text()).get("text") or "").strip()
            if t:
                c["text"] = t
                c["text_source"] = "sidecar"
                side_n += 1
    if side_n:
        save_drafts(drafts)
    print(f"sidecar filled {side_n}", flush=True)

    targets = {"rasa": [], "iv": [], "ivr": []}
    for c in clips:
        if (c.get("text") or "").strip():
            continue
        src = c.get("source")
        if src not in sources or src not in targets:
            continue
        p = resolve_wav(c["file"])
        if not p:
            continue
        h, dur = wav_info(p)
        style = ""
        if src == "rasa":
            style = p.stem.replace("scale_rasa_", "").rsplit("_", 1)[0].upper()
        targets[src].append({"clip": c, "path": p, "hash": h, "dur": dur, "style": style})

    filled = 0
    if "rasa" in sources and targets["rasa"]:
        filled += recover_rasa(targets["rasa"], drafts)
    if "iv" in sources and targets["iv"]:
        filled += recover_iv(targets["iv"], drafts)
    # ivr intentionally skipped in fast path

    still = sum(1 for c in clips if not (c.get("text") or "").strip())
    if CATALOG.exists():
        cat = json.loads(CATALOG.read_text(encoding="utf-8"))
        by_file = {c["file"]: c for c in clips}
        for row in cat.get("clips") or []:
            d = by_file.get(row.get("file"))
            if d and (d.get("text") or "").strip():
                row["text"] = d["text"]
        CATALOG.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"DONE filled={filled} still_empty={still}", flush=True)


if __name__ == "__main__":
    main()
