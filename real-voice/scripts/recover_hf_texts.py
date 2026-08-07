#!/usr/bin/env python3
"""Recover missing transcripts from HF parquet (duration+style filter → audio hash).

  .venv/bin/python scripts/recover_hf_texts.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples" / "scale"
DRAFTS = ROOT / "data" / "scale" / "drafts.json"
CATALOG = ROOT / "data" / "scale" / "pull_catalog.json"


def load_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip().strip('"')
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no HF_TOKEN")


def pcm_hash(wav, sr: int) -> str:
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    pcm = (np.clip(wav, -1, 1) * 32767.0).astype(np.int16).tobytes()
    return hashlib.sha1(pcm).hexdigest()


def wav_info(path: Path) -> tuple[str, float]:
    wav, sr = sf.read(str(path), always_2d=False)
    dur = float(len(wav) / sr)
    return pcm_hash(wav, sr), dur


def bytes_info(ab: bytes) -> tuple[str, float]:
    wav, sr = sf.read(io.BytesIO(ab), always_2d=False)
    dur = float(len(wav) / (sr or 1))
    return pcm_hash(wav, sr), dur


def audio_bytes(obj) -> bytes | None:
    if isinstance(obj, dict) and obj.get("bytes"):
        return obj["bytes"]
    return None


def resolve_wav(file_field: str) -> Path | None:
    name = Path(file_field).name
    for p in (SAMPLES / name, ROOT / "data" / "samples" / file_field):
        if p.exists():
            return p
    return None


def recover_rasa(token: str, targets: list[dict]) -> int:
    """targets: {id, path, hash, dur, style}"""
    need = {t["hash"]: t for t in targets}
    filled = 0
    files = sorted(
        f
        for f in list_repo_files("ai4bharat/Rasa", repo_type="dataset", token=token)
        if f.startswith("Hindi/train") and f.endswith(".parquet")
    )
    styles = {t["style"] for t in targets}
    # duration buckets ±0.05s
    for fpath in files:
        if not need:
            break
        print(f"Rasa {fpath} need={len(need)}", flush=True)
        local = hf_hub_download("ai4bharat/Rasa", fpath, repo_type="dataset", token=token)
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            if not need:
                break
            table = pf.read_row_group(rg, columns=["text", "audio", "style", "duration"])
            for i in range(len(table)):
                style = str(table.column("style")[i].as_py() or "")
                if style not in styles:
                    continue
                dur = float(table.column("duration")[i].as_py() or 0)
                # candidate if any target close in duration
                cands = [
                    t
                    for t in need.values()
                    if t["style"] == style and abs(t["dur"] - dur) < 0.08
                ]
                if not cands:
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
                t["clip"]["text"] = text
                t["clip"]["text_source"] = "hf_recover"
                filled += 1
                _write_sidecar(t["path"], text)
                print(f"  + {t['clip']['id']} {style}", flush=True)
    return filled


def recover_iv(token: str, targets: list[dict]) -> int:
    need = {t["hash"]: t for t in targets}
    filled = 0
    files = sorted(
        f
        for f in list_repo_files("ai4bharat/IndicVoices", repo_type="dataset", token=token)
        if f.startswith("hindi/train") and f.endswith(".parquet")
    )
    for fpath in files:
        if not need:
            break
        print(f"IV {fpath} need={len(need)}", flush=True)
        try:
            local = hf_hub_download(
                "ai4bharat/IndicVoices", fpath, repo_type="dataset", token=token
            )
        except Exception as e:
            print(f"  skip {type(e).__name__}: {e}"[:120], flush=True)
            continue
        pf = pq.ParquetFile(local)
        names = set(pf.schema_arrow.names)
        if "unsanitized_verbatim" not in names:
            continue
        # duration filter via meta first
        meta = pf.read(columns=["unsanitized_verbatim", "duration"])
        hit_idx = []
        for i in range(len(meta)):
            dur = float(meta.column("duration")[i].as_py() or 0)
            if dur > 1000:
                dur /= 16000.0
            if any(abs(t["dur"] - dur) < 0.08 for t in need.values()):
                hit_idx.append(i)
        if not hit_idx:
            continue
        hit_set = set(hit_idx)
        offset = 0
        for rg in range(pf.num_row_groups):
            cols = ["unsanitized_verbatim"]
            if "audio_filepath" in names:
                cols.append("audio_filepath")
            elif "audio" in names:
                cols.append("audio")
            else:
                break
            table = pf.read_row_group(rg, columns=cols)
            for j in range(len(table)):
                i = offset + j
                if i not in hit_set:
                    continue
                text = str(table.column("unsanitized_verbatim")[j].as_py() or "")
                ab = audio_bytes(table.column(cols[-1])[j].as_py())
                if not ab or not text:
                    continue
                try:
                    h, _ = bytes_info(ab)
                except Exception:
                    continue
                if h not in need:
                    continue
                t = need.pop(h)
                t["clip"]["text"] = text
                t["clip"]["text_source"] = "hf_recover"
                filled += 1
                _write_sidecar(t["path"], text)
                print(f"  + {t['clip']['id']}", flush=True)
                if not need:
                    break
            offset += len(table)
            if not need:
                break
    return filled


def recover_ivr(token: str, targets: list[dict]) -> int:
    need = {t["hash"]: t for t in targets}
    filled = 0
    files = sorted(
        f
        for f in list_repo_files(
            "ai4bharat/indicvoices_r", repo_type="dataset", token=token
        )
        if f.startswith("Hindi/train") and f.endswith(".parquet")
    )
    for fpath in files:
        if not need:
            break
        print(f"IVR {fpath} need={len(need)}", flush=True)
        try:
            local = hf_hub_download(
                "ai4bharat/indicvoices_r", fpath, repo_type="dataset", token=token
            )
        except Exception as e:
            print(f"  skip {type(e).__name__}", flush=True)
            continue
        pf = pq.ParquetFile(local)
        # usually one big row group
        for rg in range(pf.num_row_groups):
            if not need:
                break
            table = pf.read_row_group(rg)
            names = pf.schema_arrow.names
            for i in range(len(table)):
                dur = float(table.column("duration")[i].as_py() or 0)
                if dur > 1000:
                    dur /= 16000.0
                if not any(abs(t["dur"] - dur) < 0.08 for t in need.values()):
                    continue
                text = str(
                    table.column("text")[i].as_py()
                    or (table.column("verbatim")[i].as_py() if "verbatim" in names else "")
                    or ""
                )
                ab = audio_bytes(table.column("audio")[i].as_py())
                if not ab or not text:
                    continue
                try:
                    h, _ = bytes_info(ab)
                except Exception:
                    continue
                if h not in need:
                    continue
                t = need.pop(h)
                t["clip"]["text"] = text
                t["clip"]["text_source"] = "hf_recover"
                filled += 1
                _write_sidecar(t["path"], text)
                print(f"  + {t['clip']['id']}", flush=True)
    return filled


def _write_sidecar(wav_path: Path, text: str) -> None:
    side = Path(str(wav_path) + ".json")
    meta = json.loads(side.read_text()) if side.exists() else {}
    meta["text"] = text
    meta["text_source"] = "hf_recover"
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    token = load_token()
    drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))
    clips = drafts["clips"]

    # sidecar first
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
    print(f"sidecar filled {side_n}", flush=True)

    empty = [c for c in clips if not (c.get("text") or "").strip()]
    print(f"empty {len(empty)}", flush=True)

    buckets: dict[str, list] = defaultdict(list)
    for c in empty:
        p = resolve_wav(c["file"])
        if not p:
            print("missing", c["id"], c["file"])
            continue
        h, dur = wav_info(p)
        style = ""
        if c.get("source") == "rasa":
            # scale_rasa_anger_00.wav
            style = p.stem.replace("scale_rasa_", "").rsplit("_", 1)[0].upper()
            if style == "ANGER":
                style = "ANGER"
        buckets[c["source"]].append(
            {"clip": c, "path": p, "hash": h, "dur": dur, "style": style}
        )

    filled = 0
    if buckets["rasa"]:
        filled += recover_rasa(token, buckets["rasa"])
    if buckets["iv"]:
        filled += recover_iv(token, buckets["iv"])
    if buckets["ivr"]:
        filled += recover_ivr(token, buckets["ivr"])

    still = sum(1 for c in clips if not (c.get("text") or "").strip())
    drafts["clips"] = clips
    DRAFTS.write_text(json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if CATALOG.exists():
        cat = json.loads(CATALOG.read_text(encoding="utf-8"))
        by_file = {c["file"]: c for c in clips}
        for row in cat.get("clips") or []:
            d = by_file.get(row.get("file"))
            if d and (d.get("text") or "").strip():
                row["text"] = d["text"]
        CATALOG.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"DONE hf_filled≈{filled} still_empty={still}", flush=True)


if __name__ == "__main__":
    main()
