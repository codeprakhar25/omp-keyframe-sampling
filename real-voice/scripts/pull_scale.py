#!/usr/bin/env python3
"""Pull stratified scale wavs (beyond smoke) → data/samples/scale/ + catalog.

  .venv/bin/python scripts/pull_scale.py
  # default ~160: rasa40 / iv90 / ivr30
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "scale"
CATALOG = ROOT / "data" / "scale" / "pull_catalog.json"


def load_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip().strip('"')
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no HF_TOKEN")


def save_full(wav, sr, path: Path) -> float:
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr)
    return round(len(wav) / float(sr), 2)


def audio_bytes(obj):
    return obj.get("bytes") if isinstance(obj, dict) else None


def pull_rasa(token: str, n: int, used_text: set) -> list[dict]:
    want = {
        "HAPPY": n // 6 + 1,
        "ANGER": n // 6 + 1,
        "SAD": n // 6 + 1,
        "SURPRISE": n // 6,
        "FEAR": n // 6,
        "DISGUST": n // 6,
    }
    # trim to n
    while sum(want.values()) > n:
        for k in list(want):
            if want[k] > 0 and sum(want.values()) > n:
                want[k] -= 1
    got: dict[str, list] = defaultdict(list)
    files = sorted(
        f
        for f in list_repo_files("ai4bharat/Rasa", repo_type="dataset", token=token)
        if f.startswith("Hindi/train") and f.endswith(".parquet")
    )
    # skip first shard (smoke-heavy); start at 1
    for fpath in files[1:]:
        if all(len(got[k]) >= want[k] for k in want):
            break
        print("Rasa", fpath, flush=True)
        local = hf_hub_download("ai4bharat/Rasa", fpath, repo_type="dataset", token=token)
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            if all(len(got[k]) >= want[k] for k in want):
                break
            table = pf.read_row_group(rg)
            for i in range(len(table)):
                style = str(table.column("style")[i].as_py())
                if style not in want or len(got[style]) >= want[style]:
                    continue
                text = str(table.column("text")[i].as_py() or "")
                key = re.sub(r"\s+", " ", text).strip()[:80]
                if key in used_text:
                    continue
                audio = table.column("audio")[i].as_py()
                if not isinstance(audio, dict) or not audio.get("bytes"):
                    continue
                wav, sr = sf.read(io.BytesIO(audio["bytes"]))
                fname = f"scale_rasa_{style.lower()}_{len(got[style]):02d}.wav"
                dur = save_full(wav, sr, OUT / fname)
                if dur < 0.5:
                    continue
                used_text.add(key)
                got[style].append(
                    {
                        "source": "rasa",
                        "dataset": "ai4bharat/Rasa",
                        "file": f"scale/{fname}",
                        "text": text,
                        "native_label": style,
                        "gender": str(table.column("gender")[i].as_py() or ""),
                        "duration_s": dur,
                        "full_duration_s": dur,
                        "shard": fpath,
                    }
                )
                print(" ", fname, style, flush=True)
                if all(len(got[k]) >= want[k] for k in want):
                    break
    rows = []
    for k in want:
        rows.extend(got[k])
    print("Rasa total", len(rows), {k: len(got[k]) for k in want}, flush=True)
    return rows[:n]


def pull_iv(token: str, n: int, used_text: set) -> list[dict]:
    want = {
        "whispering": 15,
        "sigh": 20,
        "inhaling": 25,
        "thinking": 20,
        "breathing": 10,
    }
    # scale want to n
    total_w = sum(want.values())
    want = {k: max(1, int(round(v * n / total_w))) for k, v in want.items()}
    while sum(want.values()) > n:
        k = max(want, key=want.get)
        want[k] -= 1
    got: dict[str, list] = defaultdict(list)
    rx = {
        "whispering": re.compile(r"<whispering>", re.I),
        "sigh": re.compile(r"<sigh>", re.I),
        "inhaling": re.compile(r"<inhaling>", re.I),
        "thinking": re.compile(r"<(umm|uhh|hmm)>", re.I),
        "breathing": re.compile(r"<breathing>", re.I),
    }
    files = sorted(
        f
        for f in list_repo_files("ai4bharat/IndicVoices", repo_type="dataset", token=token)
        if f.startswith("hindi/train") and f.endswith(".parquet")
    )
    # spread, skip 0
    picks = files[1 :: max(1, len(files) // 20)][:20]
    for fpath in picks:
        if all(len(got[k]) >= want[k] for k in want):
            break
        print("IV", fpath, flush=True)
        local = hf_hub_download(
            "ai4bharat/IndicVoices", fpath, repo_type="dataset", token=token
        )
        pf = pq.ParquetFile(local)
        meta = pf.read(
            columns=["unsanitized_verbatim", "scenario", "gender", "state", "duration"]
        )
        hits = []
        for i in range(len(meta)):
            text = str(meta.column("unsanitized_verbatim")[i].as_py() or "")
            key = re.sub(r"\s+", " ", text).strip()[:80]
            if key in used_text:
                continue
            dur = float(meta.column("duration")[i].as_py() or 0)
            if dur > 1000:
                dur /= 16000.0
            if not (2.0 <= dur <= 40):
                continue
            bucket = None
            for b, pattern in rx.items():
                if len(got[b]) >= want[b]:
                    continue
                if pattern.search(text):
                    bucket = b
                    break
            if not bucket:
                continue
            hits.append(
                (
                    bucket,
                    i,
                    text,
                    str(meta.column("scenario")[i].as_py() or ""),
                    str(meta.column("gender")[i].as_py() or ""),
                    str(meta.column("state")[i].as_py() or ""),
                )
            )
        if not hits:
            continue
        # diversify: one spacing
        take = {}
        for bucket, i, text, scen, gender, state in hits:
            if len(got[bucket]) + sum(1 for x in take.values() if x[0] == bucket) >= want[
                bucket
            ]:
                continue
            if any(abs(i - j) < 4 for j in take):
                continue
            take[i] = (bucket, text, scen, gender, state)
        if not take:
            continue
        offset = 0
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["audio_filepath"])
            for j in range(len(table)):
                i = offset + j
                if i not in take:
                    continue
                bucket, text, scen, gender, state = take[i]
                if len(got[bucket]) >= want[bucket]:
                    continue
                ab = audio_bytes(table.column("audio_filepath")[j].as_py())
                if not ab:
                    continue
                wav, sr = sf.read(io.BytesIO(ab))
                fname = f"scale_iv_{bucket}_{len(got[bucket]):02d}.wav"
                dur = save_full(wav, sr, OUT / fname)
                used_text.add(re.sub(r"\s+", " ", text).strip()[:80])
                got[bucket].append(
                    {
                        "source": "iv",
                        "dataset": "ai4bharat/IndicVoices",
                        "file": f"scale/{fname}",
                        "text": text,
                        "native_label": f"{scen}|{bucket}",
                        "gender": gender,
                        "state": state,
                        "duration_s": dur,
                        "full_duration_s": dur,
                        "shard": fpath,
                    }
                )
                print(" ", fname, flush=True)
            offset += len(table)
            if all(len(got[k]) >= want[k] for k in want):
                break
    rows = []
    for k in want:
        rows.extend(got[k])
    print("IV total", len(rows), {k: len(got[k]) for k in want}, flush=True)
    return rows[:n]


def pull_ivr(token: str, n: int, used_text: set, per_shard: int = 4) -> list[dict]:
    """Take up to per_shard Extempore clips per parquet — avoid 1-clip-per-~500MB shard."""
    rows = []
    files = sorted(
        f
        for f in list_repo_files(
            "ai4bharat/indicvoices_r", repo_type="dataset", token=token
        )
        if f.startswith("Hindi/train") and f.endswith(".parquet")
    )
    # fewer shards needed when mining multiple per file
    step = max(1, len(files) // max(8, (n // per_shard) + 2))
    picks = files[::step]
    for fpath in picks:
        if len(rows) >= n:
            break
        print("IVR", fpath, flush=True)
        local = hf_hub_download(
            "ai4bharat/indicvoices_r", fpath, repo_type="dataset", token=token
        )
        pf = pq.ParquetFile(local)
        table = pf.read_row_group(0)
        names = pf.schema_arrow.names
        cands = []
        for i in range(len(table)):
            scen = str(table.column("scenario")[i].as_py() or "")
            if scen != "Extempore":
                continue
            dur = float(table.column("duration")[i].as_py() or 0)
            if dur > 1000:
                dur /= 16000.0
            if not (4 <= dur <= 35):
                continue
            text = str(
                table.column("text")[i].as_py()
                or table.column("verbatim")[i].as_py()
                or ""
            )
            key = re.sub(r"\s+", " ", text).strip()[:80]
            if key in used_text:
                continue
            audio = table.column("audio")[i].as_py()
            if not isinstance(audio, dict) or not audio.get("bytes"):
                continue
            gender = (
                str(table.column("gender")[i].as_py() or "") if "gender" in names else ""
            )
            state = (
                str(table.column("state")[i].as_py() or "") if "state" in names else ""
            )
            cands.append((dur, text, gender, state, audio, key))
        cands.sort(reverse=True)
        taken = 0
        for dur, text, gender, state, audio, key in cands:
            if taken >= per_shard or len(rows) >= n:
                break
            if key in used_text:
                continue
            wav, sr = sf.read(io.BytesIO(audio["bytes"]))
            fname = f"scale_ivr_{len(rows):02d}.wav"
            d = save_full(wav, sr, OUT / fname)
            used_text.add(key)
            rows.append(
                {
                    "source": "ivr",
                    "dataset": "ai4bharat/indicvoices_r",
                    "file": f"scale/{fname}",
                    "text": text,
                    "native_label": "Extempore",
                    "gender": gender,
                    "state": state,
                    "duration_s": d,
                    "full_duration_s": d,
                    "shard": fpath,
                }
            )
            taken += 1
            print(" ", fname, d, flush=True)
        print(f"  shard took {taken}", flush=True)
    print("IVR total", len(rows), flush=True)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rasa", type=int, default=40)
    ap.add_argument("--iv", type=int, default=90)
    ap.add_argument("--ivr", type=int, default=30)
    args = ap.parse_args()

    token = load_token()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "scale").mkdir(parents=True, exist_ok=True)

    used = set()
    # avoid re-pulling smoke texts
    smoke = ROOT / "data" / "smoke" / "clips.json"
    if smoke.exists():
        for c in json.loads(smoke.read_text())["clips"]:
            used.add(re.sub(r"\s+", " ", (c.get("text") or "")).strip()[:80])

    rows = []
    rows.extend(pull_rasa(token, args.rasa, used))
    rows.extend(pull_iv(token, args.iv, used))
    rows.extend(pull_ivr(token, args.ivr, used))

    for i, r in enumerate(rows):
        r["id"] = f"sc{i+1:03d}"
        r["hf_or_path"] = f"samples/{r['file']}"

    CATALOG.write_text(
        json.dumps({"n": len(rows), "clips": rows}, ensure_ascii=False, indent=2) + "\n"
    )
    from collections import Counter

    print("CATALOG", len(rows), Counter(r["source"] for r in rows), "→", CATALOG)


if __name__ == "__main__":
    main()
