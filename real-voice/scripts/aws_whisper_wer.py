#!/usr/bin/env python3
"""English WER/CER of incumbent-layout clips with faster-whisper large-v3 (the Seed-TTS-Eval English ASR).

  aws_whisper_wer.py --arms v7_26k_hien,cartesia_sonic3.6_hien --prefix eval/incumbent_v1 --langs en

Input per arm: <prefix>/inc_<arm>.json (items with id/lang/text/file) + <prefix>/inc_<arm>/wav/.
Output: <prefix>/whisper_wer_<arm>.json with per-item hypothesis, WER and CER, plus means.
Normalisation: lowercase, keep letters/digits/apostrophes, collapse spaces (same for reference and hypothesis).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BUCKET = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
REGION = os.environ.get("RV_REGION", "us-east-1")
WORK = Path(os.environ.get("RV_WORK", "/opt/rv"))


def _aws(*a):
    subprocess.run(["aws", *a, "--region", REGION, "--only-show-errors"], check=True)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower().replace("-", " ")).split())


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]



def load16(path):
    """16 kHz mono float32 for faster-whisper. Passing a file path makes it call av.open(..., metadata_errors=...),
    which newer PyAV releases reject (TypeError, 2026-10-02 cbase box); an array skips PyAV entirely."""
    import librosa
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return (librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prefix", default="eval/incumbent_v1")
    ap.add_argument("--name-tmpl", default="inc_{arm}")
    ap.add_argument("--langs", default="en")
    ap.add_argument("--asr-model", default="large-v3")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    asr = WhisperModel(a.asr_model, device="cuda", compute_type="float16")
    langs = set(a.langs.split(","))
    for arm in a.arms.split(","):
        base = a.name_tmpl.format(arm=arm)
        d = WORK / "whisper_wer" / base
        (d / "wav").mkdir(parents=True, exist_ok=True)
        try:
            _aws("s3", "cp", f"s3://{BUCKET}/{a.prefix}/{base}.json", str(d / "inc.json"))
        except subprocess.CalledProcessError:
            print(f"WHISPER_SKIP {arm}: {base}.json not on S3", flush=True)
            continue
        _aws("s3", "sync", f"s3://{BUCKET}/{a.prefix}/{base}/wav/", str(d / "wav"))
        items = [i for i in json.loads((d / "inc.json").read_text(encoding="utf-8"))["items"] if i.get("lang") in langs]
        rows = []
        for it in items:
            segs, _ = asr.transcribe(load16(d / it["file"]), language="en", beam_size=5)
            hyp = " ".join(s.text.strip() for s in segs)
            r, h = norm(it["text"]), norm(hyp)
            wer = lev(r.split(), h.split()) / max(len(r.split()), 1)
            cer = lev(r.replace(" ", ""), h.replace(" ", "")) / max(len(r.replace(" ", "")), 1)
            rows.append({"id": it["id"], "text": it["text"], "hyp": hyp, "wer": round(wer, 4), "cer": round(cer, 4)})
            print(f"  {arm} {it['id']} WER={wer:.3f} | {hyp[:60]}", flush=True)
        n = max(len(rows), 1)
        res = {"arm": arm, "asr": a.asr_model, "n": len(rows), "wer": round(sum(r["wer"] for r in rows) / n, 4),
               "cer": round(sum(r["cer"] for r in rows) / n, 4), "items": rows}
        out = WORK / "runs" / f"whisper_wer_{arm}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        _aws("s3", "cp", str(out), f"s3://{BUCKET}/{a.prefix}/whisper_wer_{arm}.json")
        print(f"WHISPER_WER {arm} n={res['n']} wer={res['wer']} cer={res['cer']}", flush=True)


if __name__ == "__main__":
    main()
