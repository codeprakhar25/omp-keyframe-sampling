#!/usr/bin/env python3
"""Inference frontend v0, reference side (2026-10-02). Cleans a cloning reference before Air ever sees it.

Why: the bad-ref tests showed the reference drives v8's dead air and stretch (seen studio ref 0/12 gapped clips,
unseen spontaneous / laptop-mic refs 4-14/18), and stock Air on the same refs is clean, so the fine-tune copies ref
hesitation. CosyVoice and IndexTTS2 do no ref cleanup at inference; they rely on clean training data. We can't
retrain cheaply, so v0 hands the model a fluent-looking ref instead:
  1. VAD (silero) on a 16 kHz copy: speech spans, min silence 150 ms, pad 50 ms
  2. trim: keep <= LEAD s before the first span and <= TRAIL s after the last
  3. pause cap: any internal gap longer than MAX_PAUSE s is cut down to MAX_PAUSE (10 ms crossfades)
  4. loudness: speech RMS -> -20 dBFS, peak capped at 0.95
  variant "fedn": spectral-gating denoise (noisereduce, noise profile = the ref's own non-speech) before 1-4.
The ref text is kept as is: trimming and pause cuts remove no words. Output ref/<name>_<variant>.wav + .json
(no "codes" key, so the harness re-encodes the cleaned wav).

  aws_ref_frontend.py --refs svh_1,pk_en1 --variants fe,fedn [--prefix eval/incumbent_v1]
  aws_ref_frontend.py --local in.wav out.wav [--denoise]       # local check, no S3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

B = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
R = os.environ.get("RV_REGION", "us-east-1")
LEAD, TRAIL, MAX_PAUSE, XFADE = 0.10, 0.15, 0.30, 0.010
TARGET_DBFS, PEAK = -20.0, 0.95


def _resample(y: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return y.astype(np.float32)
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=to)


def speech_spans(y: np.ndarray, sr: int, min_sil_ms: int = 150, pad_ms: int = 50) -> list[tuple[int, int]]:
    """Silero VAD speech spans in samples of the ORIGINAL rate. faster-whisper bundles silero (box venv);
    the silero_vad package is the local fallback."""
    y16 = _resample(y, sr, 16000)
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        ts = get_speech_timestamps(y16, VadOptions(min_silence_duration_ms=min_sil_ms, speech_pad_ms=pad_ms))
    except ImportError:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        ts = get_speech_timestamps(torch.from_numpy(y16), load_silero_vad(), sampling_rate=16000,
                                   min_silence_duration_ms=min_sil_ms, speech_pad_ms=pad_ms)
    k = sr / 16000
    return [(int(t["start"] * k), min(len(y), int(t["end"] * k))) for t in ts]


def edit_silence(y: np.ndarray, sr: int, spans, lead: float, trail: float, max_pause: float):
    """Keep the speech spans; shrink the lead, trail and every internal gap to the given caps."""
    if not spans:
        return y, {"spans": 0}
    xf = int(XFADE * sr)
    keep = [(max(0, spans[0][0] - int(lead * sr)), spans[0][1])]
    cut_s = 0.0
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        gap = s1 - e0
        if gap > max_pause * sr:   # keep half the cap after e0 and half before s1
            h = int(max_pause * sr / 2)
            keep[-1] = (keep[-1][0], e0 + h)
            keep.append((s1 - h, e1))
            cut_s += (gap - 2 * h) / sr
        else:
            keep[-1] = (keep[-1][0], e1)
    keep[-1] = (keep[-1][0], min(len(y), spans[-1][1] + int(trail * sr)))
    out = y[keep[0][0]:keep[0][1]].astype(np.float32)
    for a, b in keep[1:]:
        seg = y[a:b].astype(np.float32)
        n = min(xf, len(out), len(seg))
        if n > 0:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            out[-n:] = out[-n:] * (1 - ramp) + seg[:n] * ramp
            seg = seg[n:]
        out = np.concatenate([out, seg])
    lead_s, trail_s = spans[0][0] / sr, (len(y) - spans[-1][1]) / sr
    gaps = [(s1 - e0) / sr for (_, e0), (s1, _) in zip(spans, spans[1:])]
    return out, {"spans": len(spans), "lead_in_s": round(lead_s, 2), "trail_in_s": round(trail_s, 2),
                 "max_gap_in_s": round(max(gaps, default=0.0), 2), "pause_cut_s": round(cut_s, 2)}


def loudness(y: np.ndarray, sr: int, spans) -> np.ndarray:
    sp = np.concatenate([y[a:b] for a, b in spans]) if spans else y
    rms = float(np.sqrt(np.mean(sp ** 2)) + 1e-9)
    y = y * (10 ** (TARGET_DBFS / 20) / rms)
    pk = float(np.max(np.abs(y)) + 1e-9)
    return (y * (PEAK / pk) if pk > PEAK else y).astype(np.float32)


def denoise(y: np.ndarray, sr: int, spans) -> np.ndarray:
    try:
        import noisereduce as nr
    except ImportError:   # box venv: install on first use
        subprocess.run([shutil.which("uv") or "/usr/local/bin/uv", "pip", "install", "--python", sys.executable, "noisereduce"], check=True)
        import noisereduce as nr
    mask = np.ones(len(y), bool)
    for a, b in spans:
        mask[a:b] = False
    noise = y[mask]
    if len(noise) >= int(0.2 * sr):
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=True, prop_decrease=0.85).astype(np.float32)
    return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.85).astype(np.float32)


def process(y: np.ndarray, sr: int, do_denoise: bool):
    if y.ndim > 1:
        y = y.mean(axis=1)
    spans = speech_spans(y, sr)
    if do_denoise:
        y = denoise(y, sr, spans)
        spans = speech_spans(y, sr)   # re-detect on the cleaner signal
    y2, st = edit_silence(y, sr, spans, LEAD, TRAIL, MAX_PAUSE)
    y2 = loudness(y2, sr, speech_spans(y2, sr))
    st.update({"dur_in_s": round(len(y) / sr, 2), "dur_out_s": round(len(y2) / sr, 2), "denoise": do_denoise,
               "short": len(y2) / sr < 3.0})   # Air wants >= 3 s of ref; a cleaned ref under that is flagged, not padded
    return y2, st


def aws(*a):
    return subprocess.run(["aws", *a, "--region", R, "--only-show-errors"], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="")
    ap.add_argument("--variants", default="fe,fedn")
    ap.add_argument("--prefix", default="eval/incumbent_v1")
    ap.add_argument("--local", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--denoise", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    if a.local:
        y, sr = sf.read(a.local[0], dtype="float32")
        y2, st = process(y, sr, a.denoise)
        sf.write(a.local[1], y2, sr, subtype="PCM_16")
        print(json.dumps(st))
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for ref in filter(None, a.refs.split(",")):
            aws("s3", "cp", f"s3://{B}/{a.prefix}/ref/{ref}.wav", str(td / f"{ref}.wav"))
            aws("s3", "cp", f"s3://{B}/{a.prefix}/ref/{ref}.json", str(td / f"{ref}.json"))
            if not (td / f"{ref}.wav").exists() or not (td / f"{ref}.json").exists():
                print(f"REFFE_SKIP {ref}: wav or json missing on S3", flush=True)
                continue
            meta = json.loads((td / f"{ref}.json").read_text(encoding="utf-8"))
            meta.pop("codes", None)   # force the harness to encode the cleaned wav
            y, sr = sf.read(str(td / f"{ref}.wav"), dtype="float32")
            for v in a.variants.split(","):
                if v == "rw":   # control: unedited wav, codes dropped -> isolates the re-encode from the cleaning
                    y2, st = (y.mean(axis=1) if y.ndim > 1 else y), {"dur_in_s": round(len(y) / sr, 2),
                                                                     "dur_out_s": round(len(y) / sr, 2), "edit": None}
                else:
                    y2, st = process(y, sr, v == "fedn")
                name = f"{ref}_{v}"
                sf.write(str(td / f"{name}.wav"), y2, sr, subtype="PCM_16")
                (td / f"{name}.json").write_text(json.dumps({**meta, "dur": st["dur_out_s"], "frontend": v,
                                                             "frontend_stats": st}, ensure_ascii=False))
                for ext in ("wav", "json"):
                    aws("s3", "cp", str(td / f"{name}.{ext}"), f"s3://{B}/{a.prefix}/ref/{name}.{ext}")
                print(f"REFFE {name} {json.dumps(st)}", flush=True)


if __name__ == "__main__":
    main()
