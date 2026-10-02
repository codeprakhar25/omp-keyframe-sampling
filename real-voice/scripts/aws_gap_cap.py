#!/usr/bin/env python3
"""Inference frontend v0, output side (2026-10-02): cap dead air in generated clips after decoding.

NeuCodec has no single silent token (one fused 16-bit FSQ code per frame), so IndexTTS2's trick of collapsing runs
of a silent token in the code stream does not carry over. Instead: silero VAD on the decoded wav, then the same
edit as the ref frontend: lead <= 0.15 s, trail <= 0.25 s, any internal gap > 0.45 s cut down to 0.45 s.
A cap, not a fix: a mid-word stall or a stretched vowel is speech to the VAD and stays.

  aws_gap_cap.py --arms fxen_c_svh1_s7,... -> s3 inc_<arm>_gc.json + inc_<arm>_gc/wav/ (same layout, so every
  scorer and page builder reads a _gc arm like any other)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from aws_ref_frontend import edit_silence, speech_spans

B = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
R = os.environ.get("RV_REGION", "us-east-1")
LEAD, TRAIL, MAX_GAP = 0.15, 0.25, 0.45


def aws(*a):
    return subprocess.run(["aws", *a, "--region", R, "--only-show-errors"], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prefix", default="eval/incumbent_v1")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    for arm in a.arms.split(","):
        out = f"{arm}_gc"
        if a.skip_done and aws("s3", "ls", f"s3://{B}/{a.prefix}/inc_{out}.json").stdout.strip():
            print(f"GAPCAP_SKIP {arm}: done", flush=True)
            continue
        meta = subprocess.run(["aws", "s3", "cp", f"s3://{B}/{a.prefix}/inc_{arm}.json", "-", "--region", R],
                              capture_output=True, text=True)
        if meta.returncode != 0:
            print(f"GAPCAP_SKIP {arm}: inc json not on S3", flush=True)
            continue
        meta = json.loads(meta.stdout)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "in").mkdir()
            (td / "wav").mkdir()
            aws("s3", "sync", f"s3://{B}/{a.prefix}/inc_{arm}/wav/", str(td / "in"))
            rows, cut = [], 0.0
            for it in meta["items"]:
                f = td / "in" / Path(it["file"]).name
                if not f.exists():
                    continue
                y, sr = sf.read(str(f), dtype="float32")
                y2, st = edit_silence(y, sr, speech_spans(y, sr), LEAD, TRAIL, MAX_GAP)
                sf.write(str(td / "wav" / f.name), y2, sr, subtype="PCM_16")
                cut += st.get("pause_cut_s", 0.0)
                rows.append({**it, "system": out, "duration_s": round(len(y2) / sr, 2), "gap_cap": st})
            res = {**{k: v for k, v in meta.items() if k != "items"}, "gap_cap": {"lead": LEAD, "trail": TRAIL,
                   "max_gap": MAX_GAP, "source_arm": arm}, "items": rows}
            (td / f"inc_{out}.json").write_text(json.dumps(res, ensure_ascii=False))
            aws("s3", "sync", str(td / "wav"), f"s3://{B}/{a.prefix}/inc_{out}/wav/")
            aws("s3", "cp", str(td / f"inc_{out}.json"), f"s3://{B}/{a.prefix}/inc_{out}.json")
        print(f"GAPCAP {out} n={len(rows)} internal_cut_s={cut:.2f}", flush=True)


if __name__ == "__main__":
    main()
