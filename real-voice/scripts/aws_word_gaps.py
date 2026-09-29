#!/usr/bin/env python3
"""Dead-air metric from ASR word timings, not an energy floor (2026-09-29). aws_fx_sil.py counts hiss/hum as sound,
so on noisy refs (user laptop-mic) it read noise-filled gaps as speech; the user heard them as long pauses.

Per clip in s3 <prefix>/inc_<arm>/wav: faster-whisper large-v3 word timestamps (beam_size=1: only timings are kept,
text WER stays with aws_whisper_wer.py), then
  lead_s   = first word start          trail_s = duration - last word end
  max_gap  = longest gap between consecutive words      n_gap1 / n_gap2 = gaps >= 1 s / >= 2 s
  n_words, words_per_s over the spoken span; no_words = ASR found nothing.
Caveat: Whisper can hallucinate words over noise, which would hide a gap; read with the ear, not instead of it.

  aws_word_gaps.py --arms fxen_cz_svh1_s7,... -> s3 <prefix>/wgap_<arm>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

B = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
R = os.environ.get("RV_REGION", "us-east-1")


def aws(*a):
    return subprocess.run(["aws", *a, "--region", R, "--only-show-errors"], capture_output=True, text=True)


def gaps(words, dur):
    if not words:
        return {"dur": round(dur, 2), "no_words": True, "n_words": 0, "lead_s": round(dur, 2), "trail_s": 0.0,
                "max_gap": 0.0, "n_gap1": 0, "n_gap2": 0, "words_per_s": 0.0}
    g = [max(0.0, b.start - a.end) for a, b in zip(words, words[1:])]
    span = max(1e-6, words[-1].end - words[0].start)
    return {"dur": round(dur, 2), "no_words": False, "n_words": len(words), "lead_s": round(words[0].start, 2),
            "trail_s": round(max(0.0, dur - words[-1].end), 2), "max_gap": round(max(g, default=0.0), 2),
            "n_gap1": sum(x >= 1.0 for x in g), "n_gap2": sum(x >= 2.0 for x in g),
            "words_per_s": round(len(words) / span, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prefix", default="eval/incumbent_v1")
    ap.add_argument("--langs", default="en")
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    from faster_whisper import WhisperModel
    asr = WhisperModel(a.asr_model, device="cuda", compute_type="float16")
    langs = set(a.langs.split(","))
    for arm in a.arms.split(","):
        if a.skip_done and aws("s3", "ls", f"s3://{B}/{a.prefix}/wgap_{arm}.json").stdout.strip():
            print(f"WGAP_SKIP {arm}: done", flush=True)
            continue
        meta = subprocess.run(["aws", "s3", "cp", f"s3://{B}/{a.prefix}/inc_{arm}.json", "-", "--region", R],
                              capture_output=True, text=True)
        if meta.returncode != 0:
            print(f"WGAP_SKIP {arm}: inc json not on S3", flush=True)
            continue
        meta = json.loads(meta.stdout)
        rows = []
        with tempfile.TemporaryDirectory() as td:
            aws("s3", "sync", f"s3://{B}/{a.prefix}/inc_{arm}/wav/", td)
            for it in meta["items"]:
                if it.get("lang") not in langs:
                    continue
                f = Path(td) / Path(it["file"]).name
                if not f.exists():
                    continue
                dur = sf.info(str(f)).duration
                segs, _ = asr.transcribe(str(f), language=it.get("lang", "en"), beam_size=1, word_timestamps=True)
                words = [w for s in segs for w in (s.words or [])]
                rows.append({"id": it["id"], **gaps(words, dur)})
        n = max(len(rows), 1)
        summ = {"n": len(rows), "max_gap_mean": round(sum(r["max_gap"] for r in rows) / n, 3),
                "clips_gap2": sum(r["n_gap2"] > 0 for r in rows), "clips_gap1": sum(r["n_gap1"] > 0 for r in rows),
                "no_words": sum(r["no_words"] for r in rows),
                "words_per_s_mean": round(sum(r["words_per_s"] for r in rows) / n, 3)}
        out = Path(tempfile.gettempdir()) / f"wgap_{arm}.json"
        out.write_text(json.dumps({"arm": arm, "asr": a.asr_model, "ref": meta.get("ref"), **summ, "items": rows}))
        aws("s3", "cp", str(out), f"s3://{B}/{a.prefix}/wgap_{arm}.json")
        print(f"WGAP {arm} {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
