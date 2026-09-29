#!/usr/bin/env python3
"""Table for the flow-stack probe: per system x ref, pooled over seeds 7/11/23.

Systems: sh = stock Air (our harness), a = v8 A final, cz / cx = CosyVoice3 ref-in-LM / ref-only-in-flow, ix = IndexTTS2.
Columns: Whisper WER, word-gap dead air (clips with a >=2 s / >=1 s gap between words, mean + max of the longest gap,
no-word clips), words/s, SIM (WavLM-SV vs the ref, where scored). Reads s3 eval/incumbent_v1/{whisper_wer,wgap}_<arm>.json
and fxensim_*.json; writes data/eval/flowstack_report.json and prints markdown.

  python scripts/flowstack_report.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

B, R, P = "real-voice-mio-692707725608", "us-east-1", "eval/incumbent_v1"
SYSTEMS = [("sh", "stock Air"), ("a", "v8 A"), ("cz", "CosyVoice3 ref-in-LM"), ("cx", "CosyVoice3 ref-in-flow"),
           ("ix", "IndexTTS2")]
REFS = [("fxen", r) for r in ("ctrl", "svh1", "svh2", "svh3", "svh4", "svh5", "svh6")] + \
       [("fxpk", r) for r in ("en1", "en2", "hi1", "hi2", "mx1", "mx2")]
SIM_FILES = ["fxensim_cv", "fxensim_ix", "fxensim_sh", "fxensim_a", "fxpksim_sh", "fxpksim_a"]


def s3json(key):
    r = subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/{key}", "-", "--region", R], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def main():
    sims = {}
    for f in SIM_FILES:
        d = s3json(f"{f}.json")
        for it in (d or {}).get("items", []):
            if it["id"].startswith("en"):   # fxpksim_* also scored the Hindi/Hinglish lines; English only here
                sims.setdefault(it["arm"], []).append(it["sim"])
    rows = []
    for sysk, sysname in SYSTEMS:
        for pre, ref in REFS:
            arms = [f"{pre}_{sysk}_{ref}_s{s}" for s in (7, 11, 23)]
            g, w, sm = [], [], []
            for arm in arms:
                x = s3json(f"wgap_{arm}.json")
                g += (x or {}).get("items", [])
                y = s3json(f"whisper_wer_{arm}.json")
                w += [i["wer"] for i in (y or {}).get("items", [])]
                sm += sims.get(arm, [])
            if not g:
                continue
            mg = [i["max_gap"] for i in g]
            rows.append({"system": sysname, "key": sysk, "set": pre, "ref": ref, "n": len(g),
                         "wer": round(sum(w) / len(w), 3) if w else None,
                         "clips_gap2": sum(i["n_gap2"] > 0 for i in g), "clips_gap1": sum(i["n_gap1"] > 0 for i in g),
                         "max_gap_mean": round(sum(mg) / len(mg), 2), "max_gap_max": round(max(mg), 2),
                         "no_words": sum(i["no_words"] for i in g),
                         "wps": round(sum(i["words_per_s"] for i in g) / len(g), 2),
                         "sim": round(sum(sm) / len(sm), 3) if sm else None})
    out = Path(__file__).resolve().parent.parent / "data/eval/flowstack_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("| system | ref | n | WER | gap>=2s | gap>=1s | longest gap mean / max | no words | words/s | SIM |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: ([k for k, _ in REFS].index((r["set"], r["ref"])), [k for k, _ in SYSTEMS].index(r["key"]))):
        print(f"| {r['system']} | {r['ref']} | {r['n']} | {r['wer']} | {r['clips_gap2']} | {r['clips_gap1']} | "
              f"{r['max_gap_mean']} / {r['max_gap_max']} | {r['no_words']} | {r['wps']} | {r['sim']} |")
    print("\npooled by system (bad refs = svh1-6 + all laptop refs, ctrl excluded):")
    for k, name in SYSTEMS:
        rr = [r for r in rows if r["key"] == k and r["ref"] != "ctrl"]
        if rr:
            n = sum(r["n"] for r in rr)
            print(f"  {name:24s} n={n:3d}  gap>=2s {sum(r['clips_gap2'] for r in rr):3d}/{n}  "
                  f"gap>=1s {sum(r['clips_gap1'] for r in rr):3d}/{n}  no-words {sum(r['no_words'] for r in rr)}  "
                  f"WER {sum((r['wer'] or 0) * r['n'] for r in rr) / n:.3f}")


if __name__ == "__main__":
    main()
