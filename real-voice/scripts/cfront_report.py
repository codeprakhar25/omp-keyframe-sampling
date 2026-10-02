#!/usr/bin/env python3
"""Report for v8 C + inference frontend v0 (2026-10-02). Same bad refs, lines and seeds 7/11/23 for every system.

Systems (arm = <set>_<key>_<ref>_s<seed><suffix>):
  sh stock Air | a v8 A | ix IndexTTS2 | c v8 C raw | c+gc C, dead air capped after decode
  cfe C, cleaned ref | cfe+gc cleaned ref + gap cap | cfd C, denoised + cleaned ref | cfd+gc
Per system, pooled over the bad refs (Svarah svh1-6 + the user's laptop refs; the LibriTTS ctrl is its own row):
  word-gap dead air from Whisper timings (clips with a >=2 s / >=1 s gap), WER (Whisper en, IndicConformer hi), SIM.
"best of 3" = per ref x line, keep the seed with the fewest gaps then lowest WER: what a regenerate-on-ASR-check
loop with 3 tries would ship. It picks with the same ASR that scores it, so it is an upper bound for that loop.

  python scripts/cfront_report.py [--lang en|hi]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

B, R, P = "real-voice-mio-692707725608", "us-east-1", "eval/incumbent_v1"
CACHE = Path("/tmp/cfront_report_cache")
SYSTEMS = [("sh", "", "stock Air"), ("a", "", "v8 A"), ("ix", "", "IndexTTS2"), ("c", "", "v8 C raw"),
           ("c", "_gc", "C + gap cap"), ("cfe", "", "C + clean ref"), ("cfe", "_gc", "C + clean ref + gap cap"),
           ("cfd", "", "C + denoise ref"), ("cfd", "_gc", "C + denoise ref + gap cap"),
           ("crw", "", "C + re-encoded ref (control)")]
REFS = [("fxen", r) for r in ("ctrl", "svh1", "svh2", "svh3", "svh4", "svh5", "svh6")] + \
       [("fxpk", r) for r in ("en1", "en2", "hi1", "hi2", "mx1", "mx2")]
SEEDS = (7, 11, 23)


def sync():
    subprocess.run(["aws", "s3", "sync", f"s3://{B}/{P}/", str(CACHE), "--region", R, "--only-show-errors",
                    "--exclude", "*", "--include", "wgap_fx*.json", "--include", "whisper_wer_fx*.json",
                    "--include", "icf_wer_fx*.json", "--include", "fx*sim_*.json"], check=True)


def load(key):
    f = CACHE / key
    return json.loads(f.read_text()) if f.exists() else None


def lang_of(item_id: str) -> str:
    return "en" if item_id.startswith("en") else "hi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    sims = {}
    for f in sorted(CACHE.glob("fx*sim_*.json")):
        for it in json.loads(f.read_text()).get("items", []):
            sims[(it["arm"], it["id"])] = it["sim"]
    werf = "whisper_wer" if a.lang == "en" else "icf_wer"
    metric = "wer" if a.lang == "en" else "gcer"   # Hindi: IndicConformer normalised CER
    rows, best = [], []
    for k, suf, name in SYSTEMS:
        for pre, ref in REFS:
            clips = {}   # (line id) -> [per-seed dict]
            for s in SEEDS:
                arm = f"{pre}_{k}_{ref}_s{s}{suf}"
                g = {i["id"]: i for i in (load(f"wgap_{arm}.json") or {}).get("items", []) if lang_of(i["id"]) == a.lang}
                w = {i["id"]: i.get(metric) for i in (load(f"{werf}_{arm}.json") or {}).get("items", [])}
                for iid, gi in g.items():
                    clips.setdefault(iid, []).append({**gi, "wer": w.get(iid), "sim": sims.get((arm, iid)), "seed": s})
            if not clips:
                continue
            allc = [c for v in clips.values() for c in v]
            pick = [min(v, key=lambda c: (c["n_gap2"], c["n_gap1"], c["wer"] if c["wer"] is not None else 9,
                                          c["max_gap"])) for v in clips.values()]
            for tag, cs, out in (("all", allc, rows), ("best3", pick, best)):
                ws = [c["wer"] for c in cs if c["wer"] is not None]
                ss = [c["sim"] for c in cs if c["sim"] is not None]
                out.append({"system": name, "key": k + suf, "set": pre, "ref": ref, "n": len(cs),
                            "gap2": sum(c["n_gap2"] > 0 for c in cs), "gap1": sum(c["n_gap1"] > 0 for c in cs),
                            "no_words": sum(c["no_words"] for c in cs),
                            "wer": round(sum(ws) / len(ws), 3) if ws else None,
                            "sim": round(sum(ss) / len(ss), 3) if ss else None,
                            "wps": round(sum(c["words_per_s"] for c in cs) / len(cs), 2),
                            "max_gap_max": round(max(c["max_gap"] for c in cs), 2)})
    out = Path(f"/home/prakh/ml-resarch/real-voice/data/eval/cfront_report_{a.lang}.json")
    out.write_text(json.dumps({"all": rows, "best3": best}, indent=1))

    def pooled(rr, label):
        print(f"\n{label} ({a.lang} lines; bad refs = svh1-6 + laptop refs, ctrl excluded)")
        print(f"| system | clips | gap>=2s | gap>=1s | no words | {'WER' if a.lang == 'en' else 'CER'} | words/s | SIM |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, suf, name in SYSTEMS:
            r = [x for x in rr if x["key"] == k + suf and x["ref"] != "ctrl"]
            if not r:
                continue
            n = sum(x["n"] for x in r)
            wer = [x["wer"] * x["n"] for x in r if x["wer"] is not None]
            sim = [x["sim"] * x["n"] for x in r if x["sim"] is not None]
            nw = sum(x["n"] for x in r if x["wer"] is not None)
            ns = sum(x["n"] for x in r if x["sim"] is not None)
            print(f"| {name} | {n} | {sum(x['gap2'] for x in r)} | {sum(x['gap1'] for x in r)} | "
                  f"{sum(x['no_words'] for x in r)} | {sum(wer) / nw if nw else float('nan'):.3f} | "
                  f"{sum(x['wps'] * x['n'] for x in r) / n:.2f} | {sum(sim) / ns if ns else float('nan'):.3f} |")
        c = [x for x in rr if x["ref"] == "ctrl"]
        if c:
            print("ctrl (clean LibriTTS ref): " + "; ".join(f"{x['system']} gap2 {x['gap2']}/{x['n']} WER {x['wer']}"
                                                            for x in c))
    pooled(rows, "all seeds")
    pooled(best, "best of 3 seeds")


if __name__ == "__main__":
    main()
