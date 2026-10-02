#!/usr/bin/env python3
"""DNSMOS census of the v8 training sources (user rule 2026-10-02: DNSMOS on every source, per-source table, tier
by source). Trainset eval/neu_air_trainset_v8a2 is the superset (v8 C sources + IV-R ivr_t1 / ivr_t1v).

Rows hold NeuCodec codes, not source audio, so clips are DECODED (24 kHz -> 16 kHz) and scored: every source pays
the same codec round trip, so the per-source ranking is fair, but absolute DNSMOS reads a little under raw audio.
Sample: per (source, lang) up to --per-group clips, speakers round-robin (so one big speaker can't dominate),
seeded. Scores (aws_iv_quality_sample.Scorer): DNSMOS P.835 SIG / BAK / OVRL, brouhaha SNR / C50 / speech ratio,
plus energy pause stats.

Phase "dec" (venv312, neucodec) writes 16 kHz npy; phase "score" re-execs into venv_ivq (brouhaha + DNSMOS).
Output: <prefix>/clips.jsonl and <prefix>/summary.json (per source x lang: n, median / p10 / p90 of OVRL, SIG,
BAK, SNR, share of clips with OVRL >= 2.8 / 3.0 / 3.2 / 3.4).

  aws_dnsmos_census.py [--trainset eval/neu_air_trainset_v8a2] [--per-group 400]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BUCKET = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
REGION = os.environ.get("RV_REGION", "us-east-1")
WORK = Path(os.environ.get("RV_WORK", "/opt/rv"))
OUT = WORK / "dnsmos_census"
GATES = (2.8, 3.0, 3.2, 3.4)


def aws(*a, check=True):
    return subprocess.run(["aws", *a, "--region", REGION, "--only-show-errors"], check=check)


def phase_dec(a):
    import numpy as np
    import torch
    import torchaudio
    from neucodec import NeuCodec
    parts_dir = OUT / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    aws("s3", "sync", f"s3://{BUCKET}/{a.trainset}/", str(parts_dir), "--exclude", "*", "--include", "part-*.jsonl")
    parts = sorted(parts_dir.glob("part-*.jsonl"))
    groups = defaultdict(lambda: defaultdict(list))   # (source, lang) -> speaker -> [ids]
    for p in parts:   # pass 1: metadata only
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("split") != "train" or not r.get("duration_s") or r["duration_s"] < 1.0:
                continue
            groups[(r.get("source"), r.get("lang"))][str(r.get("speaker"))].append(r["id"])
    rng = random.Random(a.seed)
    want = {}
    for (src, lang), spk in sorted(groups.items()):
        pools = {s: rng.sample(ids, len(ids)) for s, ids in sorted(spk.items())}
        order = sorted(pools)
        rng.shuffle(order)
        picked = []
        while len(picked) < a.per_group and any(pools.values()):
            for s in order:
                if pools[s] and len(picked) < a.per_group:
                    picked.append(pools[s].pop())
        for i in picked:
            want[i] = (src, lang)
        print(f"  {src}/{lang}: {len(picked)} of {sum(len(v) for v in spk.values())} rows, {len(spk)} speakers",
              flush=True)
    print(f"sample: {len(want)} clips in {len(groups)} groups", flush=True)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to("cuda")
    (OUT / "a16").mkdir(exist_ok=True)
    meta = open(OUT / "meta.jsonl", "w", encoding="utf-8")
    n = 0
    for p in parts:   # pass 2: decode the picked rows
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in want:
                continue
            with torch.no_grad():
                y = codec.decode_code(torch.tensor(r["codes"], dtype=torch.long)[None, None, :].cuda())[0, 0]
                y = torchaudio.functional.resample(y.float().cpu(), 24000, 16000).numpy()
            np.save(OUT / "a16" / f"{n}.npy", (np.clip(y, -1, 1) * 32767).astype(np.int16))
            meta.write(json.dumps({"n": n, "id": r["id"], "source": r.get("source"), "lang": r.get("lang"),
                                   "speaker": r.get("speaker"), "dur": r.get("duration_s"),
                                   "emotion": r.get("emotion"), "text": (r.get("text") or "")[:200]},
                                  ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  decoded {n}/{len(want)}", flush=True)
    meta.close()


def pct(v, q):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 3) if v else None


def phase_score(a):
    import numpy as np
    import urllib.request
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aws_iv_quality_sample import DNSMOS_URL, Scorer, audio_stats
    dn = WORK / "sig_bak_ovr.onnx"
    if not dn.exists():
        urllib.request.urlretrieve(DNSMOS_URL, dn)
    sc = Scorer(str(dn))
    meta = [json.loads(l) for l in open(OUT / "meta.jsonl", encoding="utf-8")]
    rows = []
    with open(OUT / "clips.jsonl", "w", encoding="utf-8") as f:
        for k, m in enumerate(meta):
            x = np.load(OUT / "a16" / f"{m['n']}.npy").astype(np.float32) / 32767
            try:
                s = {**sc.score(x), **audio_stats(x, 16000)}
            except Exception as e:
                s = {"err": type(e).__name__}
            rows.append({**m, **s})
            f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            if k % 1000 == 0:
                print(f"  scored {k}/{len(meta)}", flush=True)
    by = defaultdict(list)
    for r in rows:
        if "ovrl" in r:
            by[f"{r['source']}/{r['lang']}"].append(r)
    summ = {}
    for g, rr in sorted(by.items()):
        o = [r["ovrl"] for r in rr]
        summ[g] = {"n": len(rr), "speakers": len({r["speaker"] for r in rr}),
                   **{f"ovrl_{q}": pct(o, p) for q, p in (("p10", .1), ("med", .5), ("p90", .9))},
                   "sig_med": pct([r["sig"] for r in rr], .5), "bak_med": pct([r["bak"] for r in rr], .5),
                   "snr_med": pct([r["snr"] for r in rr if "snr" in r], .5),
                   **{f"pass_{t}": round(sum(x >= t for x in o) / len(o), 3) for t in GATES}}
    (OUT / "summary.json").write_text(json.dumps({"trainset": a.trainset, "per_group": a.per_group,
                                                  "decoded": True, "groups": summ}, indent=1))
    for fn in ("clips.jsonl", "summary.json"):
        aws("s3", "cp", str(OUT / fn), f"s3://{BUCKET}/{a.prefix}/{fn}")
    print("| source/lang | n | spk | OVRL p10 / med / p90 | SIG | BAK | SNR | >=3.0 | >=3.2 |", flush=True)
    for g, s in summ.items():
        print(f"| {g} | {s['n']} | {s['speakers']} | {s['ovrl_p10']} / {s['ovrl_med']} / {s['ovrl_p90']} | "
              f"{s['sig_med']} | {s['bak_med']} | {s['snr_med']} | {s['pass_3.0']} | {s['pass_3.2']} |", flush=True)
    print(f"DNSMOS_CENSUS_DONE clips={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", default="eval/neu_air_trainset_v8a2")
    ap.add_argument("--per-group", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1002)
    ap.add_argument("--phase", default="dec", choices=("dec", "score"))
    ap.add_argument("--prefix", default="eval/data_audit_v1/dnsmos_census_v8")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    argv = ["--trainset", a.trainset, "--per-group", str(a.per_group), "--seed", str(a.seed), "--phase", "score",
            "--prefix", a.prefix]
    if a.phase == "dec":
        phase_dec(a)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from aws_iv_quality_sample import VENV, ensure_venv
        py = VENV / "bin" / "python"
        if not py.exists():
            sys.argv = [__file__, *argv]
            ensure_venv()   # builds venv_ivq, then re-execs this argv inside it
        os.execv(str(py), [str(py), __file__, *argv])
    phase_score(a)


if __name__ == "__main__":
    main()
