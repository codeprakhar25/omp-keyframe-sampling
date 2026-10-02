#!/usr/bin/env python3
"""SIM for the English bad-ref test (user ask 2026-09-28): cosine of WavLM-base-plus-sv x-vectors between each
generated clip in s3 eval/incumbent_v1/inc_<arm>/wav and the reference it cloned (ref/<ref>.wav, named by the run's
inc_<arm>.json "ref"). cross = the same clip against every OTHER reference in the batch (the no-match floor).
svh_* ref wavs are NeuCodec decodes of the codes the model was prompted with, so for them this is SIM-r.

  aws_fxen_sim.py --arms fxen_a_svh1_s7,... --out fxensim_a -> s3 eval/incumbent_v1/fxensim_a.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

B, R, P = "real-voice-mio-692707725608", "us-east-1", "eval/incumbent_v1"
SV_ID = "microsoft/wavlm-base-plus-sv"


def aws(*a):
    return subprocess.run(["aws", *a, "--region", R, "--only-show-errors"], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, WavLMForXVector
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SV_ID)
    model = WavLMForXVector.from_pretrained(SV_ID).eval().to(dev)

    def emb(path):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        w = torch.from_numpy(np.ascontiguousarray(y.mean(1)))[None]
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        inp = fe(w.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            e = model(**{k: v.to(dev) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(e, dim=-1).squeeze(0).cpu()

    work = Path("/tmp/fxen_sim")
    (work / "ref").mkdir(parents=True, exist_ok=True)
    aws("s3", "sync", f"s3://{B}/{P}/ref/", str(work / "ref"), "--exclude", "*", "--include", "*.wav")
    gens, ref_e = [], {}
    for arm in a.arms.split(","):
        d = work / arm
        d.mkdir(exist_ok=True)
        aws("s3", "sync", f"s3://{B}/{P}/inc_{arm}/wav/", str(d))
        meta = json.loads(subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/inc_{arm}.json", "-", "--region", R],
                                         capture_output=True, text=True).stdout)
        ref = meta["ref"].split("=", 1)[-1]
        # Frontend arms (2026-10-02) prompt with a cleaned copy (<ref>_fe / _fedn / _rw); voice similarity is still
        # judged against the ORIGINAL reference, the voice the user asked for.
        ref = re.sub(r"_(fe|fedn|rw)$", "", ref)
        if ref not in ref_e:
            ref_e[ref] = emb(work / "ref" / f"{ref}.wav")
        for it in meta["items"]:
            f = d / Path(it["file"]).name
            if f.exists():
                gens.append({"arm": arm, "id": it["id"], "ref": ref, "e": emb(f)})
    rows = []
    for g in gens:
        sim = float(g["e"] @ ref_e[g["ref"]])
        xs = [float(g["e"] @ e) for r, e in ref_e.items() if r != g["ref"]]
        rows.append({"arm": g["arm"], "id": g["id"], "ref": g["ref"], "sim": round(sim, 4),
                     "cross": round(sum(xs) / len(xs), 4) if xs else None})
    by = {}
    for r in rows:
        by.setdefault(r["ref"], []).append(r)
    summ = {ref: {"n": len(v), "sim": round(sum(x["sim"] for x in v) / len(v), 4),
                  "cross": round(sum(x["cross"] for x in v if x["cross"] is not None) / len(v), 4)}
            for ref, v in sorted(by.items())}
    out = work / f"{a.out}.json"
    out.write_text(json.dumps({"model": SV_ID, "by_ref": summ, "items": rows}, indent=1))
    aws("s3", "cp", str(out), f"s3://{B}/{P}/{a.out}.json")
    print(f"FXENSIM {a.out} {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
