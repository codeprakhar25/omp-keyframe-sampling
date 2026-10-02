#!/usr/bin/env python3
"""Blind ear page for the C inference frontend (2026-10-02): does ref cleaning / denoise / gap cap sound better?

Arms (all v8 C final, seed 7, same refs and lines; only the ref prep and the output post-process differ):
  c       C raw (stored ref codes, no post-process)
  c_gc    C raw + output gap cap
  cfe_gc  C + cleaned ref (VAD trim, pauses capped, loudness) + gap cap
  cfd_gc  C + cleaned + denoised ref + gap cap
Seed 7 only (no best-of-3 pick, so no cherry picking). Lines rotate across refs so each card is a new sentence.
Suites: Indian-English refs (Svarah, fxen), laptop refs speaking English, laptop refs speaking Hindi; 6 cards each.
The card plays the ORIGINAL ref (what a user would upload). Clips are staged into opaque dirs, then
build_v7_blind_page.py makes the page (per-card shuffle, md5 refusal, autosave to data/earpass/cfront_v1.json).

  python3 scripts/build_cfront_page.py [--unblind]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAIN = Path("/home/prakh/ml-resarch/real-voice")   # serve_lab serves MAIN/data
sys.path.insert(0, str(MAIN / "scripts"))
import build_v7_blind_page as bp  # noqa: E402

B, P, R = "real-voice-mio-692707725608", "eval/incumbent_v1", "us-east-1"
PROBE = "cfront_v1"
CKPT = "runs/air_v8_c/ckpt_final"
ARMS = {"c": ("c", ""), "c_gc": ("c", "_gc"), "cfe_gc": ("cfe", "_gc"), "cfd_gc": ("cfd", "_gc")}
NAMES = {"c": "C raw", "c_gc": "C + gap cap", "cfe_gc": "C + clean ref + gap cap",
         "cfd_gc": "C + denoised ref + gap cap"}
# (suite, set, ref key, ref wav, line)
CARDS = [("cf_svh", "fxen", f"svh{i}", f"svh_{i}", ("en_1", "en_2", "en_3")[(i - 1) % 3]) for i in range(1, 7)] + \
        [("cf_pk_en", "fxpk", k, f"pk_{k}", ("en_1", "en_2", "en_3")[j % 3])
         for j, k in enumerate(("en1", "en2", "hi1", "hi2", "mx1", "mx2"))] + \
        [("cf_pk_hi", "fxpk", k, f"pk_{k}", ("hi_1", "hi_3")[j % 2])
         for j, k in enumerate(("en1", "en2", "hi1", "hi2", "mx1", "mx2"))]
bp.SUITES += [
    ("cf_svh", "English, Indian-accent refs",
     "Noisy / pausey Svarah refs. Natural? Any dead air or mumbling? Every word there? Sounds like the ref?"),
    ("cf_pk_en", "English, laptop refs", "Everyday laptop-mic refs speaking English: natural, clear, ref voice?"),
    ("cf_pk_hi", "Hindi, laptop refs", "Same laptop refs speaking Hindi: natural, every word there, no cut-off ending?"),
]


def opaque(arm: str) -> str:
    return "cf1_" + hashlib.sha256(f"{PROBE}|{arm}|salt-2026-10-02".encode()).hexdigest()[:8]


def s3json(key):
    r = subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/{key}", "-", "--region", R], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def stage(arm: str, refdir: Path) -> Path:
    d = MAIN / "data/samples" / opaque(arm)
    if (d / "items.json").exists():   # rebuilds (e.g. --unblind) must not refetch
        return d
    k, suf = ARMS[arm]
    items = []
    for suite, pre, rk, rf, line in CARDS:
        run = f"{pre}_{k}_{rk}_s7{suf}"
        meta = s3json(f"inc_{run}.json")
        if not meta:
            raise SystemExit(f"missing inc_{run}.json")
        if meta.get("ckpt") != CKPT:
            raise SystemExit(f"{run}: ckpt {meta.get('ckpt')} != {CKPT}")
        text = {i["id"]: i["text"] for i in meta["items"]}[line]
        (d / "w").mkdir(parents=True, exist_ok=True)
        (d / "r").mkdir(exist_ok=True)
        f = d / "w" / f"{suite}_{rk}.wav"
        subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/inc_{run}/wav/{line}.wav", str(f), "--region", R,
                        "--only-show-errors"], check=True)
        shutil.copy(refdir / f"{rf}.wav", d / "r" / f"{rk}.wav")
        items.append({"id": f"{suite}_{rk}", "suite": suite, "text": text, "file": f"w/{suite}_{rk}.wav",
                      "ref_file": f"r/{rk}.wav"})
    (d / "items.json").write_text(json.dumps({"items": items, "ckpt": arm}, ensure_ascii=False, indent=1))
    print(f"{arm}: {len(items)} clips -> {d.name}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unblind", action="store_true")
    a = ap.parse_args()
    refdir = MAIN / "data/samples/cf1_refs"
    refdir.mkdir(parents=True, exist_ok=True)
    for rf in sorted({c[3] for c in CARDS}):
        subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/ref/{rf}.wav", str(refdir / f"{rf}.wav"), "--region", R,
                        "--only-show-errors"], check=True)
    dirs = {arm: stage(arm, refdir) for arm in ARMS}
    # per-card md5: identical clips across arms are expected only when the gap cap had nothing to cut (c vs c_gc)
    for cid in (f"{c[0]}_{c[2]}" for c in CARDS):
        h = {arm: hashlib.md5((d / "w" / f"{cid}.wav").read_bytes()).hexdigest() for arm, d in dirs.items()}
        dup = [(x, y) for i, x in enumerate(h) for y in list(h)[i + 1:] if h[x] == h[y]]
        if dup:
            print(f"  identical on {cid}: {dup}")
    sys.argv = [bp.__file__, PROBE, *(f"{arm}={d}" for arm, d in dirs.items())]
    if a.unblind:
        sys.argv += ["--unblind", ",".join(f"{k}={v}" for k, v in NAMES.items())]
    return bp.main()


if __name__ == "__main__":
    sys.exit(main())
