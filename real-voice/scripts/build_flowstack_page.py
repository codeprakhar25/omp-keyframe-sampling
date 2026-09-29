#!/usr/bin/env python3
"""Blind ear page for the flow-stack probe (2026-09-29): same bad / everyday refs, same lines, seed 7, arms hidden.

Arms: sh stock Air (our harness), a v8 A final, ix IndexTTS2, cz CosyVoice3 ref-in-LM, cx CosyVoice3 ref-in-flow.
Lines en_1 + en_2 per ref (in both items_fxen and items_fxpk). Svarah refs (fxen) and the user's laptop refs (fxpk).
Clips are copied into opaque dirs data/samples/fs1_<hash>/ (src paths never name the system), then the page is built
by build_v7_blind_page.py (per-card shuffle, md5 guard against identical arms, autosave to data/earpass/flowstack_v1.json).
Arms not on S3 yet are skipped; pass them later as late arms so earlier slots and verdicts stay put.

  python3 scripts/build_flowstack_page.py --set fxen --arms sh,a,ix [--late cz,cx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAIN = Path("/home/prakh/ml-resarch/real-voice")   # the lab server serves MAIN/data
B, P, R = "real-voice-mio-692707725608", "eval/incumbent_v1", "us-east-1"
PROBE = "flowstack_v1"   # + "_<set>" at runtime: one page per ref set, so a set's arms are all present before its build
REFS = [("fxen", k, f) for k, f in (("svh1", "svh_1"), ("svh2", "svh_2"), ("svh3", "svh_3"), ("svh4", "svh_4"),
                                    ("svh5", "svh_5"), ("svh6", "svh_6"), ("ctrl", "libritts_en_m_7127"))] + \
       [("fxpk", k, f"pk_{k}") for k in ("en1", "en2", "hi1", "hi2", "mx1", "mx2")]
LINES = ("en_1", "en_2")
NAMES = {"sh": "stock NeuTTS-Air", "a": "v8 A (our fine-tune)", "ix": "IndexTTS2", "cz": "CosyVoice3 ref-in-LM",
         "cx": "CosyVoice3 ref-in-flow"}


def opaque(arm: str) -> str:
    return "fs1_" + hashlib.sha256(f"{PROBE}|{arm}|salt-2026-09-29".encode()).hexdigest()[:8]


def s3json(key):
    r = subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/{key}", "-", "--region", R], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def stage(arm: str, refdir: Path) -> Path | None:
    d = MAIN / "data/samples" / opaque(arm)
    if (d / "items.json").exists():   # already staged: reuse (rebuilds, e.g. --unblind, must not refetch)
        return d
    items = []
    for pre, rk, rf in REFS:
        run = f"{pre}_{arm}_{rk}_s7"
        meta = s3json(f"inc_{run}.json")
        if not meta:
            continue
        texts = {i["id"]: i["text"] for i in meta["items"]}
        (d / "w").mkdir(parents=True, exist_ok=True)
        for line in LINES:
            if line not in texts:
                continue
            f = d / "w" / f"{rk}_{line}.wav"
            subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/inc_{run}/wav/{line}.wav", str(f), "--region", R,
                            "--only-show-errors"])
            if f.exists():
                (d / "r").mkdir(exist_ok=True)
                shutil.copy(refdir / f"{rf}.wav", d / "r" / f"{rk}.wav")
                items.append({"id": f"{rk}_{line}", "suite": "en", "text": texts[line], "file": f"w/{rk}_{line}.wav",
                              "ref_file": f"r/{rk}.wav"})
    if not items:
        print(f"skip {arm}: nothing on S3 yet")
        return None
    (d / "items.json").write_text(json.dumps({"items": items, "ckpt": arm}, ensure_ascii=False, indent=1))
    print(f"{arm}: {len(items)} clips -> {d.name}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="sh,a,ix")
    ap.add_argument("--late", default="")
    ap.add_argument("--set", default="fxen", choices=["fxen", "fxpk"])
    ap.add_argument("--unblind", action="store_true", help="label every slot with its system, same page + verdicts")
    a = ap.parse_args()
    global PROBE, REFS
    PROBE = f"{PROBE}_{a.set}"
    REFS = [r for r in REFS if r[0] == a.set]
    refdir = MAIN / "data/samples/fs1_refs"
    refdir.mkdir(parents=True, exist_ok=True)
    for _, _, rf in REFS:
        subprocess.run(["aws", "s3", "cp", f"s3://{B}/{P}/ref/{rf}.wav", str(refdir / f"{rf}.wav"), "--region", R,
                        "--only-show-errors"])
    specs = []
    for arm in a.arms.split(","):
        d = stage(arm, refdir)
        if d:
            specs.append(f"{arm}={d}")
    for arm in filter(None, a.late.split(",")):
        d = stage(arm, refdir)
        if d:
            specs.append(f"+{arm}={d}")
    builder = MAIN / "scripts/build_v7_blind_page.py"
    extra = ["--unblind", ",".join(f"{k}={v}" for k, v in NAMES.items())] if a.unblind else []
    return subprocess.run([sys.executable, str(builder), PROBE, *specs, *extra], cwd=MAIN).returncode


if __name__ == "__main__":
    sys.exit(main())
