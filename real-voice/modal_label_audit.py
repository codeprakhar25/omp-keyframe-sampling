"""Label audit: are the IV event tags actually AUDIBLE in the cut?

Every v8 training label came from a transcriber typing <tsk> in IndicVoices
verbatim text. TAG_FREEZE.md already warned "event != always audible", and the
pack was built catalog-only, never re-checked by ear.

If a large share of cuts have no audible event, one cause explains four
independent findings:

  * d_wrong small           - cannot learn sigh vs tsk when the audio is the same
  * more data does not help - more rows at the same noise ratio
  * row count does not predict per-tag learning
  * the tags that DID learn (laugh .092, throat_clearing .070) are the hardest
    to mislabel; the ones that failed (tsk .041, gasp .044) are the easiest

This samples N random cuts per tag and pulls the wavs so they can be judged by
ear. The output feeds data/label-audit.html.

  modal run modal_label_audit.py --per-tag 20
  modal volume get real-voice-iv-cuts /audit_v8 ./data/samples/audit_v8
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import modal

APP = "real-voice-label-audit"
VOL_MIO = "real-voice-mio"
VOL_CUTS = "real-voice-iv-cuts"
MIO = "/mio"
CUTS = "/cuts"
DATA_DIR = f"{MIO}/data_v8"
AUDIT_DIR = f"{CUTS}/audit_v8"

image = modal.Image.debian_slim(python_version="3.12").pip_install("soundfile", "numpy")

app = modal.App(APP, image=image)
vol_mio = modal.Volume.from_name(VOL_MIO, create_if_missing=False)
vol_cuts = modal.Volume.from_name(VOL_CUTS, create_if_missing=False)


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts}, timeout=60 * 60, memory=16384
)
def build(per_tag: int = 20, seed: int = 11) -> dict:
    vol_mio.reload()
    vol_cuts.reload()

    rows = [
        json.loads(l)
        for l in Path(f"{DATA_DIR}/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    tagged = [r for r in rows if (r.get("tag1") or "none") not in ("", "none")]
    by_tag: dict[str, list] = {}
    for r in tagged:
        by_tag.setdefault(r["tag1"], []).append(r)
    print({k: len(v) for k, v in sorted(by_tag.items())}, flush=True)

    rng = random.Random(seed)
    out = Path(AUDIT_DIR)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    clips, missing = [], []
    for tag, rs in sorted(by_tag.items()):
        pick = rs[:] if len(rs) <= per_tag else rng.sample(rs, per_tag)
        for r in pick:
            src = Path(CUTS, r["audio"])  # e.g. hindi_cuts/<id>.wav
            if not src.is_file():
                missing.append(r["id"])
                continue
            dest = out / f"{r['id']}.wav"
            shutil.copy(src, dest)
            clips.append(
                {
                    "id": r["id"],
                    "tag": tag,
                    "wav": f"audit_v8/{r['id']}.wav",
                    "caption": r["caption"],
                    "text": r.get("text", ""),
                    "cut_dur_s": r.get("cut_dur_s"),
                    "source_id": r.get("source_id"),
                    # to be filled by ear:
                    "audible": "",   # y | n | unsure
                    "actually": "",  # if n: what IS there (pause / nothing / other tag)
                    "notes": "",
                }
            )

    rng.shuffle(clips)  # judge blind to tag ordering
    meta = {
        "version": 1,
        "pack": "iv_hindi_v8_inline",
        "per_tag": per_tag,
        "seed": seed,
        "n": len(clips),
        "missing_wav": len(missing),
        "by_tag": dict(Counter(c["tag"] for c in clips)),
        "question": (
            "Is the tagged event ACTUALLY AUDIBLE in this clip? Judge the audio "
            "only. The caption is shown so you know what to listen for — it is "
            "not evidence."
        ),
        "answers": {
            "y": "event clearly audible",
            "n": "not there — mark what IS there in `actually`",
            "unsure": "cannot tell",
        },
        "why": (
            "Per-tag precision from this audit gets correlated against the "
            "measured d_wrong per tag. If they track, the ceiling is label "
            "noise and no objective/rank/base change moves it."
        ),
        "clips": clips,
    }
    Path(CUTS, "audit_v8_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    vol_cuts.commit()
    print(f"wrote {len(clips)} clips to {AUDIT_DIR}; missing={len(missing)}", flush=True)
    return {k: v for k, v in meta.items() if k != "clips"}


@app.local_entrypoint()
def main(per_tag: int = 20, seed: int = 11):
    print(json.dumps(build.remote(per_tag=per_tag, seed=seed), indent=2))
