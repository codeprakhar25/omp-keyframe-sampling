"""HiACC full corpus: Zenodo -> Modal volume -> manifest joined to transcripts.

HiACC (Hinglish Adult & Children Code-switched corpus), Zenodo record 15551669,
CC-BY-4.0, open access. One 531.6 MB Corpus.zip.

Why: we hold 5,176 human transcripts + code-switch labels but only 30 local
wavs. Those 5,176 aligned pairs are an ASR eval set, an ASR fine-tuning set for
code-switched Hindi-English, and the Hinglish audio the train pack is short of
(currently 29 clips, 6 with text).

  modal run modal_hiacc.py --action fetch      # download + unzip, print tree
  modal run modal_hiacc.py --action manifest   # join audio <-> transcripts
  modal run modal_hiacc.py --action stats
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "real-voice-hiacc"
VOL = "real-voice-hiacc"
ROOT = "/hiacc"
ZIP_URL = "https://zenodo.org/api/records/15551669/files/Corpus.zip/content"
RECORD = "https://zenodo.org/records/15551669"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "libsndfile1")
    .pip_install("soundfile", "numpy", "tqdm")
)

app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOL, create_if_missing=True)


@app.function(volumes={ROOT: vol}, timeout=60 * 60 * 2, memory=8192)
def fetch(force: bool = False) -> dict:
    import subprocess

    vol.reload()
    zp = Path(ROOT, "Corpus.zip")
    raw = Path(ROOT, "raw")

    if zp.is_file() and not force:
        print(f"zip already present ({zp.stat().st_size/1e6:.1f} MB)", flush=True)
    else:
        print(f"downloading {ZIP_URL}", flush=True)
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3", "-o", str(zp), ZIP_URL],
            check=True,
        )
        print(f"downloaded {zp.stat().st_size/1e6:.1f} MB", flush=True)
        vol.commit()

    if raw.exists() and not force:
        print("already extracted", flush=True)
    else:
        raw.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", str(zp), "-d", str(raw)], check=True)
        vol.commit()

    wavs = list(raw.rglob("*.wav"))
    others = [p for p in raw.rglob("*") if p.is_file() and p.suffix != ".wav"]
    dirs: dict[str, int] = {}
    for p in wavs:
        dirs[str(p.parent.relative_to(raw))] = dirs.get(str(p.parent.relative_to(raw)), 0) + 1

    print(f"\nwavs={len(wavs)}  other files={len(others)}", flush=True)
    print("\nwav counts by directory:", flush=True)
    for d, n in sorted(dirs.items(), key=lambda x: -x[1]):
        print(f"  {n:>6}  {d}", flush=True)
    print("\nnon-wav files (first 30):", flush=True)
    for p in others[:30]:
        print(f"  {p.relative_to(raw)}  {p.stat().st_size/1e3:.0f} KB", flush=True)

    vol.commit()
    return {"n_wav": len(wavs), "n_other": len(others), "dirs": dirs}


@app.function(volumes={ROOT: vol}, timeout=60 * 60 * 2, memory=16384)
def manifest() -> dict:
    """Join wavs to the human transcripts + code-switch labels.

    Label files are expected at /hiacc/labels/{adult,child}.json — push them with
    `modal volume put real-voice-hiacc <local> /labels/<name>.json` if the zip
    does not already carry them.
    """
    import shutil

    import soundfile as sf

    vol.reload()
    # 5,176 individual opens over volume FUSE is the bottleneck (killed a 12 min
    # run). Bulk-copy to local disk once, then stat headers at local speed.
    raw = Path("/tmp/hiacc_raw")
    if not raw.exists():
        print("staging raw/ to local disk ...", flush=True)
        shutil.copytree(Path(ROOT, "raw"), raw)
        print("staged", flush=True)

    # label sources: prefer ones pushed to /labels, else anything in the zip
    idx: dict[str, dict] = {}
    srcs = sorted(Path(ROOT, "labels").glob("*.json")) if Path(ROOT, "labels").is_dir() else []
    srcs += sorted(raw.rglob("code_switched_labels.json"))
    for p in srcs:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {p}: {e}", flush=True)
            continue
        # NB: match on the PATH, not the stem — "code_switched_labels" contains "ch"
        low = str(p).lower()
        grp = "child" if ("children" in low or "child" in low or "hiacc-ch" in low) else "adult"
        for r in rows:
            k = r.get("audio") or r.get("audio_filepath") or ""
            if k:
                idx.setdefault(os.path.basename(k), {"set": grp, **r})
    print(f"label rows indexed: {len(idx)} from {len(srcs)} file(s)", flush=True)

    wavs = sorted(raw.rglob("*.wav"))
    rows, miss_label, bad = [], [], []
    total_s = 0.0
    for p in wavs:
        hit = idx.get(p.name)
        if not hit:
            miss_label.append(p.name)
            continue
        try:
            info = sf.info(str(p))
            dur = info.frames / info.samplerate
        except Exception as e:
            bad.append(f"{p.name}: {e}")
            continue
        total_s += dur
        rows.append(
            {
                "id": p.stem,
                "audio": str(p.relative_to(raw)),
                "set": hit["set"],
                "text": (hit.get("transcription") or "").strip(),
                "cs_label": hit.get("label"),
                "duration_s": round(dur, 3),
                "sr": info.samplerate,
                "source": "hiacc",
                "license": "CC-BY-4.0",
                "record": RECORD,
            }
        )

    out = Path(ROOT, "manifest.jsonl")
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    stats = {
        "n_wav_found": len(wavs),
        "n_joined": len(rows),
        "n_label_rows": len(idx),
        "n_wav_without_label": len(miss_label),
        "n_label_without_wav": len(idx) - len(rows),
        "n_unreadable": len(bad),
        "hours": round(total_s / 3600, 3),
        "by_set": dict(Counter(r["set"] for r in rows)),
        "by_cs_label": dict(Counter(r["cs_label"] for r in rows)),
        "empty_text": sum(1 for r in rows if not r["text"]),
        "sample_rates": dict(Counter(r["sr"] for r in rows)),
        "manifest": str(out),
    }
    Path(ROOT, "manifest_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    vol.commit()
    print(json.dumps(stats, indent=2), flush=True)
    if miss_label[:10]:
        print("wavs with no label (first 10):", miss_label[:10], flush=True)
    return stats


@app.function(volumes={ROOT: vol}, timeout=60 * 10)
def stats() -> dict:
    vol.reload()
    p = Path(ROOT, "manifest_stats.json")
    if not p.is_file():
        return {"error": "no manifest yet — run --action manifest"}
    d = json.loads(p.read_text())
    print(json.dumps(d, indent=2), flush=True)
    return d


@app.local_entrypoint()
def main(action: str = "fetch", force: bool = False):
    if action == "fetch":
        r = fetch.remote(force=force)
    elif action == "manifest":
        r = manifest.remote()
    else:
        r = stats.remote()
    print(json.dumps(r, indent=2)[:3000])
