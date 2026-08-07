"""GPU word-align IV clips (WhisperX) → slice cut wavs for caption proposals.

GPU work: WhisperX **align** (wav2vec2) — force IV plain words onto timeline.
CPU work: soundfile slice (trivial).

Tag lock (gate/emit) lives in scripts/iv_tags.py — regenerates via
scripts/iv_caption_split.py before upload. Locked add-ons: gasp,
throat_clearing, cough, tsk, ugh, sniffle.

  modal run modal_iv_cut_slice.py --action upload
  modal run --detach modal_iv_cut_slice.py --action run
  modal run modal_iv_cut_slice.py --action pull
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

APP = "real-voice-iv-cuts"
VOL = "real-voice-iv-cuts"
ROOT = Path(__file__).resolve().parent
REMOTE = "/vol"

vol = modal.Volume.from_name(VOL, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "tqdm",
    )
    .pip_install("whisperx")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/scripts")
)

app = modal.App(APP, image=image)


@app.function(volumes={REMOTE: vol}, timeout=60 * 10)
def _mkdir() -> None:
    for rel in ("in/wavs", "align_cache", "cuts", "out", "code"):
        Path(REMOTE, rel).mkdir(parents=True, exist_ok=True)
    vol.commit()


@app.function(volumes={REMOTE: vol}, timeout=60 * 30)
def _put(payload: bytes, rel: str) -> str:
    dest = Path(REMOTE) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    vol.commit()
    return str(dest)


@app.function(volumes={REMOTE: vol}, timeout=60 * 60)
def _put_many(items: list[tuple[str, bytes]]) -> int:
    n = 0
    for rel, blob in items:
        dest = Path(REMOTE) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        n += 1
    vol.commit()
    return n


@app.function(
    gpu="A10G",
    timeout=60 * 90,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    memory=16384,
)
def run_align_slice() -> dict:
    import os

    import numpy as np
    import soundfile as sf
    import torch
    import whisperx

    # HF auth (same secret as mio)
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("huggingface")
        or ""
    ).strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        try:
            from huggingface_hub import login

            login(token=tok, add_to_git_credential=False)
        except Exception as e:
            print("hf login warn:", e)

    sys.path.insert(0, "/root/scripts")
    from iv_cut_times import cut_times_from_alignment, plain_words_from_transcript

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    job = json.loads(Path(REMOTE, "in/job.json").read_text(encoding="utf-8"))
    proposals = job["proposals"]
    clips = {c["id"]: c for c in job["clips"]}

    align_dir = Path(REMOTE, "align_cache")
    cuts_dir = Path(REMOTE, "cuts")
    out_dir = Path(REMOTE, "out")
    align_dir.mkdir(parents=True, exist_ok=True)
    cuts_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_src: dict[str, list[dict]] = {}
    for p in proposals:
        by_src.setdefault(p["source_id"], []).append(p)

    # retry model download — Modal DNS flakes happen
    model_a, metadata = None, None
    last_err = None
    for attempt in range(5):
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code="hi", device=device
            )
            break
        except Exception as e:
            last_err = e
            print(f"load_align_model attempt {attempt+1}/5 fail:", e)
            import time

            time.sleep(5 * (attempt + 1))
    if model_a is None:
        raise RuntimeError(f"align model download failed: {last_err}")
    print("align model ready")

    stats = {"clips": 0, "cuts_ok": 0, "cuts_fail": 0, "align_fail": 0}

    for sid, props in by_src.items():
        clip = clips[sid]
        wav_path = Path(REMOTE, "in/wavs", f"{sid}.wav")
        if not wav_path.is_file():
            print("MISSING wav", sid)
            for p in props:
                p["align_ok"] = False
                p["align_reason"] = "missing_wav"
                p["cut_wav"] = ""
            stats["align_fail"] += 1
            continue

        audio, sr = sf.read(str(wav_path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        duration_s = float(len(audio) / sr)

        if sr != 16000:
            import torchaudio

            t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, 16000)
            audio_16 = t.squeeze(0).numpy()
        else:
            audio_16 = np.asarray(audio, dtype=np.float32)

        text = clip.get("text") or ""
        words = plain_words_from_transcript(text)
        plain = " ".join(words)
        cache_fp = align_dir / f"{sid}.json"

        if cache_fp.is_file():
            aligned_doc = json.loads(cache_fp.read_text(encoding="utf-8"))
            print("cache hit", sid, "aligned", aligned_doc.get("n_aligned_words"))
        else:
            segments = [{"text": plain, "start": 0.0, "end": duration_s}]
            try:
                result = whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio_16,
                    device,
                    return_char_alignments=False,
                )
            except Exception as e:
                print("ALIGN FAIL", sid, e)
                stats["align_fail"] += 1
                for p in props:
                    p["align_ok"] = False
                    p["align_reason"] = f"align_exc:{type(e).__name__}"
                    p["cut_wav"] = ""
                continue

            wlist = []
            for seg in result.get("segments") or []:
                for w in seg.get("words") or []:
                    if w.get("start") is None or w.get("end") is None:
                        continue
                    wlist.append(
                        {
                            "word": w.get("word") or w.get("text") or "",
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                    )
            if not wlist:
                for w in result.get("word_segments") or []:
                    if w.get("start") is None or w.get("end") is None:
                        continue
                    wlist.append(
                        {
                            "word": w.get("word") or "",
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                    )
            aligned_doc = {
                "source_id": sid,
                "duration_s": duration_s,
                "n_ref_words": len(words),
                "n_aligned_words": len(wlist),
                "words": wlist,
            }
            cache_fp.write_text(
                json.dumps(aligned_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("aligned", sid, "words", len(wlist), "/", len(words))

        stats["clips"] += 1
        awords = aligned_doc["words"]

        for p in props:
            ct = cut_times_from_alignment(
                text,
                int(p["token_idx"]),
                int(p.get("n_words_left") or 0),
                int(p.get("n_words_right") or 0),
                awords,
                duration_s,
            )
            p["t_start_s"] = ct.get("t_start_s")
            p["t_end_s"] = ct.get("t_end_s")
            p["cut_dur_s"] = ct.get("dur_s")
            p["align_ok"] = bool(ct.get("ok"))
            p["align_reason"] = ct.get("reason") or ""
            if not ct.get("ok"):
                stats["cuts_fail"] += 1
                p["cut_wav"] = ""
                continue

            t0, t1 = float(ct["t_start_s"]), float(ct["t_end_s"])
            i0, i1 = int(t0 * sr), int(t1 * sr)
            i0 = max(0, i0)
            i1 = min(len(audio), max(i0 + 1, i1))
            out_wav = cuts_dir / f"{p['cut_id']}.wav"
            sf.write(str(out_wav), audio[i0:i1], sr)
            p["cut_wav"] = f"scale/cuts/{p['cut_id']}.wav"
            stats["cuts_ok"] += 1

        vol.commit()

    base = job.get("proposals_doc") or {}
    base["proposals"] = proposals
    by_src_out: dict[str, list] = {}
    for p in proposals:
        by_src_out.setdefault(p["source_id"], []).append(p)
    base["by_source_id"] = by_src_out
    base["slice_stats"] = stats
    base["slice_backend"] = "whisperx_align_hi"
    Path(REMOTE, "out/iv_cut_proposals.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(REMOTE, "out/slice_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    vol.commit()
    print("DONE", stats)
    return stats


def _upload_local() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from iv_cut_times import resolve_source_wav

    print("mkdir…")
    _mkdir.remote()

    props_doc = json.loads((ROOT / "data/scale/iv_cut_proposals.json").read_text(encoding="utf-8"))
    drafts = json.loads((ROOT / "data/scale/drafts.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in drafts.get("clips") or []}
    proposals = props_doc.get("proposals") or []
    sids = sorted({p["source_id"] for p in proposals})

    clips_out = []
    missing = []
    batch: list[tuple[str, bytes]] = []
    for sid in sids:
        clip = by_id.get(sid)
        if not clip:
            missing.append(sid)
            continue
        wav = resolve_source_wav(ROOT, clip)
        if not wav:
            missing.append(sid)
            continue
        batch.append((f"in/wavs/{sid}.wav", wav.read_bytes()))
        clips_out.append(
            {"id": sid, "text": clip.get("text") or "", "duration_s": clip.get("duration_s")}
        )
        if len(batch) >= 15:
            _put_many.remote(batch)
            print(f"  uploaded batch, clips so far {len(clips_out)}")
            batch = []
    if batch:
        _put_many.remote(batch)

    job = {"proposals": proposals, "clips": clips_out, "proposals_doc": props_doc}
    _put.remote(
        json.dumps(job, ensure_ascii=False).encode("utf-8"),
        "in/job.json",
    )
    print(f"uploaded clips={len(clips_out)} proposals={len(proposals)} missing={missing}")


def _pull_local() -> None:
    cuts_local = ROOT / "data/scale/cuts"
    cuts_local.mkdir(parents=True, exist_ok=True)
    align_local = ROOT / "data/scale/align_cache"
    align_local.mkdir(parents=True, exist_ok=True)
    # volume ls rooted at mount contents. get dir → nested name; flatten after.
    import shutil

    tmp_cuts = ROOT / "data/scale/_cuts_pull"
    tmp_align = ROOT / "data/scale/_align_pull"
    if tmp_cuts.exists():
        shutil.rmtree(tmp_cuts)
    if tmp_align.exists():
        shutil.rmtree(tmp_align)
    cmds = [
        ["modal", "volume", "get", VOL, "cuts", str(tmp_cuts), "--force"],
        [
            "modal",
            "volume",
            "get",
            VOL,
            "out/iv_cut_proposals.json",
            str(ROOT / "data/scale/iv_cut_proposals.json"),
            "--force",
        ],
        ["modal", "volume", "get", VOL, "align_cache", str(tmp_align), "--force"],
    ]
    for c in cmds:
        print("+", " ".join(c))
        subprocess.check_call(c)

    cuts_local.mkdir(parents=True, exist_ok=True)
    src_cuts = tmp_cuts / "cuts" if (tmp_cuts / "cuts").is_dir() else tmp_cuts
    for w in src_cuts.glob("*.wav"):
        shutil.copy2(w, cuts_local / w.name)
    shutil.rmtree(tmp_cuts, ignore_errors=True)

    align_local.mkdir(parents=True, exist_ok=True)
    src_al = tmp_align / "align_cache" if (tmp_align / "align_cache").is_dir() else tmp_align
    for j in src_al.glob("*.json"):
        shutil.copy2(j, align_local / j.name)
    shutil.rmtree(tmp_align, ignore_errors=True)

    n = len(list(cuts_local.glob("*.wav")))
    print(f"pulled {n} cut wavs → {cuts_local}")


@app.local_entrypoint()
def main(action: str = "upload"):
    if action == "upload":
        _upload_local()
    elif action == "run":
        print(run_align_slice.remote())
    elif action == "pull":
        _pull_local()
    else:
        raise SystemExit(f"unknown action {action} (upload|run|pull)")
