"""WhisperX align + slice for Hindi IV cut proposals (Modal volume only).

Reads:  /vol/hindi_split/iv_hindi_cut_proposals.json
Writes: /vol/hindi_cuts/{cut_id}.wav
        /vol/hindi_align/{source_id}.json
        /vol/hindi_split/iv_hindi_cut_proposals.json  (updated times)
        /vol/hindi_split/iv_hindi_slice_stats.json

No bulk local pull. Review UI: modal deploy + /review

  modal run --detach modal_iv_hindi_slice.py --action run
  modal deploy modal_iv_hindi_slice.py   # review URL
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

APP = "real-voice-iv-hindi-slice"
VOL = "real-voice-iv-cuts"
ROOT = Path(__file__).resolve().parent
REMOTE = "/vol"

vol = modal.Volume.from_name(VOL, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "pyarrow",
        "tqdm",
        "fastapi",
    )
    .pip_install("whisperx")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/scripts")
)

app = modal.App(APP, image=image)


def _hf_login() -> str:
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
    return tok


def _audio_from_cell(cell) -> tuple[Any, int] | tuple[None, None]:
    import soundfile as sf

    if cell is None:
        return None, None
    if isinstance(cell, dict) and cell.get("bytes"):
        arr, sr = sf.read(io.BytesIO(cell["bytes"]))
        return arr, int(sr)
    if isinstance(cell, (bytes, bytearray)):
        arr, sr = sf.read(io.BytesIO(cell))
        return arr, int(sr)
    return None, None


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    memory=16384,
)
def process_parquet(parquet_path: str, sources: list[dict]) -> dict:
    """Align+slice all sources that live in one parquet shard."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import whisperx
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/root/scripts")
    from iv_cut_times import cut_times_from_alignment, plain_words_from_transcript

    tok = _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"parquet={parquet_path} sources={len(sources)} device={device}")

    local = hf_hub_download(
        "ai4bharat/IndicVoices", parquet_path, repo_type="dataset", token=tok or None
    )
    pf = pq.ParquetFile(local)

    # row → source
    by_row = {int(s["parquet_row"]): s for s in sources}
    rows_needed = sorted(by_row.keys())

    align_dir = Path(REMOTE, "hindi_align")
    cuts_dir = Path(REMOTE, "hindi_cuts")
    align_dir.mkdir(parents=True, exist_ok=True)
    cuts_dir.mkdir(parents=True, exist_ok=True)

    model_a, metadata = whisperx.load_align_model(language_code="hi", device=device)

    updates: list[dict] = []  # per-cut fields
    stats = {"sources": 0, "align_ok_cuts": 0, "align_fail_cuts": 0, "align_fail_src": 0}

    # walk row groups, pull needed rows
    offset = 0
    for rg in range(pf.num_row_groups):
        rg_n = pf.metadata.row_group(rg).num_rows
        local_rows = [r for r in rows_needed if offset <= r < offset + rg_n]
        if not local_rows:
            offset += rg_n
            continue
        tbl = pf.read_row_group(rg)
        cols = tbl.column_names
        audio_col = (
            "audio_filepath"
            if "audio_filepath" in cols
            else ("bytes" if "bytes" in cols else None)
        )
        if audio_col is None:
            print("NO AUDIO COL", parquet_path, cols[:8])
            offset += rg_n
            continue

        for abs_row in local_rows:
            li = abs_row - offset
            src = by_row[abs_row]
            sid = src["source_id"]
            text = src.get("text") or ""
            cuts = src.get("cuts") or []

            cell = tbl.column(audio_col)[li].as_py()
            audio, sr = _audio_from_cell(cell)
            if audio is None:
                stats["align_fail_src"] += 1
                for c in cuts:
                    updates.append(
                        {
                            "cut_id": c["cut_id"],
                            "align_ok": False,
                            "align_reason": "no_audio",
                            "cut_wav": "",
                        }
                    )
                    stats["align_fail_cuts"] += 1
                continue

            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration_s = float(len(audio) / sr)

            if sr != 16000:
                t = torch.from_numpy(audio).unsqueeze(0)
                t = torchaudio.functional.resample(t, sr, 16000)
                audio_16 = t.squeeze(0).numpy()
            else:
                audio_16 = audio

            cache_fp = align_dir / f"{sid}.json"
            if cache_fp.is_file():
                aligned_doc = json.loads(cache_fp.read_text(encoding="utf-8"))
            else:
                words = plain_words_from_transcript(text)
                plain = " ".join(words)
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
                    stats["align_fail_src"] += 1
                    for c in cuts:
                        updates.append(
                            {
                                "cut_id": c["cut_id"],
                                "align_ok": False,
                                "align_reason": f"align_exc:{type(e).__name__}",
                                "cut_wav": "",
                            }
                        )
                        stats["align_fail_cuts"] += 1
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
                    json.dumps(aligned_doc, ensure_ascii=False) + "\n", encoding="utf-8"
                )

            stats["sources"] += 1
            awords = aligned_doc["words"]

            for c in cuts:
                ct = cut_times_from_alignment(
                    text,
                    int(c["token_idx"]),
                    int(c.get("n_words_left") or 0),
                    int(c.get("n_words_right") or 0),
                    awords,
                    duration_s,
                )
                if not ct.get("ok"):
                    updates.append(
                        {
                            "cut_id": c["cut_id"],
                            "align_ok": False,
                            "align_reason": ct.get("reason") or "bad_times",
                            "t_start_s": ct.get("t_start_s"),
                            "t_end_s": ct.get("t_end_s"),
                            "cut_dur_s": ct.get("dur_s"),
                            "cut_wav": "",
                        }
                    )
                    stats["align_fail_cuts"] += 1
                    continue

                t0, t1 = float(ct["t_start_s"]), float(ct["t_end_s"])
                i0, i1 = int(t0 * sr), int(t1 * sr)
                i0 = max(0, i0)
                i1 = min(len(audio), max(i0 + 1, i1))
                out_wav = cuts_dir / f"{c['cut_id']}.wav"
                sf.write(str(out_wav), audio[i0:i1], sr)
                updates.append(
                    {
                        "cut_id": c["cut_id"],
                        "align_ok": True,
                        "align_reason": "",
                        "t_start_s": ct["t_start_s"],
                        "t_end_s": ct["t_end_s"],
                        "cut_dur_s": ct["dur_s"],
                        "cut_wav": f"hindi_cuts/{c['cut_id']}.wav",
                    }
                )
                stats["align_ok_cuts"] += 1

        offset += rg_n
        vol.commit()

    vol.commit()
    print("DONE", parquet_path, stats)
    return {"parquet": parquet_path, "stats": stats, "updates": updates}


@app.function(
    timeout=60 * 60 * 6,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    memory=8192,
)
def run_all() -> dict:
    """Fan out per-parquet GPU jobs, merge cut updates into proposals on volume."""
    props_fp = Path(REMOTE, "hindi_split/iv_hindi_cut_proposals.json")
    assert props_fp.is_file(), f"missing {props_fp}"
    doc = json.loads(props_fp.read_text(encoding="utf-8"))
    proposals = doc["proposals"]
    source_meta = doc.get("source_meta") or {}

    # group cuts by source
    by_src_cuts: dict[str, list] = defaultdict(list)
    for p in proposals:
        by_src_cuts[p["source_id"]].append(p)

    by_parquet: dict[str, list[dict]] = defaultdict(list)
    for sid, cuts in by_src_cuts.items():
        meta = source_meta.get(sid) or {}
        # fallback: from first cut
        pq = meta.get("parquet") or (cuts[0].get("parquet") if cuts else None)
        row = meta.get("parquet_row")
        if row is None and cuts:
            row = cuts[0].get("parquet_row")
        text = meta.get("text") or ""
        if not pq or row is None:
            print("skip source missing parquet/row", sid)
            continue
        by_parquet[pq].append(
            {
                "source_id": sid,
                "parquet_row": int(row),
                "text": text,
                "cuts": [
                    {
                        "cut_id": c["cut_id"],
                        "token_idx": c["token_idx"],
                        "n_words_left": c.get("n_words_left"),
                        "n_words_right": c.get("n_words_right"),
                        "tag": c.get("tag"),
                    }
                    for c in cuts
                ],
            }
        )

    print(f"parquets={len(by_parquet)} sources={sum(len(v) for v in by_parquet.values())} cuts={len(proposals)}")

    # parallel GPU
    args = [(pq, srcs) for pq, srcs in sorted(by_parquet.items())]
    results = list(process_parquet.starmap(args, order_outputs=False))

    by_cut = {}
    tot = {"sources": 0, "align_ok_cuts": 0, "align_fail_cuts": 0, "align_fail_src": 0}
    for r in results:
        for k in tot:
            tot[k] += r["stats"].get(k, 0)
        for u in r["updates"]:
            by_cut[u["cut_id"]] = u

    for p in proposals:
        u = by_cut.get(p["cut_id"])
        if not u:
            p["align_ok"] = False
            p["align_reason"] = p.get("align_reason") or "not_processed"
            continue
        p.update(u)

    # rebuild by_source_id
    by_src_out: dict[str, list] = defaultdict(list)
    for p in proposals:
        by_src_out[p["source_id"]].append(p)
    doc["proposals"] = proposals
    doc["by_source_id"] = dict(by_src_out)
    doc["slice_stats"] = tot
    doc["slice_backend"] = "whisperx_align_hi"
    props_fp.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")

    slim = {
        "n_proposals": len(proposals),
        "slice_stats": tot,
        "align_ok": sum(1 for p in proposals if p.get("align_ok")),
        "by_tag_align_ok": {},
    }
    from collections import Counter

    ok_c = Counter()
    all_c = Counter()
    for p in proposals:
        all_c[p["tag"]] += 1
        if p.get("align_ok"):
            ok_c[p["tag"]] += 1
    slim["by_tag_align_ok"] = {
        t: {"ok": ok_c[t], "n": all_c[t], "rate": round(ok_c[t] / all_c[t], 3) if all_c[t] else 0}
        for t in sorted(all_c)
    }
    Path(REMOTE, "hindi_split/iv_hindi_slice_stats.json").write_text(
        json.dumps(slim, indent=2) + "\n", encoding="utf-8"
    )
    vol.commit()
    print("SLICE DONE", slim)
    return slim


# ----- review UI (5 / tag) — deploy for stable URL -----
@app.function(volumes={REMOTE: vol})
@modal.asgi_app()
def review_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse

    api = FastAPI(title="IV Hindi cut review")

    def _load():
        fp = Path(REMOTE, "hindi_split/iv_hindi_cut_proposals.json")
        if not fp.is_file():
            return None
        return json.loads(fp.read_text(encoding="utf-8"))

    @api.get("/", response_class=HTMLResponse)
    def home(n: int = 5, seed: int = 0):
        doc = _load()
        if not doc:
            return HTMLResponse("<h1>No proposals yet</h1>", status_code=404)
        props = [p for p in doc.get("proposals") or [] if p.get("align_ok") and p.get("cut_wav")]
        by_tag: dict[str, list] = defaultdict(list)
        for p in props:
            by_tag[p["tag"]].append(p)
        rng = random.Random(seed)
        cards = []
        for tag in sorted(by_tag):
            pool = by_tag[tag]
            sample = pool if len(pool) <= n else rng.sample(pool, n)
            cards.append(f"<h2>[{tag}] · {len(pool)} align✓ · showing {len(sample)}</h2>")
            for p in sample:
                wav = p.get("cut_wav") or ""
                cards.append(
                    "<div style='margin:0.8rem 0;padding:0.8rem;border:1px solid #444'>"
                    f"<b>{p.get('cut_id')}</b> "
                    f"{p.get('t_start_s')}–{p.get('t_end_s')}s "
                    f"({p.get('cut_dur_s')}s)"
                    f"<div style='color:#aaa;margin:0.4rem 0'>{p.get('caption_inline')}</div>"
                    f"<audio controls preload='none' src='/wav/{p.get('cut_id')}'></audio>"
                    f"</div>"
                )
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<body style='font-family:sans-serif;background:#111;color:#eee;padding:1.2rem'>"
            "<h1>IV Hindi cuts review</h1>"
            f"<p>align✓ total={len(props)} · seed={seed} · n={n} "
            f"<a href='/?n={n}&seed={seed+1}' style='color:#8cf'>reshuffle</a></p>"
            + "".join(cards)
            + "</body>"
        )
        return HTMLResponse(body)

    @api.get("/wav/{cut_id}")
    def wav(cut_id: str):
        # basic path safety
        if "/" in cut_id or ".." in cut_id:
            raise HTTPException(400, "bad id")
        fp = Path(REMOTE, "hindi_cuts", f"{cut_id}.wav")
        if not fp.is_file():
            raise HTTPException(404, "missing wav")
        return FileResponse(str(fp), media_type="audio/wav")

    @api.get("/stats")
    def stats():
        fp = Path(REMOTE, "hindi_split/iv_hindi_slice_stats.json")
        if not fp.is_file():
            return {"error": "no slice stats yet"}
        return json.loads(fp.read_text(encoding="utf-8"))

    return api


@app.local_entrypoint()
def main(action: str = "run"):
    if action == "run":
        print(run_all.remote())
    elif action == "stats":
        # lightweight: just print volume stats if present via run_all slim file
        print("deploy review: modal deploy modal_iv_hindi_slice.py")
        print("then open the printed ASGI URL")
    else:
        raise SystemExit("action=run|stats")
