"""IV v8: inline-caption LoRA pack → encode → train.

Fixes v7 bug: train/synth must use caption_inline (tag at event), NOT prefix.

Pack:
  - tagged cuts from hindi proposals (subset + caps)
  - 1000 untagged full IV utterances (truncate ≤15s)
  - SFT user string = row["caption"] only

  modal run --detach modal_iv_v8_train.py --action all
  modal run modal_iv_v8_train.py --action pack
  modal run --detach modal_iv_v8_train.py --action encode
  modal run --detach modal_iv_v8_train.py --action train
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal

APP = "real-voice-iv-v8-train"
VOL_MIO = "real-voice-mio"
VOL_CUTS = "real-voice-iv-cuts"
MIO = "/mio"
CUTS = "/cuts"
CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"
MODEL_ID = "SPRINGLab/Indic-Mio"
SEED = 8
N_ENCODE_SHARDS = 32
PACK_NAME = "iv_hindi_v8_inline"
ADAPTER_DIR = f"{MIO}/out/adapter_iv_v8"
# epoch-suffixed runs keep prior adapters (e.g. adapter_iv_v8_e3)
ENCODE_DIR = f"{MIO}/encoded_v8"
DATA_DIR = f"{MIO}/data_v8"
UNTAGGED_WAV_DIR = "hindi_untagged_v8"
UNTAGGED_N = 1000
UNTAGGED_MAX_S = 15.0
UNTAGGED_MIN_S = 1.5
HOLDOUT_SRC_FRAC = 0.05

# tagged allow-list + caps (None = take all survivors)
TAG_CAPS: dict[str, int | None] = {
    "thinking": 1500,
    "tsk": 1500,
    "gasp": None,  # 1418
    "throat_clearing": None,
    "ugh": None,
    "sigh": None,
    "cough": None,
    "laugh": None,
    "whispering": None,
    "sniffle": None,
}

TAG_RE = re.compile(r"<[^<>]+>")
BRACKET_TAG_RE = re.compile(r"\[([a-z_]+)\]")

vol_mio = modal.Volume.from_name(VOL_MIO, create_if_missing=True)
vol_cuts = modal.Volume.from_name(VOL_CUTS, create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "peft>=0.14.0",
        "datasets>=3.0.0",
        "soundfile",
        "numpy",
        "einops",
        "julius",
        "safetensors",
        "huggingface_hub",
        "pyarrow",
        "sentencepiece",
        "protobuf",
        "tqdm",
    )
    .pip_install("git+https://github.com/Aratako/MioCodec")
)

scan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("huggingface_hub", "pyarrow", "soundfile", "numpy")
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


def sft_user_from_row(row: dict) -> str:
    """CRITICAL: use stored caption as-is (inline for events, plain for bare).

    Never rebuild [tag] prefix from tag1 — that was the v7 bug.
    """
    cap = (row.get("caption") or "").strip()
    if not cap:
        raise ValueError(f"missing caption for row {row.get('id')}")
    tag = (row.get("tag1") or "none").strip()
    if tag and tag != "none":
        needle = f"[{tag}]"
        if needle not in cap:
            raise ValueError(
                f"inline caption missing {needle} id={row.get('id')} cap={cap[:80]}"
            )
    else:
        if BRACKET_TAG_RE.search(cap):
            raise ValueError(
                f"untagged row has bracket tag id={row.get('id')} cap={cap[:80]}"
            )
    return cap


def _stable_id(parquet_name: str, row: int) -> str:
    h = hashlib.sha1(f"{parquet_name}:{row}".encode()).hexdigest()[:10]
    return f"hi_{h}"


def _proposal_to_row(p: dict) -> dict:
    tag = p["tag"]
    text = (p.get("text_plain") or "").strip()
    caption = (p.get("caption_inline") or "").strip()
    if not caption:
        raise ValueError(f"proposal missing caption_inline: {p.get('cut_id')}")
    if f"[{tag}]" not in caption:
        raise ValueError(
            f"caption_inline missing [{tag}]: {p.get('cut_id')} {caption[:80]}"
        )
    return {
        "id": p["cut_id"],
        "audio": p.get("cut_wav") or f"hindi_cuts/{p['cut_id']}.wav",
        "text": text,
        "caption": caption,  # INLINE — train uses this
        "tag1": tag,
        "tag2": "",
        "source": "indicvoices_hi_v8_event",
        "source_id": p.get("source_id"),
        "algo_version": p.get("algo_version"),
        "native_label": p.get("raw_tag") or tag,
        "cut_dur_s": p.get("cut_dur_s"),
        "pack": PACK_NAME,
    }


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


def _plain_from_verbatim(text: str) -> str:
    """Strip <> tags; collapse whitespace."""
    t = TAG_RE.sub(" ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


@app.function(
    image=scan_image,
    timeout=60 * 30,
    secrets=[hf_secret],
    memory=4096,
    cpu=2,
)
def scan_untagged_parquet(
    parquet_path: str,
    exclude_source_ids: list[str],
    max_keep: int = 400,
) -> list[dict]:
    """ONE parquet text+duration scan (parallelize via starmap)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tok = _hf_login()
    exclude = set(exclude_source_ids)
    local = hf_hub_download(
        "ai4bharat/IndicVoices", parquet_path, repo_type="dataset", token=tok or None
    )
    pf = pq.ParquetFile(local)
    names = set(pf.schema.names)
    if "unsanitized_verbatim" not in names or "duration" not in names:
        print("skip cols", parquet_path, flush=True)
        return []
    tbl = pf.read(columns=["unsanitized_verbatim", "duration"])
    texts, durs = tbl.column(0), tbl.column(1)
    pool: list[dict] = []
    for row_i in range(len(texts)):
        text = texts[row_i].as_py()
        text = text if isinstance(text, str) else str(text or "")
        if TAG_RE.search(text):
            continue
        plain = _plain_from_verbatim(text)
        if len(plain.split()) < 3:
            continue
        try:
            dur_f = float(durs[row_i].as_py())
        except (TypeError, ValueError):
            continue
        if dur_f < UNTAGGED_MIN_S:
            continue
        sid = _stable_id(Path(parquet_path).name, row_i)
        if sid in exclude:
            continue
        pool.append(
            {
                "source_id": sid,
                "parquet": parquet_path,
                "parquet_row": row_i,
                "text": plain,
                "duration": dur_f,
            }
        )
    rng = random.Random(SEED + hash(parquet_path) % 10007)
    rng.shuffle(pool)
    short = [c for c in pool if c["duration"] <= UNTAGGED_MAX_S]
    long = [c for c in pool if c["duration"] > UNTAGGED_MAX_S]
    take = (short + long)[:max_keep]
    print(f"scan {parquet_path} pool={len(pool)} take={len(take)}", flush=True)
    return take


@app.function(
    image=scan_image,
    timeout=60 * 90,
    secrets=[hf_secret],
    memory=8192,
    cpu=2,
)
def scan_untagged_candidates(
    exclude_source_ids: list[str],
    config: str = "hindi",
    oversample: int = 4000,
) -> list[dict]:
    """Parallel per-parquet scan (fast)."""
    from huggingface_hub import HfApi

    tok = _hf_login()
    api = HfApi(token=tok or None)
    files = sorted(
        f
        for f in api.list_repo_files("ai4bharat/IndicVoices", repo_type="dataset")
        if f.endswith(".parquet") and f.startswith(f"{config}/")
    )
    # ~1k clean/file → 8 files plenty for 1k untagged (+oversample)
    n_files = min(12, len(files))
    use = files[:n_files]
    print(
        f"parallel scan files={n_files}/{len(files)} exclude={len(exclude_source_ids)}",
        flush=True,
    )
    per = max(200, (oversample // n_files) + 50)
    args = [(fp, exclude_source_ids, per) for fp in use]
    chunks = list(scan_untagged_parquet.starmap(args, order_outputs=False))
    pool: list[dict] = []
    for ch in chunks:
        pool.extend(ch or [])
    rng = random.Random(SEED)
    rng.shuffle(pool)
    short = [c for c in pool if c["duration"] <= UNTAGGED_MAX_S]
    long = [c for c in pool if c["duration"] > UNTAGGED_MAX_S]
    take = (short + long)[:oversample]
    print(
        f"DONE scan pool={len(pool)} short={len(short)} take={len(take)}",
        flush=True,
    )
    if not take:
        raise RuntimeError("untagged scan returned 0 candidates")
    return take


@app.function(
    image=scan_image,
    timeout=60 * 60,
    volumes={CUTS: vol_cuts},
    secrets=[hf_secret],
    memory=8192,
    cpu=2,
)
def extract_untagged_shard(parquet_path: str, rows: list[dict]) -> dict:
    """Decode audio for candidates in one parquet; write wav ≤15s."""
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tok = _hf_login()
    out_dir = Path(CUTS, UNTAGGED_WAV_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_row = {int(r["parquet_row"]): r for r in rows}
    need = sorted(by_row)
    local = hf_hub_download(
        "ai4bharat/IndicVoices", parquet_path, repo_type="dataset", token=tok or None
    )
    pf = pq.ParquetFile(local)

    ok_rows: list[dict] = []
    stats = {"need": len(need), "ok": 0, "fail": 0}
    offset = 0
    for rg in range(pf.num_row_groups):
        rg_n = pf.metadata.row_group(rg).num_rows
        local_rows = [r for r in need if offset <= r < offset + rg_n]
        if not local_rows:
            offset += rg_n
            continue
        tbl = pf.read_row_group(rg)
        cols = tbl.column_names
        # row-group renames schema "bytes" → "audio_filepath" (dict{bytes,path})
        audio_col = (
            "bytes"
            if "bytes" in cols
            else ("audio_filepath" if "audio_filepath" in cols else None)
        )
        if audio_col is None:
            print("NO audio col", parquet_path, cols[:8], flush=True)
            offset += rg_n
            continue
        for abs_row in local_rows:
            meta = by_row[abs_row]
            li = abs_row - offset
            cell = tbl.column(audio_col)[li].as_py()
            audio, sr = _audio_from_cell(cell)
            if audio is None:
                stats["fail"] += 1
                continue
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            max_n = int(UNTAGGED_MAX_S * sr)
            if len(audio) > max_n:
                audio = audio[:max_n]
            dur = float(len(audio) / sr)
            if dur < UNTAGGED_MIN_S:
                stats["fail"] += 1
                continue
            rid = f"{meta['source_id']}_untagged"
            wav_rel = f"{UNTAGGED_WAV_DIR}/{rid}.wav"
            sf.write(str(Path(CUTS, wav_rel)), audio, sr)
            ok_rows.append(
                {
                    "id": rid,
                    "audio": wav_rel,
                    "text": meta["text"],
                    "caption": meta["text"],  # bare — no tags
                    "tag1": "none",
                    "tag2": "",
                    "source": "indicvoices_hi_v8_untagged",
                    "source_id": meta["source_id"],
                    "parquet": parquet_path,
                    "parquet_row": abs_row,
                    "cut_dur_s": round(dur, 3),
                    "src_duration": meta.get("duration"),
                    "pack": PACK_NAME,
                }
            )
            stats["ok"] += 1
        offset += rg_n

    vol_cuts.commit()
    print(f"extract {parquet_path} {stats}", flush=True)
    return {"parquet": parquet_path, "stats": stats, "rows": ok_rows}


@app.function(volumes={MIO: vol_mio, CUTS: vol_cuts}, timeout=60 * 60, memory=8192)
def pack_tagged(seed: int = SEED) -> dict:
    """Select capped tagged proposals → rows (caption=inline)."""
    prop_fp = Path(CUTS, "hindi_split/iv_hindi_cut_proposals.json")
    doc = json.loads(prop_fp.read_text(encoding="utf-8"))
    props = [
        p
        for p in (doc.get("proposals") or [])
        if p.get("align_ok")
        and p.get("cut_wav")
        and p.get("tag") in TAG_CAPS
        and p.get("caption_inline")
    ]
    # wav exists
    ok, miss = [], 0
    for p in props:
        if Path(CUTS, p["cut_wav"]).is_file():
            ok.append(p)
        else:
            miss += 1
    props = ok

    rng = random.Random(seed)
    by_tag: dict[str, list] = defaultdict(list)
    for p in props:
        by_tag[p["tag"]].append(p)

    selected: list[dict] = []
    cap_stats = {}
    for tag, cap in TAG_CAPS.items():
        items = by_tag.get(tag, [])
        rng.shuffle(items)
        keep = items if cap is None else items[:cap]
        cap_stats[tag] = {"available": len(items), "kept": len(keep), "cap": cap}
        selected.extend(keep)

    rows = [_proposal_to_row(p) for p in selected]
    # validate inline sft string
    for r in rows:
        sft_user_from_row(r)

    # holdout by source
    by_src: dict[str, list] = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(r)
    srcs = sorted(by_src)
    rng.shuffle(srcs)
    n_hold = max(1, int(round(len(srcs) * HOLDOUT_SRC_FRAC)))
    hold_srcs = set(srcs[:n_hold])
    train_rows, hold_rows = [], []
    for sid, plist in by_src.items():
        (hold_rows if sid in hold_srcs else train_rows).extend(plist)
    rng.shuffle(train_rows)
    rng.shuffle(hold_rows)

    out = {
        "train_rows": train_rows,
        "hold_rows": hold_rows,
        "cap_stats": cap_stats,
        "wav_miss": miss,
        "exclude_source_ids": sorted(by_src.keys()),
        "n_prefix_violation": sum(
            1
            for r in rows
            if r["caption"].startswith(f"[{r['tag1']}]")
            and r["caption"].find(f"[{r['tag1']}]") == 0
            and not (r.get("text") or "").startswith("[")
            # startswith tag is OK if event truly at start; count mid vs start
        ),
        "n_mid_tag": sum(
            1
            for r in rows
            if f"[{r['tag1']}]" in r["caption"]
            and not r["caption"].startswith(f"[{r['tag1']}]")
        ),
        "n_start_tag": sum(
            1 for r in rows if r["caption"].startswith(f"[{r['tag1']}]")
        ),
    }
    # don't return huge lists through modal multiple times — write staging
    stage = Path(MIO, DATA_DIR)
    stage.mkdir(parents=True, exist_ok=True)
    Path(stage, "tagged_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    Path(stage, "tagged_holdout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in hold_rows) + "\n",
        encoding="utf-8",
    )
    Path(stage, "tagged_pack_partial.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k not in ("train_rows", "hold_rows", "exclude_source_ids")
            }
            | {
                "n_train_tagged": len(train_rows),
                "n_hold_tagged": len(hold_rows),
                "n_exclude_srcs": len(out["exclude_source_ids"]),
                "exclude_source_ids": out["exclude_source_ids"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    vol_mio.commit()
    print(
        json.dumps(
            {
                "n_train_tagged": len(train_rows),
                "n_hold_tagged": len(hold_rows),
                "cap_stats": cap_stats,
                "n_mid_tag": out["n_mid_tag"],
                "n_start_tag": out["n_start_tag"],
            },
            indent=2,
        ),
        flush=True,
    )
    return {
        "n_train_tagged": len(train_rows),
        "n_hold_tagged": len(hold_rows),
        "cap_stats": cap_stats,
        "n_mid_tag": out["n_mid_tag"],
        "n_start_tag": out["n_start_tag"],
        "n_exclude_srcs": len(out["exclude_source_ids"]),
    }


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    timeout=60 * 60 * 3,
    secrets=[hf_secret],
    memory=8192,
)
def run_pack() -> dict:
    """Tagged select → parallel untagged extract → final train/holdout jsonl."""
    vol_mio.reload()
    tmeta = pack_tagged.remote()
    # sibling container committed — must reload before reading
    vol_mio.reload()
    partial_fp = Path(MIO, DATA_DIR, "tagged_pack_partial.json")
    if not partial_fp.is_file():
        raise FileNotFoundError(
            f"missing {partial_fp} after pack_tagged — volume commit/reload race"
        )
    partial = json.loads(partial_fp.read_text())
    exclude = partial["exclude_source_ids"]

    cands = scan_untagged_candidates.remote(exclude_source_ids=exclude, oversample=4000)
    by_pq: dict[str, list] = defaultdict(list)
    for c in cands:
        by_pq[c["parquet"]].append(c)
    print(f"untagged candidates={len(cands)} parquets={len(by_pq)}", flush=True)

    args = [(pq, rows) for pq, rows in sorted(by_pq.items())]
    # parallel extract
    results = list(extract_untagged_shard.starmap(args, order_outputs=False))
    untagged: list[dict] = []
    for r in results:
        untagged.extend(r.get("rows") or [])

    rng = random.Random(SEED)
    rng.shuffle(untagged)
    # validate bare captions
    good = []
    for r in untagged:
        try:
            sft_user_from_row(r)
            good.append(r)
        except ValueError as e:
            print("skip bad untagged", e, flush=True)
    untagged = good[:UNTAGGED_N]
    if len(untagged) < UNTAGGED_N:
        print(
            f"WARN untagged got {len(untagged)} < {UNTAGGED_N}",
            flush=True,
        )

    # split untagged holdout ~5%
    n_u_hold = max(1, int(round(len(untagged) * HOLDOUT_SRC_FRAC)))
    u_hold = untagged[:n_u_hold]
    u_train = untagged[n_u_hold:]

    train_rows = [
        json.loads(l)
        for l in Path(MIO, DATA_DIR, "tagged_train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    hold_rows = [
        json.loads(l)
        for l in Path(MIO, DATA_DIR, "tagged_holdout.jsonl").read_text().splitlines()
        if l.strip()
    ]
    train_rows.extend(u_train)
    hold_rows.extend(u_hold)
    rng.shuffle(train_rows)
    rng.shuffle(hold_rows)

    # final validation
    n_inline_ok = 0
    for r in train_rows + hold_rows:
        user = sft_user_from_row(r)
        if r["tag1"] not in ("", "none") and not user.startswith(f"[{r['tag1']}]"):
            n_inline_ok += 1

    out = Path(MIO, DATA_DIR)
    def dump(name: str, rows: list[dict]) -> None:
        Path(out, name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    dump("train.jsonl", train_rows)
    dump("holdout.jsonl", hold_rows)
    dump("manifest.jsonl", train_rows + hold_rows)

    # wipe prior v8 codes so encode is clean
    enc = Path(ENCODE_DIR)
    enc.mkdir(parents=True, exist_ok=True)
    for fp in enc.glob("codes_*.jsonl"):
        fp.unlink()
    for fp in (enc / "codes.jsonl", enc / "encode_meta.json"):
        if fp.exists():
            fp.unlink()

    tag_tr = Counter(r["tag1"] for r in train_rows)
    tag_ho = Counter(r["tag1"] for r in hold_rows)
    meta = {
        "pack": PACK_NAME,
        "caption_contract": "inline_caption_field_only",
        "sft_user": "row['caption'] via sft_user_from_row — NEVER prefix rebuild",
        "n_train": len(train_rows),
        "n_holdout": len(hold_rows),
        "n_untagged_train": len(u_train),
        "n_untagged_holdout": len(u_hold),
        "untagged_max_s": UNTAGGED_MAX_S,
        "by_tag_train": dict(tag_tr.most_common()),
        "by_tag_holdout": dict(tag_ho.most_common()),
        "tagged_meta": tmeta,
        "n_mid_tag_train": sum(
            1
            for r in train_rows
            if r["tag1"] not in ("", "none")
            and not r["caption"].startswith(f"[{r['tag1']}]")
        ),
        "n_start_tag_train": sum(
            1
            for r in train_rows
            if r["tag1"] not in ("", "none")
            and r["caption"].startswith(f"[{r['tag1']}]")
        ),
        "seed": SEED,
        "adapter_dir": ADAPTER_DIR,
        "encode_dir": ENCODE_DIR,
        "data_dir": DATA_DIR,
    }
    Path(out, "pack_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # also mirror train path expected by older helpers
    Path(MIO, "data").mkdir(parents=True, exist_ok=True)
    Path(MIO, "data/train_v8.jsonl").write_text(
        Path(out, "train.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    Path(MIO, "data/pack_meta_v8.json").write_text(json.dumps(meta, indent=2) + "\n")
    vol_mio.commit()
    vol_cuts.commit()
    print(json.dumps(meta, indent=2), flush=True)
    return meta


@app.function(
    gpu="T4",
    timeout=60 * 90,
    memory=16384,
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    secrets=[hf_secret],
)
def encode_shard(shard_i: int, n_shards: int = N_ENCODE_SHARDS) -> dict:
    import torch
    from miocodec import MioCodecModel, load_audio

    _hf_login()
    vol_mio.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [
        json.loads(l)
        for l in Path(DATA_DIR, "train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    mine = [r for i, r in enumerate(rows) if i % n_shards == shard_i]
    out_fp = Path(ENCODE_DIR, f"codes_{shard_i:02d}.jsonl")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_fp.exists():
        for line in out_fp.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    codec = MioCodecModel.from_pretrained(CODEC_ID).eval().to(device)
    sr = int(codec.config.sample_rate)
    print(f"shard={shard_i} rows={len(mine)} cached={len(done)} sr={sr}", flush=True)

    n_new = 0
    with out_fp.open("a", encoding="utf-8") as ef:
        for row in mine:
            rid = row["id"]
            if rid in done:
                continue
            wav_p = Path(CUTS, row["audio"])
            if not wav_p.is_file():
                print("MISS", rid, row["audio"], flush=True)
                continue
            with torch.inference_mode():
                wave = load_audio(str(wav_p), sample_rate=sr).to(device)
                feats = codec.encode(wave)
                codes = feats.content_token_indices.detach().cpu().view(-1).tolist()
            ef.write(json.dumps({"id": rid, "codes": codes}, ensure_ascii=False) + "\n")
            ef.flush()
            n_new += 1
            if n_new % 50 == 0:
                print(f"shard={shard_i} new={n_new}/{len(mine)}", flush=True)
                vol_mio.commit()
    vol_mio.commit()
    return {"shard": shard_i, "n_new": n_new, "n_mine": len(mine), "cached": len(done)}


@app.function(volumes={MIO: vol_mio}, timeout=60 * 20)
def merge_codes() -> dict:
    vol_mio.reload()
    rows = [
        json.loads(l)
        for l in Path(DATA_DIR, "train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    need = {r["id"] for r in rows}
    shard_files = sorted(Path(ENCODE_DIR).glob("codes_*.jsonl"))
    print(f"merge shards_found={len(shard_files)} need={len(need)}", flush=True)
    merged: dict[str, list] = {}
    for fp in shard_files:
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            merged[o["id"]] = o["codes"]
    out = Path(ENCODE_DIR, "codes.jsonl")
    with out.open("w", encoding="utf-8") as ef:
        for rid, codes in merged.items():
            if rid in need:
                ef.write(json.dumps({"id": rid, "codes": codes}, ensure_ascii=False) + "\n")
    miss = sorted(need - set(merged))
    meta = {
        "n_codes": len(need & set(merged)),
        "n_train": len(need),
        "n_miss": len(miss),
        "miss_head": miss[:20],
    }
    Path(ENCODE_DIR, "encode_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    vol_mio.commit()
    print(meta, flush=True)
    return meta


def _train_core(epochs: int, lora_r: int, adapter_dir: str | None = None) -> dict:
    """LoRA train body. Safe under accelerate/torch DDP (2×A10G)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0
    adapter_dir = adapter_dir or ADAPTER_DIR
    adapter_name = Path(adapter_dir).name
    meta_name = f"train_meta_{adapter_name}.json"

    class _VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # checkpoints stay on local /tmp during DDP; final copy to volume at end
            if state.is_world_process_zero:
                print(f"ckpt saved @ step={state.global_step} (local stage)", flush=True)

    _hf_login()
    # DDP child procs often cannot see Modal Volume FUSE — use /tmp stage
    stage = Path(os.environ.get("RV_V8_STAGE", "/tmp/v8_stage"))
    train_fp = stage / "train.jsonl"
    codes_fp = stage / "codes.jsonl"
    if not train_fp.is_file() or not codes_fp.is_file():
        raise FileNotFoundError(
            f"rank{local_rank} missing staged data under {stage}: "
            f"train={train_fp.is_file()} codes={codes_fp.is_file()}"
        )
    train_rows = [
        json.loads(l)
        for l in train_fp.read_text().splitlines()
        if l.strip()
    ]
    codes_map = {}
    for line in codes_fp.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            codes_map[o["id"]] = o["codes"]
    print(
        f"[rank{local_rank}/{world_size}] train_rows={len(train_rows)} codes={len(codes_map)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    n_mid = n_bare = n_start = 0
    for row in train_rows:
        codes = codes_map.get(row["id"])
        if not codes:
            continue
        user = sft_user_from_row(row)  # INLINE / bare — never prefix rebuild
        if row["tag1"] in ("", "none"):
            n_bare += 1
        elif user.startswith(f"[{row['tag1']}]"):
            n_start += 1
        else:
            n_mid += 1
        assistant = "".join(f"<|s_{int(c)}|>" for c in codes)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        records.append({"id": row["id"], "text": text, "user": user})

    print(
        f"[rank{local_rank}] sft records={len(records)} mid={n_mid} start_tag={n_start} bare={n_bare}",
        flush=True,
    )
    assert n_mid > 0, "FATAL: zero mid-tag captions — refusing train (v7 bug guard)"
    assert n_bare > 0, "FATAL: zero untagged rows — refusing train"

    local_out = stage / adapter_name
    if is_main:
        local_out.mkdir(parents=True, exist_ok=True)
        (stage / "sft_preview_v8.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records[:8]) + "\n",
            encoding="utf-8",
        )

    ds = Dataset.from_list(records)

    # ---- prompt masking -------------------------------------------------
    # v8_e3 and v7 both trained with labels = full input_ids, so the loss was a
    # blend of caption-text prediction and audio-code prediction. The text part
    # has its own floor the LoRA cannot push down, which is why both runs
    # converged to ~4.8 from completely different data contracts. Mask every
    # position before the first <|s_N|> code token so all gradient — and the
    # reported loss — is code prediction only.
    #
    # NOTE: loss from a masked run is NOT comparable to the 4.78 of v8_e3.
    # Different label set, different scale. Compare masked runs to each other.
    _code_re = re.compile(r"^<\|s_\d+\|>$")
    code_ids = {i for t, i in tokenizer.get_vocab().items() if _code_re.match(t)}
    assert code_ids, "FATAL: no <|s_N|> code tokens in tokenizer vocab"
    if is_main:
        print(f"[mask] code token ids in vocab: {len(code_ids)}", flush=True)

    def tok_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )
        labels = []
        for ids in out["input_ids"]:
            first = next((j for j, t in enumerate(ids) if t in code_ids), None)
            if first is None:
                # every code token lost to truncation — row teaches nothing
                labels.append([-100] * len(ids))
                continue
            lab = list(ids)
            for j in range(first):
                lab[j] = -100
            labels.append(lab)
        out["labels"] = labels
        return out

    ds = ds.map(
        tok_fn, batched=True, remove_columns=[c for c in ds.column_names if c != "id"]
    )

    if is_main:
        n_pos = n_kept = n_dead = 0
        probe = min(2000, len(ds))
        for ex in ds.select(range(probe)):
            lab = ex["labels"]
            k = sum(1 for x in lab if x != -100)
            n_pos += len(lab)
            n_kept += k
            n_dead += k == 0
        print(
            f"[mask] probe {probe} rows: kept {n_kept}/{n_pos} label positions "
            f"({n_kept/max(1,n_pos):.1%}); rows with zero labels: {n_dead}",
            flush=True,
        )
        assert n_kept > 0, "FATAL: prompt masking removed every label"
        assert n_dead < probe * 0.05, (
            f"FATAL: {n_dead}/{probe} rows lost all code tokens to truncation — "
            "raise max_length or shorten cuts"
        )

    # NO device_map="auto" — breaks DDP. Trainer/accelerate places shards.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    if is_main:
        model.print_trainable_parameters()

    out_dir = str(local_out)
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    resume = str(ckpts[-1]) if ckpts else None
    print(f"[rank{local_rank}] resume_from={resume} out={out_dir}", flush=True)

    # keep global batch ~8: per_device=1 × world × accum
    accum = max(1, 8 // max(1, world_size))
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        # v8_e3 kept only steps 2500/3000/3405 — all past the plateau, so the
        # checkpoint sweep could not discriminate. All movement was in the
        # first ~250 steps; keep that region.
        save_steps=200,
        save_total_limit=8,
        bf16=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )

    def collate(features):
        pad_id = tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        callbacks=[_VolCommit()],
    )
    trainer.train(resume_from_checkpoint=resume)
    if trainer.is_world_process_zero():
        import shutil

        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        # copy adapter + preview onto Modal volume
        vol_mio.reload()
        Path(adapter_dir).parent.mkdir(parents=True, exist_ok=True)
        if Path(adapter_dir).exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(out_dir, adapter_dir)
        preview = stage / "sft_preview_v8.jsonl"
        if preview.is_file():
            shutil.copy(preview, Path(MIO, "out/sft_preview_v8.jsonl"))
        meta = {
            "model_id": MODEL_ID,
            "codec_id": CODEC_ID,
            "epochs": epochs,
            "lora_r": lora_r,
            "n_train": len(records),
            "n_mid": n_mid,
            "n_start_tag": n_start,
            "n_bare": n_bare,
            "adapter": adapter_dir,
            "pack": PACK_NAME,
            "caption_contract": "inline",
            "prompt_masked": True,
            "loss_note": "code-token loss only (prompt masked) — NOT comparable to unmasked v7/v8_e3 loss",
            "resumed_from": resume,
            "n_gpu": world_size,
            "grad_accum": accum,
        }
        Path(MIO, "out", meta_name).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        # also keep canonical meta pointer for default adapter path
        if adapter_dir == ADAPTER_DIR:
            Path(MIO, "out/train_meta_iv_v8.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        vol_mio.commit()
        print("SAVED", adapter_dir, meta, flush=True)
        return meta
    return {"rank": local_rank, "ok": True}


@app.function(
    gpu="A10G:2",
    timeout=60 * 60 * 8,
    memory=32768,
    volumes={MIO: vol_mio},
    secrets=[hf_secret],
)
def train(
    epochs: int = 2,
    lora_r: int = 16,
    adapter_name: str = "",
) -> dict:
    """2×A10G DDP via accelerate notebook_launcher.

    Stage train/codes to /tmp first — DDP workers miss Volume FUSE mounts.
    """
    import shutil
    import torch
    from accelerate import notebook_launcher

    vol_mio.reload()
    name = adapter_name or (
        f"adapter_iv_v8_e{epochs}" if epochs != 2 else "adapter_iv_v8"
    )
    adapter_dir = f"{MIO}/out/{name}"
    stage = Path("/tmp/v8_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(Path(DATA_DIR, "train.jsonl"), stage / "train.jsonl")
    shutil.copy(Path(ENCODE_DIR, "codes.jsonl"), stage / "codes.jsonl")
    os.environ["RV_V8_STAGE"] = str(stage)
    print(
        f"staged -> {stage} train_bytes={(stage/'train.jsonl').stat().st_size} "
        f"codes_bytes={(stage/'codes.jsonl').stat().st_size} "
        f"adapter={name} epochs={epochs}",
        flush=True,
    )

    n_gpu = torch.cuda.device_count()
    print(f"train launch n_gpu={n_gpu}", flush=True)

    def _go():
        _train_core(epochs, lora_r, adapter_dir=adapter_dir)

    if n_gpu > 1:
        notebook_launcher(_go, num_processes=n_gpu)
    else:
        _go()
    vol_mio.reload()
    meta_fp = Path(MIO, "out", f"train_meta_{name}.json")
    if meta_fp.is_file():
        return json.loads(meta_fp.read_text())
    # fallback canonical
    meta_fp = Path(MIO, "out/train_meta_iv_v8.json")
    if meta_fp.is_file():
        return json.loads(meta_fp.read_text())
    return {"error": "no train meta written"}


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    timeout=60 * 60 * 12,
    secrets=[hf_secret],
)
def run_all(epochs: int = 2, lora_r: int = 16) -> dict:
    pmeta = run_pack.remote()
    print("PACK", pmeta, flush=True)
    enc = list(
        encode_shard.starmap(
            [(i, N_ENCODE_SHARDS) for i in range(N_ENCODE_SHARDS)],
            order_outputs=False,
        )
    )
    print("ENCODE", enc, flush=True)
    mmeta = merge_codes.remote()
    print("MERGE", mmeta, flush=True)
    if mmeta["n_miss"] > 50:
        raise RuntimeError(f"too many encode misses: {mmeta['n_miss']}")
    tmeta = train.remote(epochs=epochs, lora_r=lora_r)
    print("TRAIN", tmeta, flush=True)
    return {"pack": pmeta, "encode": mmeta, "train": tmeta}


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    timeout=60 * 60 * 6,
    secrets=[hf_secret],
)
def run_encode() -> dict:
    enc = list(
        encode_shard.starmap(
            [(i, N_ENCODE_SHARDS) for i in range(N_ENCODE_SHARDS)],
            order_outputs=False,
        )
    )
    mmeta = merge_codes.remote()
    return {"shards": enc, "merge": mmeta}


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    timeout=60 * 60 * 10,
    secrets=[hf_secret],
    memory=8192,
)
def finish_untagged_encode_train(epochs: int = 2, lora_r: int = 16) -> dict:
    """FAST path: tagged already packed+encoded. Add 1k untagged (parallel),
    encode only new IDs, merge, train. Does NOT wipe existing codes."""
    vol_mio.reload()
    vol_cuts.reload()
    train_fp = Path(DATA_DIR, "train.jsonl")
    hold_fp = Path(DATA_DIR, "holdout.jsonl")
    assert train_fp.is_file(), "missing data_v8/train.jsonl — pack tagged first"

    train_rows = [
        json.loads(l) for l in train_fp.read_text().splitlines() if l.strip()
    ]
    hold_rows = [
        json.loads(l) for l in hold_fp.read_text().splitlines() if l.strip()
    ]
    # drop any prior bare rows (retry-safe)
    train_rows = [r for r in train_rows if (r.get("tag1") or "none") != "none"]
    hold_rows = [r for r in hold_rows if (r.get("tag1") or "none") != "none"]
    exclude = sorted(
        {
            r["source_id"]
            for r in train_rows + hold_rows
            if r.get("source_id")
        }
    )
    print(
        f"finish: tagged_train={len(train_rows)} tagged_hold={len(hold_rows)} exclude={len(exclude)}",
        flush=True,
    )

    cands = scan_untagged_candidates.remote(
        exclude_source_ids=exclude, oversample=2500
    )
    by_pq: dict[str, list] = defaultdict(list)
    for c in cands:
        by_pq[c["parquet"]].append(c)
    print(f"untagged cands={len(cands)} parquets={len(by_pq)}", flush=True)
    results = list(
        extract_untagged_shard.starmap(
            [(pq, rows) for pq, rows in sorted(by_pq.items())],
            order_outputs=False,
        )
    )
    untagged: list[dict] = []
    for r in results:
        untagged.extend(r.get("rows") or [])
    good = []
    for r in untagged:
        try:
            sft_user_from_row(r)
            good.append(r)
        except ValueError as e:
            print("skip", e, flush=True)
    rng = random.Random(SEED)
    rng.shuffle(good)
    untagged = good[:UNTAGGED_N]
    if len(untagged) < UNTAGGED_N:
        raise RuntimeError(f"untagged only {len(untagged)} < {UNTAGGED_N}")

    n_u_hold = max(1, int(round(len(untagged) * HOLDOUT_SRC_FRAC)))
    u_hold, u_train = untagged[:n_u_hold], untagged[n_u_hold:]
    train_rows.extend(u_train)
    hold_rows.extend(u_hold)
    rng.shuffle(train_rows)
    rng.shuffle(hold_rows)

    for r in train_rows + hold_rows:
        sft_user_from_row(r)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> None:
        Path(DATA_DIR, name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    dump("train.jsonl", train_rows)
    dump("holdout.jsonl", hold_rows)
    dump("manifest.jsonl", train_rows + hold_rows)

    tag_tr = Counter(r["tag1"] for r in train_rows)
    meta = {
        "pack": PACK_NAME,
        "caption_contract": "inline_caption_field_only",
        "n_train": len(train_rows),
        "n_holdout": len(hold_rows),
        "n_untagged_train": len(u_train),
        "n_untagged_holdout": len(u_hold),
        "by_tag_train": dict(tag_tr.most_common()),
        "n_mid_tag_train": sum(
            1
            for r in train_rows
            if r["tag1"] not in ("", "none")
            and not r["caption"].startswith(f"[{r['tag1']}]")
        ),
        "finish_path": True,
        "seed": SEED,
    }
    Path(DATA_DIR, "pack_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    Path(MIO, "data/pack_meta_v8.json").write_text(json.dumps(meta, indent=2) + "\n")
    vol_mio.commit()
    vol_cuts.commit()
    print("PACK+", json.dumps(meta, indent=2), flush=True)

    # encode: keep existing shard caches; only new IDs written
    enc = list(
        encode_shard.starmap(
            [(i, N_ENCODE_SHARDS) for i in range(N_ENCODE_SHARDS)],
            order_outputs=False,
        )
    )
    print("ENCODE", enc, flush=True)
    mmeta = merge_codes.remote()
    print("MERGE", mmeta, flush=True)
    if mmeta["n_miss"] > 20:
        raise RuntimeError(f"encode misses {mmeta['n_miss']}")
    if mmeta["n_codes"] < len(train_rows) - 20:
        raise RuntimeError(f"codes {mmeta['n_codes']} << train {len(train_rows)}")

    tmeta = train.remote(epochs=epochs, lora_r=lora_r)
    print("TRAIN", tmeta, flush=True)
    return {"pack": meta, "encode": mmeta, "train": tmeta}


def _spawn(call, label: str) -> None:
    print(f"SPAWNED {label} call_id={call.object_id}", flush=True)
    print("Local exit OK. Watch https://modal.com/apps", flush=True)


@app.local_entrypoint()
def main(
    action: str = "pack",
    epochs: int = 2,
    lora_r: int = 16,
    adapter_name: str = "",
):
    if action == "pack":
        # pack has internal parallel extract — wait for meta (CPU-bound ~tens of min)
        print(run_pack.remote())
    elif action == "encode":
        _spawn(run_encode.spawn(), "encode_v8")
    elif action == "train":
        _spawn(
            train.spawn(epochs=epochs, lora_r=lora_r, adapter_name=adapter_name),
            "train_v8",
        )
    elif action == "all":
        _spawn(run_all.spawn(epochs=epochs, lora_r=lora_r), "all_v8")
    elif action == "finish":
        # tagged+codes already done — only untagged + encode-new + train
        _spawn(
            finish_untagged_encode_train.spawn(epochs=epochs, lora_r=lora_r),
            "finish_v8",
        )
    elif action == "pack_tagged_only":
        print(pack_tagged.remote())
    else:
        raise SystemExit("action=pack|encode|train|all|finish|pack_tagged_only")
