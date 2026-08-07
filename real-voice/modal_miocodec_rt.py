"""P0.1 MioCodec round-trip kill-switch.

Sample Hindi IV cuts from volume → encode → decode with (a) own global emb
and (b) neutral speaker emb → write wavs for ear.

  modal run modal_miocodec_rt.py
  modal volume get real-voice-iv-cuts /vol/hindi_codec_rt ./data/samples/codec_rt
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import modal

APP = "real-voice-miocodec-rt"
VOL = "real-voice-iv-cuts"
REMOTE = "/vol"
CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"
OUT_REL = "hindi_codec_rt"
SEED = 7

# 4 whisper · 2 sigh · 2 inhaling · 2 thinking (speech-ish control)
SAMPLE_PLAN = {
    "whispering": 4,
    "sigh": 2,
    "inhaling": 2,
    "thinking": 2,
}

vol = modal.Volume.from_name(VOL, create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "soundfile",
        "numpy",
        "einops",
        "julius",
        "safetensors",
        "huggingface_hub",
        "tqdm",
    )
    .pip_install("git+https://github.com/Aratako/MioCodec")
)

app = modal.App(APP, image=image)


def _hf_login() -> None:
    import os

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


def _pick_samples(proposals: list[dict], seed: int = SEED) -> list[dict]:
    by_tag: dict[str, list] = defaultdict(list)
    for p in proposals:
        if not p.get("align_ok") or not p.get("cut_wav"):
            continue
        by_tag[p["tag"]].append(p)
    rng = random.Random(seed)
    picked: list[dict] = []
    used_ids: set[str] = set()
    for tag, n in SAMPLE_PLAN.items():
        pool = [x for x in by_tag.get(tag, []) if x["cut_id"] not in used_ids]
        # prefer mid-length cuts (more audible event + speech)
        pool.sort(key=lambda x: abs(float(x.get("cut_dur_s") or 0) - 2.5))
        take = pool[: max(n * 3, n)]
        rng.shuffle(take)
        for p in take[:n]:
            picked.append(p)
            used_ids.add(p["cut_id"])
    return picked


def _pick_neutral_ref(proposals: list[dict], exclude_sources: set[str], seed: int = SEED) -> dict:
    """Speech-forward cut from a different source for neutral global emb."""
    rng = random.Random(seed + 99)
    pool = [
        p
        for p in proposals
        if p.get("align_ok")
        and p.get("cut_wav")
        and p.get("tag") in ("thinking", "tsk")
        and p.get("source_id") not in exclude_sources
        and 1.5 <= float(p.get("cut_dur_s") or 0) <= 4.0
    ]
    if not pool:
        pool = [
            p
            for p in proposals
            if p.get("align_ok")
            and p.get("cut_wav")
            and p.get("source_id") not in exclude_sources
        ]
    rng.shuffle(pool)
    return pool[0]


@app.function(
    gpu="T4",
    timeout=60 * 40,
    memory=16384,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
)
def run_rt(seed: int = SEED) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from miocodec import MioCodecModel, load_audio

    _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_path = Path(REMOTE, "hindi_split/iv_hindi_cut_proposals.json")
    doc = json.loads(prop_path.read_text(encoding="utf-8"))
    proposals = doc.get("proposals") or []
    samples = _pick_samples(proposals, seed=seed)
    if len(samples) < 8:
        raise RuntimeError(f"too few samples: {len(samples)}")

    exclude = {p.get("source_id") for p in samples}
    neutral = _pick_neutral_ref(proposals, exclude, seed=seed)
    print(
        f"samples={len(samples)} neutral_ref={neutral['cut_id']} tag={neutral['tag']}",
        flush=True,
    )

    out_dir = Path(REMOTE, OUT_REL)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.is_file():
            old.unlink()

    codec = MioCodecModel.from_pretrained(CODEC_ID).eval().to(device)
    sr = int(codec.config.sample_rate)
    print(f"codec={CODEC_ID} sr={sr} device={device}", flush=True)

    with torch.inference_mode():
        neu_wav = load_audio(
            str(Path(REMOTE, neutral["cut_wav"])), sample_rate=sr
        ).to(device)
        neu_feats = codec.encode(neu_wav)
        neu_g = neu_feats.global_embedding
        if neu_g.dim() > 1:
            neu_g = neu_g.squeeze(0)

    # copy neutral ref original for ear context
    neu_src = Path(REMOTE, neutral["cut_wav"])
    neu_dst = out_dir / f"_neutral_ref_{neutral['cut_id']}_orig.wav"
    neu_dst.write_bytes(neu_src.read_bytes())

    rows_out = []
    for i, p in enumerate(samples):
        cid = p["cut_id"]
        tag = p["tag"]
        src = Path(REMOTE, p["cut_wav"])
        if not src.is_file():
            print("MISS", cid, flush=True)
            continue
        # keep original
        (out_dir / f"{i:02d}_{tag}_{cid}_orig.wav").write_bytes(src.read_bytes())

        with torch.inference_mode():
            wave = load_audio(str(src), sample_rate=sr).to(device)
            feats = codec.encode(wave)
            codes = feats.content_token_indices
            own_g = feats.global_embedding
            if own_g.dim() > 1:
                own_g = own_g.squeeze(0)

            rt_own = codec.decode(
                global_embedding=own_g, content_token_indices=codes
            )
            rt_neu = codec.decode(
                global_embedding=neu_g, content_token_indices=codes
            )

        own_arr = rt_own.detach().float().cpu().numpy().reshape(-1)
        neu_arr = rt_neu.detach().float().cpu().numpy().reshape(-1)
        sf.write(str(out_dir / f"{i:02d}_{tag}_{cid}_rt_own.wav"), own_arr, sr)
        sf.write(str(out_dir / f"{i:02d}_{tag}_{cid}_rt_neutral.wav"), neu_arr, sr)

        row = {
            "i": i,
            "cut_id": cid,
            "tag": tag,
            "source_id": p.get("source_id"),
            "cut_dur_s": p.get("cut_dur_s"),
            "caption_inline": p.get("caption_inline"),
            "n_codes": int(codes.numel()),
            "files": {
                "orig": f"{i:02d}_{tag}_{cid}_orig.wav",
                "rt_own": f"{i:02d}_{tag}_{cid}_rt_own.wav",
                "rt_neutral": f"{i:02d}_{tag}_{cid}_rt_neutral.wav",
            },
        }
        rows_out.append(row)
        print(f"OK {i:02d} [{tag}] codes={row['n_codes']} {cid}", flush=True)

    meta = {
        "algo": "P0.1 MioCodec round-trip",
        "codec_id": CODEC_ID,
        "sr": sr,
        "seed": seed,
        "sample_plan": SAMPLE_PLAN,
        "neutral_ref": {
            "cut_id": neutral["cut_id"],
            "tag": neutral["tag"],
            "source_id": neutral.get("source_id"),
            "caption_inline": neutral.get("caption_inline"),
            "file": neu_dst.name,
        },
        "listen_guide": {
            "orig": "ground-truth slice",
            "rt_own": "content tokens + SAME clip global emb",
            "rt_neutral": "content tokens + NEUTRAL speaker global emb",
            "verdict": (
                "survive both → train OK; "
                "own only → event in global emb, drop whisper this pass; "
                "die both → content stream blind, rethink"
            ),
        },
        "rows": rows_out,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    vol.commit()
    return {
        "n": len(rows_out),
        "out": f"{REMOTE}/{OUT_REL}",
        "neutral": neutral["cut_id"],
        "tags": {r["tag"]: sum(1 for x in rows_out if x["tag"] == r["tag"]) for r in rows_out},
    }


@app.local_entrypoint()
def main(seed: int = SEED):
    print(run_rt.remote(seed=seed))
    print(f"pull: modal volume get {VOL} {REMOTE}/{OUT_REL} ./data/samples/codec_rt")
