"""Pack Hindi IV cuts → MioCodec encode → Indic-Mio LoRA (Modal).

Cuts stay on volume real-voice-iv-cuts. Manifest + codes + adapter on
real-voice-mio.

Long jobs use .spawn() so local exits immediately — laptop/WSL close OK.
Watch progress on https://modal.com/apps (do not block on .remote()).

  modal run modal_iv_mio_train.py --action all
  # or: pack | encode | train | merge_train

  modal volume get real-voice-mio out/adapter_iv_v7 ./data/scale/train/adapter_iv
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import modal

APP = "real-voice-iv-mio-train"
VOL_MIO = "real-voice-mio"
VOL_CUTS = "real-voice-iv-cuts"
MIO = "/mio"
CUTS = "/cuts"
CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"
MODEL_ID = "SPRINGLab/Indic-Mio"
SPEECH_OFFSET = 151669
HOLDOUT_SRC_FRAC = 0.05
SEED = 7
N_ENCODE_SHARDS = 32

EMO_MAP = {
    "angry": "<angry>",
    "sad": "<sad>",
    "excited": "<happy>",
    "fear": "<fear>",
    "surprise": "<surprise>",
    "disgust": "<disgust>",
}
EVENT_TAGS = {
    "pause",
    "inhaling",
    "breathing",
    "thinking",
    "sigh",
    "whispering",
    "laugh",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
}

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
        "sentencepiece",
        "protobuf",
        "tqdm",
    )
    .pip_install("git+https://github.com/Aratako/MioCodec")
)

app = modal.App(APP, image=image)


def _hf_login() -> None:
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


def user_text_from_row(row: dict) -> str:
    body = (row.get("text") or "").strip()
    t1 = (row.get("tag1") or "none").strip()
    t2 = (row.get("tag2") or "").strip()
    events = []
    emo = None
    for t in (t1, t2):
        if not t or t == "none":
            continue
        if t in EMO_MAP:
            emo = EMO_MAP[t]
        else:
            events.append(f"[{t}]")
    prefix = "".join(events)
    if prefix:
        body = f"{prefix} {body}".strip()
    if emo:
        body = f"{body} {emo}".strip()
    return body


def _proposal_to_row(p: dict) -> dict:
    tag = p["tag"]
    text = (p.get("text_plain") or "").strip()
    return {
        "id": p["cut_id"],
        "audio": p.get("cut_wav") or f"hindi_cuts/{p['cut_id']}.wav",
        "text": text,
        "caption": p.get("caption_inline") or f"[{tag}] {text}".strip(),
        "tag1": tag,
        "tag2": "",
        "source": "indicvoices_hi_v7",
        "source_id": p.get("source_id"),
        "algo_version": p.get("algo_version"),
        "native_label": p.get("raw_tag") or tag,
    }


@app.function(
    volumes={MIO: vol_mio, CUTS: vol_cuts},
    timeout=60 * 30,
    memory=8192,
)
def pack(holdout_frac: float = HOLDOUT_SRC_FRAC, seed: int = SEED) -> dict:
    """Build train/holdout jsonl on mio vol from sliced Hindi proposals."""
    prop_fp = Path(CUTS, "hindi_split/iv_hindi_cut_proposals.json")
    doc = json.loads(prop_fp.read_text(encoding="utf-8"))
    props = [
        p
        for p in (doc.get("proposals") or [])
        if p.get("align_ok") and p.get("cut_wav")
    ]
    # verify wav exists
    ok = []
    miss = 0
    for p in props:
        if Path(CUTS, p["cut_wav"]).is_file():
            ok.append(p)
        else:
            miss += 1
    props = ok

    by_src: dict[str, list] = defaultdict(list)
    for p in props:
        by_src[p["source_id"]].append(p)
    srcs = sorted(by_src)
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_hold = max(1, int(round(len(srcs) * holdout_frac)))
    hold_srcs = set(srcs[:n_hold])
    train_rows, hold_rows = [], []
    for sid, plist in by_src.items():
        rows = [_proposal_to_row(p) for p in plist]
        (hold_rows if sid in hold_srcs else train_rows).extend(rows)

    rng.shuffle(train_rows)
    rng.shuffle(hold_rows)

    out = Path(MIO, "data")
    out.mkdir(parents=True, exist_ok=True)
    Path(MIO, "encoded").mkdir(parents=True, exist_ok=True)
    Path(MIO, "out").mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    dump(out / "train.jsonl", train_rows)
    dump(out / "holdout.jsonl", hold_rows)
    dump(out / "manifest.jsonl", train_rows + hold_rows)

    tag_tr = Counter(r["tag1"] for r in train_rows)
    tag_ho = Counter(r["tag1"] for r in hold_rows)
    meta = {
        "pack": "iv_hindi_v7_align_ok",
        "algo_version": doc.get("algo_version"),
        "n_train": len(train_rows),
        "n_holdout": len(hold_rows),
        "n_train_sources": len(srcs) - n_hold,
        "n_holdout_sources": n_hold,
        "wav_miss": miss,
        "holdout_frac": holdout_frac,
        "seed": seed,
        "by_tag_train": dict(tag_tr.most_common()),
        "by_tag_holdout": dict(tag_ho.most_common()),
        "cuts_volume": VOL_CUTS,
        "mio_volume": VOL_MIO,
        "p01_codec_rt": "PASS",
    }
    (out / "pack_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    vol_mio.commit()
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [
        json.loads(l)
        for l in Path(MIO, "data/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    mine = [r for i, r in enumerate(rows) if i % n_shards == shard_i]
    out_fp = Path(MIO, f"encoded/codes_{shard_i:02d}.jsonl")
    done = set()
    if out_fp.exists():
        for line in out_fp.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    codec = MioCodecModel.from_pretrained(CODEC_ID).eval().to(device)
    sr = int(codec.config.sample_rate)
    print(
        f"shard={shard_i} rows={len(mine)} cached={len(done)} sr={sr}",
        flush=True,
    )

    n_new = 0
    with out_fp.open("a", encoding="utf-8") as ef:
        for j, row in enumerate(mine):
            rid = row["id"]
            if rid in done:
                continue
            wav_p = Path(CUTS, row["audio"])
            if not wav_p.is_file():
                print("MISS", rid, flush=True)
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
    # shards commit from other containers — must reload before glob
    vol_mio.reload()
    rows = [
        json.loads(l)
        for l in Path(MIO, "data/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    need = {r["id"] for r in rows}
    shard_files = sorted(Path(MIO, "encoded").glob("codes_*.jsonl"))
    print(f"merge shards_found={len(shard_files)} need={len(need)}", flush=True)
    merged: dict[str, list] = {}
    for fp in shard_files:
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            merged[o["id"]] = o["codes"]
    out = Path(MIO, "encoded/codes.jsonl")
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
    Path(MIO, "encoded/encode_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    vol_mio.commit()
    print(meta, flush=True)
    return meta


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    volumes={MIO: vol_mio},
    secrets=[hf_secret],
)
def train(epochs: int = 2, lora_r: int = 16, max_rows: int = 0) -> dict:
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

    class _VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            vol_mio.commit()
            print(f"vol.commit @ step={state.global_step}", flush=True)

    _hf_login()
    vol_mio.reload()
    train_rows = [
        json.loads(l)
        for l in Path(MIO, "data/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    if max_rows:
        train_rows = train_rows[:max_rows]

    codes_map = {}
    for line in Path(MIO, "encoded/codes.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            codes_map[o["id"]] = o["codes"]
    print(f"train_rows={len(train_rows)} codes={len(codes_map)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    for row in train_rows:
        codes = codes_map.get(row["id"])
        if not codes:
            continue
        user = user_text_from_row(row)
        assistant = "".join(f"<|s_{int(c)}|>" for c in codes)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        records.append({"id": row["id"], "text": text, "user": user})
    print(f"sft records={len(records)}", flush=True)
    Path(MIO, "out/sft_preview.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records[:5]) + "\n",
        encoding="utf-8",
    )

    ds = Dataset.from_list(records)

    def tok_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    ds = ds.map(tok_fn, batched=True, remove_columns=[c for c in ds.column_names if c != "id"])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
    model.print_trainable_parameters()

    out_dir = f"{MIO}/out/adapter_iv_v7"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    resume = str(ckpts[-1]) if ckpts else None
    print(f"resume_from={resume}", flush=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
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
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "model_id": MODEL_ID,
        "codec_id": CODEC_ID,
        "epochs": epochs,
        "lora_r": lora_r,
        "n_train": len(records),
        "adapter": out_dir,
        "pack": "iv_hindi_v7_align_ok",
        "p01_codec_rt": "PASS",
        "resumed_from": resume,
    }
    Path(MIO, "out/train_meta_iv_v7.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    vol_mio.commit()
    print("SAVED", out_dir, flush=True)
    return meta


@app.function(volumes={MIO: vol_mio, CUTS: vol_cuts}, timeout=60 * 60 * 10, secrets=[hf_secret])
def run_all(epochs: int = 2, lora_r: int = 16) -> dict:
    """Pack → parallel encode → merge → LoRA train."""
    pmeta = pack.local()
    print("PACK", pmeta, flush=True)
    enc = list(
        encode_shard.starmap(
            [(i, N_ENCODE_SHARDS) for i in range(N_ENCODE_SHARDS)],
            order_outputs=False,
        )
    )
    print("ENCODE shards", enc, flush=True)
    # remote merge = fresh volume mount (local() misses sibling commits)
    mmeta = merge_codes.remote()
    print("MERGE", mmeta, flush=True)
    if mmeta["n_miss"] > 100:
        raise RuntimeError(f"too many encode misses: {mmeta['n_miss']}")
    tmeta = train.remote(epochs=epochs, lora_r=lora_r)
    print("TRAIN", tmeta, flush=True)
    return {"pack": pmeta, "encode": mmeta, "train": tmeta}


@app.function(volumes={MIO: vol_mio}, timeout=60 * 60 * 10, secrets=[hf_secret])
def run_merge_train(epochs: int = 2, lora_r: int = 16) -> dict:
    """Merge codes then train (one Modal-side chain — safe if laptop closes)."""
    mmeta = merge_codes.remote()
    print("MERGE", mmeta, flush=True)
    if mmeta["n_miss"] > 100:
        raise RuntimeError(f"too many encode misses: {mmeta['n_miss']}")
    tmeta = train.remote(epochs=epochs, lora_r=lora_r)
    print("TRAIN", tmeta, flush=True)
    return {"encode": mmeta, "train": tmeta}


@app.function(volumes={MIO: vol_mio, CUTS: vol_cuts}, timeout=60 * 60 * 3, secrets=[hf_secret])
def run_encode() -> dict:
    """Parallel encode + merge (Modal-side — safe if laptop closes)."""
    enc = list(
        encode_shard.starmap(
            [(i, N_ENCODE_SHARDS) for i in range(N_ENCODE_SHARDS)],
            order_outputs=False,
        )
    )
    mmeta = merge_codes.remote()
    print("ENCODE+MERGE", mmeta, flush=True)
    return {"shards": enc, "merge": mmeta}


def _spawn_and_exit(call, label: str) -> None:
    """Fire-and-forget: local process exits; GPU job keeps running on Modal."""
    print(f"SPAWNED {label} call_id={call.object_id}", flush=True)
    print("Local exit OK — close laptop fine. Watch https://modal.com/apps", flush=True)
    print("Status later: modal app list   |   logs on the app run page", flush=True)


@app.local_entrypoint()
def main(action: str = "all", epochs: int = 2, lora_r: int = 16):
    # Long GPU jobs: .spawn() then exit (do NOT block on .remote() — laptop/WSL
    # kill cancels the waiting client and can cancel the Modal input).
    if action == "pack":
        print(pack.remote())  # short CPU — ok to wait
    elif action == "encode":
        _spawn_and_exit(run_encode.spawn(), "encode")
    elif action == "train":
        _spawn_and_exit(train.spawn(epochs=epochs, lora_r=lora_r), "train")
    elif action == "merge_train":
        _spawn_and_exit(
            run_merge_train.spawn(epochs=epochs, lora_r=lora_r), "merge_train"
        )
    elif action == "all":
        _spawn_and_exit(run_all.spawn(epochs=epochs, lora_r=lora_r), "all")
    else:
        raise SystemExit("action=pack|encode|train|merge_train|all")
