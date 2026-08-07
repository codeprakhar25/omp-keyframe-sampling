"""Indic-Mio LoRA on Modal — tag-controllable Hindi/Hinglish TTS.

  # 1) upload train pack to Volume
  modal run modal_indic_mio.py::upload

  # 2) encode + LoRA (detach)
  modal run --detach modal_indic_mio.py::train

  # 3) pull adapter locally
  modal volume get real-voice-mio /vol/out/adapter ./data/scale/train/adapter

See data/scale/train/FINETUNE_PLAN.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "real-voice-mio"
VOL = "real-voice-mio"
ROOT = Path(__file__).resolve().parent
LOCAL_TRAIN = ROOT / "data" / "scale" / "train"
REMOTE = "/vol"

vol = modal.Volume.from_name(VOL, create_if_missing=True)
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
SPEECH_OFFSET = 151669


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
    """Map freeze tags → Indic-Mio style user string."""
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
        elif t in EVENT_TAGS:
            events.append(f"[{t}]")
        else:
            events.append(f"[{t}]")
    prefix = "".join(events)
    if prefix:
        body = f"{prefix} {body}".strip()
    if emo:
        body = f"{body} {emo}".strip()
    return body


@app.function(volumes={REMOTE: vol}, timeout=60 * 10)
def _mkdir() -> None:
    Path(f"{REMOTE}/data/wavs").mkdir(parents=True, exist_ok=True)
    Path(f"{REMOTE}/out").mkdir(parents=True, exist_ok=True)
    Path(f"{REMOTE}/encoded").mkdir(parents=True, exist_ok=True)
    vol.commit()


@app.function(volumes={REMOTE: vol}, timeout=60 * 45)
def _put(payload: bytes, rel: str) -> str:
    dest = Path(REMOTE) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    vol.commit()
    return str(dest)


@app.function(volumes={REMOTE: vol}, timeout=60 * 60)
def _put_many(items: list[tuple[str, bytes]]) -> int:
    n = 0
    for name, blob in items:
        dest = Path(f"{REMOTE}/data/wavs") / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        n += 1
    vol.commit()
    return n


def _upload_local() -> None:
    """Upload data/scale/train → Volume /vol/data."""
    print("mkdir…")
    _mkdir.remote()

    files = [
        LOCAL_TRAIN / "train.jsonl",
        LOCAL_TRAIN / "holdout.jsonl",
        LOCAL_TRAIN / "manifest.jsonl",
    ]
    for f in files:
        assert f.exists(), f
        print("put", f.name, f.stat().st_size)
        _put.remote(f.read_bytes(), f"data/{f.name}")

    wavs = sorted((LOCAL_TRAIN / "wavs").glob("*.wav"))
    print(f"uploading {len(wavs)} wavs…")
    batch: list[tuple[str, bytes]] = []
    total = 0
    for w in wavs:
        batch.append((w.name, w.read_bytes()))
        if len(batch) >= 20:
            total += _put_many.remote(batch)
            print(f"  {total}/{len(wavs)}")
            batch = []
    if batch:
        total += _put_many.remote(batch)
    print(f"uploaded wavs={total}")

@app.function(
    gpu="A10G",
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
    memory=32768,
)
def train(
    model_id: str = "SPRINGLab/Indic-Mio",
    codec_id: str = "Aratako/MioCodec-25Hz-24kHz",
    epochs: int = 2,
    lora_r: int = 16,
    max_rows: int = 0,
) -> dict:
    """Encode wavs with MioCodec → LoRA SFT on Indic-Mio → save adapter."""
    import torch
    from datasets import Dataset
    from miocodec import MioCodecModel, load_audio
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    train_path = Path(f"{REMOTE}/data/train.jsonl")
    rows = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    if max_rows:
        rows = rows[:max_rows]
    print(f"train rows={len(rows)}", flush=True)

    # --- encode ---
    enc_path = Path(f"{REMOTE}/encoded/codes.jsonl")
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if enc_path.exists():
        for line in enc_path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done[o["id"]] = o["codes"]
        print(f"resume encode: {len(done)} cached", flush=True)

    codec = MioCodecModel.from_pretrained(codec_id).eval().to(device)
    sr = codec.config.sample_rate
    print(f"codec sr={sr}", flush=True)

    with enc_path.open("a", encoding="utf-8") as ef:
        for i, row in enumerate(rows):
            rid = row["id"]
            if rid in done:
                continue
            wav_p = Path(f"{REMOTE}/data") / row["audio"]
            if not wav_p.exists():
                print("MISS", rid, wav_p, flush=True)
                continue
            with torch.inference_mode():
                waveform = load_audio(str(wav_p), sample_rate=sr).to(device)
                feats = codec.encode(waveform)
                codes = feats.content_token_indices.detach().cpu().view(-1).tolist()
            done[rid] = codes
            ef.write(json.dumps({"id": rid, "codes": codes}, ensure_ascii=False) + "\n")
            ef.flush()
            if (i + 1) % 10 == 0:
                print(f"encoded {i+1}/{len(rows)} last_len={len(codes)}", flush=True)
                vol.commit()
    vol.commit()
    del codec
    torch.cuda.empty_cache()
    print(f"encode done n={len(done)}", flush=True)

    # --- build SFT strings ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    for row in rows:
        codes = done.get(row["id"])
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
    Path(f"{REMOTE}/out/sft_preview.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records[:3]) + "\n",
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
        model_id,
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out_dir = f"{REMOTE}/out/adapter"
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
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
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "model_id": model_id,
        "codec_id": codec_id,
        "epochs": epochs,
        "lora_r": lora_r,
        "n_train": len(records),
        "emo_map": EMO_MAP,
    }
    Path(f"{REMOTE}/out/train_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    vol.commit()
    print("SAVED", out_dir, flush=True)
    return meta


@app.function(
    gpu="A10G",
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    timeout=60 * 60,
    memory=32768,
)
def synth(
    model_id: str = "SPRINGLab/Indic-Mio",
    codec_id: str = "Aratako/MioCodec-25Hz-24kHz",
    adapter_dir: str = f"{REMOTE}/out/adapter",
    pairs_path: str = f"{REMOTE}/data/eval_pairs.json",
    ref_wav: str = f"{REMOTE}/data/wavs/s01.wav",
    use_lora: bool = True,
    out_subdir: str = "eval_pairs",
) -> dict:
    """Synthesize tag-swap A/B wavs → /vol/{out_subdir}/.

    use_lora=True  → finetuned adapter (default)
    use_lora=False → base Indic-Mio only (control)
    """
    import numpy as np
    import soundfile as sf
    import torch
    from miocodec import MioCodecModel, load_audio
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    out_dir = Path(f"{REMOTE}/{out_subdir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = MioCodecModel.from_pretrained(codec_id).eval().to(device)
    sr = codec.config.sample_rate
    # speaker / style prior from a real train clip
    with torch.inference_mode():
        ref = load_audio(ref_wav, sample_rate=sr).to(device)
        ref_feats = codec.encode(ref)
        global_emb = ref_feats.global_embedding

    tok_src = adapter_dir if use_lora and Path(adapter_dir).exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if use_lora:
        model = PeftModel.from_pretrained(base, adapter_dir)
        print("LOADED LoRA", adapter_dir, flush=True)
    else:
        model = base
        print("LOADED base (no LoRA)", model_id, flush=True)
    model.eval()

    def make_user(text: str, tag: str) -> str:
        return user_text_from_row({"text": text, "tag1": tag or "none", "tag2": ""})

    def gen_codes(user: str) -> list[int]:
        messages = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        codes = [
            int(t) - SPEECH_OFFSET
            for t in gen.tolist()
            if SPEECH_OFFSET <= int(t) < SPEECH_OFFSET + 12800
        ]
        return codes

    def decode_wav(codes: list[int]) -> np.ndarray:
        if not codes:
            return np.zeros(sr // 2, dtype=np.float32)
        # MioCodec.decode expects (seq_len,) indices + (dim,) global
        idx = torch.tensor(codes, dtype=torch.long, device=device)
        g = global_emb
        if g.dim() > 1:
            g = g.squeeze(0)
        with torch.inference_mode():
            wav = codec.decode(
                global_embedding=g,
                content_token_indices=idx,
            )
        arr = wav.detach().float().cpu().numpy().reshape(-1)
        return arr

    n_ok = 0
    for p in pairs:
        pid = p["id"]
        text = p["text"]
        for side, tag in (("a", p["tag_a"]), ("b", p["tag_b"])):
            user = make_user(text, tag)
            print(f"synth {pid}_{side} tag={tag} user={user[:80]}", flush=True)
            codes = gen_codes(user)
            print(f"  codes={len(codes)}", flush=True)
            audio = decode_wav(codes)
            path = out_dir / f"{pid}_{side}.wav"
            sf.write(str(path), audio, sr)
            n_ok += 1
        vol.commit()

    vol.commit()
    return {
        "n_wavs": n_ok,
        "out": str(out_dir),
        "sr": sr,
        "use_lora": use_lora,
        "out_subdir": out_subdir,
    }


@app.local_entrypoint()
def main(action: str = "upload"):
    """modal run modal_indic_mio.py --action upload|train|synth|synth_base"""
    if action == "upload":
        _upload_local()
    elif action == "train":
        print(train.remote())
    elif action == "synth":
        local_pairs = ROOT / "data" / "scale" / "eval_synth_pairs.json"
        assert local_pairs.exists(), "run scripts/tag_swap_eval.py --init first"
        print("upload pairs…")
        _put.remote(local_pairs.read_bytes(), "data/eval_pairs.json")
        print(synth.remote(use_lora=True, out_subdir="eval_pairs"))
    elif action == "synth_base":
        local_pairs = ROOT / "data" / "scale" / "eval_synth_pairs.json"
        assert local_pairs.exists(), "run scripts/tag_swap_eval.py --init first"
        print("upload pairs…")
        _put.remote(local_pairs.read_bytes(), "data/eval_pairs.json")
        print(synth.remote(use_lora=False, out_subdir="eval_pairs_base"))
    else:
        raise SystemExit(f"unknown action {action}")
