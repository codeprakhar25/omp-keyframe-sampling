"""Orpheus Hindi LoRA on Modal — tag-controllable TTS (2nd base bake-off).

  # 1) upload train pack
  modal run modal_orpheus.py --action upload

  # 2) SNAC encode + LoRA (detach)
  modal run --detach modal_orpheus.py --action train

  # 3) synth tag-swap (LoRA / base)
  modal run --detach modal_orpheus.py --action synth
  modal run --detach modal_orpheus.py --action synth_base

Same Modal account as Indic-Mio; separate volume `real-voice-orpheus`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "real-voice-orpheus"
VOL = "real-voice-orpheus"
ROOT = Path(__file__).resolve().parent
LOCAL_TRAIN = ROOT / "data" / "scale" / "train"
REMOTE = "/vol"

DEFAULT_MODEL = "canopylabs/3b-hi-pretrain-research_release"
SNAC_ID = "hubertsiuzdak/snac_24khz"
SPEAKER = "hi"

# Orpheus special token ids (Llama-3.2-3B extended)
TOKENISER_LENGTH = 128256
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2  # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3  # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4  # 128260
START_OF_AI = TOKENISER_LENGTH + 5  # 128261
END_OF_AI = TOKENISER_LENGTH + 6  # 128262
PAD_TOKEN = TOKENISER_LENGTH + 7  # 128263
AUDIO_OFFSET = TOKENISER_LENGTH + 10  # 128266

# Orpheus has some verbal event tags; emotions are not native → bracket tags
ORPHEUS_NATIVE = {
    "laugh": "<laugh>",
    "sigh": "<sigh>",
    # gasp ~ surprise-ish event; keep surprise as bracket emo
}
EVENT_TAGS = {"pause", "inhaling", "thinking", "whispering", "laugh", "sigh"}
EMO_TAGS = {"angry", "sad", "excited", "fear", "surprise", "disgust"}

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
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "tqdm",
        "snac",
    )
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


def user_text_from_row(row: dict, speaker: str = SPEAKER) -> str:
    """Build Orpheus prompt text: `{speaker}: [tags…] transcript`.

    Emotions → `[angry]` etc (LoRA teach). Native Orpheus events where known.
    """
    body = (row.get("text") or "").strip()
    t1 = (row.get("tag1") or "none").strip()
    t2 = (row.get("tag2") or "").strip()
    prefix_parts: list[str] = []
    for t in (t1, t2):
        if not t or t == "none":
            continue
        if t in ORPHEUS_NATIVE:
            prefix_parts.append(ORPHEUS_NATIVE[t])
        elif t in EMO_TAGS or t in EVENT_TAGS:
            prefix_parts.append(f"[{t}]")
        else:
            prefix_parts.append(f"[{t}]")
    prefix = "".join(prefix_parts)
    if prefix:
        body = f"{prefix} {body}".strip()
    return f"{speaker}: {body}"


def snac_codes_to_ids(codes: list) -> list[int]:
    """Flatten SNAC hierarchical codes → Orpheus interleaved audio token ids."""
    all_codes: list[int] = []
    # codes = [layer0 [B,T], layer1 [B,2T], layer2 [B,4T]]
    t = codes[0].shape[1]
    for i in range(t):
        all_codes.append(int(codes[0][0][i].item()) + AUDIO_OFFSET)
        all_codes.append(int(codes[1][0][2 * i].item()) + AUDIO_OFFSET + 4096)
        all_codes.append(int(codes[2][0][4 * i].item()) + AUDIO_OFFSET + 2 * 4096)
        all_codes.append(int(codes[2][0][(4 * i) + 1].item()) + AUDIO_OFFSET + 3 * 4096)
        all_codes.append(int(codes[1][0][(2 * i) + 1].item()) + AUDIO_OFFSET + 4 * 4096)
        all_codes.append(int(codes[2][0][(4 * i) + 2].item()) + AUDIO_OFFSET + 5 * 4096)
        all_codes.append(int(codes[2][0][(4 * i) + 3].item()) + AUDIO_OFFSET + 6 * 4096)
    return all_codes


def remove_duplicate_frames(vals: list[int]) -> list[int]:
    if len(vals) % 7 != 0:
        vals = vals[: (len(vals) // 7) * 7]
    if not vals:
        return vals
    result = vals[:7]
    for i in range(7, len(vals), 7):
        if vals[i] != result[-7]:
            result.extend(vals[i : i + 7])
    return result


def redistribute_codes(code_list: list[int]):
    """Inverse of snac_codes_to_ids → 3 tensors for SNAC.decode."""
    import torch

    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 6) // 7):
        base = 7 * i
        if base + 6 >= len(code_list):
            break
        layer_1.append(code_list[base])
        layer_2.append(code_list[base + 1])
        layer_3.append(code_list[base + 2])
        layer_3.append(code_list[base + 3])
        layer_2.append(code_list[base + 4])
        layer_3.append(code_list[base + 5])
        layer_3.append(code_list[base + 6])
    codes = [
        torch.tensor(layer_1, dtype=torch.long).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.long).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.long).unsqueeze(0),
    ]
    return codes


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
    gpu="A100",
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
    memory=65536,
)
def train(
    model_id: str = DEFAULT_MODEL,
    epochs: int = 2,
    lora_r: int = 32,
    max_rows: int = 0,
    max_length: int = 4096,
) -> dict:
    """SNAC-encode wavs → Orpheus LoRA SFT → save adapter."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.transforms as T
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
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

    enc_path = Path(f"{REMOTE}/encoded/codes.jsonl")
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, list[int]] = {}
    if enc_path.exists():
        for line in enc_path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done[o["id"]] = o["codes"]
        print(f"resume encode: {len(done)} cached", flush=True)

    snac = SNAC.from_pretrained(SNAC_ID).eval().to(device)
    print("SNAC loaded", flush=True)

    with enc_path.open("a", encoding="utf-8") as ef:
        for i, row in enumerate(rows):
            rid = row["id"]
            if rid in done:
                continue
            wav_p = Path(f"{REMOTE}/data") / row["audio"]
            if not wav_p.exists():
                print("MISS", rid, wav_p, flush=True)
                continue
            try:
                audio, sr = sf.read(str(wav_p), always_2d=False)
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=-1)
                waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
                if sr != 24000:
                    waveform = T.Resample(orig_freq=sr, new_freq=24000)(waveform)
                waveform = waveform.unsqueeze(0).to(device)  # [1,1,T]
                with torch.inference_mode():
                    codes = snac.encode(waveform)
                ids = remove_duplicate_frames(snac_codes_to_ids(codes))
            except Exception as e:
                print(f"encode fail {rid}: {e}", flush=True)
                continue
            done[rid] = ids
            ef.write(json.dumps({"id": rid, "codes": ids}, ensure_ascii=False) + "\n")
            ef.flush()
            if (i + 1) % 10 == 0:
                print(f"encoded {i+1}/{len(rows)} last_len={len(ids)}", flush=True)
                vol.commit()
    vol.commit()
    del snac
    torch.cuda.empty_cache()
    print(f"encode done n={len(done)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = PAD_TOKEN

    records = []
    for row in rows:
        codes = done.get(row["id"])
        if not codes:
            continue
        user = user_text_from_row(row)
        text_ids = tokenizer.encode(user, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        input_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + codes
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )
        if len(input_ids) > max_length:
            # keep prompt + truncated audio + closers
            keep_tail = 2  # EOS + EOA
            prompt_len = 1 + len(text_ids) + 1 + 1 + 1  # SOH..SOA+SOS
            max_audio = max_length - prompt_len - keep_tail
            max_audio = (max_audio // 7) * 7
            if max_audio < 7:
                continue
            codes_t = codes[:max_audio]
            input_ids = (
                [START_OF_HUMAN]
                + text_ids
                + [END_OF_HUMAN]
                + [START_OF_AI]
                + [START_OF_SPEECH]
                + codes_t
                + [END_OF_SPEECH]
                + [END_OF_AI]
            )
        records.append(
            {
                "id": row["id"],
                "input_ids": input_ids,
                "labels": input_ids[:],
                "attention_mask": [1] * len(input_ids),
                "user": user,
            }
        )
    print(f"sft records={len(records)}", flush=True)
    Path(f"{REMOTE}/out/sft_preview.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"id": r["id"], "user": r["user"], "len": len(r["input_ids"])},
                ensure_ascii=False,
            )
            for r in records[:3]
        )
        + "\n",
        encoding="utf-8",
    )

    ds = Dataset.from_list(
        [
            {
                "input_ids": r["input_ids"],
                "labels": r["labels"],
                "attention_mask": r["attention_mask"],
            }
            for r in records
        ]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
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
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

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
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [PAD_TOKEN] * pad)
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
        "codec_id": SNAC_ID,
        "epochs": epochs,
        "lora_r": lora_r,
        "n_train": len(records),
        "speaker": SPEAKER,
        "prompt": "{speaker}: [tags] text",
    }
    Path(f"{REMOTE}/out/train_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    vol.commit()
    print("SAVED", out_dir, flush=True)
    return meta


@app.function(
    gpu="A100",
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    timeout=60 * 90,
    memory=65536,
)
def synth(
    model_id: str = DEFAULT_MODEL,
    adapter_dir: str = f"{REMOTE}/out/adapter",
    pairs_path: str = f"{REMOTE}/data/eval_pairs.json",
    use_lora: bool = True,
    out_subdir: str = "eval_pairs",
    max_new_tokens: int = 1800,
) -> dict:
    """Synthesize tag-swap A/B wavs → /vol/{out_subdir}/."""
    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    out_dir = Path(f"{REMOTE}/{out_subdir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    snac = SNAC.from_pretrained(SNAC_ID).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = PAD_TOKEN

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if use_lora:
        # prefer adapter root; fall back to latest checkpoint-*
        ad = Path(adapter_dir)
        if not (ad / "adapter_config.json").exists():
            ckpts = sorted(ad.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                ad = ckpts[-1]
        model = PeftModel.from_pretrained(base, str(ad))
        print("LOADED LoRA", ad, flush=True)
    else:
        model = base
        print("LOADED base (no LoRA)", model_id, flush=True)
    model.eval()

    def make_user(text: str, tag: str) -> str:
        return user_text_from_row({"text": text, "tag1": tag or "none", "tag2": ""})

    def gen_audio_ids(user: str) -> list[int]:
        text_ids = tokenizer.encode(user, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        prompt = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
        )
        input_ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
        attn = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH,
                pad_token_id=PAD_TOKEN,
            )
        gen = out[0][input_ids.shape[1] :].tolist()
        # strip trailing end tokens; keep audio-range only
        audio = []
        for t in gen:
            if t == END_OF_SPEECH:
                break
            if AUDIO_OFFSET <= t < AUDIO_OFFSET + 7 * 4096:
                # undo layer offsets → raw codebook indices 0..4095 for redistribute
                # redistribute expects already-offset-removed per-slot values in canopy
                # convention: subtract AUDIO_OFFSET + slot*4096 when packing inverse
                audio.append(t)
        return audio

    def ids_to_snac_codes(token_ids: list[int]) -> list:
        """Convert model audio token ids → SNAC codebook indices tensors."""
        import torch as _torch

        raw = []
        for i, t in enumerate(token_ids):
            slot = i % 7
            raw.append(t - AUDIO_OFFSET - slot * 4096)
        # clamp junk
        raw = [max(0, min(4095, x)) for x in raw]
        return redistribute_codes(raw)

    def decode_wav(token_ids: list[int]) -> np.ndarray:
        if len(token_ids) < 7:
            return np.zeros(24000 // 2, dtype=np.float32)
        # trim to multiple of 7
        token_ids = token_ids[: (len(token_ids) // 7) * 7]
        codes = ids_to_snac_codes(token_ids)
        codes = [c.to(device) for c in codes]
        with torch.inference_mode():
            audio_hat = snac.decode(codes)
        arr = audio_hat.detach().float().cpu().numpy().reshape(-1)
        return arr

    n_ok = 0
    for p in pairs:
        pid = p["id"]
        text = p["text"]
        for side, tag in (("a", p["tag_a"]), ("b", p["tag_b"])):
            user = make_user(text, tag)
            print(f"synth {pid}_{side} tag={tag} user={user[:100]}", flush=True)
            tids = gen_audio_ids(user)
            print(f"  audio_tokens={len(tids)}", flush=True)
            audio = decode_wav(tids)
            path = out_dir / f"{pid}_{side}.wav"
            sf.write(str(path), audio, 24000)
            n_ok += 1
        vol.commit()

    vol.commit()
    return {
        "n_wavs": n_ok,
        "out": str(out_dir),
        "sr": 24000,
        "use_lora": use_lora,
        "out_subdir": out_subdir,
    }


@app.local_entrypoint()
def main(action: str = "upload"):
    """modal run modal_orpheus.py --action upload|train|synth|synth_base"""
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
