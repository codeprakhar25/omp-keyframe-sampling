"""v9: tag-contrastive, event-window-weighted LoRA.

Why this exists
---------------
Teacher-forced ablation on v8_e3 measured the actual defect:

  LoRA gain over base      1.067 nats
  d_wrong (tag identity)   0.052 nats   -> ~5% of what the LoRA learned
  best 1 s window          0.200 nats   -> 3.9x concentration, peak at the tag

So the model learned "a bracket means an event happens near here" but barely
learned WHICH event, and nothing in the SFT objective ever asked it to. Row
count does not predict per-tag identity (laugh @426 beats tsk @1412), so more
data is not the lever.

v9 changes the objective, not the data:

  1. event-window weighting  — upweight code tokens near the tag. The window is
     placed from the caption alone: tag word-fraction ~= audio time-fraction
     (measured 0.435 vs profile peak 0.40-0.45, so no aligner needed).

  2. tag-contrastive hinge   — same clip, same codes, wrong tag in the caption.
     Push NLL(wrong) above NLL(right) by `margin`, scored on the window only.
     Hinge, not raw maximisation, so it stops once the gap is achieved and
     cannot run away.

  L = CE_right_all
    + alpha * CE_right_window
    + beta  * max(0, margin - (CE_wrong_window - CE_right_window))

Costs 2 forwards per step. Reuses data_v8 / encoded_v8 — no re-pack, no encode.

  modal run --detach modal_iv_v9_contrastive.py --epochs 1
  modal run modal_tag_ablation.py --adapter adapter_iv_v9_contrast --n 300
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import modal

APP = "real-voice-iv-v9-contrast"
VOL_MIO = "real-voice-mio"
MIO = "/mio"
MODEL_ID = "SPRINGLab/Indic-Mio"
CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"
DATA_DIR = f"{MIO}/data_v8"
ENCODE_DIR = f"{MIO}/encoded_v8"
OUT_DIR = f"{MIO}/out"
PACK_NAME = "iv_hindi_v8_inline"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "numpy",
    )
)

app = modal.App(APP, image=image)
vol_mio = modal.Volume.from_name(VOL_MIO, create_if_missing=False)


def swap_tag(caption: str, tag: str, other: str) -> str:
    return caption.replace(f"[{tag}]", f"[{other}]", 1)


@app.function(gpu="A10G", volumes={MIO: vol_mio}, timeout=60 * 60 * 6, memory=32768)
def train(
    epochs: int = 1,
    lora_r: int = 16,
    alpha_window: float = 2.0,
    beta_contrast: float = 1.0,
    margin: float = 0.5,
    win_tokens: int = 25,
    adapter_name: str = "adapter_iv_v9_contrast",
    max_rows: int = 0,
) -> dict:
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    vol_mio.reload()

    codes_map = {}
    with open(f"{ENCODE_DIR}/codes.jsonl") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                codes_map[o["id"]] = o["codes"]
    rows = [
        json.loads(l)
        for l in Path(f"{DATA_DIR}/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rows = [r for r in rows if r["id"] in codes_map]
    if max_rows:
        rows = rows[:max_rows]
    all_tags = sorted(
        {r["tag1"] for r in rows if (r.get("tag1") or "none") not in ("", "none")}
    )
    print(f"rows={len(rows)} tags={all_tags}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    code_re = re.compile(r"^<\|s_\d+\|>$")
    code_ids = {i for t, i in tok.get_vocab().items() if code_re.match(t)}
    assert code_ids, "no code tokens in vocab"

    def render(user: str, codes: list) -> str:
        return tok.apply_chat_template(
            [
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": "".join(f"<|s_{int(c)}|>" for c in codes),
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def encode(text: str):
        ids = tok(text, truncation=True, max_length=2048).input_ids
        first = next((j for j, t in enumerate(ids) if t in code_ids), None)
        return ids, first

    rng = random.Random(7)
    feats, n_tagged, n_bare = [], 0, 0
    for r in rows:
        tag = (r.get("tag1") or "none").strip()
        cap = r["caption"]
        codes = codes_map[r["id"]]
        ids, first = encode(render(cap, codes))
        if first is None:
            continue
        lab = [-100] * first + ids[first:]

        # event window from the caption's word position (no aligner needed)
        n_code = len(ids) - first
        if tag not in ("", "none") and f"[{tag}]" in cap:
            words = cap.split()
            pos = next((j for j, w in enumerate(words) if f"[{tag}]" in w), 0)
            frac = pos / max(1, len(words) - 1)
            centre = first + int(frac * n_code)
            lo = max(first, centre - win_tokens // 2)
            hi = min(len(ids), centre + win_tokens // 2)
            other = rng.choice([t for t in all_tags if t != tag]) if len(all_tags) > 1 else tag
            w_ids, w_first = encode(render(swap_tag(cap, tag, other), codes))
            n_tagged += 1
        else:  # bare row: plain LM only, no window, no contrast
            lo = hi = -1
            other = ""
            w_ids, w_first = [], -1
            n_bare += 1

        feats.append(
            {
                "input_ids": ids,
                "labels": lab,
                "win_lo": lo,
                "win_hi": hi,
                "wrong_ids": w_ids,
                "wrong_first": w_first if w_first is not None else -1,
                "first": first,
            }
        )

    print(f"features={len(feats)} tagged={n_tagged} bare={n_bare}", flush=True)
    ds = Dataset.from_list(feats)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    model.config.use_cache = False

    pad_id = tok.pad_token_id

    def collate(fs):
        # batch size 1 keeps the two-forward bookkeeping simple and exact
        f = fs[0]
        out = {
            "input_ids": torch.tensor([f["input_ids"]]),
            "labels": torch.tensor([f["labels"]]),
            "win_lo": f["win_lo"],
            "win_hi": f["win_hi"],
            "first": f["first"],
        }
        if f["wrong_ids"]:
            out["wrong_ids"] = torch.tensor([f["wrong_ids"]])
            out["wrong_first"] = f["wrong_first"]
        return out

    stats = {"n": 0, "n_tag": 0, "ce": 0.0, "ce_win": 0.0, "gap": 0.0, "hinge": 0.0}

    class ContrastTrainer(Trainer):
        # NOTE: with an overridden compute_loss, Trainer does NOT divide by
        # gradient_accumulation_steps (it assumes we already normalised). The
        # smoke run showed logged loss = 8x the true value and grad_norm 24 vs
        # ~1.0 for v8 — i.e. an 8x effective LR. We divide explicitly.
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            ids = inputs["input_ids"].to(model.device)
            labels = inputs["labels"].to(model.device)
            first = int(inputs["first"])
            lo, hi = int(inputs["win_lo"]), int(inputs["win_hi"])

            logits = model(input_ids=ids).logits[0, :-1].float()
            tgt = ids[0, 1:]
            nll = F.cross_entropy(logits, tgt, reduction="none")  # per position
            code_mask = torch.zeros_like(nll, dtype=torch.bool)
            code_mask[first - 1 :] = True
            ce_all = nll[code_mask].mean()

            loss = ce_all
            ce_win = torch.tensor(0.0, device=loss.device)
            gap = torch.tensor(0.0, device=loss.device)
            hinge = torch.tensor(0.0, device=loss.device)

            if lo >= 0 and hi > lo + 2 and "wrong_ids" in inputs:
                win = torch.zeros_like(nll, dtype=torch.bool)
                win[max(0, lo - 1) : max(0, hi - 1)] = True
                if win.any():
                    ce_win = nll[win].mean()
                    loss = loss + alpha_window * ce_win

                    w_ids = inputs["wrong_ids"].to(model.device)
                    w_first = int(inputs["wrong_first"])
                    w_logits = model(input_ids=w_ids).logits[0, :-1].float()
                    w_nll = F.cross_entropy(w_logits, w_ids[0, 1:], reduction="none")
                    # same relative window in the wrong-tag sequence
                    off = w_first - first
                    wlo = max(0, lo - 1 + off)
                    whi = min(len(w_nll), hi - 1 + off)
                    if whi > wlo + 2:
                        ce_win_wrong = w_nll[wlo:whi].mean()
                        gap = ce_win_wrong - ce_win
                        hinge = torch.clamp(margin - gap, min=0.0)
                        loss = loss + beta_contrast * hinge
                        stats["n_tag"] += 1
                        stats["gap"] += float(gap)
                        stats["hinge"] += float(hinge)

            stats["n"] += 1
            stats["ce"] += float(ce_all)
            stats["ce_win"] += float(ce_win)
            if stats["n"] % 500 == 0:
                k, kt = stats["n"], max(1, stats["n_tag"])
                print(
                    f"    [contrast] seen={k}  ce={stats['ce']/k:.4f} "
                    f"ce_win={stats['ce_win']/k:.4f} gap={stats['gap']/kt:+.4f} "
                    f"hinge={stats['hinge']/kt:.4f}",
                    flush=True,
                )
            return loss / self.args.gradient_accumulation_steps

    out_dir = f"/tmp/{adapter_name}"
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
        save_steps=200,
        save_total_limit=8,
        bf16=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = ContrastTrainer(
        model=model, args=args, train_dataset=ds, data_collator=collate
    )
    trainer.train()

    import shutil

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol_mio.reload()
    dest = f"{OUT_DIR}/{adapter_name}"
    if Path(dest).exists():
        shutil.rmtree(dest)
    shutil.copytree(out_dir, dest)

    k = max(1, stats["n"])
    meta = {
        "model_id": MODEL_ID,
        "codec_id": CODEC_ID,
        "pack": PACK_NAME,
        "adapter": dest,
        "epochs": epochs,
        "lora_r": lora_r,
        "objective": "CE_all + alpha*CE_window + beta*hinge(margin - (CE_win_wrong - CE_win_right))",
        "alpha_window": alpha_window,
        "beta_contrast": beta_contrast,
        "margin": margin,
        "win_tokens": win_tokens,
        "n_rows": len(feats),
        "n_tagged": n_tagged,
        "n_bare": n_bare,
        "final_mean_ce": round(stats["ce"] / k, 5),
        "final_mean_ce_window": round(stats["ce_win"] / k, 5),
        "final_mean_gap": round(stats["gap"] / max(1, stats["n_tag"]), 5),
        "final_mean_hinge": round(stats["hinge"] / max(1, stats["n_tag"]), 5),
        "note": "gap = CE(wrong tag, window) - CE(right tag, window). Training pushes it toward `margin`.",
    }
    Path(OUT_DIR, f"train_meta_{adapter_name}.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    vol_mio.commit()
    print("SAVED", dest, json.dumps(meta, indent=2), flush=True)
    return meta


@app.local_entrypoint()
def main(
    epochs: int = 1,
    lora_r: int = 16,
    alpha_window: float = 2.0,
    beta_contrast: float = 1.0,
    margin: float = 0.5,
    adapter_name: str = "adapter_iv_v9_contrast",
    max_rows: int = 0,
):
    r = train.remote(
        epochs=epochs,
        lora_r=lora_r,
        alpha_window=alpha_window,
        beta_contrast=beta_contrast,
        margin=margin,
        adapter_name=adapter_name,
        max_rows=max_rows,
    )
    print(json.dumps(r, indent=2))
