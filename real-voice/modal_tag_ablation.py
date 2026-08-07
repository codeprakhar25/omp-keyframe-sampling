"""Teacher-forced tag ablation — does the tag reach the model at all?

No generation, no ear. For each held-out clip we already have the TRUE audio
codes. Score those same codes three ways and compare:

  right  caption exactly as trained          "हाँ [sigh] मैं अभी बताता हूँ"
  none   the tag removed                     "हाँ मैं अभी बताता हूँ"
  wrong  the tag swapped for another         "हाँ [tsk] मैं अभी बताता हूँ"

  d_none  = NLL(none)  - NLL(right)   > 0 -> having a tag helps predict the audio
  d_wrong = NLL(wrong) - NLL(right)   > 0 -> WHICH tag it is matters

d_wrong is the one that defines tag control. If it is ~0 the model is ignoring
tag identity, and no amount of extra data, epochs or rank will fix that — it is
a conditioning problem, not a capacity problem.

  modal run modal_tag_ablation.py --adapter adapter_iv_v8_e3
  modal run modal_tag_ablation.py --adapter ""                  # base control
  modal run modal_tag_ablation.py --adapter adapter_iv_v8_e1_masked --n 400
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import modal

APP = "real-voice-tag-ablation"
VOL_MIO = "real-voice-mio"
MIO = "/mio"
MODEL_ID = "SPRINGLab/Indic-Mio"
DATA_DIR = f"{MIO}/data_v8"
ENCODE_DIR = f"{MIO}/encoded_v8"
OUT_DIR = f"{MIO}/out"

BRACKET_TAG_RE = re.compile(r"\[([a-z_]+)\]")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "numpy",
    )
)

app = modal.App(APP, image=image)
vol_mio = modal.Volume.from_name(VOL_MIO, create_if_missing=False)


def strip_tag(caption: str, tag: str) -> str:
    out = caption.replace(f"[{tag}]", " ")
    return re.sub(r"\s+", " ", out).strip()


def swap_tag(caption: str, tag: str, other: str) -> str:
    return caption.replace(f"[{tag}]", f"[{other}]", 1)


@app.function(gpu="A10G", volumes={MIO: vol_mio}, timeout=60 * 60 * 2, memory=32768)
def ablate(adapter: str = "adapter_iv_v8_e3", n: int = 300, seed: int = 7) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vol_mio.reload()

    # ---- data: prefer holdout, fall back to train rows that have codes ----
    codes_map: dict[str, list] = {}
    with open(f"{ENCODE_DIR}/codes.jsonl") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                codes_map[o["id"]] = o["codes"]

    def load(fp: str) -> list[dict]:
        p = Path(fp)
        if not p.is_file():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    holdout = load(f"{DATA_DIR}/holdout.jsonl")
    train = load(f"{DATA_DIR}/train.jsonl")

    def usable(rows: list[dict]) -> list[dict]:
        return [
            r
            for r in rows
            if r["id"] in codes_map
            and (r.get("tag1") or "none") not in ("", "none")
            and f"[{r['tag1']}]" in (r.get("caption") or "")
        ]

    pool, split = usable(holdout), "holdout"
    if len(pool) < 20:
        pool, split = usable(train), "train (holdout not encoded — memorisation caveat applies)"

    all_tags = sorted({r["tag1"] for r in pool})
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:n]
    print(f"split={split} n={len(pool)} tags={all_tags}", flush=True)
    if not pool:
        return {"error": "no usable rows with codes + inline tag"}

    # ---- model ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"{OUT_DIR}/{adapter}")
        print(f"adapter loaded: {adapter}", flush=True)
    else:
        print("BASE model, no adapter (control arm)", flush=True)
    model.eval()

    code_re = re.compile(r"^<\|s_\d+\|>$")
    code_ids = {i for t, i in tok.get_vocab().items() if code_re.match(t)}
    assert code_ids, "no code tokens in vocab"

    @torch.no_grad()
    def nll(user: str, codes: list) -> tuple[float, int]:
        """mean negative log-likelihood over the true code tokens, in nats."""
        assistant = "".join(f"<|s_{int(c)}|>" for c in codes)
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(text, truncation=True, max_length=2048).input_ids
        first = next((j for j, t in enumerate(ids) if t in code_ids), None)
        if first is None:
            return float("nan"), 0
        x = torch.tensor([ids], device="cuda")
        logits = model(x).logits[0, :-1].float()
        tgt = x[0, 1:]
        lp = torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0]
        # positions >= first are code tokens; label index i predicts token i+1
        mask = torch.zeros_like(lp, dtype=torch.bool)
        mask[first - 1 :] = True
        return float(-lp[mask].mean()), int(mask.sum())

    rows = []
    for i, r in enumerate(pool, 1):
        tag, cap, codes = r["tag1"], r["caption"], codes_map[r["id"]]
        others = [t for t in all_tags if t != tag]
        wrong_tag = rng.choice(others) if others else tag
        try:
            n_right, ntok = nll(cap, codes)
            n_none, _ = nll(strip_tag(cap, tag), codes)
            n_wrong, _ = nll(swap_tag(cap, tag, wrong_tag), codes)
        except Exception as e:
            print(f"  [{i}] {r['id']} FAIL {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(
            {
                "id": r["id"],
                "tag": tag,
                "wrong_tag": wrong_tag,
                "n_code_tokens": ntok,
                "nll_right": round(n_right, 5),
                "nll_none": round(n_none, 5),
                "nll_wrong": round(n_wrong, 5),
                "d_none": round(n_none - n_right, 5),
                "d_wrong": round(n_wrong - n_right, 5),
            }
        )
        if i % 25 == 0:
            print(f"  [{i}/{len(pool)}] running", flush=True)

    def summarise(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        dn = [r["d_none"] for r in rs]
        dw = [r["d_wrong"] for r in rs]

        def stat(v: list[float]) -> dict:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
            se = sd / math.sqrt(len(v))
            return {
                "mean": round(m, 5),
                "sd": round(sd, 5),
                "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)],
                "frac_positive": round(sum(1 for x in v if x > 0) / len(v), 4),
            }

        return {
            "n": len(rs),
            "nll_right_mean": round(sum(r["nll_right"] for r in rs) / len(rs), 5),
            "d_none": stat(dn),
            "d_wrong": stat(dw),
        }

    by_tag = {}
    for t in all_tags:
        sub = [r for r in rows if r["tag"] == t]
        if sub:
            by_tag[t] = summarise(sub)

    res = {
        "adapter": adapter or "BASE (no adapter)",
        "split": split,
        "overall": summarise(rows),
        "by_tag": by_tag,
        "reading": (
            "d_wrong ~ 0 means the model ignores WHICH tag it was given -> "
            "conditioning is broken, more data/epochs/rank will not help. "
            "d_wrong > 0 with 95% CI clear of 0 means tag identity is used, and "
            "any ear failure is a decoding/sampling problem instead."
        ),
        "rows": rows,
    }

    name = (adapter or "base").replace("/", "_")
    out = Path(OUT_DIR, f"tag_ablation_{name}.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    vol_mio.commit()

    o = res["overall"]
    print("\n==== TAG ABLATION ====", flush=True)
    print(f"adapter : {res['adapter']}", flush=True)
    print(f"split   : {split}   n={o['n']}", flush=True)
    print(f"NLL(right)        = {o['nll_right_mean']}", flush=True)
    print(f"d_none  (no tag)  = {o['d_none']['mean']}  CI{o['d_none']['ci95']}  pos={o['d_none']['frac_positive']}", flush=True)
    print(f"d_wrong (bad tag) = {o['d_wrong']['mean']}  CI{o['d_wrong']['ci95']}  pos={o['d_wrong']['frac_positive']}", flush=True)
    print(f"wrote {out}", flush=True)
    return {k: v for k, v in res.items() if k != "rows"}


@app.function(gpu="A10G", volumes={MIO: vol_mio}, timeout=60 * 60 * 2, memory=32768)
def profile(adapter: str = "adapter_iv_v8_e3", n: int = 200, seed: int = 7,
            win_tokens: int = 25) -> dict:
    """WHERE along the audio does the tag matter?

    The whole-clip d_wrong averages the event (~0.5 s) over the whole cut
    (~6-8 s), so a strong local effect would be diluted ~10x. Instead of
    guessing a window, measure the per-token profile:

      * delta[t] = logp_right[t] - logp_wrong[t] for every code token
      * binned by absolute position, and by position relative to where the tag
        sits in the caption (word fraction)
      * best contiguous `win_tokens` window (25 tokens = 1 s at 25 Hz)

    concentration = best_window_delta / overall_delta.
      ~1  -> effect is flat; dilution is NOT the story, whole-clip number is honest
      >>1 -> effect is local; event-window loss weighting is the right fix
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vol_mio.reload()

    codes_map = {}
    with open(f"{ENCODE_DIR}/codes.jsonl") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                codes_map[o["id"]] = o["codes"]

    rows_all = [
        json.loads(l)
        for l in Path(f"{DATA_DIR}/train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    pool = [
        r
        for r in rows_all
        if r["id"] in codes_map
        and (r.get("tag1") or "none") not in ("", "none")
        and f"[{r['tag1']}]" in (r.get("caption") or "")
    ]
    all_tags = sorted({r["tag1"] for r in pool})
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:n]
    print(f"n={len(pool)} tags={all_tags}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"{OUT_DIR}/{adapter}")
    model.eval()

    code_re = re.compile(r"^<\|s_\d+\|>$")
    code_ids = {i for t, i in tok.get_vocab().items() if code_re.match(t)}

    @torch.no_grad()
    def token_logps(user: str, codes: list):
        assistant = "".join(f"<|s_{int(c)}|>" for c in codes)
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(text, truncation=True, max_length=2048).input_ids
        first = next((j for j, t in enumerate(ids) if t in code_ids), None)
        if first is None:
            return None
        x = torch.tensor([ids], device="cuda")
        logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
        return lp[first - 1 :].cpu()

    NB = 20
    abs_bins = [[] for _ in range(NB)]
    rel_bins = [[] for _ in range(NB)]  # centred on estimated tag position
    overall, best_win, tag_fracs = [], [], []
    per_tag_best: dict[str, list] = {}

    for i, r in enumerate(pool, 1):
        tag, cap = r["tag1"], r["caption"]
        others = [t for t in all_tags if t != tag]
        wrong = rng.choice(others)
        try:
            a = token_logps(cap, codes_map[r["id"]])
            b = token_logps(swap_tag(cap, tag, wrong), codes_map[r["id"]])
        except Exception as e:
            print(f"  [{i}] {r['id']} FAIL {type(e).__name__}", flush=True)
            continue
        if a is None or b is None or len(a) != len(b) or len(a) < win_tokens * 2:
            continue
        d = (a - b)  # >0 where the right tag predicts better
        T = len(d)
        overall.append(float(d.mean()))

        # where the tag sits in the caption, as a word fraction
        words = cap.split()
        pos = next((j for j, w in enumerate(words) if f"[{tag}]" in w), 0)
        frac = pos / max(1, len(words) - 1)
        tag_fracs.append(frac)

        for t in range(T):
            abs_bins[min(NB - 1, int(t / T * NB))].append(float(d[t]))
            # relative: -1 .. +1 around the estimated tag position
            rel = (t / T) - frac
            rel_bins[min(NB - 1, int((rel + 1) / 2 * NB))].append(float(d[t]))

        cs = d.cumsum(0)
        sums = cs[win_tokens - 1 :].clone()
        sums[1:] -= cs[: -win_tokens]
        bw = float(sums.max() / win_tokens)
        best_win.append(bw)
        per_tag_best.setdefault(tag, []).append(bw)
        if i % 25 == 0:
            print(f"  [{i}/{len(pool)}]", flush=True)

    def m(v):
        return round(sum(v) / len(v), 5) if v else None

    ov, bw = m(overall), m(best_win)
    res = {
        "adapter": adapter or "BASE",
        "n": len(overall),
        "win_tokens": win_tokens,
        "win_seconds": win_tokens / 25.0,
        "overall_d_wrong": ov,
        "best_window_d_wrong": bw,
        "concentration": round(bw / ov, 3) if ov else None,
        "mean_tag_position_frac": m(tag_fracs),
        "abs_profile": [m(b) for b in abs_bins],
        "rel_profile_centred_on_tag": [m(b) for b in rel_bins],
        "per_tag_best_window": {k: m(v) for k, v in sorted(per_tag_best.items())},
        "reading": (
            "concentration ~1 means the tag effect is spread evenly over the whole "
            "clip — dilution is not the explanation and the whole-clip number is "
            "the honest one. concentration >> 1 means the effect is local to the "
            "event, and weighting the loss on that window is the right fix."
        ),
    }
    out = Path(OUT_DIR, f"tag_profile_{(adapter or 'base').replace('/', '_')}.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    vol_mio.commit()

    print("\n==== TAG PROFILE ====", flush=True)
    print(f"adapter          : {res['adapter']}  n={res['n']}", flush=True)
    print(f"overall d_wrong  : {ov}", flush=True)
    print(f"best {res['win_seconds']}s window : {bw}", flush=True)
    print(f"concentration    : {res['concentration']}x", flush=True)
    print(f"abs profile      : {res['abs_profile']}", flush=True)
    print(f"wrote {out}", flush=True)
    return {k: v for k, v in res.items() if k != "rel_profile_centred_on_tag"}


@app.local_entrypoint()
def main(
    adapter: str = "adapter_iv_v8_e3",
    n: int = 300,
    seed: int = 7,
    action: str = "ablate",
):
    if action == "profile":
        r = profile.remote(adapter=adapter, n=n, seed=seed)
    else:
        r = ablate.remote(adapter=adapter, n=n, seed=seed)
    print(json.dumps(r, indent=2)[:4000])
