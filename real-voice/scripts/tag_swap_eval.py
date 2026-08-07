#!/usr/bin/env python3
"""Init / score tag-swap A/B eval pairs (post-LoRA listen test).

  .venv/bin/python scripts/tag_swap_eval.py --init
  # put wavs in data/scale/eval_pairs/<pair_id>_a.wav and _b.wav
  # fill judgments in data/scale/eval.json via tag-swap.html
  .venv/bin/python scripts/tag_swap_eval.py --score
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE = ROOT / "data" / "scale"
EVAL = SCALE / "eval.json"
PAIRS = SCALE / "eval_pairs"

# Seed hold-out texts — synthesize later on Fireworks endpoint
SEED_PAIRS = [
    {
        "id": "e01",
        "text": "तुम हमेशा देर से आते हो",
        "tag_a": "angry",
        "tag_b": "sad",
        "family": "emotion",
    },
    {
        "id": "e02",
        "text": "आज मौसम बहुत अच्छा है",
        "tag_a": "excited",
        "tag_b": "none",
        "family": "emotion",
    },
    {
        "id": "e03",
        "text": "मुझे नहीं पता क्या करूँ",
        "tag_a": "fear",
        "tag_b": "sad",
        "family": "emotion",
    },
    {
        "id": "e04",
        "text": "क्या सच में ये हो गया",
        "tag_a": "surprise",
        "tag_b": "none",
        "family": "emotion",
    },
    {
        "id": "e05",
        "text": "मतलब वो बात अलग है",
        "tag_a": "thinking",
        "tag_b": "pause",
        "family": "hesitation",
    },
    {
        "id": "e06",
        "text": "हां मैं अभी बताता हूँ",
        "tag_a": "inhaling",
        "tag_b": "none",
        "family": "breath",
    },
    {
        "id": "e07",
        "text": "ठीक है फिर कल मिलते हैं",
        "tag_a": "sigh",
        "tag_b": "none",
        "family": "breath",
    },
    {
        "id": "e08",
        "text": "यहाँ कोई नहीं सुन रहा",
        "tag_a": "whispering",
        "tag_b": "none",
        "family": "whisper",
    },
    {
        "id": "e09",
        "text": "यह बिल्कुल गलत है",
        "tag_a": "disgust",
        "tag_b": "angry",
        "family": "emotion",
    },
    {
        "id": "e10",
        "text": "I think हमें अभी निकलना चाहिए",
        "tag_a": "thinking",
        "tag_b": "excited",
        "family": "hinglish",
    },
]


def init_eval() -> None:
    SCALE.mkdir(parents=True, exist_ok=True)
    PAIRS.mkdir(parents=True, exist_ok=True)
    pairs = []
    for s in SEED_PAIRS:
        pairs.append(
            {
                **s,
                "caption_a": f"[{s['tag_a']}] {s['text']}" if s["tag_a"] != "none" else s["text"],
                "caption_b": f"[{s['tag_b']}] {s['text']}" if s["tag_b"] != "none" else s["text"],
                "wav_a": f"eval_pairs/{s['id']}_a.wav",
                "wav_b": f"eval_pairs/{s['id']}_b.wav",
                # blinded fields for UI
                "pick": "",  # "a" | "b" | "same" | "unsure"
                "heard_tag": "",  # which tag ear thinks is stronger on picked side
                "notes": "",
                "correct": None,
            }
        )
    doc = {
        "version": 1,
        "howto": "Synth via modal_indic_mio.py --action synth → judge in tag-swap.html",
        "pass_rule": "emotion family ≥70% ear-correct; breath/whisper ≥60%",
        "pairs": pairs,
    }
    EVAL.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # slim list for Modal synth
    synth = [{"id": p["id"], "text": p["text"], "tag_a": p["tag_a"], "tag_b": p["tag_b"]} for p in pairs]
    synth_path = SCALE / "eval_synth_pairs.json"
    synth_path.write_text(json.dumps(synth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {EVAL} ({len(pairs)} pairs)")
    print(f"wrote {synth_path}")
    print(f"wav dir {PAIRS}")


def score() -> None:
    doc = json.loads(EVAL.read_text(encoding="utf-8"))
    # For each pair, "correct" if pick matches the side whose tag is the intended contrast
    # Simple protocol: listener told "which is more TAG_A-like?" → pick a => correct
    # UI should ask: which clip matches tag_a better?
    judged = [p for p in doc["pairs"] if p.get("pick") in ("a", "b", "same", "unsure")]
    if not judged:
        print("no judgments yet")
        return
    # correct = picked a (tag_a is the asked target in UI)
    ok = sum(1 for p in judged if p.get("pick") == "a" or p.get("correct") is True)
    # also count explicit correct flag
    ok = sum(
        1
        for p in judged
        if p.get("correct") is True or (p.get("correct") is None and p.get("pick") == "a")
    )
    by = {}
    for p in judged:
        fam = p.get("family") or "other"
        by.setdefault(fam, {"n": 0, "ok": 0})
        by[fam]["n"] += 1
        if p.get("correct") is True or (p.get("correct") is None and p.get("pick") == "a"):
            by[fam]["ok"] += 1
    print(f"overall {ok}/{len(judged)} = {ok/len(judged):.0%}")
    for fam, st in by.items():
        print(f"  {fam}: {st['ok']}/{st['n']} = {st['ok']/st['n']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.init:
        init_eval()
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
