#!/usr/bin/env python3
"""Render one labeled contact sheet (thumbnail grid) per manifest item, for the
S1 gold-evidence PIN PASS.

For each item: load the candidate frames (same sampling the harness uses), tile them
in a grid, label each tile with its frame INDEX + seconds, and write the question on
top. You open the PNGs, find the frame that shows the MCQA answer, and note its index.

Then `apply_pins.py` turns {id: answer_index} into tight gold_evidence_seconds spans.

Usage:
  python3 scripts/make_contact_sheets.py --manifest data/manifest.s1.json --out guiworld_sheets
"""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw

from harness.media import load_frames


def sheet(frames, question, cols=4, thumb=380, pad=24, gold=None):
    gold = set(gold or [])
    n = len(frames)
    rows = (n + cols - 1) // cols
    W = cols * (thumb + 8) + 8
    H = pad + 8 + rows * (thumb + pad)
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), question[:140], fill=(240, 240, 240))
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        x = 6 + c * (thumb + 8)
        y = pad + 8 + r * (thumb + pad)
        im = fr.image.copy()
        im.thumbnail((thumb, thumb))
        canvas.paste(im, (x, y))
        is_gold = fr.index in gold
        if is_gold:  # green box around the needle frames
            d.rectangle([x - 3, y - 3, x + im.size[0] + 2, y + im.size[1] + 2], outline=(40, 230, 40), width=4)
        sec = f"{fr.seconds:.1f}s" if fr.seconds is not None else "-"
        tag = "  <== NEEDLE" if is_gold else ""
        d.text((x + 2, y + im.size[1] + 4), f"#{fr.index}  {sec}{tag}",
                fill=(40, 230, 40) if is_gold else (170, 170, 170))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.s1.json")
    ap.add_argument("--out", default="guiworld_sheets")
    ap.add_argument("--dump-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(args.manifest))
    index_lines = []
    for it in items:
        frames = load_frames(it, dump_fps=args.dump_fps, max_frames=args.max_frames)
        img = sheet(frames, f"{it['id']}: {it['question']}", gold=it.get("gold_evidence_frames"))
        path = os.path.join(args.out, f"{it['id']}.png")
        img.save(path)
        q1 = it["question"].replace("\n", "  ||  ")
        index_lines.append(f"{it['id']}\tgold={it['gold_answer']}\tn_frames={len(frames)}\tQ={q1}")
        print("wrote", path, f"({len(frames)} frames)")
    open(os.path.join(args.out, "_questions.tsv"), "w").write("\n".join(index_lines))
    # emit a pins template to fill in: id -> answer frame index (int) or [i,j]
    tmpl = {it["id"]: None for it in items}
    open(os.path.join(args.out, "pins.template.json"), "w").write(json.dumps(tmpl, indent=1))
    print("\nFill", os.path.join(args.out, "pins.template.json"),
          "-> set each id to the frame INDEX (#) showing the answer; save as pins.json")


if __name__ == "__main__":
    main()
