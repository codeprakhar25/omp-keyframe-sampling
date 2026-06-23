#!/usr/bin/env python3
"""Chatterbox TTS -> WAV. Phase 1 uses default voice; pass --voice <wav> to clone.

Run inside the chatterbox venv:
  envs/chatterbox/bin/python scripts/tts_chatterbox.py \
      --text "hello, this is a test" --out out/test.wav
  # clone (phase 2):
  envs/chatterbox/bin/python scripts/tts_chatterbox.py \
      --text "..." --voice data/voice/me.wav --out out/me.wav
"""
import argparse
import sys

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="text to speak")
    ap.add_argument("--out", default="out/test.wav", help="output wav path")
    ap.add_argument("--voice", default=None, help="reference wav for voice clone")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tts] device={device} clone={'yes' if args.voice else 'no (default voice)'}")

    model = ChatterboxTTS.from_pretrained(device=device)
    print(model)
    if args.voice:
        wav = model.generate(args.text, audio_prompt_path=args.voice)
    else:
        wav = model.generate(args.text)

    ta.save(args.out, wav, model.sr)
    dur = wav.shape[-1] / model.sr
    print(f"[tts] wrote {args.out}  ({dur:.1f}s @ {model.sr}Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
