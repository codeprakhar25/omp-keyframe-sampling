# avatar-lab

Hands-on experiment: build a talking-head video **of myself** from open-source
models, on a 6GB GPU. Test what's possible local vs RunPod. Learn the stack.

## Hardware

- RTX 3050 **6GB** Laptop (Ampere, CUDA 12.x), WSL2, Python 3.12 system, `uv`
  for env mgmt, ffmpeg present, ~848G free.
- **6GB VRAM is the binding constraint.** Light stages run local; heavy
  (SDXL/Flux LoRA train, LatentSync, video-gen) go to RunPod.

## Pipeline

```
[photo of me] + [voice → script]
      │                │
 (LoRA, phase 3)   TTS clone (Chatterbox)
      │                │
      └──► talking-head (photo+audio→video) ◄─┘
                   │
            polish: face-restore (GFPGAN) + frame-interp (RIFE) → smooth
```

## Phases

- **Phase 1 — local loop ($0):** prove end-to-end with photo + TTS *default*
  voice. Chatterbox → SadTalker → RIFE+GFPGAN. Rough but complete.
- **Phase 2 — upgrade:** record my voice → Chatterbox *clone*; swap
  SadTalker → **LatentSync** on RunPod for quality lip-sync.
- **Phase 3 — likeness LoRA:** SD1.5 LoRA local (proof) → SDXL/Flux LoRA on
  RunPod → generate new images of me → feed into animate.

## Model picks (June 2026)

| Stage | Local (6GB) | RunPod (quality) | Notes |
|---|---|---|---|
| TTS / voice clone | **Chatterbox** (MIT), XTTS-v2 | — | Chatterbox beat ElevenLabs 65% blind test |
| Talking head | **SadTalker**, MuseTalk | **LatentSync** (ByteDance), Hallo3, Sonic | SadTalker = old deps → py3.10 venv |
| Face restore | GFPGAN / CodeFormer | — | light |
| Frame interp | Practical-RIFE | — | light |
| Image gen | SD1.5 | SDXL / Flux | |
| LoRA train | SD1.5 LoRA | SDXL (12GB+), Flux (24GB) | |
| Text/img→video | LTX-Video (tight) | Wan2.1, HunyuanVideo | optional track |

## Env strategy

Per-tool isolated venvs under `envs/` (their deps conflict). `uv` pins python
per tool (SadTalker → 3.10; Chatterbox/RIFE/GFPGAN → 3.12 ok).

## Layout

- `data/photos/` — my stills (gitignored, PII)
- `data/voice/` — my voice samples (gitignored)
- `scripts/` — glue between stages
- `out/` — generated audio/video (gitignored)
- `envs/` — per-tool venvs (gitignored)

## Data capture spec

- **Talking-head photo:** 1 frontal, neutral face, mouth closed, even light,
  plain bg, face fills frame, ≥512px, sharp.
- **Voice:** 20-60s single clean take, quiet room, no music, WAV. ~10s min.
- **LoRA:** 15-30 varied photos (angle/light/expression/bg), 1024px.

## Setup notes / gotchas

- **Chatterbox env (`envs/chatterbox`, py3.12):** `uv pip install chatterbox-tts`
  pulls torch 2.6.0+cu124, transformers 5.2.0, resemble-perth 1.0.1.
  Frozen lock: `scripts/chatterbox.requirements.txt`.
- **GOTCHA:** chatterbox's watermarker (`perth`) imports `pkg_resources`, which
  `setuptools>=81` removed. Fix: `uv pip install "setuptools<81"` (pinned
  80.10.2). Without it: `TypeError: 'NoneType' object is not callable` at
  `ChatterboxTTS.from_pretrained`.
- Run: `envs/chatterbox/bin/python scripts/tts_chatterbox.py --text "..." --out out/x.wav`

## Status

- 2026-06-16: scaffolded. **Phase 1 stage A (TTS) DONE** — Chatterbox default
  voice, 7.6s WAV @24kHz, ~18s gen on RTX 3050. Have photos, no voice yet
  (default voice used now; clone at Phase 2).
- 2026-06-17: **Phase 1 stage B (SadTalker) DONE** — `profile.jpeg` + `test.wav`
  -> `out/sadtalker/*.mp4`, 256x256 @25fps, 7.6s, ~67s render on 6GB (no OOM).
  Config: `--size 256 --preprocess crop --still --batch_size 1`.
- NEXT: Phase 1 stage C = polish (GFPGAN face-restore + Practical-RIFE interp);
  optionally re-run at `--size 512 --preprocess full` for bigger frame.

### SadTalker setup notes (`envs/sadtalker`, py3.10)
- torch 2.0.1+cu118 / torchvision 0.15.2+cu118 / torchaudio 2.0.2 (Ampere ok;
  tv 0.15 keeps `functional_tensor` that `basicsr==1.4.2` imports).
- **Same setuptools>=81 trap as Chatterbox:** `librosa` imports `pkg_resources`
  -> pin `setuptools<81` (80.10.2) after the reqs install.
- Checkpoints (~2.7G) in `tools/SadTalker/checkpoints` + `gfpgan/weights` via
  `scripts/download_models.sh`. GOTCHA: a killed download left a **truncated**
  `mapping_00229-model.pth.tar` (117MB vs 155MB) -> `PytorchStreamReader ...
  failed finding central directory`. Re-fetch fixes.
- Run from inside `tools/SadTalker/` (uses relative `./checkpoints`); pass
  abs paths for `--source_image` / `--driven_audio` / `--result_dir`.

### Disk / WSL note
- WSL2 rootfs vhdx lives on **C: (only ~20G free)** — that is the real ceiling,
  not the 830G `df` shows on `/`. Keep local footprint lean; `uv cache clean`
  after each big install (reclaimed 10.5G here). Heavy stuff -> RunPod.
