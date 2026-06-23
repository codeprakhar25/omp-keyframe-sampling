# Pod scripts — Wan 2.2 + FLUX.1 LoRA pipeline (RunPod)

Reproducible scripts for the from-scratch avatar pipeline, run on a RunPod GPU
(RTX PRO 4500 Blackwell 32GB, sm_120 → needs **cu128** torch wheels).

> Heavy artifacts (weights, mp4, photos) are gitignored by design — these are
> just the code + configs.

## Env setup (per tool, on /workspace so it survives overlay wipes)

```bash
# Wan inference venv
python -m venv /workspace/avatar/wanenv
source /workspace/avatar/wanenv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "diffusers>=0.36" transformers accelerate ftfy imageio imageio-ffmpeg safetensors
export HF_HOME=/workspace/hf_cache

# ai-toolkit (Flux LoRA training) venv
git clone --depth 1 https://github.com/ostris/ai-toolkit.git /workspace/avatar/ai-toolkit
python -m venv /workspace/avatar/aienv
source /workspace/avatar/aienv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r /workspace/avatar/ai-toolkit/requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # re-pin cu128 after reqs
```

**Secrets:** never hardcode the HF token. Export it in your shell:
`export HF_TOKEN=...`  (needed for gated FLUX.1-dev download). Rotate if leaked.

## Gotchas learned
- Blackwell sm_120 → cu128 torch (cu124 = "no kernel image available").
- Wan VAE decode OOMs on 32GB → `pipe.enable_model_cpu_offload()` +
  `pipe.vae.enable_tiling()` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- ai-toolkit pins no transformers → grabs bleeding-edge; needs `torchaudio`
  installed, and an explicit `neg: ""` in the sample block (newer transformers
  rejects a None negative prompt).
- I2V uses `WanImageToVideoPipeline` (NOT `WanPipeline`, which has no `image` arg).

## Pipeline order
1. `prakhar_lora.yaml` — train FLUX.1-dev LoRA on your photos (ai-toolkit):
   `python ai-toolkit/run.py prakhar_lora.yaml`
2. `gen_keyframe.py` — FLUX.1-dev + LoRA → tennis keyframe (3/4 framing best).
3. `wan_t2v.py` — Wan 2.2 text-to-video capability tests.
4. `wan_i2v.py` — Wan 2.2 image-to-video from a keyframe (identity-stable,
   motion muted → use VACE + a driving clip for a real swing).

## Findings (capability map)
- T2V: big free motion + ball survives, but no personal identity.
- I2V: identity locked to frame 1, motion muted.
- → real swing needs **VACE** pose-control (keyframe + real forehand driving clip).
