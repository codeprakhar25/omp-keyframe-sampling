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
- **Wan VACE-14B OOMs with `enable_model_cpu_offload()`** (28GB transformer lands
  on-card at once) → use **`enable_sequential_cpu_offload()`** (fits, ~30s/step).
- ai-toolkit pins no transformers → grabs bleeding-edge; needs `torchaudio`
  installed, and an explicit `neg: ""` in the sample block (newer transformers
  rejects a None negative prompt).
- I2V uses `WanImageToVideoPipeline` (NOT `WanPipeline`, which has no `image` arg).
- `source <venv>` BEFORE `nohup python ...` or you get `No module named diffusers`
  (system python). Restart wipes overlay `/` → reinstall `ffmpeg` (apt).
- `hf download REPO --local-dir DIR` — pass **no extra positionals**; extra args
  (e.g. `--exclude X Y`) get read as explicit filenames → fetches junk.
- LivePortrait `requirements.txt` 1st line is `-r requirements_base.txt` →
  install from the repo dir so the relative `-r` resolves.

## Talking-head + swing pipelines (this repo)
- `liveportrait_run.sh` — still keyframe + driving clip → motion video (clean,
  stitched). Out length = driving length. NOT audio-driven.
- `extract_pose.py` — donor swing clip → OpenPose skeleton frames (VACE control).
  (DWpose avoided: mmpose/mmcv has no cu128/torch2.7 wheel + no nvcc = dead end.)
- `wan_vace.py` — pose + keyframe → real forehand swing (best tennis result).

### ⚠️ LatentSync relip is BROKEN on Blackwell
The lower-face shows a washed translucent **box** (the VAE-reconstructed affine
face-crop pasted back). Ruled out: resolution (crop+upscale), fp16→**fp32**,
**deepcache off**, and it boxes on LatentSync's OWN demo too. Root cause: the
torch-2.7/cu128 port regresses the relip; LatentSync's good env is torch-2.5/cu121
which **can't run on sm_120**. → For talking-head use **SadTalker** (still→talk,
clean, runs on a 6GB local GPU via `scripts/run_sadtalker.sh`), or run LatentSync
on a **non-Blackwell** GPU (cu121).

## Pipeline order
1. `prakhar_lora.yaml` — train FLUX.1-dev LoRA on your photos (ai-toolkit):
   `python ai-toolkit/run.py prakhar_lora.yaml`
2. `gen_keyframe.py` — FLUX.1-dev + LoRA → tennis keyframe (3/4 framing best).
3. `wan_t2v.py` / `wan_i2v.py` — Wan 2.2 capability tests (T2V free-motion/no-id;
   I2V id-locked/motion-muted).
4. **Swing:** `extract_pose.py` (donor → skeleton) → `wan_vace.py` (VACE swing).
5. **Talking-head:** `liveportrait_run.sh` (still → motion) → SadTalker for lips
   (LatentSync relip broken on Blackwell — see above).

## Findings (capability map)
- T2V: big free motion + ball survives, but no personal identity.
- I2V: identity locked to frame 1, motion muted.
- VACE: pose-control = real swing + identity from keyframe (glasses can drop).
- LivePortrait: clean stitched head/expr motion (needs a driving clip).
- SadTalker: clean still→talk (audio-driven), the working talking-head route.
