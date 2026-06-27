# Cross-Modal LoRA — Video Diffusion Learning Log

Follow-on to the avatar-lab model survey. Goal: learn image+video diffusion
**from the inside** by training LoRAs on FLUX (image) and CogVideoX (video),
then attempting a cross-model FLUX→CogVideoX transfer. Skills, not papers.

Platform: RunPod Blackwell (RTX PRO 4500 / sm_120 / cu128), finetrainers,
diffusers. All heavy artifacts (`.safetensors`, `.mp4`) gitignored — see the
`data/lora_weights/` and `out/cog_lora/` listings at bottom for what's local.

---

## 1. What got done

| step | status | artifact |
|---|---|---|
| FLUX identity LoRA (image) | ✅ pre-existing | `data/lora_weights/prakhar_flux_lora.safetensors` (r32, d3072) |
| CogVideoX crush LoRA (learning) | ✅ | quick 60-step + long 1000-step (`crush_long`) |
| **CogVideoX personal LoRA v1** | ✅ (buggy — see §3) | `data/lora_weights/person_v1/` (squished) |
| **CogVideoX personal LoRA v2** | ✅ (fixed) | `data/lora_weights/person_v2/` (native portrait) |
| Cross-modal ProLoRA projection | ✅ built, ⏸ gen paused | `data/lora_weights/flux2cog_projected.safetensors` |

---

## 2. Architecture facts learned (concepts 0–3, all passed)

- **3D VAE**: spatial 8× + temporal 4× compression. `T_lat = 1 + (T_in−1)/4`.
  CogVideoX latent channels C=16.
- **Full-3D attention** (CogVideoX/Wan) vs factorized (AnimateDiff): O(N²) over
  *joint* space+time tokens → 49-frame is ~29× slower than 13-frame.
- **DiT 2D→3D**: positional encoding *supplies coordinates*; attention is
  permutation-invariant without it (no RoPE ⇒ frame 5 has no relation to frame 2).
- **LoRA math**: `h = W₀x + (α/r)·B·A·x`, `A=[r×d]`, `B=[d×r]`.
  `A·x` is an **r-vector** (the bottleneck), not a matrix. A squeezes d→r, B expands r→d.
  - `d` = model hidden width (host architecture: FLUX 3072, CogVideoX 1920).
  - `r` = rank = the LoRA's own capacity (how many independent update directions).
- **Transfer crux**: to move FLUX's LoRA into CogVideoX you MUST re-dimension
  `d` (3072→1920, forced by host activation width) but KEEP `r` (the learned
  content, not host-bound). Match the container rank to the **source** (FLUX r32),
  don't pad to 64 — padding fabricates untrained directions.

---

## 3. THE BUG: aspect squish (the session's key finding)

Personal LoRA **v1 was trained on horizontally-squished faces.**

- Phone clips are **portrait**: `ffprobe` → W=480, H=848 (aspect 0.57).
- Training bucket `[49, 480, 848]` = finetrainers `(frames, **H, W**)` → forced
  target H=480, W=848 = **landscape** (aspect 1.77).
- `reshape_mode: bicubic` = pure resize, **no crop, no aspect preservation**
  (confirmed in `finetrainers/functional/video.py:resize_to_nearest_bucket_video`
  → `bicubic_resize_video(v, (target_h, target_w))`).
- Net: every frame height-crushed 848→480 + width-stretched 480→848. **All 20
  clips. The face the LoRA learned was flat and wide.** This was the real
  likeness ceiling — not footage diversity.

**Fix (v2):** bucket `[49, 848, 480]` (= source H×W exactly → zero resize/crop/squish)
+ `reshape_mode: resize_crop` (preserve-aspect fallback). Gen at `height=848, width=480`.

**Result:** v2 final loss **0.13** (vs v1 0.17–0.27 — undistorted data fits easier),
portrait in → portrait out. A/B (same prompts/seed): `out/cog_lora/person_prompts/`
(v1 squished) vs `out/cog_lora/person_prompts_v2/` (v2 native).

> Lesson: always `ffprobe` your source clips and confirm the bucket order matches
> orientation **before** a long training run.

---

## 4. ProLoRA capstone (built, gen paused)

Cross-model FLUX→CogVideoX weight surgery. Per attention module:
```
P     : [1920 × 3072]                 # random ORTHONORMAL-row proj, scaled √(3072/1920)
B_cog = P @ B_flux  : [1920×32]       # re-dimension output d
A_cog = A_flux @ Pᵀ : [32×1920]       # re-dimension input d
r=32 kept (the FLUX-learned bottleneck)
```
Built locally with numpy: `pod_scripts/cogvideo/prolora_project.py` →
`flux2cog_projected.safetensors` (240 tensors = 120 modules × A/B).

**Three baked-in breakage sources** (the "why it breaks" lesson):
1. **Block count** — FLUX has only 19 attn-complete double-blocks; CogVideoX
   needs 30 → mapped `i mod 19` (arbitrary).
2. **Attention semantics** — FLUX `attn` = 2D space-only; CogVideoX `attn1` =
   full 3D space-time. Same slot name, different job.
3. **Random P** — orthonormal but arbitrary; FLUX feature dims don't *mean* the
   same thing in CogVideoX space.

Gen script staged: `pod_scripts/cogvideo/prolora_gen.py` (base control + projected
@scale 1.0/2.0, FLUX trigger = "prakhar man"). **Not yet run** — expect
garbage / no identity transfer = the demonstration. Run on next pod after env setup.

---

## 5. Pod / infra gotchas

- **MooseFS `/workspace` has a PER-POD quota** (not the global 580T). HF
  re-download → `OSError [Errno 122] Disk quota exceeded`. Put HF cache on the
  **local overlay**: `HF_HOME=/root/hf_local` (20G, but **wiped on pod stop**).
- **HF Xet client deadlocks on MooseFS FUSE** → `export HF_HUB_DISABLE_XET=1`.
- **Overlay wipe** kills apt packages → reinstall ffmpeg each pod (`pod_setup.sh` does it).
- **The 5-pin env chain** (why `pod_setup.sh` exists): torch must be **2.7.0**
  (not 2.11 — lost `torchvision.io.video_reader` + `_AttentionOp`); `datasets==3.3.2`
  (5.0 returns lazy VideoDecoder, breaks `.size(0)`); `torchcodec==0.4.0`.
- **Killing a run**: kill the **torchrun elastic agent** (`pgrep -f *_train.sh`),
  not just `train.py` — the agent respawns workers. `pkill` of a FUSE-stuck proc
  drops ssh with exit 255 — reconnect + verify.
- **Storage trim**: `finetrainers_step_*` dirs are training-RESUME state (3.6G ea),
  NOT the LoRA. Safe to delete once `lora_weights/` is saved (freed 46G this session).

---

## 6. Volume migration (286G → small)

Network volume `stable_chocolate_marten`: **286 GB = $20/mo**, billed on
*provisioned* size (deleting files saves nothing; can't shrink; bills 24/7).
RunPod **S3 API** can read/write the volume pod-free (`s3api-eu-ro-1.runpod.io`).

Plan: migrate keepers to a **~40G** volume (~$3/mo), delete the 286G one.

| keep on small vol | discard (rebuildable) |
|---|---|
| `ftenv` (13G venv, same `/workspace/avatar/ftenv` path) | hf_cache (33G — re-download) |
| `finetrainers` (112M) | HuMo / Wan / LatentSync / DiffSynth repos |
| trained LoRAs (2.1G) + 20 clips (13M) | aienv / dsenv / wanenv (other-exp venvs) |
| all scripts (now in git) | — |

Precious unique keepers (2.1G) also pulled **local** as backup (`data/lora_weights/`).
`experiment/` (8.9G) on the volume is a **different project** — leave it.

---

## 7. Next steps (when back)

1. Judge v2 vs v1 likeness visually (esp. `talking_closeup` pair).
2. Run the paused ProLoRA gen (`prolora_gen.py`) → observe the break.
3. Optional v3: outdoor/varied footage for diversity (separate axis from the squish fix).
4. Rebuild env on small-volume pod: `bash pod_scripts/cogvideo/pod_setup.sh`.
