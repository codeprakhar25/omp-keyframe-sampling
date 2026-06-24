"""Wan2.1-VACE-14B pose-controlled tennis-swing gen (the real-swing path).

Identity comes from the FLUX-LoRA keyframe (reference_images); MOTION comes from
a donor forehand clip via an OpenPose skeleton (see extract_pose.py).

Model: Wan-AI/Wan2.1-VACE-14B-diffusers (~71G). Pipeline: WanVACEPipeline (diffusers>=0.38).

GOTCHA (the important one): on a 32GB card `enable_model_cpu_offload()` OOMs — the
14B transformer (~28GB bf16) lands on-card all at once. Use
`enable_sequential_cpu_offload()` (per-submodule swap) -> fits, ~30s/step
(49f/30steps ~= 15min). Also: num_frames must be 4k+1 (49 ok). And remember to
`source wanenv` before launching or you get `No module named diffusers`.
"""
import torch, glob
from diffusers import WanVACEPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

MODEL = "/workspace/avatar/wan_vace_14b"
POSE  = "/workspace/avatar/vace_pose"          # openpose skeleton frames (extract_pose.py)
REF   = "/workspace/avatar/wan_in/key_threeq_s0.png"   # FLUX-LoRA keyframe = identity
W, H, N = 832, 480, 49

vae  = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()   # NOT model_cpu_offload -> that OOMs on 32GB
pipe.vae.enable_tiling()

pf    = sorted(glob.glob(f"{POSE}/*.png"))[:N]
video = [Image.open(p).convert("RGB").resize((W, H)) for p in pf]   # control = pose
mask  = [Image.new("L", (W, H), 255) for _ in range(N)]             # all-white = generate everything
ref   = Image.open(REF).convert("RGB").resize((W, H))
print("frames:", len(video), "ref:", ref.size, flush=True)

prompt = ("prakhar man playing tennis, hitting a powerful forehand, holding a tennis racket, "
          "athletic sportswear, grass court, dynamic athletic swing, follow through, "
          "photorealistic, sharp focus, highly detailed face")
neg = "blurry, deformed, extra limbs, warped face, two rackets, low quality, static, frozen"

out = pipe(prompt=prompt, negative_prompt=neg, video=video, mask=mask,
           reference_images=[ref], height=H, width=W, num_frames=N,
           conditioning_scale=1.0, num_inference_steps=30, guidance_scale=5.0,
           generator=torch.Generator().manual_seed(0)).frames[0]
export_to_video(out, "/workspace/avatar/wan_out/vace_forehand.mp4", fps=16)
print("VACE_DONE", flush=True)

# Launch:
#   source /workspace/avatar/wanenv/bin/activate
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python wan_vace.py
# Result: identity/outfit/racket/ball/court HOLD; known miss = glasses dropped
# (add "wearing glasses" to prompt / try 720p for crisper face).
