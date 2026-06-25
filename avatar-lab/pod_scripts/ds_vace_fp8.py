"""DiffSynth-Studio Wan2.1-VACE-14B, fp8 on-card (no offload) = fast VACE swing.
Reuses local t5/VAE (Wan2.1 family shares them); only the VACE-14B DiT downloads.
Identity = our FLUX-LoRA keyframe (vace_reference_image); motion = our pose video.
"""
import torch, glob
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

W21 = "/workspace/avatar/HuMo/weights/Wan2.1-T2V-1.3B"   # local t5/vae/tokenizer (shared)
DIT = sorted(glob.glob("/workspace/avatar/vace14b_hf/diffusion_pytorch_model*.safetensors"))  # local HF shards
POSE = "/workspace/avatar/vace_pose_video.mp4"
REF  = "/workspace/avatar/wan_in/key_threeq_s0.png"
H, W = 480, 832            # start 480p to validate fp8 speed; bump to 720x1280 next

# DiT: fp8 weights on-card (no offload -> kills offload bottleneck), bf16 compute
fp8 = dict(offload_device="cuda", offload_dtype=torch.float8_e4m3fn,
           onload_device="cuda", onload_dtype=torch.float8_e4m3fn,
           computation_device="cuda", computation_dtype=torch.bfloat16)
# t5/VAE: keep on CPU, stream to GPU per-use (t5 11G can't co-reside with fp8 DiT on 32G)
cpu_off = dict(offload_device="cpu", offload_dtype=torch.bfloat16,
               onload_device="cpu", onload_dtype=torch.bfloat16,
               computation_device="cuda", computation_dtype=torch.bfloat16)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=DIT, skip_download=True, **fp8),
        ModelConfig(path=f"{W21}/models_t5_umt5-xxl-enc-bf16.pth", skip_download=True, **cpu_off),
        ModelConfig(path=f"{W21}/Wan2.1_VAE.pth", skip_download=True, **cpu_off),
    ],
    tokenizer_config=ModelConfig(path=f"{W21}/google/umt5-xxl/", skip_download=True),
    vram_limit=26,
)

prompt = ("prakhar man playing tennis, hitting a powerful forehand, holding a tennis racket, "
          "wearing glasses, athletic sportswear, grass court, dynamic athletic swing, follow through, "
          "photorealistic, sharp focus, highly detailed face")
neg = "blurry, deformed, extra limbs, warped face, two rackets, low quality, static, no glasses"

import time; t0 = time.time()
video = pipe(
    prompt=prompt, negative_prompt=neg,
    vace_video=VideoData(POSE, height=H, width=W),
    vace_reference_image=Image.open(REF).convert("RGB").resize((W, H)),
    num_inference_steps=30, seed=0, tiled=True,
)
print("GEN_SECONDS", round(time.time() - t0, 1), flush=True)
save_video(video, "/workspace/avatar/wan_out/vace_fp8.mp4", fps=16, quality=5)
print("DS_VACE_DONE", flush=True)
