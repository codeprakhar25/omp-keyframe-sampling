"""Wan 2.2 TI2V-5B text-to-video capability test (no input image).
Run in the wanenv venv. T2V gives big free motion + ball survives, but no
personal identity (text can't lock a real face -> that's why we train a LoRA
and use I2V/VACE for identity).
"""
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
H, W, NUM_FRAMES, STEPS = 480, 832, 49, 30
NEG = "blurry, deformed, extra limbs, warped face, low quality, cartoon, distorted hands"
CASES = {
    "forehand": "a tennis player hitting a forehand on a grass court, full body, dynamic athletic swing, photorealistic",
    "speech":   "a man giving a motivational speech on a stage, gesturing with hands, medium shot, photorealistic",
}
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()

for name, prompt in CASES.items():
    out = pipe(prompt=prompt, negative_prompt=NEG, height=H, width=W,
               num_frames=NUM_FRAMES, num_inference_steps=STEPS,
               generator=torch.Generator().manual_seed(0)).frames[0]
    export_to_video(out, f"/workspace/avatar/wan_out/t2v_{name}.mp4", fps=24)
    print("SAVED", name, flush=True)
