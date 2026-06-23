"""Wan 2.2 TI2V-5B image-to-video from a keyframe.
Run in the wanenv venv. Identity stays locked to frame 1; motion is muted
(use VACE + a real driving clip for a crisp swing).
"""
import sys, torch
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/workspace/avatar/wan_in/key_threeq_s0.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/avatar/wan_out/i2v.mp4"
W, H = 480, 832            # portrait 480p bucket
PROMPT = "a man hitting a powerful tennis forehand, swinging the racket through the ball, dynamic athletic motion, follow through, on a tennis court"
NEG = "static, frozen, blurry, deformed, extra limbs, warped face, two rackets, low quality"

def fit(p, w, h):
    im = Image.open(p).convert("RGB"); sw, sh = im.size; s = max(w / sw, h / sh)
    im = im.resize((round(sw * s), round(sh * s))); nw, nh = im.size
    l, t = (nw - w) // 2, (nh - h) // 2
    return im.crop((l, t, l + w, t + h))

vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()   # avoid VAE-decode OOM

out = pipe(image=fit(IMG, W, H), prompt=PROMPT, negative_prompt=NEG,
           height=H, width=W, num_frames=49, num_inference_steps=30,
           generator=torch.Generator().manual_seed(0)).frames[0]
export_to_video(out, OUT, fps=24)
print("SAVED", OUT, flush=True)
