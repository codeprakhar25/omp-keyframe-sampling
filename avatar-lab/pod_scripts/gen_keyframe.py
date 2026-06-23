"""FLUX.1-dev + trained LoRA -> tennis keyframe images.
Run in the ai-toolkit venv (diffusers + flux). Requires: export HF_TOKEN=<token>.
Insight: full-body => face too small; 3/4 body keeps the face recognizable
(a forehand reads in the upper body), so 3/4 is the better keyframe for Wan I2V.
"""
import torch
from diffusers import FluxPipeline

LORA = "/workspace/avatar/lora_out/prakhar_flux_lora/prakhar_flux_lora.safetensors"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights(LORA)              # adds B*A*scale into transformer attn layers
pipe.enable_model_cpu_offload()           # swap T5/DiT/VAE so 12B fits 32GB

JOBS = [
    ("full",   "prakhar man playing tennis on a grass court, holding a tennis racket, athletic sportswear, full body, athletic ready stance, side view, wearing glasses, bright soft daylight, photorealistic, sharp focus, highly detailed face", 896, 1152, [0, 1, 2]),
    ("threeq", "prakhar man on a grass tennis court holding a tennis racket, three quarter body shot, athletic sportswear, wearing glasses, looking at camera, bright daylight, photorealistic, highly detailed face", 832, 1024, [0, 1]),
]
for name, prompt, W, H, seeds in JOBS:
    for s in seeds:
        img = pipe(
            prompt=prompt, width=W, height=H,
            num_inference_steps=28, guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(s),
            joint_attention_kwargs={"scale": 1.0},   # LoRA strength
        ).images[0]
        img.save(f"/workspace/avatar/wan_in/key_{name}_s{s}.png")
        print("SAVED", name, s, flush=True)
print("KEYFRAME_DONE", flush=True)
