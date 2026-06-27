import torch, os
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

PROJ="/workspace/avatar/flux2cog_projected.safetensors"
OUT="/workspace/avatar/prolora_out"; os.makedirs(OUT, exist_ok=True)
prompt="prakhar man, close-up portrait talking to the camera, plain background, photorealistic"

pipe=CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload(); pipe.vae.enable_tiling()

def gen(tag):
    g=torch.Generator(device="cuda").manual_seed(42)
    v=pipe(prompt=prompt, num_frames=49, num_inference_steps=30, guidance_scale=6.0,
           height=480, width=720, generator=g).frames[0]
    p=f"{OUT}/{tag}.mp4"; export_to_video(v, p, fps=8); print("SAVED", p, flush=True)

# control: base CogVideoX, no LoRA
gen("base_noLoRA")
# transfer: projected FLUX->CogVideoX LoRA
try:
    pipe.load_lora_weights(PROJ, adapter_name="flux2cog")
    pipe.set_adapters(["flux2cog"], [1.0]); print("LoRA LOADED ok", flush=True)
    gen("projected_s1.0")
    pipe.set_adapters(["flux2cog"], [2.0]); gen("projected_s2.0")  # overdrive: force the broken signal
except Exception as e:
    print("LOAD/GEN FAILED:", repr(e), flush=True)
print("ALL DONE", flush=True)
