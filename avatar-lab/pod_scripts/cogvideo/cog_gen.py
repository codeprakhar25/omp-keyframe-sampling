import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

P="/workspace/avatar/cog_lora_out/lora_weights/000060/pytorch_lora_weights.safetensors"
prompt="PIKA_CRUSH A red toy car is crushed by a large hydraulic press, flattening it as if under a press."

pipe=CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b",torch_dtype=torch.float16)
pipe.load_lora_weights(P, adapter_name="crush")
pipe.set_adapters(["crush"], [1.0])
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()

g=torch.Generator(device="cuda").manual_seed(42)
print("generating...")
vid=pipe(prompt=prompt, num_frames=49, num_inference_steps=30, guidance_scale=6.0,
         height=480, width=720, generator=g).frames[0]
out="/workspace/avatar/cog_lora_out/test_crush_lora.mp4"
export_to_video(vid, out, fps=8)
print("SAVED", out)
