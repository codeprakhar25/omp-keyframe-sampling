import torch, os, sys
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
LORA="/workspace/avatar/person_lora_v2_out/lora_weights/001500/pytorch_lora_weights.safetensors"
OUT="/workspace/avatar/person_gen_v2_out"; os.makedirs(OUT, exist_ok=True)
SCALE=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
PROMPTS={
 "outdoor_walk":"PRAKH_PERSON, a bearded man, walking down a sunny city street, outdoor daylight, smiling, cinematic",
 "talking_closeup":"PRAKH_PERSON, a bearded man, close-up portrait talking to the camera, soft indoor lighting, natural expression",
 "cafe_sit":"PRAKH_PERSON, a bearded man, sitting at a cafe table holding a coffee cup, warm ambient light",
 "nod_smile":"PRAKH_PERSON, a bearded man, nodding and smiling, head and shoulders, plain background, studio lighting",
}
pipe=CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
pipe.load_lora_weights(LORA, adapter_name="prakh"); pipe.set_adapters(["prakh"],[SCALE])
pipe.enable_sequential_cpu_offload(); pipe.vae.enable_tiling()
for name,prompt in PROMPTS.items():
    g=torch.Generator(device="cuda").manual_seed(42)
    print(f"=== {name} scale={SCALE} ===",flush=True)
    v=pipe(prompt=prompt,num_frames=49,num_inference_steps=30,guidance_scale=6.0,
           height=848,width=480,generator=g).frames[0]          # PORTRAIT now
    p=f"{OUT}/{name}_s{SCALE}.mp4"; export_to_video(v,p,fps=8); print("SAVED",p,flush=True)
print("ALL DONE",flush=True)
