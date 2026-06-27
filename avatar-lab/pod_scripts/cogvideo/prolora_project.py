import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

FLUX = "/home/prakh/ml-resarch/avatar-lab/data/lora_weights/prakhar_flux_lora.safetensors"
OUT  = "/home/prakh/ml-resarch/avatar-lab/data/lora_weights/flux2cog_projected.safetensors"
D_FLUX, D_COG, R = 3072, 1920, 32
N_COG_BLOCKS = 30          # CogVideoX-2b transformer_blocks
ATTN = ["to_q","to_k","to_v","to_out.0"]

# ---- build P: random ORTHONORMAL-row projection [D_COG x D_FLUX] (JL-style, distance-preserving) ----
rng = np.random.default_rng(0)
M = rng.standard_normal((D_FLUX, D_COG)).astype(np.float32)
Q, _ = np.linalg.qr(M)              # Q:[D_FLUX x D_COG], orthonormal columns
P = Q.T.copy()                      # [D_COG x D_FLUX], orthonormal rows  -> P P^T = I
P *= np.sqrt(D_FLUX / D_COG)        # norm-preserving scale
print(f"P shape {P.shape}  P@P.T diag mean {np.diag(P@P.T).mean():.3f}")

f = safe_open(FLUX, "numpy")
keys = set(f.keys())

# FLUX has only 19 DOUBLE-blocks (transformer_blocks) with to_q/k/v/to_out.0.
# CogVideoX needs 30. -> map cog block i  <-  flux double-block (i % 19).  (arbitrary; part of why it breaks)
N_FLUX_DOUBLE = 19
out = {}
mapped = 0
for i in range(N_COG_BLOCKS):
    src = i % N_FLUX_DOUBLE
    for mod in ATTN:
        ka = f"transformer.transformer_blocks.{src}.attn.{mod}.lora_A.weight"
        kb = f"transformer.transformer_blocks.{src}.attn.{mod}.lora_B.weight"
        if ka not in keys or kb not in keys:
            print("MISSING", ka); continue
        A = f.get_tensor(ka).astype(np.float32)   # [32, 3072]
        B = f.get_tensor(kb).astype(np.float32)   # [3072, 32]
        A_cog = A @ P.T                            # [32, 1920]
        B_cog = P @ B                              # [1920, 32]
        out[f"transformer.transformer_blocks.{i}.attn1.{mod}.lora_A.weight"] = A_cog
        out[f"transformer.transformer_blocks.{i}.attn1.{mod}.lora_B.weight"] = B_cog
        mapped += 1
        if i==0 and mod=="to_q":
            print(f"sample to_q: A {A.shape}->{A_cog.shape}  B {B.shape}->{B_cog.shape}")

save_file(out, OUT)
print(f"WROTE {OUT}  tensors={len(out)}  modules={mapped}  (expect {N_COG_BLOCKS*len(ATTN)})")
