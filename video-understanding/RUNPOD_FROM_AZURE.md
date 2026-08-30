# RunPod from Azure — restore + PyTorch playbook

Azure holds **data + models + code**, not the old venvs.  
Blind `pip install -r freeze.txt` will **fail or fight CUDA**. Do this instead.

Related: [`AZURE_BACKUP.md`](./AZURE_BACKUP.md) (blob layout + creds).

---

## Hard truth

| Copied to Azure | Not copied |
|---|---|
| Qwen3-VL-8B, LongCLIP, videos, embeds, picks, `lmms-eval`, harness/scripts | `lmmsenv/`, `slmenv/` (Blackwell-tied) |
| `slm-lab/env_freeze/lmmsenv.freeze.txt` (partial — **no `torch` line**) | Working GPU wheels |

So: **install torch for the GPU you rent now**, then layer app deps, then pull Azure trees.

---

## 0) Pick a pod

- **GPU:** 24 GB class enough for Qwen3-VL-8B @ k8 full-res (`batch_size=1`) — saw ~20 GB used.
- **Disk:** ≥400 GB if restoring videomme+lvbench+models; or restore **only what you need** for the next run.
- **Image:** Ubuntu + CUDA driver matching the GPU (RunPod template with CUDA 12.x is fine).
- Optional: attach a **new** empty network volume if you want persistence across pod stops.

---

## 1) Auth + tools on the pod

```bash
# Azure CLI or just azcopy
curl -fsSL -o /tmp/azcopy.tgz https://aka.ms/downloadazcopy-v10-linux
tar -xzf /tmp/azcopy.tgz -C /tmp
install -m 755 "$(find /tmp -name azcopy -type f | head -1)" /usr/local/bin/azcopy

# Creds: paste from human / local .azure_backup.env (never commit)
export AZURE_SA=slmlabsponsored
export AZURE_CONTAINER=slm-lab
export AZURE_STORAGE_KEY='...'          # portal Access keys
# or SAS:
export AZURE_DEST="https://${AZURE_SA}.blob.core.windows.net/${AZURE_CONTAINER}"
export AZURE_SAS='...'                  # quote if putting in a file; contains &
```

Mint SAS from a machine with `az login` if needed (see `AZURE_BACKUP.md`).

---

## 2) Python + PyTorch (do this BEFORE big Azure pulls optional)

Prefer **official torch wheel for this machine’s CUDA**, not the old freeze.

```bash
python3 -m venv /workspace/lmmsenv
source /workspace/lmmsenv/bin/activate
pip install -U pip wheel

# Example — pick the index that matches `nvidia-smi` CUDA:
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

If CUDA unavailable → wrong wheel / driver; fix before continuing.

---

## 3) App deps (lmms-eval / Qwen path)

```bash
source /workspace/lmmsenv/bin/activate

# Core stack used in prior runs (versions from freeze where safe)
pip install \
  "transformers>=4.51" accelerate datasets evaluate \
  decord av einops qwen-vl-utils \
  sentencepiece protobuf pillow numpy

# Bring patched lmms-eval from Azure, then editable install
mkdir -p /workspace
azcopy copy \
  "${AZURE_DEST}/lmms-eval/*?${AZURE_SAS}" \
  /workspace/lmms-eval/ --recursive=true

cd /workspace/lmms-eval
pip install -e . --no-deps   # or with deps if clean; resolve conflicts carefully
# if needed: pip install -r requirements.txt  (after torch already installed)
```

**LongCLIP / scorer env** (lighter): separate venv or same if versions allow.

```bash
# slm-lab code + LongCLIP repo from Azure
azcopy copy "${AZURE_DEST}/slm-lab/harness/*?${AZURE_SAS}" /workspace/slm-lab/harness/ --recursive=true
azcopy copy "${AZURE_DEST}/slm-lab/scripts/*?${AZURE_SAS}" /workspace/slm-lab/scripts/ --recursive=true
azcopy copy "${AZURE_DEST}/slm-lab/Long-CLIP/*?${AZURE_SAS}" /workspace/slm-lab/Long-CLIP/ --recursive=true
```

Optional: use freeze as a **hint list**, not a hard pin:

```bash
# download freeze
azcopy copy "${AZURE_DEST}/slm-lab/env_freeze/lmmsenv.freeze.txt?${AZURE_SAS}" /tmp/lmmsenv.freeze.txt
# install non-torch lines only, or pip install package==ver one-by-one when something imports fail
```

---

## 4) Restore data from Azure (minimal vs full)

Mirror old paths so env vars / scripts keep working:

```bash
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1
mkdir -p /workspace/hf/hub /workspace/slm-lab/results
```

### Minimal — one LVBench T1 arm

```bash
# answerer weights
azcopy copy "${AZURE_DEST}/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/*?${AZURE_SAS}" \
  /workspace/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/ --recursive=true

# videos for that bench
azcopy copy "${AZURE_DEST}/hf/lvbench/*?${AZURE_SAS}" \
  /workspace/hf/lvbench/ --recursive=true

# picks + embeds for selector
azcopy copy "${AZURE_DEST}/slm-lab/results/picks_lmmseval/*?${AZURE_SAS}" \
  /workspace/slm-lab/results/picks_lmmseval/ --recursive=true
# pull only needed embeds_* / scores if regenerating picks
```

### Full lab restore (~200–300 GB)

Pull in parallel (2–3 azcopy jobs): `hf/hub/...`, `hf/videomme`, `hf/lvbench`, `slm-lab/results`, `slm-lab/features`, `slm-lab/data`.

---

## 5) Smoke before overnight runs

```bash
source /workspace/lmmsenv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/lmms-eval:/workspace/slm-lab

python - <<'PY'
import torch
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", local_files_only=True)
print("processor ok", type(p))
assert torch.cuda.is_available()
x = torch.zeros(1, device="cuda")
print("cuda alloc ok", x.device)
PY

# one-doc lmms-eval dry path if you have a tiny task; else import picks_utils
python -c "import lmms_eval; print('lmms_eval', lmms_eval.__file__)"
```

Only then launch full `lmms_eval` / `restier_run` / T1 selector evals.

---

## 6) Difficulty map

| Step | Hard? | Notes |
|---|---|---|
| Azure → disk restore | Easy | azcopy; already proven ~Gb/s from RunPod DC |
| PyTorch + CUDA match | **Medium** | Main footgun — wrong wheel = no CUDA |
| `lmms-eval` + Qwen3-VL | Medium | Pin transformers that support Qwen3-VL; watch API drift |
| LongCLIP scorer | Easy–medium | Separate small stack; needs ckpt under HF hub |
| Full 300G restore | Time/cost | Prefer **minimal pull** per experiment |

---

## Recommended order (one sitting)

1. Rent GPU pod (24 GB+).  
2. Venv + **torch CUDA smoke**.  
3. azcopy **Qwen + one bench videos + picks**.  
4. Install `lmms-eval` editable + deps.  
5. 1-video / 10-sample eval smoke.  
6. Scale pull + overnight.

---

## Cost note

- Azure storage hold: ~$5–7/mo @ ~300 GB Hot LRS.  
- RunPod: pay GPU only when running; disk on pod/volume while attached.  
- Don’t re-upload whole volume each time — **pull subsets**.

---

## If stuck

1. `nvidia-smi` + `torch.cuda.is_available()` mismatch → reinstall torch for that CUDA.  
2. HF download attempts despite Azure trees → `HF_HUB_OFFLINE=1` + correct `HF_HOME`.  
3. Import errors → install missing package from freeze **name only**, keep torch untouched.  
4. OOM → confirm k8/`batch_size=1`; don’t dual-load `hf_gpu1` hack.
