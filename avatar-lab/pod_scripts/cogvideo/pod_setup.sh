#!/bin/bash
# ============================================================================
# pod_setup.sh — rebuild the CogVideoX/finetrainers env (ftenv) on a fresh
# RunPod Blackwell (sm_120, cu128) pod. ~20 min. Reproduces the working stack
# that took a 5-deep version-pin debug to find (see CROSSMODAL_LORA_LOG.md).
#
# Assumes: network volume mounted at /workspace, this repo's pod_scripts on it.
# Run:  bash pod_setup.sh
# ============================================================================
set -e
cd /workspace/avatar

# --- 0. overlay (/root, /usr) is WIPED on every pod stop -> reinstall ffmpeg ---
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true

# --- 1. venv ---
python -m venv /workspace/avatar/ftenv
source /workspace/avatar/ftenv/bin/activate
pip install -q --upgrade pip

# --- 2. PINNED torch stack (NOT latest!) ---------------------------------
# torch 2.11's torchvision 0.26 dropped torchvision.io.video_reader (finetrainers
# needs it) AND torch 2.11 removed _AttentionOp. torch 2.7.0 has BOTH. cu128 = sm_120.
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --- 3. finetrainers + deps ---
git clone --depth 1 https://github.com/huggingface/finetrainers.git /workspace/avatar/finetrainers || true
pip install -q -r /workspace/avatar/finetrainers/requirements.txt || true
pip install -q "git+https://github.com/huggingface/diffusers.git"
pip install -q "peft>=0.6.0" accelerate transformers safetensors imageio imageio-ffmpeg ftfy

# --- 4. video-decode pins (the part that bites) --------------------------
# datasets 5.0 returns lazy torchcodec.VideoDecoder; finetrainers calls
# sample["video"].size(0) -> needs eager decord/torchvision VideoReader.
# datasets 3.3.2 (finetrainers floor) returns that. torchcodec 0.4.0 pairs torch 2.7.
pip install -q "datasets==3.3.2" "torchcodec==0.4.0"

# --- 5. re-pin torch (a dep above may have dragged cpu/cu124 torch back) ---
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "POD_SETUP_DONE"
python -c "import torch,torchvision,diffusers,datasets,torchcodec; \
print('torch',torch.__version__,'tv',torchvision.__version__,'diffusers',diffusers.__version__, \
'datasets',datasets.__version__,'cap',torch.cuda.get_device_capability())"
python -c "import torchvision.io.video_reader; print('VIDEO_READER_OK')"

# ---------------------------------------------------------------------------
# RUNTIME ENV (export before any train/gen; HF cache MUST be on local overlay,
# NOT /workspace — MooseFS FUSE deadlocks the HF Xet client + hits volume quota):
#   export HF_HOME=/root/hf_local HF_HUB_DISABLE_XET=1
#   source /workspace/avatar/.hf_env   # contains: export HF_TOKEN=...
# ---------------------------------------------------------------------------
