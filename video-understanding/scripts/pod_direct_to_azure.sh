#!/usr/bin/env bash
# Run ON RunPod CPU pod with network volume mounted (same DC as volume: EU-RO-1).
# Direct disk → Azure. No local laptop hop. Does NOT delete anything on the volume.
#
# Usage on pod:
#   export AZURE_SA=slmlabsponsored
#   export AZURE_CONTAINER=slm-lab
#   export AZURE_STORAGE_KEY='...'   # from Azure portal / az storage account keys list
#   bash pod_direct_to_azure.sh
set -euo pipefail

: "${AZURE_SA:?}"
: "${AZURE_CONTAINER:?}"
: "${AZURE_STORAGE_KEY:?}"

# volume layouts vary
ROOT=""
for cand in /workspace/slm-lab /runpod-volume/slm-lab /workspace; do
  if [[ -d "$cand/results/embeds_vmm" ]] || [[ -d "$cand/slm-lab/results/embeds_vmm" ]]; then
    ROOT="$cand"
    [[ -d "$cand/slm-lab/results" ]] && ROOT="$cand/slm-lab"
    break
  fi
done
if [[ -z "$ROOT" ]]; then
  echo "Cannot find slm-lab on volume. Mounted paths:"; ls -la /workspace /runpod-volume 2>/dev/null || true
  exit 1
fi
echo "ROOT=$ROOT"

# install azcopy if missing
if ! command -v azcopy >/dev/null; then
  echo "installing azcopy..."
  cd /tmp
  curl -fsSL -o azcopy.tgz https://aka.ms/downloadazcopy-v10-linux
  tar -xzf azcopy.tgz
  AZB=$(find /tmp -maxdepth 2 -type f -name azcopy | head -1)
  install -m 755 "$AZB" /usr/local/bin/azcopy
fi
azcopy --version | head -1

# SAS (1 day) via REST using account key — no az CLI required
EXP=$(date -u -d '+1 day' '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v+1d '+%Y-%m-%dT%H:%M:%SZ')
# simpler: use azcopy with account key via env
export AZCOPY_ACCOUNT_NAME="$AZURE_SA"
export AZCOPY_ACCOUNT_KEY="$AZURE_STORAGE_KEY"

DEST="https://${AZURE_SA}.blob.core.windows.net/${AZURE_CONTAINER}"

copy_one() {
  local src="$1" dst_path="$2"
  [[ -e "$src" ]] || { echo "skip missing $src"; return 0; }
  echo "==== COPY $src -> ${DEST}/${dst_path}"
  # --overwrite=ifSourceNewer keeps RunPod + Azure safe; never deletes source
  azcopy copy "$src" "${DEST}/${dst_path}" \
    --recursive=true \
    --overwrite=ifSourceNewer \
    --log-level=WARNING
}

# IMPORTANT trees only; exclude videos by not copying videos/
copy_one "$ROOT/results/embeds_vmm"     "slm-lab/results/embeds_vmm"
copy_one "$ROOT/results/embeds_lvbench" "slm-lab/results/embeds_lvbench"
copy_one "$ROOT/results/embeds_text"    "slm-lab/results/embeds_text"
copy_one "$ROOT/results/embeds"         "slm-lab/results/embeds"
copy_one "$ROOT/results/embeds_lc"      "slm-lab/results/embeds_lc"
copy_one "$ROOT/results/scores"         "slm-lab/results/scores"
copy_one "$ROOT/results/picks_lmmseval" "slm-lab/results/picks_lmmseval"
copy_one "$ROOT/harness"                "slm-lab/harness"
copy_one "$ROOT/scripts"                "slm-lab/scripts"

# data without videos: copy subdirs explicitly
for sub in qvh subtitles synthetic synthetic_hard; do
  copy_one "$ROOT/data/$sub" "slm-lab/data/$sub"
done
# manifests at data root (no mp4)
if [[ -d "$ROOT/data" ]]; then
  mkdir -p /tmp/data_meta
  find "$ROOT/data" -maxdepth 1 -type f ! -name '*.mp4' ! -name '*.webm' ! -name '*.mkv' \
    -exec cp -a {} /tmp/data_meta/ \;
  copy_one /tmp/data_meta "slm-lab/data"
fi

for d in aks_ablation dppmm_ablation focus_ablation lvbench_eval videomme_eval restier \
         lmmseval_matrix_clean aks_vmm dppmm_vmm focus_vmm picks; do
  copy_one "$ROOT/results/$d" "slm-lab/results/$d"
done

# loose bank files
for f in mega_bank_2026-07-23.json RESULTS_SUMMARY.txt; do
  [[ -f "$ROOT/results/$f" ]] && azcopy copy "$ROOT/results/$f" "${DEST}/slm-lab/results/$f" --overwrite=ifSourceNewer
done

echo "ALL_DONE $(date -u)"
echo "RunPod volume untouched. Verify in Azure portal Storage browser → container slm-lab"
