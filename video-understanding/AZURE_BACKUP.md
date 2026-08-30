# Azure backup — agent handoff

> **UPDATED 2026-08-30 — ACCOUNT MOVED.** The storage account `slmlab89c4a8` was
> **deleted** in the 2026-08-04 billing migration (244 GB moved off the card-billed
> PAYG subscription onto the $10k sponsorship credit). Its DNS name no longer
> resolves, so every command below that names it will fail with
> `Failed to resolve 'slmlab89c4a8.blob.core.windows.net'` — which reads like a
> network problem but is a dead account.
>
> | Field | Old (dead) | Current |
> |---|---|---|
> | Storage account | `slmlab89c4a8` | **`slmlabsponsored`** |
> | Blob endpoint | `https://slmlab89c4a8.blob.core.windows.net` | **`https://slmlabsponsored.blob.core.windows.net`** |
> | Container | `slm-lab` | `slm-lab` (unchanged) |
> | Prefix layout | — | unchanged |
>
> The rest of this document has been updated in place to the new account;
> resource group (`rg-slm-lab-backup`) and container (`slm-lab`) are unchanged,
> and the subscription is now `ba212d9e-bba5-4bcd-bc4b-5b1c649f1e3a` (sponsored).
> Verified 2026-08-30 against the live container:
> `slm-lab/data/videos/` 756 blobs / 26.8 GiB · `hf/videomme/` 1644 / 94.5 GiB ·
> `hf/lvbench/` 103 / 57.7 GiB.
>
> Read SAS lives in AWS SSM Parameter Store at `/video-selector/azure-sas`
> (us-east-1, SecureString, `sp=rl`, container-scoped).
>
> **Gotcha:** videos are under `hf/videomme/` and `hf/lvbench/` — *not* bare
> `videomme/` / `lvbench/`. Listing the bare prefixes returns 0 blobs and looks
> like the data is gone.

Copy of RunPod network volume (EU-RO-1 / `cifnbc8jpx`) → Azure Blob.  
**Purpose:** survive RunPod credit expiry; spin answerer / selector work without re-downloading benches.

Verified 2026-07-23: azcopy waves finished with **0 failed**; spot counts matched volume (Qwen 25, videomme 1644, lvbench 103, data/videos 756, features 22878, embeds_vmm 2700, …).

---

## Where it lives

| Field | Value |
|---|---|
| Subscription | `Azure subscription 1` (`ba212d9e-bba5-4bcd-bc4b-5b1c649f1e3a`) |
| Resource group | `rg-slm-lab-backup` |
| Region | `eastus` |
| Storage account | `slmlabsponsored` |
| SKU | `Standard_LRS` (Hot) |
| Container | `slm-lab` |
| Blob endpoint | `https://slmlabsponsored.blob.core.windows.net` |
| Portal | Storage account → **Storage browser** → container `slm-lab` |

**URI pattern:**  
`https://slmlabsponsored.blob.core.windows.net/slm-lab/<prefix>/...`

---

## Credentials (do not commit keys)

Local secrets file (gitignored):  
`video-understanding/.azure_backup.env`

Expected vars:

```bash
AZURE_RG=rg-slm-lab-backup
AZURE_SA=slmlabsponsored
AZURE_CONTAINER=slm-lab
AZURE_BLOB=https://slmlabsponsored.blob.core.windows.net
AZURE_STORAGE_KEY=<from portal Access keys or az CLI>
# optional short-lived:
AZURE_DEST=https://slmlabsponsored.blob.core.windows.net/slm-lab
AZURE_SAS=<container SAS with racwl>
```

**Get key (human / agent with `az login`):**

```bash
az storage account keys list \
  -n slmlabsponsored -g rg-slm-lab-backup \
  --query "[0].value" -o tsv
```

**Mint SAS (copy/list for ~2 days):**

```bash
az storage container generate-sas \
  --account-name slmlabsponsored \
  --account-key "$AZURE_STORAGE_KEY" \
  --name slm-lab \
  --permissions racwl \
  --expiry "$(date -u -d '+2 days' '+%Y-%m-%dT%H:%MZ')" \
  -o tsv
```

Quote `AZURE_SAS` in env files — token contains `&`.

---

## How another agent should access

### A. Azure CLI (list / download)

```bash
source video-understanding/.azure_backup.env   # or export vars

# list prefix
az storage blob list \
  --account-name "$AZURE_SA" --account-key "$AZURE_STORAGE_KEY" \
  --container-name "$AZURE_CONTAINER" \
  --prefix "slm-lab/results/embeds_vmm/" \
  --num-results 20 -o table

# download tree
az storage blob download-batch \
  --account-name "$AZURE_SA" --account-key "$AZURE_STORAGE_KEY" \
  --source "$AZURE_CONTAINER" \
  --destination ./restore \
  --pattern "slm-lab/results/picks_lmmseval/*"
```

### B. azcopy (fast restore to a GPU pod)

```bash
export AZURE_DEST="https://slmlabsponsored.blob.core.windows.net/slm-lab"
# SAS from generate-sas above
azcopy copy "${AZURE_DEST}/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/*?${AZURE_SAS}" \
  /workspace/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/ \
  --recursive=true
```

### C. Python (`azure-storage-blob`)

```python
from azure.storage.blob import BlobServiceClient
client = BlobServiceClient(
    f"https://{AZURE_SA}.blob.core.windows.net",
    credential=AZURE_STORAGE_KEY,
)
container = client.get_container_client("slm-lab")
for b in container.list_blobs(name_starts_with="slm-lab/results/scores/"):
    print(b.name, b.size)
```

### Suggested layout on a new compute pod

Mirror RunPod paths so existing scripts keep working:

```
/workspace/hf/...          ← from Azure hf/
/workspace/lmms-eval/...   ← from Azure lmms-eval/
/workspace/slm-lab/...     ← from Azure slm-lab/
```

**Do not** restore `slmenv` / `lmmsenv` from Azure (not uploaded). Use:

```
slm-lab/env_freeze/lmmsenv.freeze.txt
slm-lab/env_freeze/slmenv.freeze.txt
```

→ `pip install` on matching CUDA torch build for the new GPU.

---

## Bucket layout (container `slm-lab`)

Top-level prefixes:

```
hf/
  hub/
    models--Qwen--Qwen3-VL-8B-Instruct/     # P0 answerer ~17G
    models--BeichenZhang--LongCLIP-L/       # P0 scorer ~1.6G
    models--google--siglip-so400m-patch14-384/
    models--Salesforce--blip2-itm-vit-g/
    models--BAAI--bge-small-en-v1.5/
    datasets--lmms-lab--Video-MME/          # HF meta JSON (not videos)
    datasets--lmms-lab--LVBench/
    datasets--longvideobench--LongVideoBench/
    datasets--LVHaystack--LongVideoHaystack/
  videomme/                                 # ~95G video cache
  lvbench/                                  # ~58G video cache
lmms-eval/                                  # patched tasks / runners
slm-lab/
  harness/
  scripts/
  Long-CLIP/
  features/                                 # QVH / moment features ~8.7G
  env_freeze/
  data/
    videos/
    subtitles/
    qvh/
    synthetic/   synthetic_hard/
    (manifests *.json at data/ root — no bulk mp4 at root except skipped whispr)
  results/
    embeds/  embeds_vmm/  embeds_lvbench/  embeds_lc/  embeds_text/
    scores/
    picks_lmmseval/
    aks_ablation/  focus_ablation/  dppmm_ablation/
    aks_vmm/  focus_vmm/  dppmm_vmm/
    lvbench_eval/  videomme_eval/  restier/  lmmseval_matrix_clean/
    qvh_* / older ablations (dpp_, ab_, ksweep_, …)
    mega_bank_2026-07-23.json  RESULTS_SUMMARY.txt
```

### Verified blob counts (2026-07-23)

| Prefix | Blobs |
|---|---|
| `hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/` | 25 |
| `hf/videomme/` | 1644 |
| `hf/lvbench/` | 103 |
| `slm-lab/data/videos/` | 756 |
| `slm-lab/features/` | 22878 |
| `slm-lab/results/embeds_vmm/` | 2700 |
| `slm-lab/results/embeds_lvbench/` | 1549 |
| `lmms-eval/` | 3481 |

Approx total parked: **~200–300 GB** (Hot LRS ≈ **$5–7/mo**).

---

## Intentionally NOT on Azure

| Path | Why |
|---|---|
| `lmmsenv/`, `slm-lab/slmenv/` | CUDA/host-specific; use freeze + reinstall |
| `hf_gpu1/` | Dup Qwen dual-load hack |
| RunPod junk logs / smoke / persist | noise |
| Full pip wheel caches | skip |

---

## Related local files

| File | Role |
|---|---|
| `video-understanding/.azure_backup.env` | Secrets (gitignored) |
| `video-understanding/scripts/pod_direct_to_azure.sh` | Early single-shot uploader |
| `video-understanding/results_mega_table.html` | Banked metrics notebook |
| RunPod volume (until deleted) | Original `cifnbc8jpx` @ EU-RO-1 — still valid backup of record on RunPod side |

---

## Run model — Blob store + GPU pod scratch (not a live mount)

Azure Blob ≠ RunPod network volume. **Do not** point decord/`from_pretrained` at raw HTTPS blobs.
Pattern: **stage-in → run → stage-out** on pod SSD (~40–80G enough per batch).

```text
Blob (eastus / slm-lab)  --azcopy-->  /workspace/scratch (pod disk)
                                      HF_HOME, picks, batch videos, Qwen
                                      lmms-eval run
                     results --azcopy--> Blob
```

| Scratch component | Size (FOCUS VMM long hA) |
|---|---|
| Qwen3-VL-8B | ~17G |
| Batch videos (100 unique / 300 Q) | ~19G |
| picks + qid2vid + harness | ≪1G |
| lmms-eval checkout | ~0.5G |
| **Peak** | **~40G** (+ venv) |

Full `hf/videomme` (~95G) stays on Blob — pull **only** videos for the picks JSON.

### Scripts (local repo `video-understanding/scripts/`)

| Script | Role |
|---|---|
| `azure_stage_answerer.sh` | Pull picks + batch mp4s + optional Qwen/lmms-eval/freeze into `--scratch` |
| `vmm_videos_for_picks.py` | `question_id` → `videoID` → `hf/videomme/data/<id>.mp4` |
| `azure_upload_results.sh` | Push result dir back to Blob |

`vmm_qid2vid.json` lives at  
`slm-lab/results/picks_lmmseval/vmm_qid2vid.json` (2700 qids).

### CPU smoke (done 2026-07-23 on RunPod CPU `:33203`)

Pulled from Blob into `/workspace/azure_scratch_smoke`: map + `picks_focus_lc_vmm_long_k8_hA.json` + sample mp4 (`0Jbc3Ah4EIc.mp4` ~111M). azcopy OK with SAS from `.azure_backup.env`.

### GPU job recipe

```bash
# on GPU pod (azcopy + az CLI; disk ≥80G recommended)
source .azure_backup.env   # AZURE_SA / KEY or SAS
bash scripts/azure_stage_answerer.sh \
  --scratch /workspace/scratch \
  --picks-blob slm-lab/results/picks_lmmseval/picks_focus_lc_vmm_long_k8_hA.json \
  --with-qwen --with-lmms-eval

export HF_HOME=/workspace/scratch/hf HF_HUB_OFFLINE=1
export PYTHONPATH=/workspace/scratch/lmms-eval:...
# rebuild venv once from slm-lab/env_freeze/*.freeze.txt (CUDA match)
# then run existing run_focus_vmm_long_half.sh pointing picks at scratch path

bash scripts/azure_upload_results.sh \
  /workspace/scratch/slm-lab/results/focus_vmm/long_hA \
  slm-lab/results/focus_vmm/long_hA
```

Same Azure region as storage (`eastus`) ⇒ cheap/no egress. Cross-cloud (RunPod EU → Blob US) already paid on upload; download back to EU costs egress.

---

## Agent checklist — restore for LVBench / Video-MME answerer

1. `az login` (or use key/SAS from `.azure_backup.env`).
2. Stage via `azure_stage_answerer.sh` (Qwen + **batch** videos + picks) — not full bench trees unless needed.
3. Pull `lmms-eval` (patched yamls / `picks_utils`).
4. Rebuild venv from `slm-lab/env_freeze/*.freeze.txt`.
5. Set `HF_HOME=<scratch>/hf`, `HF_HUB_OFFLINE=1`.
6. Upload results with `azure_upload_results.sh`.

**Copy-only history:** uploads used `azcopy --overwrite=ifSourceNewer`. Never deleted RunPod volume as part of this backup.
