# FOCUS VMM long + Azure stage handoff (2026-07-24)

Status: FOCUS VMM long **hA** running on pod `:19114` (RTX PRO 4500). Continue here until done / day pod ready.  
Later: **new day pod + fresh network volume** (re-stage from Azure; don’t assume this euro-3 volume attaches).

## Goal
Finish T1 VMM FOCUS long (900 Q = 3×300) then LDDR VMM long; bank scores; update mega table. RunPod credits tight; Azure = cold store.

## Azure (source of truth)
| | |
|--|--|
| SA | `slmlabsponsored` · eastus · Hot LRS · RG `rg-slm-lab-backup` |
| Container | `slm-lab` |
| Endpoint | `https://slmlabsponsored.blob.core.windows.net/slm-lab/<prefix>/` |
| Secrets | `video-understanding/.azure_backup.env` (gitignored) — KEY + SAS |
| Docs | `video-understanding/AZURE_BACKUP.md` |

**Do NOT** mount Blob live for decord/Qwen. Pattern: **azcopy stage-in → run → azcopy results out**.

### Must stage (P0)
- `hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/` (~17G) — hub blobs/trees format
- `hf/videomme/data/<videoID>.mp4` — **batch only** via picks (hA=100 vids ~19G)
- `slm-lab/results/picks_lmmseval/` incl. `picks_focus_lc_vmm_long_k8_{hA,hB,hC}.json` + `vmm_qid2vid.json`
- `lmms-eval/` (patched picks tasks)
- `slm-lab/harness` + `scripts` + `env_freeze/`

### Skip
- `hf_gpu1`, full `hf/videomme` 95G unless needed, `slmenv`/`lmmsenv` from Azure (rebuild)

## Pod specs that work
| | Min | Rec |
|--|-----|-----|
| Container disk | 40G | **60–80G** |
| Network volume | 80G | **100G** |
| GPU | 24GB+ Blackwell (sm_120) | PRO 4500 32GB |
| Layout | all under `/workspace` OR torch on container, weights+vids on volume |

**Fail modes seen:** container 20G → pip ENOSPC; volume 45G → no room for cu130 after Qwen+vids.

## Fast day-pod bootstrap
```bash
# 1) tools + env
# scp .azure_backup.env + scripts/azure_stage_answerer.sh + vmm_videos_for_picks.py + azure_upload_results.sh
export PIP_CACHE_DIR=/workspace/.pip-cache TMPDIR=/workspace/tmp HF_HOME=/workspace/hf
mkdir -p $PIP_CACHE_DIR $TMPDIR
# install azcopy (tar --no-same-owner)

# 2) stage (videos loop in azure_stage_answerer is flaky — use parallel after)
source .azure_backup.env
bash scripts/azure_stage_answerer.sh --scratch /workspace \
  --picks-blob slm-lab/results/picks_lmmseval/picks_focus_lc_vmm_long_k8_hA.json \
  --with-qwen --with-lmms-eval
# then parallel azcopy xargs -P 8 for videos_to_pull.txt

# 3) venv + Blackwell torch
python3 -m venv /workspace/lmmsenv
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# + transformers accelerate decord qwen-vl-utils datasets ... ; PYTHONPATH=/workspace/lmms-eval

# 4) CRITICAL — materialize Qwen snapshots (Azure hub lacks snapshots/; blob IDs ≠ tree names)
# map weight shards BY FILE SIZE to blobs/, symlink into snapshots/<rev>/
# rev = refs/main = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
SNAP=.../snapshots/0c351dd...
# model_args pretrained=$SNAP,device_map=auto

# 5) Video-MME dataset offline
# once: unset HF_HUB_OFFLINE; load_dataset("lmms-lab/Video-MME","videomme")  # caches under hf/datasets/
# then: HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 6) run
export VMM_PICKS_OMP_LC_K8=.../picks_focus_lc_vmm_long_k8_hA.json
HALF=hA bash slm-lab/scripts/run_focus_vmm_long_half.sh
# coverage: line count samples jsonl == 300 (doc_id is local index — do NOT match qids)
```

Scripts (local): `video-understanding/scripts/{azure_stage_answerer,azure_upload_results,vmm_videos_for_picks}.py|.sh`

## Banked T1 numbers (remember)
**LVB k8 overall:** OMP .6223 · AKS .6021 · FOCUS .5819 · LDDR-select .6320 (+0.97pt)

**VMM k8:** OMP .7233/.6000/.5433/.6222  
AKS .6967/.5567/.4822/.5785  
FOCUS .6667/.5433/long TBD  
LDDR .7389/.5933/long TBD  

Mega table: `video-understanding/results_mega_table.html`

## Open queue (after hA)
1. FOCUS long hB, hC (300 each) — same pod if volume persists
2. LDDR VMM long (can 3×300)
3. Upload results → Azure via `azure_upload_results.sh`
4. Update mega table
5. Optional: LVBench T1

## Pod lessons
- Dual GPU `from_pretrained` hang on Blackwell (`request_wait_answer`) — one model load at a time
- SSH key: `~/.ssh/runpod`
- Azure Startups $1k credits: UI shows, Sponsorship sub **not** provisioned (`InvalidPrincipalId` ManagedServices) — ignore for now; RunPod + Blob
- qid `601-1` ≠ youtube `videoID` — use `vmm_qid2vid.json`

## Current pod (may die)
`ssh root@213.173.110.111 -p 19114 -i ~/.ssh/runpod` — euro-3 volume, FOCUS hA in flight ~97/300 @ save time.
