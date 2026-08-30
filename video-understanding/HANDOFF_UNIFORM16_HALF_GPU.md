# Handoff — uniform-16 @50% GPU run (Claim B control)

**Status (2026-07-28):** CPU plumbing **DONE** and banked on Azure. Next agent = **GPU only**.  
Do **not** re-do patch / yaml / join / smoke unless Azure artifacts missing.

> ### ⚠️ CORRECTION 2026-07-28 (later) — the version on Blob is BROKEN, re-upload before staging
>
> Review of the appended patch found **two run-blocking defects**. Both fixed locally in
> `slm-lab/cpu_smoke/uniform16_half/bundle/longvideobench/` and in
> `slm-lab/scripts/uniform16_half_patch.py`. **The copies live on Blob still have them.**
>
> 1. **`NameError` on the first doc.** `longvideobench_doc_to_visual_uniform16_half` called
>    `_video_path(doc)` — a helper that **does not exist** in `picks_utils.py`. Every other
>    `doc_to_visual` in that file resolves the path with
>    `_resolve_dataset_dir("longvideobench_val_v.yaml", "video_subdir", "videos/")` +
>    `os.path.join(cache_dir, doc["video_path"])`. Now does the same.
> 2. **Wrong sampling rule — a temporal-coverage confound, not a crash.** The patch used
>    `np.linspace(0, len(vr)-1, 16)`. The banked uniform-8 baseline goes through
>    `_load_video_uniform`, which uses stride `int(duration*fps/k)*i`. Those differ at the tail:
>    stride\@k8 stops at **87.5%** of the video, linspace reaches **100%**. The new arm would
>    have been handed the last **12.5% of every video that the baseline never sees** — a
>    coverage effect that would have been read as a resolution effect. Sampling is now copied
>    verbatim from `_load_video_uniform`. (k16 stride reaches 93.7%; that 6.2pp is intrinsic to
>    k=8→16 under the stock rule, not a patch artifact — footnote it, don't fix it.)
>
> Verified after fix: `py_compile` clean, no undefined calls (AST scan), no `LVB_PICKS*` refs in
> either yaml, stale "Residual-tiered / OMP-8" yaml comments replaced. The smoke's **resize**
> half still stands (mean pixel_ratio 0.46, path untouched); its **`idxs_head`/`idxs_tail`** are
> now stale — the new index math was re-verified arithmetically instead (no videos needed).
>
> **Budget note:** measured ratio 0.46, so 16 × 0.46 = **7.4** frame-equivalents vs 8 × 1.0
> baseline — the new arm is ~**8% UNDER** budget, making any gain conservative. Say this in the
> paper rather than claiming exact parity.
>
> ### Token math — "50%" is a CAP, not the realized ratio
>
> `tokens = pixels / 1024`. The arm sets `max_pixels = 0.5 × full_render`, then `smart_resize`
> floors both dims to a multiple of `FACTOR=32` (`floor_by_factor`, **not** round) — so the
> realized ratio always lands **below** 0.5. Per video the ratio is a **constant across all 16
> frames** (frames share native resolution); it **varies across videos**. Simulated (reproduces
> the smoke's `901120 → 414720` exactly):
>
> | native | full tok | half tok | ratio | 16f vs 8f budget |
> |--|--|--|--|--|
> | 1920x1080 | 1508 | 720 | 0.478 | 0.955 |
> | 1280x720 | 880 | 405 | 0.460 | 0.921 |
> | 1024x576 | 576 | 264 | 0.458 | 0.917 |
> | 854x480 | 405 | 180 | 0.444 | 0.889 |
> | **640x360** | 220 | 180 | **0.818** | **1.636** |
> | **320x240** | 221 | 221 | **1.000** | **2.000** |
>
> **The floor is the hazard.** `min_pixels = 256×28×28` = 196 tokens. Videos at or near it
> **cannot be halved**, so 16 frames costs up to **2×** the uniform-8 baseline — for those the
> equal-budget framing is false and any gain is bought with double the tokens.
>
> Mitigating: Claim B's own OMP-16@50% arm used the **same cap and floor on the same videos**,
> so this control stays apples-to-apples with Claim B either way — only selection differs. And
> the banked restier audit removed 47–68% of tokens on the real corpus, implying most LVB videos
> sit well above the floor. Encouraging, not proof.
>
> **NEW MANDATORY PRE-GPU GATE:** during staging (CPU, free) ffprobe all 395 videos, compute
> exact per-video token counts for both arms, and report the realized pooled ratio plus the
> count of floor-hitting videos. If a nontrivial share hit the floor, stratify or exclude them
> **explicitly** — do not discover it in the McNemar.

Related: `slm-lab/RUN_UNIFORM16_HALF.md` (science plan) · `slm-lab/cpu_smoke/uniform16_half/` (local mirror).

---

## What this run is

| Arm | Frames | Res | Role |
|--|--|--|--|
| **NEW** | 16 uniform | ~50% pixels (`smart_resize` cap) | this GPU job |
| **BASELINE (banked)** | 8 uniform | full | McNemar only — **do not re-run** |

Equal-budget control for Claim B: if uniform-16@50% also gains ~+2.4pp like OMP-16@50% vs OMP-8, then Claim B is a **VLM frames-vs-pixels** effect (selector not required). Either outcome publishable.

Bins: LongVideoBench **600s** (n=412) + **3600s** (n=564). Pooled n=976.

---

## Azure — how access works here

| | |
|--|--|
| Storage account | `slmlabsponsored` |
| Container | `slm-lab` |
| Endpoint | `https://slmlabsponsored.blob.core.windows.net` |
| Secrets | `video-understanding/.azure_backup.env` (gitignored) — `AZURE_SA`, `AZURE_CONTAINER`, `AZURE_STORAGE_KEY`, optional `AZURE_SAS` |
| Docs | `video-understanding/AZURE_BACKUP.md` |

**Pattern:** stage-in from Blob → run on pod → **upload results before stop**.  
This job uses **container disk only** (no network volume). **Stop = disk gone.**

Prefer **account key** (`az storage blob … --account-key`) over SAS — SAS in sourced env files often breaks on `&`.

**SSH to RunPod:** use `~/.ssh/runpod` (not `id_ed25519` — that key fails on these pods).

**Note on Azure subscriptions:** lab Blob lives on the **old** backup account/key in `.azure_backup.env`. A newer Startups-sponsored sub (`…gmail162…`) may be active for *new* spend — **do not assume** `rg-slm-lab-backup` is on the current CLI default tenant. Blob via **key** still works regardless of which tenant `az account show` points at.

---

## CPU step — already done (do not redo)

Pod used: `103.196.86.88:29514` (CPU, container ~45G). Artifacts uploaded.

| Check | Result |
|--|--|
| Patch | `_load_video_uniform_k_frac` + `longvideobench_doc_to_visual_uniform16_half` appended to LVB `picks_utils.py` |
| Yamls | `longvideobench_val_uniform16half_{3600s,600s}.yaml` |
| Join assert | 564 + 412 qids · **0 missing** on Azure |
| Unique videos needed | **395** (not full 753 mp4s) · ~**25.7 GiB** |
| Smoke | 16 linspace frames · mean pixel_ratio **~0.46** (smart_resize / MINP floor — expected, not a bug) |

### Azure paths (CPU outputs)

```
slm-lab/cpu_smoke/uniform16_half/README.md
slm-lab/cpu_smoke/uniform16_half/join_assert.json
slm-lab/cpu_smoke/uniform16_half/smoke_extract.json
slm-lab/cpu_smoke/uniform16_half/videos_needed_600_3600.txt   ← pull list for GPU
slm-lab/cpu_smoke/uniform16_half/uniform16_half_cpu.tgz

# live task files (already overwritten on Blob)
lmms-eval/lmms_eval/tasks/longvideobench/picks_utils.py
lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_uniform16half_3600s.yaml
lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_uniform16half_600s.yaml
```

Local mirror: `slm-lab/cpu_smoke/uniform16_half/`.

---

## Correct McNemar baselines (important)

`RUN_UNIFORM16_HALF.md` once cited **.4770 / .5850** — those are **uniform-16 full**, **wrong** for this control.

Use **uniform-8 full** (`longvideobench_val_i_*_k8`):

| Bin | Acc | Samples on Azure |
|--|--|--|
| 3600 | **.4716** | `slm-lab/results/lmmseval_matrix_clean/k8_3600/Qwen__Qwen3-VL-8B-Instruct/20260716_193815_samples_longvideobench_val_i_3600s_k8.jsonl` (564 lines ✓) |
| 600 | **.5534** | `slm-lab/results/lmmseval_matrix_clean/k8_600/Qwen__Qwen3-VL-8B-Instruct/20260716_134540_samples_longvideobench_val_i_600s_k8.jsonl` (412 lines ✓) |

McNemar helper: `slm-lab/scripts/restier_mcnemar.py` (on Azure). Runner: `slm-lab/scripts/restier_run.sh` (on Azure; may be absent from local git tree).

---

## GPU pod — stage checklist

**Hardware:** 1× GPU ≥24GB VRAM · **container disk ≥100GB** (safer 120) · no network volume.

| What | Azure prefix / note | ~Size |
|--|--|--|
| Qwen3-VL-8B | `hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/` | ~16.3 GiB |
| Videos | **only** names in `videos_needed_600_3600.txt` under `slm-lab/data/videos/` | ~25.7 GiB |
| lmms-eval + patched LVB tasks | `lmms-eval/` (at least tasks/longvideobench + model code) | varies |
| Scripts | `slm-lab/scripts/restier_run.sh`, `restier_mcnemar.py`, … | small |
| Baseline jsonl | `k8_3600` + `k8_600` samples above | small |

**Qwen hub note (from prior pods):** Azure hub may lack `snapshots/`. Materialize snapshot / symlink weight shards before `pretrained=…` (see `HANDOFF_FOCUS_AZURE.md`).

**Layout suggestion:**
```
/workspace/
  hf/…Qwen…
  slm-lab/data/videos/*.mp4     # 395 files
  lmms-eval/
  slm-lab/scripts/
  slm-lab/results/…             # outputs + banked baselines
```

---

## GPU run commands (sketch)

```bash
# after stage + venv + torch (Blackwell: cu130 as before)
# CRITICAL first: RESTIER_DEBUG=1 on 1–2 docs; confirm pixel ~half
# confirm answerer module is chat/ path with do_resize=False (not simple/)

# bs=1, NSHARD=1 — NEVER doc-shard (corrupt coverage history)
bash slm-lab/scripts/restier_run.sh 0 longvideobench_val_uniform16half_3600s uni16half_3600 <ENV> ""
bash slm-lab/scripts/restier_run.sh 0 longvideobench_val_uniform16half_600s  uni16half_600  <ENV> ""
```

This arm **reads no picks file** (linspace inline). If runner requires a picks env, pass empty / dummy per script help.

**Gates before believing numbers:**
1. Unique qids: 3600=564, 600=412 (not line count alone).
2. `RESTIER_DEBUG` shows half-ish pixels; 0/0 McNemar discordant = bug until proven otherwise.
3. `token_audit.py` ≈ 1.0× vs uniform-8 full token cost.
4. Upload results to Azure **before** stopping container.

**Suggested upload prefix:**
```
slm-lab/results/restier/uni16half_3600/
slm-lab/results/restier/uni16half_600/
slm-lab/results/restier/uni16half_summary.json   # accuracies + McNemar vs k8 baselines
```

---

## After numbers

1. McNemar vs banked uniform-8 (per bin + pooled like Claim B).  
2. Update `PAPER_DRAFT.md` Claim B wording:
   - uniform also ~+2pp → soften (budget geometry; selector optional for B)
   - uniform flat → selector matters for B  
3. Tick checklist item in draft §9.

---

## Explicitly out of scope for this handoff

- Re-running CPU patch/smoke  
- top-k / AKS @50% grid  
- Full 753-video pull (waste)  
- Assuming network volume persistence  

---

## One-liner for the next agent

CPU done + on Azure. Stage Qwen + **395** videos + patched lmms-eval; run `uniform16half` 600/3600; McNemar vs **uniform-8** `.5534`/`.4716`; upload; stop.
