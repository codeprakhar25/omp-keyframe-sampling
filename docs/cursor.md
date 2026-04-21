# Cursor notes — clothing fit recommendation & fine-tuning

This file is the **research + honest eval** view for this repo. The full step-by-step plan (code, timeline, stack) lives in [`docs/README.md`](./README.md). Project context for agents is in [`docs/claude.md`](./claude.md). **Prior research & algorithm choices** (papers, GBDT/TF-IDF/LoRA/QLoRA/SFT/DPO) are in the sections *Prior research and how this project sits on it* and *Algorithms and technical strategies* below. The repo-root **OpenCode** analysis is summarized [under OpenCode analysis](#opencode-analysis); see [`opencode.md`](../opencode.md) for full code snippets.

---

## What you are building

**Task:** From user body info, item info, and optional review text, predict whether an item will feel **too small**, **about right**, or **too large** — framed as classes `small | fit | large`.

**Production-shaped output** (contract to lock early):

```json
{
  "predicted_fit": "small|fit|large",
  "confidence": 0.85,
  "recommended_action": "size_down|stay|size_up",
  "explanation": "short, user-safe rationale"
}
```

You do **not** need an LLM to solve the classification core; an LLM is optional for **structured JSON + explanations** once a baseline proves the signal is learnable.

---

## Dataset (anchor everything here)

**Kaggle:** [Clothing Fit Dataset for Size Recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation) (Misra et al.; RecSys’18 framing).

**Reality check:**

- Two sources (**ModCloth**, **RentTheRunway**) → different missingness and text richness; treat them as related but not identical distributions.
- **Class imbalance:** `"fit"` dominates; model quality lives or dies on **minority classes** (`small`, `large`). Optimize for **macro-F1** and per-class recall, not headline accuracy.
- **Reviews are gold** but dangerous: random splits leak future language into training. **`README.md`’s time-ordered split** is non-negotiable if timestamps exist; if not, split by **user** or **item** groups to approximate leakage control.
- Tabular fields are noisy (height/weight strings, bust sizes). Parsing + missingness indicators matter as much as “which LLM.”

---

## Prior research and how this project sits on it

### Foundational work (this exact dataset / task)

- **Misra, Wan & McAuley — *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces* (RecSys 2018).**  
  [ACM link](https://dl.acm.org/doi/10.1145/3240323.3240398) — Introduces the **fit-as-semantics** view: “small / fit / large” is not arbitrary; it connects **user measurements**, **item sizing**, and **language** in reviews. The Kaggle release is the public artifact of that research program. Your job is **not** to reproduce the paper’s full metric-space machinery on day one; it *is* to respect the same **leakage discipline** and **imbalance** they highlight when you build splits and metrics.

- **Kaggle bundle — [Clothing Fit Dataset for Size Recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation).**  
  Two retailers (**ModCloth**, **RentTheRunway**), overlapping schema, different missingness. Prior work treats this as a **hybrid tabular + text** problem: measurements and categories give a prior; reviews refine or contradict.

### Adjacent empirical / industry write-ups (good for column semantics, not “truth”)

- **Rent The Runway–style deep dives** (e.g. blog analyses of columns, “fit” vs rating) help you **engineer features** and sanity-check labels. Treat blog numbers as **illustrative**; always recompute on your split.

### LLMs + fashion / recommendation (why LoRA / DPO show up in our stack)

- **“Decoding Style”–class work (e.g. instruction-tuned LMs + LoRA for outfit / style tasks)** — Example direction: [arXiv 2409.12150](https://arxiv.org/html/2409.12150v1) (*Decoding Style*, fashion recommendation with Mistral-class models and preference-style training). **Takeaway for us:** small open LMs can be **specialized** to narrow domains with **parameter-efficient** updates; **DPO** (or similar) appears when the product cares about **ranking explanations**, not only accuracy.

- **Parameter-efficient fine-tuning — LoRA** — [Hu et al., LoRA (2021)](https://arxiv.org/abs/2106.09685). **Takeaway:** train low-rank adapters on attention (and often MLP) projections instead of full weights; far less VRAM and storage.

- **QLoRA** — [Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314). **Takeaway:** keep the **base model in 4-bit** while adapters and optimizer states stay in higher precision so consumer GPUs can **fine-tune** multi-billion-parameter LMs.

### Where “research” stops and “your experiment” starts

Papers and blogs suggest **families** of methods (GBDT + text, hybrid recsys, PEFT for LMs). **No** off-the-shelf paper gives you a drop-in “best” checkpoint for *your* split, **cold-start** policy, and **JSON API**. The strategy below is: **cite the lineage**, then **lock an eval protocol** and let baselines + ablations decide.

---

## Algorithms and technical strategies (what we run)

### A. Core learning problem (both tracks)

- **Formal task:** multi-class classification with ordered-ish semantics → treat as **3-class** `small | fit | large` (not ordinal regression in v1 unless you add that experiment later).
- **Primary scores:** **macro-F1** + **per-class recall** on `small` and `large`; confusion matrix always. **Accuracy** is a misleading headline when “fit” dominates.
- **Calibration (optional but recommended if you expose `confidence`):** map model scores to **reliability** (e.g. **Platt scaling** or **isotonic regression** on a validation fold) so “0.85” means something operationally.

### B. Track 1 — Tabular + text baselines (no LLM generation)

| Component | Algorithm / method | Why it fits this dataset |
|-----------|-------------------|---------------------------|
| **Tree ensembles (main workhorses)** | **CatBoost**, **XGBoost**, **LightGBM** — all **gradient-boosted decision trees (GBDT)** with different handling of **speed**, **default hyperparameters**, and **categorical** features. | Strong on **mixed types** (numeric heights/weights after parsing, high-cardinality `category`, `size`, `body_type`). CatBoost’s **ordered boosting** and native categoricals reduce leakage from naive target encoding. |
| **Ensembling (Phase 1+)** | **Stacking** or simple **soft-voting** of diverse GBDTs + a shallow **logistic regression** or ridge meta-learner (see `opencode.md`). | Raises the **ceiling** so the LLM track must beat a serious non-neural baseline, not a single under-tuned model. |
| **Text → features (sparse)** | **TF–IDF** with word + **character n-grams** (captures “tight”, “runs big”, typos). | Cheap, strong for **short reviews**; interpretable coefficients if you use linear heads on top (optional). |
| **Text → features (dense)** | **Sentence embeddings** (e.g. `sentence-transformers` small models like **MiniLM**-family) pooled to one vector per review. | Captures **semantic** fit language beyond keyword overlap; concatenate to tabular features or late-fuse in a second-stage model. |
| **Class imbalance** | **Class weights** in GBDT loss; **stratified** CV; optionally **downsample** majority in mini-batches for neural baselines only. | Stops the model from always predicting “fit”. **SMOTE** and heavy synthetic oversampling are *optional* and often risky for text+tabular hybrids—prefer weights + better features first. |
| **Leakage control (not an “algorithm” but part of strategy)** | **Group splits** by `user_id` / `item_id`, or **time-based** splits if timestamps exist. | Same algorithm with a **random** split can look SOTA and fail in production. |

**Practical ordering:** one strong **CatBoost** (or XGBoost) pipeline with clean features → add **TF-IDF or embeddings** → then **second model + stack** if you need a harder bar for the LLM.

### C. Track 2 — LLM fine-tuning (generation + JSON)

| Stage | Algorithm / method | Role |
|-------|-------------------|------|
| **Base model** | Causal **decoder-only** Transformer (e.g. **Qwen2.5 / Qwen3** instruct checkpoints). | Instruction following + enough capacity to **condition** on measurements + item + review snippet. |
| **Efficient adaptation** | **LoRA** on selected projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, and often MLP `gate_proj` / `up_proj` / `down_proj`). | Updates a **small** set of parameters; full weights frozen. |
| **Memory-efficient base load** | **4-bit quantization (QLoRA)** — NF4 weights + BF16 LoRA adapters per [QLoRA](https://arxiv.org/abs/2305.14314). | Makes **8B-class** training feasible on **~16 GB** VRAM with sensible batching. |
| **Training objective (v1)** | **Supervised fine-tuning (SFT)** — maximize likelihood of the **assistant** JSON (and optional rationale) given a fixed **user** prompt template. | Standard **TRL `SFTTrainer`** + **PEFT** pattern; easiest to debug and to validate **JSON parse rate**. |
| **Training objective (v2+)** | **DPO** (Direct Preference Optimization) or related **preference** losses — compare two candidate explanations for the same input. | Improves **human preference** on text without a full RLHF stack; see Unsloth DPO docs when SFT is stable. |cd .git
| **Infra / speed** | **Unsloth**-accelerated kernels + **HF Accelerate** / **bitsandbytes**. | Faster iteration cycles for a first implementation pass. |
| **Inference later** | **vLLM** (throughput) or **llama.cpp** **GGUF** (edge / CPU). | Deployment is separate from **training** algorithm choice. |

### D. System strategies (how algorithms are composed)

- **Two-tier routing:** GBDT produces class + probability → if confidence high and class “easy”, **skip** LLM; else call LLM for **JSON + explanation** (cost/latency control).
- **Guardrails:** **Pydantic** (or JSON Schema) **validate → retry once** with “fix JSON” hint → else **fallback** to GBDT label. This is **not** optional for production-shaped APIs.
- **Cold-start slice:** same algorithms, but **evaluated** on users/items **unseen** in training — the metric that tells you if you’re learning **generalizable** fit vs memorizing IDs.

---

## Recommended game plan (first implementation, concepts already known)

Order matters for first-timers: **measurement harness → baseline → optional LLM**.

### Phase A — Baseline you can ship (do this first)

1. **Download & audit:** label distribution, missing rates, duplicate `(user_id, item_id)`, ModCloth vs RTR breakdown.
2. **Define splits:** time-based if possible; else group split by user (and report a **cold-start** slice: users/items unseen in train).
3. **Parse & feature:** numeric height/weight, bust parsing, category, size; optional BMI; **TF-IDF or small sentence embeddings** on review text.
4. **Train a strong tabular model:** CatBoost / XGBoost / LightGBM with **class weights** or focal-style handling; tune on **macro-F1**.
5. **Calibration (optional but grown-up):** reliability of any “confidence” you expose from the baseline (e.g. predicted probabilities).

**Exit criterion:** You have a reproducible number on a **held-out** set + a clear confusion matrix for `small / fit / large`. Until this exists, fine-tuning money is speculative.

### Phase B — LLM fine-tuning (only after Phase A)

Use the baseline to answer: *Is there enough signal in inputs + text for this to be non-trivial?* If macro-F1 is already respectable, an LLM may **explain** and **format** more than it **classifies**.

1. **Format:** instruction → structured fields → **strict JSON** in the assistant turn (see `README.md` prompt sketch).
2. **Method:** **QLoRA** (4-bit) + **LoRA** on an instruction-tuned open model — e.g. **Qwen2.5-7B/8B** or **Qwen3-8B** family with a permissive license for your use case. **Unsloth + TRL** is a sane first pipeline (speed + VRAM).
3. **Compare fairly:** same test set as baseline; same metrics; add **JSON validity rate** and latency.

**Deploy rule of thumb (from `README.md`, still agree):** ship the LLM path only if it **meaningfully** beats the baseline on macro-F1 / minority recall *and* JSON is reliable enough with a **fallback** to the tabular model.

### Phase C — Production-minded layers

- **Fallback:** malformed JSON or timeout → baseline prediction (and a generic explanation if needed).
- **Cost/latency:** two-tier routing — high-confidence baseline predictions may skip the LLM for speed.
- **Monitoring:** class distribution drift, JSON failure rate, cold-start accuracy.

---

## Honest evaluation of the current repo docs

### What is already strong

- **Baseline-before-LLM** discipline is correct and underused in industry prototypes.
- **Macro-F1 + minority recall** focus matches the problem; accuracy would lie to you.
- **Leakage-aware splitting** is called out; that single choice separates toy metrics from useful ones.
- **QLoRA + instruction JSON** is a reasonable first fine-tuning stack for structured output.
- **`claude.md`** gives agents a compact, consistent context — good for automation.

### What I would tighten or question

1. **Row counts / exact schema:** treat Kaggle’s page as source of truth when you implement; minor discrepancies across blog summaries are normal.
2. **“Beat baseline by +3 macro-F1”** is a reasonable heuristic, not physics — on small test sets, use confidence intervals or bootstrap; don’t over-interpret tiny gaps.
3. **8B scale:** for a first **implementation** pass, a **7B / 3B** instruction model often iterates faster and still validates the pipeline; scale up once metrics justify GPU time.
4. **SFT-only story:** ranking **Explanations** (DPO / preference pairs) is a later upgrade; SFT on clean JSON is enough v1 if labels are reliable.

### Biggest first-timer trap

Skipping a **realistic eval slice** (cold users/items) and then “fine-tuning a big model” that only memorizes collaborative patterns from leaked splits. **Fix splits and baseline first.**

---

## OpenCode analysis

Full write-up: [`opencode.md`](../opencode.md) (repository root).

Checked against this file: OpenCode agrees with **`README.md` / `claude.md`** on track (baseline → QLoRA, macro-F1, leakage-safe splits, JSON fallback). It mainly **raises the bar** on baselines, eval, and prod hygiene.

### Status snapshot (from OpenCode)

| Phase | Model | Status |
|-------|-------|--------|
| Baseline | CatBoost + TF-IDF/embeddings | Not started |
| LLM | QLoRA on Qwen3-8B (Unsloth) | Not started |
| Deployment | vLLM or llama.cpp | Future |

### Priority order (OpenCode — use as a backlog)

| Priority | Item | Cursor take |
|----------|------|----------------|
| P0 | Cold-start test slice | **Mandatory** before trusting any number. |
| P0 | Enhanced features (BMI, category–size mapping) | Cheap wins for baseline ceiling. |
| P1 | XGBoost + LightGBM (+ optional stacking) | Harder baseline so the LLM must earn its keep. |
| P1 | JSON schema validation (e.g. Pydantic) + retry | Prefer over regex-only; pair with baseline fallback. |
| P2 | DPO vs SFT | Only after SFT JSON is stable; needs preference data. |
| P2 | Two-tier inference (skip LLM when baseline is confident) | Matches Phase C above — cost/latency. |
| P3 | Uncertainty (MC dropout / multi-LoRA ensemble) | Nice for “ask for more measurements”; not v0. |
| P3 | Smaller LLM for pipeline shakeout | Aligns with “7B/3B first” in this doc. |

### Improvement themes (full detail in `opencode.md`)

1. **Model size:** Consider **Qwen2.5-7B** or **3B** (or Phi-4-mini for JSON) before locking 8B — iterate faster; upgrade when justified.
2. **Baselines:** Add **XGBoost / LightGBM** and optionally **stacking** so “beat baseline” is meaningful.
3. **Features:** BMI; **category-specific size semantics**; sentence embeddings; light **sentiment / fit keywords** (“tight”, “runs small”).
4. **Eval:** **Calibration** curves and confidence bins on top of macro-F1; cold-start slice **before** major training.
5. **JSON:** **Pydantic**-style validation + limited retries, then baseline — not regex alone.
6. **Objectives:** **DPO** as optional pass for explanation quality (Unsloth DPO path).
7. **Uncertainty:** Later — variance across dropout or LoRA seeds for “defer / ask user.”
8. **Imbalance:** Beyond CatBoost weights — weighted sampling / focal-style ideas for LLM training if needed.
9. **Routing:** High-confidence baseline → return without LLM; use LLM when uncertain or for minority classes.
10. **Monitoring early:** distribution drift, JSON retry rate, cold-start accuracy — not only at “week 6.”

**Honest eval:** OpenCode is **directionally right** and consistent with this repo. The only caveat: **stacking three GBDTs** is more engineering than a single CatBoost run — do it **after** one solid baseline + cold-start eval, unless you enjoy tuning ensembles before you have metrics.

---

## Cross-reference

| Doc | Role |
|-----|------|
| [`README.md`](./README.md) | Full game plan: phases, code sketches, metrics, timeline, extended reference list |
| [`claude.md`](./claude.md) | Short agent context: decisions, stack, checklist |
| [`opencode.md`](../opencode.md) | OpenCode analysis: priorities, improvements, stacking/code snippets |
| This file (`cursor.md`) | Research lineage, algorithm rationale, honest eval, OpenCode alignment |

---

## Suggested next actions (concrete)

1. Create conda/venv and install baseline stack (`pandas`, `scikit-learn`, `catboost`, optional `xgboost`).
2. Ingest Kaggle files; print label counts and missingness; implement one **group-aware** split + document it in a short `EXPERIMENT.md` or notebook header.
3. Train baseline; save metrics + confusion matrix.
4. Only then clone a minimal Unsloth/TRL notebook and overfit a **tiny** subset to verify JSON formatting; scale to full SFT.

When Phase A metrics are logged, revisit this file and update a single line: **“Baseline macro-F1 on [split] = X — decision: proceed / pause LLM.”**
