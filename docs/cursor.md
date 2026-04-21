# Cursor notes — clothing fit recommendation & fine-tuning

This file is the **research + honest eval** view for this repo. The full step-by-step plan (code, timeline, stack) lives in [`docs/README.md`](./README.md). Project context for agents is in [`docs/claude.md`](./claude.md). The repo-root **OpenCode** analysis is summarized [below](#opencode-analysis); see [`opencode.md`](../opencode.md) for full code snippets.

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
| [`README.md`](./README.md) | Full game plan: phases, code sketches, metrics, timeline, references |
| [`claude.md`](./claude.md) | Short agent context: decisions, stack, checklist |
| [`opencode.md`](../opencode.md) | OpenCode analysis: priorities, improvements, stacking/code snippets |

---

## Suggested next actions (concrete)

1. Create conda/venv and install baseline stack (`pandas`, `scikit-learn`, `catboost`, optional `xgboost`).
2. Ingest Kaggle files; print label counts and missingness; implement one **group-aware** split + document it in a short `EXPERIMENT.md` or notebook header.
3. Train baseline; save metrics + confusion matrix.
4. Only then clone a minimal Unsloth/TRL notebook and overfit a **tiny** subset to verify JSON formatting; scale to full SFT.

When Phase A metrics are logged, revisit this file and update a single line: **“Baseline macro-F1 on [split] = X — decision: proceed / pause LLM.”**
