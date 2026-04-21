# Project Context for Claude

## What this project is
Fine-tuning an open-source LLM for clothing fit recommendation — predicting whether a clothing item will fit a user as `small`, `fit`, or `large` given their body measurements, the item info, and optionally their review text.

## Dataset
- Source: [Kaggle — rmisra/clothing-fit-dataset-for-size-recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)
- Two sub-datasets: **ModCloth** and **RentTheRunway**
- ~192,544 rows, 15 features
- Key columns: `fit` (target), `height`, `weight`, `age`, `bust_size`, `body_type`, `size`, `category`, `review_text`, `review_summary`, `rating`, `user_id`, `item_id`
- Original paper: *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces* — Misra, Wan, McAuley (RecSys'18)

## User context
- First time implementing fine-tuning (knows the concepts, working on the practical implementation)
- Goal: production-usable clothing fit recommendation system
- Primary output: `{"predicted_fit": "small|fit|large", "confidence": ..., "recommended_action": "size_down|stay|size_up", "explanation": "..."}`

## Chosen approach
1. **Baseline first**: CatBoost multiclass with tabular + TF-IDF/embedding features from reviews
2. **LLM fine-tuning**: QLoRA (4-bit) on `Qwen/Qwen3-8B` using Unsloth + TRL SFTTrainer
3. **Only deploy LLM** if it beats baseline by ≥3 macro-F1 points on held-out test

## Why these model choices
- **Qwen3-8B**: Apache-2.0 license, state-of-the-art reasoning, 128k context, instruction-following
- **Unsloth**: 2–5x faster training vs vanilla HuggingFace, 80% less VRAM — important for first-timer with limited GPU
- **QLoRA**: only trains LoRA adapters (~1% of params) on top of 4-bit quantized base — fits on 16GB VRAM

## Key technical decisions
- Splits must be **time-based** (not random) to prevent review-text leakage
- Primary metric: **macro-F1** (not accuracy — "fit" class dominates ~70%)
- Missing measurements treated as signal (add `has_height`, `has_weight` indicator columns)
- JSON output validation is mandatory — add fallback to baseline classifier on parse failure
- LoRA rank r=16 on Q/K/V/O + gate/up/down projections

## Hardware requirements
- Baseline (CatBoost): CPU only
- QLoRA fine-tuning (8B): 16–24 GB VRAM (RTX 3090/4090 or cloud: RunPod A100 ~$1.50/hr)
- Post-fine-tune inference: 8 GB VRAM (4-bit quant)

## Tooling stack
- `unsloth`, `trl`, `peft`, `bitsandbytes`, `accelerate`, `transformers`
- `catboost`, `xgboost`, `scikit-learn`
- `sentence-transformers` for review embeddings
- `wandb` for experiment tracking
- `vllm` or `llama.cpp` (GGUF) for serving

## Current status
- [ ] Environment setup
- [ ] Data downloaded and audited
- [ ] Baseline CatBoost trained
- [ ] QLoRA fine-tuning run
- [ ] Comparison + evaluation
- [ ] Deployment

## Related files
- `docs/README.md` — full game plan with code snippets, timeline, evaluation protocol
- `docs/cursor.md` — research synthesis, critique of the plan, OpenCode alignment, first-timer pitfalls
- `opencode.md` (repo root) — baseline/LLM improvements, priorities P0–P3, pydantic JSON pattern
