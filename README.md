# Clothing Fit Prediction with QLoRA

This repository is a learning-focused machine learning project on clothing size and fit prediction. The goal is to predict whether a rented or purchased clothing item will fit a user as `small`, `fit`, or `large`, using a mix of tabular features and review text.

The project starts with classical ML baselines, then fine-tunes an instruction-tuned language model using QLoRA and compares the results on the same held-out evaluation setup.

## Project Status

Current stage: experimental research prototype.

The first fine-tuning run shows that a QLoRA-tuned LLM can improve minority-class performance over the baseline, especially for `small` and `large` fit predictions. The project is not production-ready yet. The next work is focused on leakage checks, cold-start evaluation, and ablation experiments.

## Problem

Online clothing fit is difficult because the same item size can feel different depending on:

- user measurements such as height, weight, age, bust size, and body type
- item category and selected size
- review text describing tightness, looseness, length, shoulders, waist, sleeves, etc.
- strong class imbalance where most examples are labeled `fit`

The prediction target is:

```json
{
  "predicted_fit": "small | fit | large",
  "confidence": 0.85,
  "recommended_action": "size_up | stay | size_down",
  "explanation": "Short explanation based on the user, item, and review context."
}
```

## Dataset

The project is based on the public clothing fit recommendation dataset from Rent the Runway / ModCloth:

- Dataset: [Kaggle Clothing Fit Dataset for Size Recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)
- Paper: *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces* by Misra, Wan, and McAuley

The raw data is not committed to this repository. It should be downloaded separately and placed under `data/`.

## Approach

### 1. Baseline Modeling

The baseline track uses classical ML on structured and text-derived features. The saved baseline report includes:

- CatBoost macro-F1: `0.5178`
- Small-class F1: `0.4084`
- Large-class F1: `0.4062`
- TF-IDF baseline macro-F1: `0.5707`

This gives a practical benchmark before spending GPU time on language model fine-tuning.

### 2. Prompt Dataset Construction

Each row is converted into an instruction-style supervised fine-tuning example. The prompt includes user information, item information, and optionally the review text. The target response is strict JSON.

The training data is class-balanced to reduce the dominance of the majority `fit` class:

- `small`: up to 10,000 examples
- `large`: up to 10,000 examples
- `fit`: up to 20,000 examples

This is important because accuracy alone can be misleading when most items fit correctly. Macro-F1 and per-class F1 are more useful here.

### 3. QLoRA Fine-Tuning

The first completed fine-tuning run used:

- Base model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Method: QLoRA with PEFT LoRA adapters
- Library stack: Unsloth, Transformers, TRL, PEFT, bitsandbytes
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Max sequence length: `768`
- Epochs: `3`
- Learning rate: `2e-4`
- Effective batch size: `16`

The LoRA adapters target attention and MLP projection layers:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

QLoRA keeps the base model quantized and mostly frozen, then trains a small set of adapter weights. This makes fine-tuning feasible without full-model training.

## Results

Current saved evaluation results:

| Metric | Baseline | Fine-tuned LLM |
|---|---:|---:|
| Macro-F1 | `0.5178` | `0.7312` |
| Small F1 | `0.4084` | `0.6926` |
| Large F1 | `0.4062` | `0.6442` |
| JSON validity | `1.0000` | `0.9998` |

Additional saved observations:

- Zero-shot macro-F1: `0.3184`
- Random baseline F1 estimate: `0.3330`
- Original paper F1 estimate: `0.6350`
- Human ceiling estimate: `0.8750`
- LLM judge accurate percentage: `0.52`
- LLM judge good percentage: `0.50`
- LLM judge excellent percentage: `0.06`

The fine-tuned model passes the current evaluation gate:

```text
LLM macro-F1 >= baseline macro-F1 + 0.03
JSON validity >= 98%
```

## Repository Layout

```text
.
|-- 01_eda_baseline.ipynb        # baseline exploration and classical ML benchmark
|-- 02_llm_finetune.ipynb        # QLoRA fine-tuning notebook
|-- 03_evaluation.ipynb          # evaluation and analysis notebook
|-- results/                     # saved metrics, predictions, and plots
|-- trained-info/                # lightweight adapter metadata and fine-tuning notes
|-- docs/                        # planning notes and research references
`-- data/                        # local dataset location, not committed
```

Large generated model artifacts are intentionally not committed. Files such as `.safetensors`, full tokenizer files, W&B logs, virtual environments, and raw data should be stored outside normal Git history or managed with Git LFS / Hugging Face Hub.

## Key Learnings

- Macro-F1 is a better metric than accuracy for imbalanced fit prediction.
- Class balancing has a large impact on `small` and `large` prediction quality.
- Review text carries strong signal, but it may also introduce leakage depending on the intended production use case.
- QLoRA is a practical way to fine-tune an LLM by training only adapter weights instead of the full model.
- Strict JSON validation is essential if an LLM is used as part of an application pipeline.
- Strong benchmark results are not enough; cold-start and leakage-safe evaluation are needed before claiming production usefulness.

## Limitations

This is a learning and research project, not a production recommender system.

Known limitations:

- The current result needs a deeper leakage audit.
- Review text may contain direct clues such as "too tight" or "too large".
- Cold-start performance for unseen users and unseen items still needs to be measured.
- The saved model card generated by PEFT is incomplete and should be expanded before publishing model artifacts.
- Heavy model files are excluded from Git and need a separate storage/publishing workflow.

## Next Experiments

The next planned experiments are:

1. Cold-start evaluation for unseen users and unseen items.
2. Ablation runs comparing metadata-only, review-only, and metadata-plus-review inputs.
3. Leakage audit around duplicate users, duplicate items, and review-derived label clues.
4. LoRA rank sweep with `r=8`, `r=16`, and `r=32`.
5. Learning-rate sweep with `2e-4`, `1e-4`, and `5e-5`.
6. Dropout comparison between `lora_dropout=0` and `lora_dropout=0.05`.
7. Confidence calibration for the JSON `confidence` field.

The detailed experiment plan is in [`docs/experiments.md`](docs/experiments.md). A first audit script is available at `scripts/experiment_audit.py`:

```bash
python scripts/build_splits.py --data-dir data
python scripts/experiment_audit.py --data-dir data --results-dir results
```

`build_splits.py` uses a source-balanced split by default so RentTheRunway and ModCloth are represented in each split.

## How to Reproduce

1. Download the dataset from Kaggle and place it under `data/`.
2. Run `scripts/build_splits.py` to create `data/train.parquet`, `data/val.parquet`, and `data/test.parquet`.
3. Run `01_eda_baseline.ipynb` to build the baseline metrics.
4. Run `02_llm_finetune.ipynb` on a CUDA GPU environment.
5. Run `03_evaluation.ipynb` to generate evaluation reports and plots.

The fine-tuning notebook was designed for a cloud GPU environment. A local laptop is enough for reading the notebooks and baseline analysis, but not for efficient QLoRA training.

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [Unsloth documentation](https://docs.unsloth.ai/)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer)
- [Original clothing fit dataset paper](https://dl.acm.org/doi/10.1145/3240323.3240398)
