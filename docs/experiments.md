# Next Experiments

This document tracks the next evaluation work after the first QLoRA run. The goal is to learn whether the current result is genuinely useful or mostly helped by data leakage / review text shortcuts.

## Why This Matters

The first fine-tuned model reached strong headline metrics:

- Macro-F1: `0.7312`
- Small F1: `0.6926`
- Large F1: `0.6442`
- JSON validity: `0.9998`

Those numbers are promising, but they are not enough by themselves. Clothing-fit prediction can be misleading because reviews often contain direct phrases like "too small", "runs large", or "fit perfectly". If the intended product predicts fit before a future user buys/rents an item, using that same user's review text would not be available at inference time.

## Experiment 1: Split And Leakage Audit

Question: are train, validation, and test cleanly separated?

Before running the audit, build the parquet splits from the raw Kaggle JSON files:

```bash
python scripts/build_splits.py --data-dir data
```

The default split strategy is `source_balanced`: it splits RentTheRunway and
ModCloth separately, then combines each train/validation/test partition. This
avoids the earlier issue where a global sort could leave validation/test
dominated by one source.

To reproduce the original notebook-style split for comparison:

```bash
python scripts/build_splits.py --data-dir data --split-strategy global_time --output-prefix global_time_
python scripts/experiment_audit.py --data-dir data --results-dir results
```

Expected raw files:

```text
data/renttherunway_final_data.json
data/modcloth_final_data.json
```

Checks:

- Class distribution for each split.
- Column availability for `user_id`, `item_id`, and review columns.
- User overlap between train and validation/test, if user IDs exist.
- Item overlap between train and validation/test, if item IDs exist.
- Exact review-text overlap across splits.
- Label-clue phrase coverage in review text.

Expected output:

- `results/experiment_audit.json`
- `results/experiment_audit.md`

Run:

```bash
python scripts/experiment_audit.py --data-dir data --results-dir results
```

Important: the original notebook output suggests one earlier parquet export did not preserve `user_id` or `item_id`. The `scripts/build_splits.py` script intentionally keeps both IDs so true cold-start evaluation can be computed.

## Experiment 2: Cold-Start Evaluation

Question: does the model work for unseen users and unseen items?

Slices:

- Seen user + seen item
- Unseen user + seen item
- Seen user + unseen item
- Unseen user + unseen item

Why this matters:

- A recommender often sees new users and new products.
- Random row-level splits can inflate metrics if the same users/items appear in train and test.
- Cold-start performance is closer to production behavior.

Requirements:

- `user_id` preserved in `train.parquet`, `val.parquet`, and `test.parquet`
- `item_id` preserved in `train.parquet`, `val.parquet`, and `test.parquet`
- full prediction file with one row per test example

## Experiment 3: Input Ablations

Question: which input source is carrying the performance?

Variants:

| Variant | Input Fields | What It Tests |
|---|---|---|
| `metadata_only` | height, weight, age, body type, bust size, size, category, source | Can structured information predict fit without review leakage? |
| `review_only` | review text and summary | Is the model mostly reading explicit fit phrases? |
| `metadata_plus_review` | metadata + review text | Current full-information setting |

Recommended order:

1. Start with non-LLM baselines for each variant.
2. Then run LLM inference/fine-tuning only on the variants that are informative.
3. Compare macro-F1, small F1, large F1, and JSON validity.

## Experiment 4: Hyperparameter Sweep

Only run this after the leakage and ablation audits.

Suggested sweep:

| Experiment | Values | Reason |
|---|---|---|
| LoRA rank | `8`, `16`, `32` | Measures whether adapter capacity is limiting performance. |
| Learning rate | `2e-4`, `1e-4`, `5e-5` | Lower LR may reduce noisy or overconfident updates. |
| LoRA dropout | `0`, `0.05` | Unsloth is fastest with `0`; `0.05` may regularize. |
| Epochs | `3`, `5` | More epochs may help if validation loss is still improving. |

## Decision Rules

Treat the current model as promising but not final until:

- Cold-start metrics are reported.
- Review-only vs metadata-only ablation is measured.
- Any direct review-label clue leakage is quantified.
- Full test predictions are saved, not only samples.

The most important next question is:

> Does the model still perform well when the input resembles what would actually be available at prediction time?
