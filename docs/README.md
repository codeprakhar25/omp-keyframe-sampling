# Clothing Fit Recommendation — Fine-Tuning Game Plan

## TL;DR
If your goal is **accurate size prediction** (`small / fit / large`), run two tracks in parallel:

1. **Track A (baseline first, ~Week 1–2)**: CatBoost/XGBoost + text features — fast, interpretable, hard to beat.
2. **Track B (LLM, ~Week 3–5)**: QLoRA fine-tune **`Qwen/Qwen3-8B`** for structured output + user-facing explanation.

Only keep the LLM in prod if it beats the baseline on macro-F1 **and** minority-class recall.

---

## The Dataset

**Source:** [Kaggle — Clothing Fit Dataset for Size Recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)  
**Paper:** *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces* — Misra, Wan, McAuley (RecSys'18)  
**Two sub-datasets:** ModCloth + RentTheRunway

| Field | Type | Notes |
|---|---|---|
| `fit` | label | `small / fit / large` — your target |
| `user_id` / `item_id` | IDs | for collaborative signals |
| `rating` | numeric | 1–5 |
| `review_text` / `review_summary` | text | rich signal |
| `height` / `weight` / `age` | numeric | user measurements |
| `bust_size` / `body_type` | categorical | user shape |
| `size` | categorical | size ordered |
| `category` | categorical | item type |
| `rented_for` | categorical | context (RentTheRunway only) |

**Key facts:**
- ~192,544 rows, 15 features
- `fit` is heavily imbalanced — "fit" dominates; "small" and "large" are the hard classes
- Missing measurements are common — treat missingness as signal, not noise
- ModCloth has more complete measurement data; RentTheRunway has more review text

---

## Problem Framing (decide before writing any code)

Primary task: **3-class classification** → `small`, `fit`, `large`

Output contract (lock this early):
```json
{
  "predicted_fit": "small|fit|large",
  "confidence": 0.85,
  "recommended_action": "size_down|stay|size_up",
  "explanation": "Your height/weight profile suggests this item runs large in the shoulders."
}
```

Optional future tasks:
- Top-2 recommendation with confidence margin
- Cold-start inference (new user with only measurements, no history)

---

## Phase 0 — Setup (Day 1, ~2 hours)

Before touching models, set up your environment:

```bash
# Create conda env
conda create -n clothfit python=3.11 -y
conda activate clothfit

# Core stack
pip install pandas polars scikit-learn catboost xgboost
pip install transformers datasets peft trl bitsandbytes accelerate
pip install unsloth   # 2-5x faster fine-tuning, 80% less VRAM
pip install sentence-transformers  # for review embeddings
pip install wandb     # experiment tracking

# Jupyter
pip install jupyter ipywidgets
```

Hardware reality check:
- **CatBoost baseline**: runs on CPU, no GPU needed
- **QLoRA fine-tuning (8B model)**: needs at least **16 GB VRAM** (24 GB recommended)
  - RTX 3090 / 4090 works; otherwise use Google Colab Pro or RunPod (A100 ~$1.50/hr)
- **Inference after fine-tuning**: 8 GB VRAM is enough with 4-bit quantization

---

## Phase 1 — Data Audit & Preprocessing (Week 1)

### 1.1 Exploratory audit
```python
import pandas as pd

df = pd.read_json("renttherunway_final_data.json", lines=True)
print(df["fit"].value_counts(normalize=True))   # check imbalance
print(df.isnull().mean().sort_values(ascending=False))  # check missingness
```

### 1.2 Critical preprocessing steps
1. Normalize fit labels — drop/relabel anything not in `{small, fit, large}`
2. Parse measurements into standardized numeric:
   - height: convert "5ft 4in" → 64 inches
   - weight: strip "lbs", cast to float
   - bust_size: extract numeric + letter band
3. Add missingness indicator columns (`has_height`, `has_weight`, etc.)
4. De-duplicate near-identical `(user_id, item_id)` pairs — keep most recent
5. Build splits **by timestamp** (never random split — that leaks future reviews):
   - `train`: oldest 70%, `valid`: next 15%, `test`: newest 15%
   - Also build cold-start slices: users/items not in train

### 1.3 Feature engineering for baseline
```python
# Measurement delta (key signal)
df["size_weight_ratio"] = df["weight_lbs"] / df["size_numeric"]
df["height_bucket"] = pd.cut(df["height_inches"], bins=[0,60,64,68,72,100])

# Review features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))
review_feats = tfidf.fit_transform(df["review_text"].fillna(""))
```

---

## Phase 2 — Track A: Baseline Model (Week 1–2)

**Goal:** establish a hard benchmark before spending GPU time.

```python
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="MultiClass",
    class_weights=[1.0, 0.4, 1.0],  # downweight dominant "fit" class
    cat_features=["body_type", "category", "size", "height_bucket"],
    eval_metric="TotalF1",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

**Evaluate with this exact metric set (use for all models):**
```python
print(classification_report(y_test, preds, target_names=["small","fit","large"]))
# Focus on: macro-F1, recall on small + large
```

Expected baseline macro-F1: ~0.55–0.65 (anything above is a win for LLM track to beat).

---

## Phase 3 — Track B: LLM Fine-Tuning (Week 3–4)

### 3.1 Prompt format (build dataset in this shape)

Each training example is a `(instruction, input, output)` tuple:

```
### Instruction:
You are a clothing fit assistant. Given user measurements and an item,
predict fit class and return strict JSON only.

### Input:
User: height=5ft4in, weight=135lbs, body_type=hourglass, age=28, bust=34C
Item: category=dress, size=M (size chart: XS/S/M/L/XL)
Review: "Ordered my usual medium but the waist was very tight"

### Response:
{"predicted_fit":"small","recommended_action":"size_up","explanation":"Review mentions tightness at waist and measurements suggest this item runs small for your proportions."}
```

### 3.2 Fine-tuning with Unsloth + QLoRA (fastest path for first-timers)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",  # pre-quantized, fastest download
    max_seq_length=1024,
    dtype=None,        # auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # LoRA rank — start here, try 32 if GPU memory allows
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # memory saver
    random_state=42,
)
```

Training config:
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb",
        output_dir="./qwen3-clothfit",
    ),
)
trainer.train()
```

### 3.3 Why Qwen3-8B specifically
- Apache-2.0 license (commercially usable)
- State-of-the-art reasoning for its size class
- Excellent instruction following — important for strict JSON output
- 128k context window (overkill here but future-proof)
- Unsloth has a pre-quantized version so you skip quantization setup

Fallback if Qwen3-8B is too large for your GPU: use `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

---

## Phase 4 — Evaluation & Comparison (Week 4–5)

Run both models on the **exact same held-out test set**:

| Metric | Baseline (CatBoost) | LLM (QLoRA) |
|---|---|---|
| Macro-F1 | ? | ? |
| Recall — small | ? | ? |
| Recall — large | ? | ? |
| JSON validity rate | 100% | needs checking |
| Inference latency | <10ms | ~500ms–2s |
| Cold-start performance | ? | ? |

**Decision rule:** keep LLM in prod only if:
- macro-F1 ≥ baseline + 3 points **AND**
- minority recall (small/large) ≥ baseline + 5 points **AND**
- JSON validity ≥ 98% on test set

---

## Phase 5 — Hardening & Deployment (Week 5–6)

### JSON output validation (non-negotiable for LLM)
```python
import json, re

def parse_llm_output(raw: str) -> dict | None:
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            assert result["predicted_fit"] in {"small","fit","large"}
            return result
    except Exception:
        return None  # trigger fallback to baseline classifier
```

### Deployment architecture
```
User request
    ↓
[Measurement + item featurizer]
    ↓
[CatBoost baseline] ← always runs, returns fallback
    ↓
[LLM explainer] ← runs if latency budget allows
    ↓
Policy layer: map predicted_fit → size_up/stay/size_down
    ↓
Response to user
```

### Serving the fine-tuned model
```bash
# Export to GGUF for local CPU/GPU serving via llama.cpp
model.save_pretrained_gguf("clothfit-qwen3-q4", tokenizer, quantization_method="q4_k_m")

# Or serve via vLLM for production throughput
pip install vllm
vllm serve ./qwen3-clothfit --dtype auto --max-model-len 1024
```

---

## Experiment Tracking Checklist

Use W&B (or MLflow). Log for every run:
- [ ] Dataset split stats (class distribution per split)
- [ ] Training loss curve
- [ ] Validation macro-F1 per epoch
- [ ] Per-class precision/recall/F1 on test
- [ ] Confusion matrix
- [ ] JSON validity rate (LLM runs)
- [ ] Inference latency (p50/p95)
- [ ] LoRA hyperparams used (r, alpha, dropout, LR)

---

## Common First-Timer Mistakes to Avoid

1. **Random split instead of time-based** — leaks future reviews into training, inflates metrics
2. **Optimizing accuracy instead of macro-F1** — accuracy looks great when "fit" is 70% of data
3. **Skipping the baseline** — you won't know if the LLM is actually helping
4. **Not validating JSON output** — LLMs occasionally generate malformed JSON, crashes prod
5. **Fine-tuning all layers** — LoRA rank=16 on Q/K/V/O projections is enough; don't full fine-tune
6. **Ignoring cold-start** — most real users are new; test it explicitly
7. **Using training data leakage signals** — don't include `rating` as input if rating correlates with fit label at test time

---

## 6-Week Timeline

| Week | Goal | Done? |
|---|---|---|
| 1 | Data audit + leakage-safe split + EDA | [ ] |
| 2 | CatBoost baseline + text ablations | [ ] |
| 3 | QLoRA fine-tuning on Qwen3-8B (first run) | [ ] |
| 4 | Hyperparameter sweep + comparison vs baseline | [ ] |
| 5 | Hardening (JSON validation, fallback, latency) | [ ] |
| 6 | Shadow deploy + drift monitoring dashboard | [ ] |

---

## Tooling Stack

| Purpose | Tool |
|---|---|
| Data wrangling | pandas, polars |
| Baseline model | CatBoost, XGBoost |
| Text embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM fine-tuning | Unsloth + TRL SFTTrainer + PEFT |
| Quantization | bitsandbytes (QLoRA 4-bit) |
| Experiment tracking | Weights & Biases |
| Serving | vLLM or llama.cpp (GGUF) |
| Eval | scikit-learn + custom JSON validator |

---

## Research References

1. [Kaggle dataset page](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation) — task context, data files
2. [RecSys'18 paper — fit semantics framing](https://dl.acm.org/doi/10.1145/3240323.3240398) — the original paper for this dataset
3. [RentTheRunway dataset deep-dive (Shaped.ai)](https://www.shaped.ai/blog/renttherunway-dataset-deep-dive-into-fashion-fit-context-and-recommendation-challenges) — column-level analysis
4. [Decoding Style — LLM fine-tuning for outfit recommendation (arXiv)](https://arxiv.org/html/2409.12150v1) — Mistral 7B + LoRA + DPO approach, 81% AUC
5. [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) — fastest path to QLoRA
6. [Fine-tune Llama 3.1 with Unsloth (HuggingFace blog)](https://huggingface.co/blog/mlabonne/sft-llama3) — walkthrough
7. [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer) — official fine-tuning workflow
8. [PEFT LoRA docs](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) — LoRA theory + usage
9. [QLoRA paper](https://arxiv.org/abs/2305.14314) — 4-bit quantized fine-tuning
10. [LoRA paper](https://arxiv.org/abs/2106.09685) — original LoRA method
