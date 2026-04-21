# OpenCode Analysis: Clothing Fit Recommendation Fine-Tuning

## Objective Summary

Fine-tune an open-source LLM for clothing fit recommendation — predicting whether a clothing item will fit a user as `small`, `fit`, or `large` given:
- User body measurements (height, weight, age, bust_size, body_type)
- Item info (size, category)
- Optional review text

**Target Output:**
```json
{
  "predicted_fit": "small|fit|large",
  "confidence": 0.85,
  "recommended_action": "size_down|stay|size_up",
  "explanation": "..."
}
```

## Current Approach (from docs/claude.md + docs/README.md)

| Phase | Model | Status |
|-------|-------|--------|
| Baseline | CatBoost multiclass + TF-IDF/embeddings | Not started |
| LLM | QLoRA fine-tune on Qwen3-8B (Unsloth) | Not started |
| Deployment | vLLM or llama.cpp | Future |

**Key Technical Decisions:**
- Time-based train/val/test splits (leakage prevention)
- Primary metric: Macro-F1 (not accuracy — "fit" class ~70%)
- Missing measurements → indicator columns (`has_height`, `has_weight`)
- LoRA rank r=16 on Q/K/V/O + gate/up/down projections
- QLoRA 4-bit quantization for 16GB VRAM constraint

---

## Improvements for Better Performance

### 1. Model Architecture Options

| Option | Recommendation | Rationale |
|--------|-------------|----------|
| **Current: Qwen3-8B** | Consider downgrading first | 8B may be overkill; 4-bit inference still needs ~8GB |
| Alternative: Qwen2.5-7B | Try if 8B OOM | Similar performance, slightly faster |
| Alternative: Qwen2.5-3B | Best for limited VRAM | Can run on 8GB, good instruction-following |
| Alternative: Phi-4-mini | Check for better JSON generation | Microsoft model, strong reasoning |

**Action:** Start with smaller model to iterate faster, upgrade if baseline macro-F1 > 0.60

### 2. Stronger Baseline Models

The docs only mention CatBoost. Add **ensemble diversity:**

```python
# Add to baseline comparison
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier

# Stacking: CatBoost + XGBoost + LightGBM with logistic regression meta-learner
# This gives harder benchmark for LLM to beat
```

**Expected baseline improvement:** +2-5 macro-F1 points vs single model

### 3. Enhanced Feature Engineering

| Current | Missing | Add |
|---------|---------|-----|
| Raw height/weight | | BMI calculation |
| | | Size chart normalization per category |
| | | User shape latent embeddings (from user_id history) |
| | | Item fit-tendency per brand/seller |
| TF-IDF on review | | Sentence embeddings (sentence-transformers) |
| | | Sentiment + keyword extraction (tight/long/run small) |

**Action:** Add category-specific size mapping — "size M" in dresses ≠ "size M" in jeans

### 4. Better Evaluation Protocol

Add these metrics beyond macro-F1:

```python
# Calibration: predicted confidence should match actual accuracy
from sklearn.calibration import calibration_curve
plot_calibration_curve(y_test, probs)  # reliability diagram

# Confidence bins: does p=0.90 actually ≈ 90% correct?
bin_by_confidence(preds, probs, y_test)

# Cold-start: test on users NOT in training
# This is the real production scenario
```

**Action:** Build explicit cold-start test slice BEFORE starting model work

### 5. JSON Output Hardening (Critical)

LLMs generate malformed JSON. Current fallback is baseline classifier, but better:

```python
# Add schema validation with retry
from pydantic import BaseModel

class FitResponse(BaseModel):
    predicted_fit: Literal["small", "fit", "large"]
    confidence: float
    recommended_action: Literal["size_down", "stay", "size_up"]
    explanation: str

def parse_with_retry(raw: str, max_retries=3) -> FitResponse:
    for _ in range(max_retries):
        try:
            return FitResponse.model_validate_json(raw)
        except:
            raw = regenerate_with_hints(raw)  # ask LLM to fix
    return fallback_to_baseline()  # CatBoost prediction
```

### 6. Alternative Fine-Tuning Objectives

Current approach: SFT (Supervised Fine-Tuning) with instruction pairs

**Consider:** DPO (Direct Preference Optimization) instead
- Collect or generate preference pairs: "this explanation is better than that one"
- DPO often produces more natural output than simple SFT
- Works well with Unsloth: see https://unsloth.ai/dpo

### 7. Uncertainty Quantification

Real production systems need to know "I don't know":

```python
# Method 1: MC Dropout at inference
logits_list = [model(input, dropout=True) for _ in range(10)]
variance = np.std(logits_list)

# Method 2: Ensemble of LoRA adapters
# Train 3 LoRAs with different seeds, variance = uncertainty

# If confidence variance > threshold → request more info from user
```

### 8. Class Imbalance Handling

Current: CatBoost uses `class_weights=[1.0, 0.4, 1.0]`

**Better approaches for LLM:**
- Focal loss in training
- Data augmentation: generate synthetic minority examples
- Weighted sampling in batch construction

### 9. Deployment Architecture Improvement

Current design:
```
User → Baseline → LLM explainer → Response
```

**Better:**
```
User → 
  [Fast: CatBoost] → confidence > 0.80? → Return (no LLM)
  [Slower: LLM] → confidence < 0.60 OR minority class → Enhance with explanation
```

Two-tier: skip LLM for high-confidence baseline predictions

### 10. Monitoring & Drift

Add from Day 1, not Week 6:

```python
# Track in production
monitor(
    prediction_distribution,      # shift in class balance?
    confidence_histogram,         # becoming overconfident?
    json_retry_rate,           # LLM output quality drift
    cold_start_accuracy,        # new user performance
)
```

---

## Recommended Priority Order

| Priority | Improvement | Why |
|----------|------------|-----|
| P0 | Cold-start test slice | Without this, you can't trust eval |
| P0 | Enhanced features (BMI, category-size map) | Baseline quality |
| P1 | XGBoost + LightGBM ensemble | Harder baseline |
| P1 | JSON schema validation | Production critical |
| P2 | DPO vs SFT comparison | Better output quality |
| P2 | Two-tier inference | Cost/quality tradeoff |
| P3 | Uncertainty quantification | Trust signals |
| P3 | Smaller model iteration | Faster experiments |

---

## Summary

The current approach is sound for a first attempt. Key gaps to address:

1. **Stronger baselines** — CatBoost alone is too weak; add XGBoost/LightGBM ensemble
2. **Feature gaps** — BMI, category-specific size charts, sentiment extraction
3. **Cold-start evaluation** — most real users are new; test this explicitly
4. **JSON hardening** — pydantic validation + retry > regex fallback
5. **Architecture** — two-tier inference (baseline fast path + LLM enhancement)

Focus on getting a reliable cold-start test slice BEFORE training any models — this determines whether the project succeeds in production.