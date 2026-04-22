# Codex view — current behavior

## Current behavior snapshot
The repo currently behaves like a **planning-first project** with strong documentation but no implemented training pipeline yet.

- Strategy is clear: baseline first (CatBoost/GBDT), then QLoRA LLM if metrics justify it.
- Evaluation intent is mostly correct: macro-F1 and minority-class recall are treated as primary.
- Production constraints are acknowledged early: strict JSON output, validation, fallback behavior.
- Execution is still pending: environment setup, data audit, baseline run, and fine-tuning are all not started.

## What is good right now
- The docs avoid the common mistake of jumping directly to fine-tuning.
- Leakage and class imbalance are recognized as first-order risks.
- The target output contract is specific enough to implement consistently.

## What needs tightening
- Move from research notes to reproducible code paths and experiment artifacts.
- Freeze one split protocol and keep it constant across all model comparisons.
- Add explicit JSON-schema/Pydantic validation behavior to avoid brittle inference responses.

## My practical position
Current behavior is **directionally strong but execution-light**. The next milestone should be a reproducible baseline result on a leakage-safe split. After that, LLM work should be judged strictly against that baseline on the same test slice.
