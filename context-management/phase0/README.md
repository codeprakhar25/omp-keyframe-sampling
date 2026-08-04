# Phase-0 harness

Flat vs Hier retrieve on canonical `data/toy` (or later full synthetic).

## Setup

```bash
cd context-management/phase0
pip install -r requirements.txt
# OPENAI_API_KEY in phase0/.env or exported
```

## Smoke (3 queries, retrieval+QA)

```bash
python3 -m harness.run --data data/toy --methods flat,hier --k 5 --k-global 1 --smoke 3
```

## BM25 (default ranker — no embeddings)

```bash
python3 -m harness.run --data data/hard_v1 --ranker bm25 --methods flat,hier --k 5 --k-global 1
```

## Embed ranker

```bash
python3 -m harness.run --data data/hard_v1 --ranker embed --methods flat,hier --k 5 --k-global 1
```

Lead metrics: **recall@k + QA EM/F1**. `precision_in_scope` is diagnostic only (≈1 for hard Hier by construction).


Outputs under `runs/<timestamp>_flat-hier/`: `config.json`, `metrics.json`, `predictions.jsonl`, `delta.json`.

Embed cache: `runs/_embed_cache/`.
