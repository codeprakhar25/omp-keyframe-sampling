# Visual-token allocation in long-video MLLMs

Code and results for *Select, Compress, Reinvest: A Controlled Study of
Visual-Token Allocation in Long-Video MLLMs*.

A video language model given an hour of footage can afford roughly eight frames
out of 3,600. This repository contains the harness used to study how that budget
should be spent, under a single fixed scorer, prompt boundary, resolution policy
and answering model, so that one decision varies at a time.

Benchmarks: LongVideoBench, Video-MME, LVBench. Answerer: Qwen3-VL-8B-Instruct,
with GPT-5-mini and InternVL3 used for transfer checks. Scorer: frozen LongCLIP,
with a SigLIP swap as a robustness check.

## Layout

```
harness/          selection rules, scoring, answerer adapters, metrics
harness/lmmseval_patch/   lmms-eval task definitions for every benchmark, bin and budget
scripts/          pick generation, evaluation drivers, statistics, figures
data/             frame selections, per-item predictions
```

### harness/

| file | contents |
|---|---|
| `selectors.py` | the selection rules: uniform, top-k, OMP, DPP/MMR variants, AKS, FOCUS |
| `replay_selectors.py` | replays published selection schedules against our scorer |
| `embeds.py` | frozen-encoder embedding cache |
| `answerers.py` | answerer adapters |
| `metrics.py` | accuracy and paired-test primitives |
| `media.py` | frame decode and resize |
| `text.py` | question-stem construction; see the note on the prompt boundary below |
| `manifest.py`, `mcq_extract.py`, `union_retrieval.py`, `run.py` | manifest build, answer parsing, retrieval, driver |

### scripts/

Pick generation (`gen_*_picks.py`, `gen_topk.py`) writes frame selections for one
rule. `export_picks_lmmseval.py` converts them into the form the evaluation
harness consumes. `preflight_picks.py`, `cov_gate.py`, `audit_picks.py` and
`audit_coverage.py` are the gates: they assert that every question id the run
will consume is actually present, and refuse to mark a run complete otherwise.

Scoring runs through `dump_longclip_all.py` and `dump_embeds.py` to build the
embedding cache, then `score_from_embeds.py` and `dump_scores.py`.

Statistics are `pooled_mcnemar.py`, `cross_budget_mcnemar.py`, `claim_a_tost.py`,
`t1_paired_lvbench.py`, `t1_paired_videomme.py` and `interaction_same_env.py`.
`published_vs_matched.py` builds the table comparing our matched harness against
the numbers reported in the cited papers. Figures come from
`make_paper_figures.py` and `make_protocol_figure.py`.

## Requirements

Python 3.11+, PyTorch, and `lmms-eval` for the evaluation runs. See
`requirements.txt`. Frame decoding uses decord. On Blackwell-class GPUs
(sm_120) PyTorch must be built against CUDA 12.8 or newer; older builds load
weights and then fail with "no kernel image".

## Reproducing a result

```bash
# 1. build the manifest for a benchmark and duration bin
python scripts/build_longvideobench.py --bin 600

# 2. cache frozen scorer embeddings (GPU)
python scripts/dump_longclip_all.py --bin 600

# 3. generate selections for one rule
python scripts/gen_omp_picks.py --bin 600 --k 8
python scripts/export_picks_lmmseval.py --picks picks_omp_lc_600_k8.json

# 4. gate before spending GPU time
python scripts/preflight_picks.py --picks picks_omp_lc_600_k8.json

# 5. evaluate
bash scripts/lmmseval_run.sh

# 6. paired test against another arm
python scripts/pooled_mcnemar.py arm_a/ arm_b/
```

`run_matrix_k8.sh` runs the full eight-frame selector matrix for one bin.
Paths in the shell drivers default to a pod layout under `/workspace` and are
overridable with `SLM`, `LMMS`, `HF_HOME` and `PYBIN`.

## Two things worth knowing before you use this

**Batch size is a protocol axis, not a speed knob.** Padding plus non-associative
float accumulation flips greedy decoding on near-ties, so results move with batch
size even at temperature 0. Every number in the paper was produced at
`batch_size=1`. Never compare arms run at different batch sizes.

**The text handed to the scorer matters.** Scoring against the question stem and
scoring against stem-plus-options differ by two to four points. `text.py` builds
the stem; the paper reports both settings.

## Data

`data/picks/` holds the selected frame indices for every rule, bin and budget.
`data/predictions/` holds the per-item lmms-eval predictions permitted by the
benchmark licenses, covering the selection-axis arms: uniform, top-k and OMP
under LongCLIP and SigLIP on the LongVideoBench duration bins, the Video-MME
duration splits, and the blind-floor control.

Not every arm reported in the paper is included. The per-arm LVBench runs, the
spatial-compression arms, and the InternVL3 transfer arms are not in the
release, so `scripts/t1_paired_lvbench.py` and `scripts/claim_a_tost.py` need
prediction files you regenerate yourself; each script names what it expects and
reads its input directory from an environment variable. Benchmark video files
are not included; obtain them from the original benchmark releases. The cached
scorer embeddings are also excluded for size, and can be rebuilt from step 2
above.

## Citation

```bibtex
@article{khatri2026allocation,
  title  = {Select, Compress, Reinvest: A Controlled Study of Visual-Token
            Allocation in Long-Video MLLMs},
  author = {Khatri, Prakhar},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
