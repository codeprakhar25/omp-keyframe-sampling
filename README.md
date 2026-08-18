# omp-keyframe-sampling

A controlled evaluation harness for frame-selection and visual-token-budget rules in
video MLLMs. Everything here re-runs uniform sampling, top-k, AKS, FOCUS, classical
Orthogonal Matching Pursuit (OMP), and an LDDR MinMax stage-1 replay under one frozen
LongCLIP scorer, so the numbers are comparable across selectors instead of each one
bringing its own scorer/prompt/answerer.

OMP is used here as a cheap 1993 reference selector (Pati et al.), not a method this
repo introduces. The harness is the contribution: fix the scorer and the evaluation
path, vary only the selection rule and the per-frame token budget.

## What's here

- `harness/` — scorer, frame decode/resize, selector implementations, eval driver,
  lmms-eval task configs (`harness/lmmseval_patch/`) for LongVideoBench across every
  duration bin and selector/scorer combination.
- `scripts/` — pick generation for each selector (`gen_omp_picks.py`, `gen_topk.py`,
  `gen_aks_picks.py`, `gen_focus_picks.py`, `gen_dpp_*.py`, ...), embedding/scoring
  utilities, coverage gates, paired significance tests (McNemar, TOST equivalence),
  and pod-level run drivers for LongVideoBench, Video-MME, and LVBench.

Results, cached embeddings, and the paper are intentionally not in this repo — this is
the harness, not the write-up or the data.

## Method, briefly

- Frames are scored once against the question *stem only* with LongCLIP; every
  selector rule reads the same cached score vector, so a comparison isolates the
  selection rule rather than the scorer.
- OMP picks greedily by residual correlation (`e_i^T q_{r-1}`), then projects the
  residual against the selected span (Gram-Schmidt) before the next pick.
- Per-frame resolution can be swept independently of which timestamps are selected,
  to separate "which frames" from "how many tokens per frame."
- Significance is paired (McNemar on per-question correct/incorrect) or an
  equivalence test (TOST) where the claim is "no meaningful difference," not a null
  result treated as proof of equivalence.

## Requirements

Python 3.11, `lmms-eval`, `torch`, a LongCLIP checkpoint. See
`scripts/requirements_viz.txt` for the analysis/visualization dependencies.

## License

MIT — see `LICENSE`.
