#!/usr/bin/env python3
"""Emit lmms-eval task yamls over (scorer, bin, k).

Two families:
  longvideobench_val_picks_{sc}_{bin}s        -> $LVB_PICKS_{SC}       (k-agnostic; calibration matrix)
  longvideobench_val_picks_{sc}_{bin}s_k{k}   -> $LVB_PICKS_{SC}_K{k}  (budget matrix)
  longvideobench_val_i_{bin}s_k{k}            -> uniform@k, frames pathway

Separate env var per (scorer, k) is what lets ONE lmms-eval process hold several
scorers AND several budgets at once -> the whole matrix on ONE model load.

The _i tasks are the honest control for injected picks: injected frames enter as
IMAGES, so uniform@k must come through the same PIL pathway, NOT native-video _v.

Run from repo root:  python3 harness/lmmseval_patch/gen_picks_tasks.py
"""
from pathlib import Path

HERE = Path(__file__).parent
# k=8 is the LDDR Table 1 grid (Qwen3-VL-8B is #F=8 at EVERY bin) -> the only
# budget where our rows are comparable to a published number. 16/32 are the
# budget-scaling arms. k=6 dropped 2026-07-15: it matched no published cell and
# its one claim (sel@6 > uniform@16) came back p=0.095 ns.
BINS = (15, 60, 600, 3600)
KS = (8, 16, 32, 64)
SCORERS = {"sig": "SigLIP-so400m", "lc": "LongCLIP-L"}

HEAD = """dataset_path: longvideobench/LongVideoBench
dataset_kwargs:
  token: True
  cache_dir: datasets/longvideobench
  video: True
  force_download: False
  local_files_only: False
"""

TAIL = """test_split: validation
process_docs: !function picks_utils.filter_{bin}s
doc_to_visual: !function {visual}
doc_to_text: !function utils.longvideobench_doc_to_text
doc_to_target: "correct_choice"
generation_kwargs:
  max_new_tokens: 32
  temperature: 0
  do_sample: False
process_results: !function utils.longvideobench_process_results
metric_list:
  - metric: lvb_acc
    aggregation: !function utils.longvideobench_aggregate_results
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "Answer with the option's letter from the given choices directly.\\n"
"""

n = 0

# --- picks arms: k-agnostic (calibration) + per-k (budget) -------------------
# METHODS: "" = flat top-k (the name stays bare for backward compat with the
# existing picks files); "omp_" = OMP pick-math over the same scores+embeds.
# Both are generated NOW so an OMP arm is one env var away, not a round-trip:
# the picks JSON is the only thing that differs, and it is free to produce once
# the scores exist. Same reason every k is emitted -- k costs nothing at pick time.
METHODS = {"": "top-{k}", "omp_": "OMP-{k}"}

for meth, mpretty in METHODS.items():
    for sc, pretty in SCORERS.items():
        for b in BINS:
            for k in (None,) + KS:
                suffix = "" if k is None else f"_k{k}"
                env = (f"LVB_PICKS_{meth.upper()}{sc.upper()}"
                       + ("" if k is None else f"_K{k}"))
                visual = (f"picks_utils.longvideobench_doc_to_visual_picks_"
                          f"{meth}{sc}" + suffix)
                task = f"longvideobench_val_picks_{meth}{sc}_{b}s{suffix}"
                body = (
                    HEAD
                    + f"task: {task}\n"
                    + f"# {pretty} {mpretty.format(k=k or 'k')} picks, {b}s bin. Reads ${env}.\n"
                    + f"# Control = longvideobench_val_i_{b}s"
                    + ("" if k is None else f"_k{k}")
                    + " (same frames pathway), NOT native-video _v.\n"
                    + TAIL.format(bin=b, visual=visual)
                )
                (HERE / f"{task}.yaml").write_text(body)
                n += 1

# --- uniform@k frames-pathway controls --------------------------------------
# NOTE: k is bound in the FUNCTION (picks_utils._doc_to_visual_i_k{k}), NOT via
# dataset_kwargs.max_num_frames. lmms-eval splats dataset_kwargs into
# datasets.load_dataset(), so an extra key raises
#   ValueError: BuilderConfig ParquetConfig(...) doesn't have a 'max_num_frames' key
# at task-load time, before any sample runs. (Cost a whole budget-matrix run
# 2026-07-15; the handoff doc's claim that _i reads it from dataset_kwargs is
# only half-true — utils reads it back for subtitles, but stock never sets it.)
for b in BINS:
    for k in KS:
        task = f"longvideobench_val_i_{b}s_k{k}"
        body = (
            HEAD
            + f"task: {task}\n"
            + f"# uniform@{k} via the PIL frames pathway = control for injected picks@{k}.\n"
            + TAIL.format(bin=b, visual=f"picks_utils.longvideobench_doc_to_visual_i_k{k}")
            + "    # papers disable subs for selection ablations; stock _i defaults True\n"
            + "    insert_interleave_subtitles: False\n"
        )
        (HERE / f"{task}.yaml").write_text(body)
        n += 1

print(f"wrote {n} task yamls: picks {{sig,lc}} x {BINS} x {{None,{KS}}} + val_i x {BINS} x {KS}")
