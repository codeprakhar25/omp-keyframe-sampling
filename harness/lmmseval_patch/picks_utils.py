"""slm-lab pick-injection for lmms-eval LongVideoBench.

Drop-in extension for `lmms_eval/tasks/longvideobench/`. Adds a `doc_to_visual`
that, instead of uniform sampling, extracts the EXACT frame timestamps our
selector chose (from a picks JSON keyed by LVB question id).

Wire-up (on the pod's lmms-eval clone):
  1. copy this file next to the stock `utils.py`
  2. copy `longvideobench_val_picks.yaml` into the same dir
  3. export LVB_PICKS=/path/to/picks.json   (our exporter's output)
  4. --tasks longvideobench_val_picks

Picks JSON: {qid: [sec, sec, ...]} (temporally sorted seconds from clip start),
produced by scripts/export_picks_lmmseval.py in slm-lab. qid == LVB doc["id"],
direct join.

Frames are returned as PIL images — the same pathway as the stock `_i`
(frames) arm, which is a first-class LVB variant. The paper-comparable control
for an injected arm is therefore uniform@k under the SAME frames pathway
(longvideobench_val_i with max_num_frames=k), NOT the native-video _v number.
"""
import functools
import json
import os
from pathlib import Path

import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval import utils as lmms_utils


def _resolve_dataset_dir(task_yaml_name, subdir_key, default_subdir):
    """Self-contained copy of the stock utils resolver (avoids a fragile
    relative import through lmms-eval's task loader)."""
    yaml_path = Path(__file__).parent / task_yaml_name
    with open(yaml_path) as f:
        safe = "".join(l for l in f.readlines() if "!function" not in l)
    cfg = yaml.safe_load(safe)
    dataset_kwargs = cfg["dataset_kwargs"]
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    cache_dir = lmms_utils.resolve_cache_dir(dataset_kwargs["cache_dir"], base_dir=hf_home)
    return os.path.join(cache_dir, dataset_kwargs.get(subdir_key, default_subdir)), dataset_kwargs


@functools.lru_cache(maxsize=8)
def _load_picks_env(envvar):
    """Load the picks JSON named by `envvar`. Cached per env var so ONE
    lmms-eval process can hold several picks arms at once (see the *_sig /
    *_lc doc_to_visual pair below) — that is what lets the whole
    scorer x bin matrix batch into a single model load."""
    path = os.environ.get(envvar)
    if not path:
        raise RuntimeError(
            f"this task needs {envvar}=/path/to/picks.json (qid -> [seconds]); "
            f"produce it with export_picks_lmmseval.py"
        )
    with open(path) as f:
        picks = json.load(f)
    eval_logger.info(f"[picks] {envvar}: loaded {len(picks)} qids from {path}")
    return picks


def _load_picks():
    return _load_picks_env("LVB_PICKS")


def _extract_frames_at(video_file, seconds):
    """Return PIL frames nearest to each requested timestamp (seconds)."""
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    n = len(vr)
    idxs = [min(n - 1, max(0, int(round(s * fps)))) for s in seconds]
    frames = vr.get_batch(idxs)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def _duration_group_int(doc):
    g = doc.get("duration_group")
    try:
        return int(g)
    except (TypeError, ValueError):
        s = str(g).rstrip("s")
        try:
            return int(s)
        except ValueError:
            return None


def filter_short_bins(dataset):
    """Keep only LongVideoBench duration_group ∈ {15, 60}.

    Thesis-first validation: short bins before 600s/3600s. Used by the
    `*_short` task yamls so lmms-eval never loads the long tails.
    """
    return dataset.filter(lambda doc: _duration_group_int(doc) in (15, 60))


def filter_15s(dataset):
    """15s bin only — fast gate vs LDDR Uniform@8 = 70.9."""
    return dataset.filter(lambda doc: _duration_group_int(doc) == 15)


def filter_60s(dataset):
    return dataset.filter(lambda doc: _duration_group_int(doc) == 60)


def filter_600s(dataset):
    """600s only — kicked after 15s gate PASSes."""
    return dataset.filter(lambda doc: _duration_group_int(doc) == 600)


def filter_3600s(dataset):
    """3600s bin — the longest LVB group, and the one where LDDR Table 1 shows the
    largest selection gain over uniform (Qwen3-VL-8B #F=8: uniform 45.2 -> LD 55.9)."""
    return dataset.filter(lambda doc: _duration_group_int(doc) == 3600)


def _doc_to_visual_from(doc, envvar):
    picks = _load_picks_env(envvar)
    qid = doc["id"]
    cache_dir, _ = _resolve_dataset_dir(
        "longvideobench_val_v.yaml", "video_subdir", "videos/"
    )
    video_path = os.path.join(cache_dir, doc["video_path"])
    if qid not in picks:
        # no injected pick for this id: fail loud rather than silently biasing
        # the bin with a fallback sampling that isn't the method under test.
        raise KeyError(
            f"[picks] qid {qid} absent from {envvar} — incomplete picks file; "
            f"generate picks for the full bin before running."
        )
    return _extract_frames_at(video_path, picks[qid])


def longvideobench_doc_to_visual_picks(doc):
    return _doc_to_visual_from(doc, "LVB_PICKS")


def longvideobench_doc_to_visual_picks_sig(doc):
    """SigLIP-so400m top-k picks; reads $LVB_PICKS_SIG."""
    return _doc_to_visual_from(doc, "LVB_PICKS_SIG")


def longvideobench_doc_to_visual_picks_lc(doc):
    """LongCLIP-L top-k picks; reads $LVB_PICKS_LC."""
    return _doc_to_visual_from(doc, "LVB_PICKS_LC")


def _load_video_uniform(video_file, duration, max_num_frames):
    """Byte-for-byte copy of stock utils.load_video's sampling.

    Kept identical on purpose: this is the CONTROL arm for injected picks, so any
    drift here (rounding, fps handling, the min(k, int(duration)) clamp) would
    show up as a fake selection effect.
    """
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))
    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    frames = vr.get_batch(frame_indices)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def _make_uniform_fn(k, name):
    def _fn(doc):
        cache_dir, _ = _resolve_dataset_dir(
            "longvideobench_val_v.yaml", "video_subdir", "videos/"
        )
        video_path = os.path.join(cache_dir, doc["video_path"])
        return _load_video_uniform(video_path, doc["duration"], k)
    _fn.__name__ = name
    _fn.__doc__ = f"uniform@{k} frames, stock sampling — control for injected picks@{k}"
    return _fn


# uniform@k with k bound to the FUNCTION, e.g. ..._doc_to_visual_i_k6.
# Why not dataset_kwargs.max_num_frames (as the handoff claimed): lmms-eval
# splats dataset_kwargs straight into datasets.load_dataset(), so an extra key
# blows up with
#   ValueError: BuilderConfig ParquetConfig(...) doesn't have a 'max_num_frames' key
# before a single sample runs. utils.py DOES read that key back out of the yaml
# for its subtitle path, which is why the claim looked true — but stock val_i.yaml
# never sets it (defaults to 16). Binding k per-function sidesteps the whole thing
# and lets several budgets coexist in one process.
for _k in (6, 8, 16, 32, 64):
    _uname = f"longvideobench_doc_to_visual_i_k{_k}"
    globals()[_uname] = _make_uniform_fn(_k, _uname)


def _make_picks_fn(envvar, name):
    def _fn(doc):
        return _doc_to_visual_from(doc, envvar)
    _fn.__name__ = name
    _fn.__doc__ = f"top-k picks; reads ${envvar}"
    return _fn


# Per-(method, scorer, k) doc_to_visual, e.g. longvideobench_doc_to_visual_picks_sig_k8
# -> $LVB_PICKS_SIG_K8, and ..._picks_omp_lc_k8 -> $LVB_PICKS_OMP_LC_K8. Generated so a
# single process can hold several METHODS and BUDGETS at once (topk@8 alongside OMP@8
# alongside uniform@8) -> the whole matrix on ONE model load, which is the only way the
# arm count stays affordable. lmms-eval's !function loader does a plain getattr, so
# module globals are fine.
# Method "" = flat top-k, keeping the bare historical names.
for _meth in ("", "omp_"):
    for _sc in ("sig", "lc"):
        for _k in (6, 8, 16, 32, 64):
            _name = f"longvideobench_doc_to_visual_picks_{_meth}{_sc}_k{_k}"
            _env = f"LVB_PICKS_{_meth.upper()}{_sc.upper()}_K{_k}"
            globals()[_name] = _make_picks_fn(_env, _name)
        # k-agnostic (calibration) variant
        _name = f"longvideobench_doc_to_visual_picks_{_meth}{_sc}"
        if _name not in globals():
            globals()[_name] = _make_picks_fn(
                f"LVB_PICKS_{_meth.upper()}{_sc.upper()}", _name)
