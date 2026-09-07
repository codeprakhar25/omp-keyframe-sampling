"""Uniform-16 @ ~50% pixels — the missing control for Claim B (§5.6).

Claim B is currently OMP-16@50% vs OMP-8@full-res (+2.36pt pooled, p=.0346, n=976). Both arms
use OMP picks, so it isolates frames-vs-pixels at fixed budget but does NOT show the selector is
needed. This control repeats the comparison with the selector removed:

    uniform-16 @ ~50% pixels   vs   uniform-8 @ full res   (banked: 3600 .4716, 600 .5534)

If uniform also gains ~+2.4pt, the equal-budget win is a pure VLM frames-vs-pixels property and
OMP contributes nothing to it -- the same way `uniform50` killed the rank claim.

APPEND the source below to the LongVideoBench picks_utils (the module that already defines
`longvideobench_doc_to_visual_restier`). It reuses that file's imports (`VideoReader`, `cpu`,
`Image`, `torch`, `os`) and its `_resolve_dataset_dir` helper.

TWO CORRECTIONS vs the first draft of this patch (2026-07-28) -- both were run-blocking:
  1. it called a `_video_path(doc)` helper that DOES NOT EXIST in picks_utils -> NameError on
     the first doc. Now resolves the path the same way every other doc_to_visual here does.
  2. it sampled with `np.linspace(0, len(vr)-1, k)`, which is NOT the rule the banked uniform-8
     baseline used. That baseline goes through `_load_video_uniform`: stride
     `int(duration*fps/k)*i`, so its last frame sits at 87.5% of the video. linspace reaches
     100%, handing the new arm the final 12.5% of every video that the baseline never sees --
     a temporal-coverage confound masquerading as a resolution effect. Sampling is now copied
     verbatim from `_load_video_uniform`.

Budget arithmetic: cap = 0.5 * bh * bw per frame; measured mean pixel_ratio is ~0.46 (smart_resize
FACTOR rounding + the MINP floor), so 16 x 0.46 = 7.4 frame-equivalents vs 8 x 1.0 for the
baseline -- the new arm sits ~8% UNDER budget, which makes any gain conservative. Confirm with
token_audit.py before trusting the comparison.
"""

UNIFORM16_HALF_SRC = '''

def _load_video_uniform_frac(video_file, duration, max_num_frames, frac):
    """Uniform sampling BYTE-FOR-BYTE identical to `_load_video_uniform`, each frame then
    capped at `frac` of its full-res rendered pixels.

    The sampling block below is copied verbatim from `_load_video_uniform` on purpose: the
    paired baseline for this arm is the banked uniform-8 run, which went through that exact
    function. Any drift (linspace instead of stride, len(vr) instead of duration*fps, reaching
    the last frame instead of stopping at stride*(k-1)) would change temporal coverage and show
    up as a fake resolution effect. The resize block is copied from `_extract_frames_restier`
    for the same reason on the pixel axis.
    """
    from qwen_vl_utils import smart_resize

    FACTOR, MINP, MAXP = 32, 256 * 28 * 28, 1605632

    # --- sampling: verbatim from _load_video_uniform ---
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))
    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    batch = vr.get_batch(frame_indices)
    batch = batch.numpy() if isinstance(batch, torch.Tensor) else batch.asnumpy()

    # --- resize: verbatim from _extract_frames_restier ---
    out, dbg = [], []
    for fr in batch:
        img = Image.fromarray(fr).convert("RGB")
        W, H = img.size
        bh, bw = smart_resize(H, W, factor=FACTOR, min_pixels=MINP, max_pixels=MAXP)
        if frac >= 0.999:
            th, tw = bh, bw
        else:
            cap = max(MINP, int(frac * bh * bw))
            th, tw = smart_resize(H, W, factor=FACTOR, min_pixels=MINP, max_pixels=cap)
        if (tw, th) != (W, H):
            img = img.resize((tw, th), Image.BICUBIC)
        img._tier_max_pixels = max(MINP, th * tw)
        out.append(img)
        dbg.append((frac, tw * th, bw * bh))
    if os.environ.get("RESTIER_DEBUG"):
        print(
            "[RESTIER_DEBUG] uniform16-half n=%d idx=%s (frac, px, full_px)=%s"
            % (len(out), frame_indices, dbg),
            flush=True,
        )
    return out


def longvideobench_doc_to_visual_uniform16_half(doc):
    """Claim-B control arm: 16 uniform frames at ~50% pixels ~= the uniform-8 full-res budget.

    Paired against the banked uniform-8 full-res arm (`longvideobench_val_i_*_k8`:
    3600s .4716, 600s .5534). Reads NO picks file -- frames are chosen inline by the same
    stride rule that arm used, so the only differences are frame count and resolution.
    """
    cache_dir, _ = _resolve_dataset_dir(
        "longvideobench_val_v.yaml", "video_subdir", "videos/"
    )
    video_path = os.path.join(cache_dir, doc["video_path"])
    return _load_video_uniform_frac(video_path, doc["duration"], 16, 0.5)
'''

if __name__ == "__main__":
    print(UNIFORM16_HALF_SRC)
