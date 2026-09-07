#!/usr/bin/env python3
"""Per-frame embedding cache for offline selector replay (AdaRD / Adaptive-Greedy / DPP).

Companion to dump_scores.py: that cached scalar SigLIP relevance; diversity/coverage
selectors additionally need the VECTORS -- SigLIP image embeddings (Gram matrix for
AdaRD log-det) and DINOv2 embeddings (facility-location coverage for Adaptive Greedy).
One GPU pass here, then every selector variant replays on CPU forever
(harness/replay_selectors.py via scripts/replay_picks.py).

GPU-only to RUN, import-safe offline (torch lazy). Pod usage (from /workspace/slm-lab):

    PYTHONPATH=. HF_HOME=/workspace/hf python3 scripts/dump_embeds.py --bins 600

Output: one npz per item under --out-dir/{id}.npz with
    siglip (N,1152) fp16 L2-normalized | dinov2 (N,1024) fp16 L2-normalized | times (N) fp32
600s bin ~= 100 items x ~300-600 frames -> ~150-300MB total; scp the dir down after.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from harness.media import iter_frames, load_frames
from harness.manifest import check_gold_reliable


def build_embedders(siglip_id: str, dino_id: str, longclip: bool = False,
                    use_dino: bool = True, use_siglip: bool = True,
                    lc_repo: str = "Long-CLIP",   # repo clone, relative to slm-lab root
                    lc_ckpt_repo: str = "BeichenZhang/LongCLIP-L",
                    lc_ckpt_file: str = "longclip-L.pt"):
    """Return embed_fn(images) -> dict of L2-normalized fp16 embeddings.

    Towers are opt-out: siglip (N,1152), dinov2 (N,1024), longclip (N,768).
    longclip is the IMAGE tower of LongCLIP-L, so pick-math (OMP/AdaRD Gram
    matrices, DPP kernels) can run in LongCLIP's own geometry instead of
    borrowing SigLIP's. Without it, a "LongCLIP" pick-math arm is really
    LongCLIP scores on SigLIP geometry, which is a different claim.

    use_siglip=False exists for GPU MEMORY, not compute. This stage is CPU-bound
    on decode (cv2 walks every native frame) while the GPU sits at ~0% util, so
    throughput is set by how many shards fit in VRAM. fp32 SigLIP-so400m is 3.5GB
    of the ~5.2GB weight footprint; dropping it takes a shard ~7.0GB -> ~2.7GB and
    lets 11 shards run where 4 did. Measured 2026-07-16: 4 shards on the 3600s bin
    tracked 13-15h against a 112-core box that was 78% idle.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig = sig_proc = None
    if use_siglip:
        sig = AutoModel.from_pretrained(siglip_id).to(device).eval()
        sig_proc = AutoProcessor.from_pretrained(siglip_id)
    dino = dino_proc = None
    if use_dino:
        dino = AutoModel.from_pretrained(dino_id).to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained(dino_id)

    lc_model = lc_prep = None
    if longclip:
        import sys
        from huggingface_hub import hf_hub_download
        if lc_repo not in sys.path:
            sys.path.insert(0, lc_repo)
        from model import longclip as _lc  # noqa: E402 (repo package)
        ck = hf_hub_download(lc_ckpt_repo, lc_ckpt_file)
        lc_model, lc_prep = _lc.load(ck, device=device)
        lc_model.eval()

    use_ac = device == "cuda"

    def _norm(x):
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)

    def _encode_chunk(chunk):
        """Encode ONE batch of PIL images -> {tower: fp16 array}. Per-image, no cross-image
        state (LayerNorm ViTs, not BatchNorm), so results are independent of batch boundaries
        -- streaming in batches of 32 is numerically identical to encoding the whole video."""
        out = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            if sig is not None:
                pv = sig_proc(images=chunk, return_tensors="pt").to(device)
                feat = sig.get_image_features(**pv)
                if not torch.is_tensor(feat):  # transformers 5.x returns a ModelOutput
                    feat = feat.pooler_output if getattr(feat, "pooler_output", None) is not None \
                        else feat.last_hidden_state[:, 0]
                out["siglip"] = _norm(feat.float())
            if dino is not None:
                pv = dino_proc(images=chunk, return_tensors="pt").to(device)
                d = dino(**pv)
                feat = d.pooler_output if getattr(d, "pooler_output", None) is not None \
                    else d.last_hidden_state[:, 0]
                out["dinov2"] = _norm(feat.float())
            if lc_model is not None:
                px = torch.stack([lc_prep(im) for im in chunk]).to(device)
                out["longclip"] = _norm(lc_model.encode_image(px).float())
        return out

    def _finalize(parts):
        # Emit ONLY keys we actually computed. A placeholder "siglip" would be worse than
        # omitting it: harness.embeds.load_image_embed checks key PRESENCE, so a fake key
        # reads as a cached embed and silently scores garbage.
        got = {k: np.concatenate(v) for k, v in parts.items() if v}
        if not got:
            raise SystemExit("REFUSING: every tower disabled — nothing to embed.")
        return got

    def embed_fn(images, batch_size=64):
        """List path (kept for any non-streaming caller): holds all images, batches internally."""
        parts = {}
        for i in range(0, len(images), batch_size):
            for k, v in _encode_chunk(images[i:i + batch_size]).items():
                parts.setdefault(k, []).append(v)
        return _finalize(parts)

    def embed_stream(frame_iter, batch_size=32):
        """Streaming path: pull frames in batches, encode, KEEP ONLY the vectors, drop the
        raw frames. Peak RAM = one batch of PIL images (~batch_size x 2.7MB), not the whole
        video. Returns (times[N] float32, {tower: emb[N,D] fp16}). This is what lets a shard
        run in ~2.5GB instead of ~6GB and fit many shards under a 57.7GB cgroup."""
        times, parts, chunk = [], {}, []

        def flush():
            if not chunk:
                return
            for k, v in _encode_chunk([f.image for f in chunk]).items():
                parts.setdefault(k, []).append(v)
            times.extend(f.seconds for f in chunk)
            chunk.clear()  # drop refs to the PIL images -> memory reclaimed before next batch

        for fr in frame_iter:
            chunk.append(fr)
            if len(chunk) >= batch_size:
                flush()
        flush()
        return np.asarray(times, dtype=np.float32), _finalize(parts)

    return embed_fn, embed_stream



def _norm_bin(b) -> str:
    """LVB bins are '600s'/'3600s' -> '600'/'3600'. Video-MME is 'short'/'medium'/'long'.
    NEVER use str.replace('s','') — that turns 'short' into 'hort'."""
    s = str(b)
    if s.endswith("s") and s[:-1].isdigit():
        return s[:-1]
    return s

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--siglip", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--dino", default="facebook/dinov2-large")
    ap.add_argument("--use-dino", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-use-dino skips the DINOv2 tower entirely (~1/3 of the GPU "
                         "work). Nothing in the selection pipeline reads the dinov2 key: "
                         "scorers use siglip/longclip, OMP/AdaRD use the same. Only pass "
                         "--use-dino if a method that actually consumes it exists.")
    ap.add_argument("--use-siglip", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-use-siglip skips the SigLIP tower. This is a GPU MEMORY knob, "
                         "not a compute one: the stage is CPU-bound on cv2 decode while the "
                         "GPU idles at ~0%%, so throughput = how many shards fit in VRAM. "
                         "fp32 SigLIP-so400m is 3.5GB of a ~7.0GB shard; dropping it gives "
                         "~2.7GB and ~11 shards instead of 4. The npz then carries NO siglip "
                         "key -- harness.embeds.load_image_embed will report that scorer as "
                         "uncached rather than return wrong vectors.")
    ap.add_argument("--dedup-by-video", action=argparse.BooleanOptionalAction, default=False,
                    help="Image embeds are a property of the VIDEO, not the question, but "
                         "the cache is keyed per qid. LVB reuses one video across several "
                         "questions (3600s: 564 items over 188 videos), so the default "
                         "re-decodes and re-encodes each video ~3x. With this flag the video "
                         "is embedded once and the sibling qids are HARDLINKED to it — same "
                         "layout, same bytes, ~3x less work. Do not combine with --num-shards: "
                         "two shards holding sibling qids would both miss the cache and race.")
    ap.add_argument("--longclip", action=argparse.BooleanOptionalAction, default=True,
                    help="also cache LongCLIP-L image embeds (768-d) so pick-math can "
                         "run in LongCLIP geometry, not just SigLIP's. Refills npz files "
                         "written before this flag existed.")
    ap.add_argument("--dump-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=3600)
    ap.add_argument("--bins", nargs="+", default=["600"],
                    help="length_bin values to embed (default 600)")
    ap.add_argument("--gold-reliable-only", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-gold-reliable-only keeps text-Qs (gold_reliable=False)")
    ap.add_argument("--video-root", default="",
                    help="decode from LOCAL staged videos (same basename under this dir) "
                         "instead of the manifest's media_path. mfs per-frame grabs are "
                         "network-latency-bound and collapse under shard concurrency; a local "
                         "root (e.g. /dev/shm/v3600) makes decode local-speed.")
    ap.add_argument("--out-dir", default="results/embeds")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="run N processes over disjoint items. Decode (ffmpeg) is the "
                         "bottleneck and the GPU idles ~75%% between bursts, so shards "
                         "scale near-linearly. GPU mem is the cap: ~3 models/process.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="lower it to fit more shards on one card")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    # refuse a filter that would silently drop every item (see harness/manifest.py)
    check_gold_reliable(manifest, args.gold_reliable_only, args.manifest)
    bins = set(_norm_bin(b) for b in args.bins)
    print(f"towers: siglip={args.use_siglip} longclip={args.longclip} dino={args.use_dino} "
          f"| shard {args.shard}/{args.num_shards} | bins={args.bins}")
    embed_fn, embed_stream = build_embedders(
        args.siglip, args.dino, longclip=args.longclip,
        use_dino=args.use_dino, use_siglip=args.use_siglip)
    os.makedirs(args.out_dir, exist_ok=True)

    # dedup map: video_file -> sibling qids in scope. Built from the SAME filters as the
    # loop, so a sibling is only ever reused when it would have been embedded identically.
    sibs, vid_of, _shard_of = {}, {}, {}
    if args.dedup_by_video:
        for it in manifest:
            bb = _norm_bin(it.get("length_bin", "all"))
            if bb not in bins:
                continue
            if args.gold_reliable_only and not it.get("gold_reliable"):
                continue
            vf = it.get("video_file") or it.get("media_path")
            vid_of[it["id"]] = vf
            sibs.setdefault(vf, []).append(it["id"])
        dup = sum(len(v) - 1 for v in sibs.values())
        print(f"dedup-by-video: {len(sibs)} videos across {len(vid_of)} items "
              f"-> {dup} hardlinks instead of re-encodes")
        for _i, _vf in enumerate(sorted(sibs)):
            _shard_of[_vf] = _i % max(args.num_shards, 1)
        if args.num_shards > 1:
            mine = sum(1 for v in _shard_of.values() if v == args.shard)
            print(f"shard {args.shard}/{args.num_shards}: {mine} videos (deterministic split)")

    def _sibling_src(qid):
        """A finished sibling npz for the same video that has every key we need.

        `want` must track the ENABLED towers, not a hardcoded set: with
        --no-use-siglip a hardcoded {"siglip"} is satisfied by nothing, so this
        returns None for every sibling, dedup silently stops linking, and each
        video gets decoded 3x -- the run just takes 3x longer and says nothing.
        """
        want = set()
        if args.use_siglip:
            want.add("siglip")
        if args.longclip:
            want.add("longclip")
        if args.use_dino:
            want.add("dinov2")
        for q in sibs.get(vid_of.get(qid), []):
            if q == qid:
                continue
            p2 = os.path.join(args.out_dir, f"{q}.npz")
            if not os.path.exists(p2):
                continue
            try:
                if want.issubset(set(np.load(p2).files)):
                    return p2
            except Exception:
                continue   # half-written/corrupt sibling: ignore, do not link to it
        return None

    n = skipped = failed = linked = 0
    seen = 0
    for item in manifest:
        b = _norm_bin(item.get("length_bin", "all"))
        if b not in bins:
            continue
        if args.gold_reliable_only and not item.get("gold_reliable"):
            continue
        # shard AFTER the bin/gold filters so the split is even across workers.
        # With --dedup-by-video, shard on the VIDEO, not the item: sibling qids must
        # land in the SAME worker or two shards both miss the cache, race on the same
        # decode, and one os.link() hits an existing path. Video-sharding also keeps
        # the expensive part (a full sequential cv2 walk of a 3600s file) done once.
        seen += 1
        if args.num_shards > 1:
            if args.dedup_by_video:
                # NOT hash(): str hashing is salted per-process (PYTHONHASHSEED), so
                # each worker would build a DIFFERENT video->shard map -- some videos
                # embedded twice, others never, and nothing would say so. Index into a
                # sorted video list instead: identical in every process, and balanced.
                if _shard_of.get(vid_of.get(item["id"])) != args.shard:
                    continue
            elif (seen - 1) % args.num_shards != args.shard:
                continue
        out = os.path.join(args.out_dir, f"{item['id']}.npz")
        if os.path.exists(out):
            # an existing npz from an older run may lack the longclip key; refill it
            if args.longclip:
                try:
                    if "longclip" in np.load(out).files:
                        skipped += 1
                        continue
                except Exception:
                    pass  # unreadable -> re-dump below
                print(f"refill {item['id']} (no longclip key)")
            else:
                skipped += 1
                continue
        if args.dedup_by_video:
            src = _sibling_src(item["id"])
            if src:
                # same video -> same 1fps frames -> byte-identical embeds. Hardlink, so
                # the cache layout stays per-qid and every downstream reader is unchanged.
                # `out` may already exist and still reach here: the refill path above falls
                # through when an OLD npz lacks a key we now need (e.g. written before
                # longclip was added). os.link() onto an existing path raises FileExistsError,
                # so drop the stale file first -- that is exactly what refill means.
                if os.path.exists(out):
                    os.remove(out)
                os.link(src, out)
                linked += 1
                print(f"link {item['id']} -> {os.path.basename(src)} (same video)")
                continue
        # --video-root: decode from a LOCAL staged copy instead of the network volume.
        # cv2 grabs every native frame; over mfs each grab is a network round-trip, so even
        # 4 concurrent shards collapsed to ~4.85 min/video (vs 33s solo) and 12 wedged the box
        # (load 225, zero throughput) on 2026-07-16. Same basename under a local root (e.g.
        # /dev/shm) makes every grab a local read. Only the media_path is swapped; id/keys/
        # dedup are unchanged.
        if args.video_root:
            item = {**item, "media_path": os.path.join(
                args.video_root, os.path.basename(item.get("media_path", "")))}
        # Stream: never hold a whole video in RAM (the 57.7GB-cgroup OOM on 2026-07-16).
        times, got = embed_stream(
            iter_frames(item, dump_fps=args.dump_fps, max_frames=args.max_frames),
            batch_size=args.batch_size)
        if len(times) == 0:
            # loud: a silently-absent item is what leaves a bin at 411/412 and
            # aborts a fail-loud batched run later.
            failed += 1
            print(f"!! NO FRAMES for {item['id']} bin={b} — item will be MISSING from the cache")
            continue
        np.savez_compressed(out, times=times, **got)
        n += 1
        print(f"[{n}] {item['id']} bin={b} nf={len(times)} keys={sorted(got)} -> {out}")

    print(f"\nwrote {n} items to {args.out_dir}/ (linked {linked}, skipped {skipped} existing, {failed} FAILED)")
    if failed:
        print(f"!! {failed} items have no frames — bins are INCOMPLETE. Fix before "
              f"replaying pick-math or injecting into lmms-eval (fail-loud on missing qid).")


if __name__ == "__main__":
    main()
