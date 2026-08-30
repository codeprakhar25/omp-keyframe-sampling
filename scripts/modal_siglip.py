"""Modal app for the SigLIP scorer ablation (paper item 1).

Runs in stages so a failure never repeats the expensive part, and so the cheap
gate always runs before anything billable:

    modal run modal_siglip.py::gate     # CPU ~10 min, ~$0.05   COVERAGE GATE
    modal run modal_siglip.py::stage    # CPU ~20 min, ~$1      videos + model -> Volume
    modal run modal_siglip.py::picks    # CPU ~5 min,  ~$0      text embeds + picks
    modal run --detach modal_siglip.py::run_all   # GPU ~8-10 h, ~$18-22

`gate` is a hard gate, not a formality. The cached embeds on Azure are a MIXED
fill: some items carry siglip+dinov2, others only longclip. If the 600s bin is not
fully covered by the siglip tower, the missing items must be re-encoded on a GPU,
which changes the cost of this experiment. Measure before spending.

Nothing here re-runs a banked arm. LongCLIP arms are re-run alongside SigLIP on
purpose: comparing a SigLIP arm measured today against the banked LongCLIP numbers
from the primary stack would repeat exactly the cross-environment mistake that
produced the item-3 confound (slm-lab/CORRECT_FINDINGS.md, 2026-08-21).
"""

import json
import os

import modal

APP_NAME = "siglip-scorer"
VOL_NAME = "slm-lab-sig"
ROOT = "/vol"

AZ_EMBEDS = "slm-lab/results/embeds"
EMBEDS_DIR = f"{ROOT}/embeds"

# unique qids per LongVideoBench duration bin -- the coverage denominator
EXPECT = {"15s": 189, "60s": 172, "600s": 412, "3600s": 564}
TARGET_BIN = "600s"

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
az_secret = modal.Secret.from_name("azure-backup")

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("azure-storage-blob==12.22.0", "numpy==1.26.4")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "lvb_id2bin.json"),
        "/root/lvb_id2bin.json",
    )
)

app = modal.App(APP_NAME)


def _client():
    from azure.storage.blob import BlobServiceClient

    sa = os.environ["AZURE_SA"]
    svc = BlobServiceClient(
        f"https://{sa}.blob.core.windows.net", credential=os.environ["AZURE_STORAGE_KEY"]
    )
    return svc.get_container_client(os.environ["AZURE_CONTAINER"])


@app.function(
    image=cpu_image, volumes={ROOT: vol}, secrets=[az_secret],
    cpu=4.0, memory=8192, timeout=60 * 45,
)
def gate(force: bool = False):
    """Pull the embed cache and report siglip coverage per duration bin.

    Cheap and idempotent: files already on the Volume are skipped, so a re-run
    after a timeout resumes instead of re-downloading.
    """
    import concurrent.futures as cf

    import numpy as np

    os.makedirs(EMBEDS_DIR, exist_ok=True)
    cc = _client()

    names = [b.name for b in cc.list_blobs(name_starts_with=f"{AZ_EMBEDS}/")
             if b.name.endswith(".npz")]
    print(f"blobs on Azure: {len(names)}", flush=True)

    def fetch(name):
        dest = os.path.join(EMBEDS_DIR, os.path.basename(name))
        if not force and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return 0
        with open(dest, "wb") as fh:
            fh.write(cc.download_blob(name).readall())
        return 1

    got = 0
    with cf.ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(fetch, names), 1):
            got += r
            if i % 300 == 0:
                print(f"  {i}/{len(names)}", flush=True)
    print(f"downloaded {got} new, {len(names) - got} already present", flush=True)
    vol.commit()

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    per_bin = {b: {"total": 0, "siglip": 0, "longclip": 0, "both": 0} for b in EXPECT}
    corrupt = []

    for qid, b in id2bin.items():
        per_bin[b]["total"] += 1
        path = os.path.join(EMBEDS_DIR, qid + ".npz")
        if not os.path.exists(path):
            continue
        try:
            keys = set(np.load(path).files)
        except Exception:
            corrupt.append(qid)
            continue
        if "siglip" in keys:
            per_bin[b]["siglip"] += 1
        if "longclip" in keys:
            per_bin[b]["longclip"] += 1
        if {"siglip", "longclip"} <= keys:
            per_bin[b]["both"] += 1

    print("\nbin      total  siglip  longclip   both   siglip%", flush=True)
    for b in ("15s", "60s", "600s", "3600s"):
        s = per_bin[b]
        assert s["total"] == EXPECT[b], f"{b}: manifest has {s['total']}, expected {EXPECT[b]}"
        pct = s["siglip"] / s["total"] * 100 if s["total"] else 0
        print(f"{b:>6}  {s['total']:6d}  {s['siglip']:6d}  {s['longclip']:8d}  "
              f"{s['both']:5d}  {pct:6.1f}%", flush=True)
    if corrupt:
        print(f"\nCORRUPT: {len(corrupt)} -> {corrupt[:5]}", flush=True)

    t = per_bin[TARGET_BIN]
    ok = t["siglip"] == t["total"]
    print(f"\nGATE bin={TARGET_BIN}: {'PASS' if ok else 'FAIL'} "
          f"({t['siglip']}/{t['total']} siglip, {t['both']}/{t['total']} both towers)",
          flush=True)
    if not ok:
        print(f"  -> {t['total'] - t['siglip']} items need SigLIP re-encoding on a GPU. "
              f"Budget that before running stage/run_all.", flush=True)
    return {"per_bin": per_bin, "gate_pass": ok, "corrupt": len(corrupt)}


# ---------------------------------------------------------------- picks (CPU)

TARGET_BINS = ("600s", "60s")
K = 8
PICKS_DIR = f"{ROOT}/picks"
SIGLIP_ID = "google/siglip-so400m-patch14-384"

# Banked LongCLIP picks are reused as-is. Pick generation is deterministic CPU math
# over cached embeddings, so it carries no environment dependence -- only the
# answerer eval does, and that is why every arm is re-run here together.
BANKED_LC = {
    "topk": "picks_topk_lc_lvb_all_k8.json",
    "omp": "picks_omp_lc_lvb_all_k8.json",
    "aks": "picks_aks_lc_lvb_all_k8.json",
}

pick_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "transformers==4.51.3", "numpy==1.26.4", "sentencepiece==0.2.0",
        "protobuf==5.28.3", "azure-storage-blob==12.22.0", "pillow==11.0.0",
    )
    .add_local_dir(os.path.join(os.path.dirname(__file__), "harness"), "/root/harness")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "manifest.lvb.allbins.json"),
        "/root/manifest.json",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "lvb_id2bin.json"),
        "/root/lvb_id2bin.json",
    )
    .add_local_dir(os.path.join(os.path.dirname(__file__), "data"), "/root/banked",
                   ignore=["manifest.lvb.allbins.json", "lvb_id2bin.json"])
)


@app.function(
    image=pick_image, volumes={ROOT: vol}, cpu=8.0, memory=16384, timeout=60 * 40,
)
def picks():
    """SigLIP stem-only text embeds -> top-k / OMP / AKS picks for the target bins.

    The query is the question STEM. Answer options must never reach the scorer:
    the fused stem+options string moved ~40% of selected frames on LVB-600s and can
    reverse the ordering between selectors. harness.text.question_stem is the single
    source of truth for that boundary, so it is used rather than re-split here.
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.replay_selectors import aks_indices, omp_indices
    from harness.text import question_stem

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    manifest = {r["id"]: r for r in json.load(open("/root/manifest.json"))}
    ids = [q for q, b in id2bin.items() if b in TARGET_BINS]
    ids.sort()
    print(f"target bins {TARGET_BINS}: {len(ids)} items", flush=True)

    # ---- SigLIP text tower, stem only ------------------------------------
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(SIGLIP_ID).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    stems = [question_stem(manifest[q]) for q in ids]
    print(f"sample stem: {stems[0][:110]!r}", flush=True)

    qvecs = {}
    B = 32
    with torch.no_grad():
        for i in range(0, len(stems), B):
            chunk = stems[i:i + B]
            # SigLIP requires fixed-length padding; the default dynamic padding
            # silently changes the text embedding.
            tok = proc(text=chunk, padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt")
            feats = model.get_text_features(**tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for j, q in enumerate(ids[i:i + B]):
                qvecs[q] = feats[j].float().numpy()
            if i % (B * 8) == 0:
                print(f"  text {i}/{len(stems)}", flush=True)
    print(f"encoded {len(qvecs)} stems", flush=True)

    # ---- picks per selector ----------------------------------------------
    out = {"topk": {}, "omp": {}, "aks": {}}
    missing, flat = [], 0
    for n, qid in enumerate(ids, 1):
        p = os.path.join(EMBEDS_DIR, qid + ".npz")
        if not os.path.exists(p):
            missing.append(qid)
            continue
        z = np.load(p)
        if "siglip" not in z.files:
            missing.append(qid)
            continue
        times = np.asarray(z["times"], dtype=np.float32)
        E = np.asarray(z["siglip"], dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        q = qvecs[qid]
        scores = E @ q

        k = min(K, len(times))
        tk = sorted(int(i) for i in np.argsort(-scores)[:k])
        out["topk"][qid] = [round(float(times[i]), 4) for i in tk]
        out["omp"][qid] = [round(float(times[i]), 4) for i in omp_indices(q, E, k)]
        out["aks"][qid] = [round(float(times[i]), 4)
                           for i in aks_indices(scores.tolist(), times.tolist(), k)]
        if float(scores.max() - scores.min()) < 1e-9:
            flat += 1
        if n % 100 == 0:
            print(f"  picks {n}/{len(ids)}", flush=True)

    # Coverage is asserted over the ids the run will CONSUME, not over file
    # existence: one absent id has previously killed a multi-hour GPU run at 60%.
    assert not missing, f"{len(missing)} ids lack siglip embeds: {missing[:5]}"
    for sel in out:
        assert len(out[sel]) == len(ids), f"{sel}: {len(out[sel])} != {len(ids)}"
        assert all(len(v) == K for v in out[sel].values()), f"{sel}: non-k picks"
    print(f"flat-score items: {flat}", flush=True)

    os.makedirs(PICKS_DIR, exist_ok=True)
    for sel, d in out.items():
        fp = f"{PICKS_DIR}/picks_{sel}_sig_k8.json"
        json.dump(d, open(fp, "w"))
        print(f"  wrote {fp}: {len(d)} ids", flush=True)

    # ---- banked LongCLIP picks, filtered to the same ids ------------------
    keep = set(ids)
    for sel, fn in BANKED_LC.items():
        src = json.load(open(f"/root/banked/{fn}"))
        sub = {q: v for q, v in src.items() if q in keep}
        assert len(sub) == len(ids), f"banked {sel}: {len(sub)} != {len(ids)}"
        fp = f"{PICKS_DIR}/picks_{sel}_lc_k8.json"
        json.dump(sub, open(fp, "w"))
        print(f"  wrote {fp}: {len(sub)} ids (banked LongCLIP)", flush=True)

    # ---- how different are the two scorers? (free diagnostic) ------------
    print("\npick overlap, SigLIP vs LongCLIP (mean frames shared out of 8):", flush=True)
    ov = {}
    for sel in ("topk", "omp", "aks"):
        lc = json.load(open(f"{PICKS_DIR}/picks_{sel}_lc_k8.json"))
        sg = out[sel]
        per = [len(set(lc[q]) & set(sg[q])) for q in ids]
        ov[sel] = float(np.mean(per))
        print(f"   {sel:5s} {ov[sel]:.2f} / {K}   ({ov[sel]/K:.0%} agreement)", flush=True)

    vol.commit()
    return {"n_items": len(ids), "bins": list(TARGET_BINS), "overlap_vs_longclip": ov}


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=4.0, memory=8192, timeout=60 * 20)
def overlap_diag():
    """Per-bin SigLIP-vs-LongCLIP pick agreement against a random-selection baseline.

    Raw overlap is not interpretable on its own: with k=8 drawn from ~600 candidates
    the chance overlap is ~0.11 frames, but from ~60 candidates it is ~1.07. A pooled
    number mixes those two regimes and can make near-chance agreement look meaningful.
    """
    import numpy as np

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    picks = {}
    for sel in ("topk", "omp", "aks"):
        for sc in ("sig", "lc"):
            picks[(sel, sc)] = json.load(open(f"{PICKS_DIR}/picks_{sel}_{sc}_k8.json"))

    # candidate-pool size per item, from the cached embeds
    pool = {}
    for qid in picks[("omp", "sig")]:
        pool[qid] = int(np.load(os.path.join(EMBEDS_DIR, qid + ".npz"))["times"].shape[0])

    print(f"{'bin':>6} {'sel':>5} {'n':>5} {'obs':>6} {'rand':>6} {'ratio':>6} "
          f"{'pool_med':>9}", flush=True)
    out = {}
    for b in ("60s", "600s"):
        ids = [q for q in picks[("omp", "sig")] if id2bin[q] == b]
        med_pool = float(np.median([pool[q] for q in ids]))
        for sel in ("topk", "omp", "aks"):
            sg, lc = picks[(sel, "sig")], picks[(sel, "lc")]
            obs = float(np.mean([len(set(sg[q]) & set(lc[q])) for q in ids]))
            # E|A n B| for two independent uniform k-subsets of an n-pool = k*k/n
            rand = float(np.mean([min(K, pool[q]) ** 2 / max(pool[q], 1) for q in ids]))
            ratio = obs / rand if rand > 0 else float("nan")
            out[f"{b}_{sel}"] = {"n": len(ids), "obs": round(obs, 3),
                                 "rand": round(rand, 3), "ratio": round(ratio, 2)}
            print(f"{b:>6} {sel:>5} {len(ids):>5} {obs:>6.2f} {rand:>6.2f} "
                  f"{ratio:>6.2f}x {med_pool:>9.0f}", flush=True)

    # Do the two scorers at least agree with THEMSELVES across selectors?
    # If SigLIP-topk and SigLIP-OMP also barely overlap, the low cross-scorer number
    # is about pick diversity generally, not about the scorer swap.
    print("\nwithin-scorer overlap (topk vs omp, same scorer):", flush=True)
    for sc in ("sig", "lc"):
        for b in ("60s", "600s"):
            ids = [q for q in picks[("omp", sc)] if id2bin[q] == b]
            v = float(np.mean([len(set(picks[("topk", sc)][q]) & set(picks[("omp", sc)][q]))
                               for q in ids]))
            print(f"   {sc:>3} {b:>6}: {v:.2f} / {K}", flush=True)
            out[f"within_{sc}_{b}"] = round(v, 3)
    return out


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=4.0, memory=8192, timeout=60 * 20)
def grid_check():
    """Are the banked LongCLIP pick timestamps on the same grid as the embeds cache?

    Dump the literal values before trusting any overlap number. Within-scorer
    top-k/OMP overlap below 1.0 is impossible on shared scores (OMP's first pick IS
    top-k's first pick), so a sub-1.0 reading means the two pick sets are not
    expressed on a common time basis.
    """
    import numpy as np

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    lc = json.load(open(f"{PICKS_DIR}/picks_topk_lc_k8.json"))
    sg = json.load(open(f"{PICKS_DIR}/picks_topk_sig_k8.json"))

    for b in ("60s", "600s"):
        ids = sorted(q for q in lc if id2bin[q] == b)
        q = ids[0]
        t = np.load(os.path.join(EMBEDS_DIR, q + ".npz"))["times"]
        print(f"\n=== {b}  qid={q}  pool={len(t)} ===", flush=True)
        print(f"  embeds times[:8] : {[round(float(x),3) for x in t[:8]]}", flush=True)
        print(f"  embeds times step: {float(t[1]-t[0]):.4f}", flush=True)
        print(f"  banked LC picks  : {lc[q]}", flush=True)
        print(f"  my SigLIP picks  : {sg[q]}", flush=True)

        # what fraction of banked LC timestamps exist in the embeds grid at all?
        hit = 0
        tot = 0
        for qid in ids:
            tt = set(np.round(np.load(os.path.join(EMBEDS_DIR, qid + ".npz"))["times"]
                              .astype(np.float64), 4).tolist())
            for v in lc[qid]:
                tot += 1
                hit += (round(float(v), 4) in tt)
        print(f"  banked LC timestamps present in embeds grid: {hit}/{tot} "
              f"({hit/tot:.1%})", flush=True)

        hit2 = tot2 = 0
        for qid in ids:
            tt = set(np.round(np.load(os.path.join(EMBEDS_DIR, qid + ".npz"))["times"]
                              .astype(np.float64), 4).tolist())
            for v in sg[qid]:
                tot2 += 1
                hit2 += (round(float(v), 4) in tt)
        print(f"  my SigLIP timestamps present in embeds grid: {hit2}/{tot2} "
              f"({hit2/tot2:.1%})", flush=True)
    return "done"


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=4.0, memory=8192, timeout=60 * 20)
def banked_lc_audit():
    """Do the banked LongCLIP top-k and OMP picks come from the same query vector?

    Invariant: OMP's first pick is argmax(q.e), which is exactly top-k's highest-scoring
    frame, so |topk & omp| >= 1 for every item scored from one query. My SigLIP picks are
    the positive control -- they are generated from a single query in one pass here.
    """
    import collections

    import numpy as np

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    P = {(s, c): json.load(open(f"{PICKS_DIR}/picks_{s}_{c}_k8.json"))
         for s in ("topk", "omp") for c in ("lc", "sig")}

    for sc in ("sig", "lc"):
        print(f"\n===== scorer={sc} ({'CONTROL' if sc=='sig' else 'BANKED'}) =====", flush=True)
        for b in ("60s", "600s"):
            ids = sorted(q for q in P[("topk", sc)] if id2bin[q] == b)
            share = [len(set(P[("topk", sc)][q]) & set(P[("omp", sc)][q])) for q in ids]
            zero = sum(1 for v in share if v == 0)
            print(f"  {b}: n={len(ids)}  mean|topk&omp|={np.mean(share):.2f}  "
                  f"items sharing ZERO frames={zero} ({zero/len(ids):.0%})", flush=True)
            print(f"     distribution: "
                  f"{dict(sorted(collections.Counter(share).items()))}", flush=True)

    # concrete side-by-side on one failing item
    ids = sorted(q for q in P[("topk", "lc")] if id2bin[q] == "600s")
    bad = [q for q in ids if not (set(P[("topk", "lc")][q]) & set(P[("omp", "lc")][q]))]
    if bad:
        q = bad[0]
        print(f"\nexample banked-LC item with ZERO shared frames: {q}", flush=True)
        print(f"   topk_lc: {P[('topk','lc')][q]}", flush=True)
        print(f"   omp_lc : {P[('omp','lc')][q]}", flush=True)
        print(f"   (a shared first pick is mandatory if both used one query)", flush=True)
    return "done"


def _to_idx(times, secs):
    """Map pick timestamps to frame indices in the embeds grid (nearest match).

    Pick files on disk carry different float precisions -- some rounded to 1 decimal
    when written, some full float64. Comparing timestamps with `set()` therefore
    silently reports non-overlap for identical frames. Always compare indices.
    """
    import numpy as np

    t = np.asarray(times, dtype=np.float64)
    out = []
    for s in secs:
        j = int(np.argmin(np.abs(t - float(s))))
        out.append(j)
    return set(out)


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=4.0, memory=8192, timeout=60 * 25)
def overlap_v2():
    """Index-based SigLIP-vs-LongCLIP pick agreement, with a random baseline.

    Supersedes overlap_diag, whose timestamp set-intersection was corrupted by
    inconsistent float rounding across pick files.
    """
    import numpy as np

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    P = {(s, c): json.load(open(f"{PICKS_DIR}/picks_{s}_{c}_k8.json"))
         for s in ("topk", "omp", "aks") for c in ("lc", "sig")}
    ids_all = sorted(P[("omp", "sig")])

    times, idx = {}, {}
    for q in ids_all:
        times[q] = np.load(os.path.join(EMBEDS_DIR, q + ".npz"))["times"].astype(np.float64)
        for key in P:
            idx[(key, q)] = _to_idx(times[q], P[key][q])

    # sanity invariant: OMP pick 1 == top-k argmax, so |topk & omp| >= 1 per item
    print("INVARIANT |topk & omp| >= 1 (same scorer):", flush=True)
    ok = True
    for sc in ("sig", "lc"):
        bad = [q for q in ids_all
               if not (idx[(("topk", sc), q)] & idx[(("omp", sc), q)])]
        print(f"   {sc}: {len(bad)}/{len(ids_all)} items violate "
              f"{'  <-- FAIL' if bad else '  OK'}", flush=True)
        ok &= not bad

    print(f"\n{'bin':>6} {'sel':>5} {'n':>5} {'obs':>6} {'rand':>6} {'ratio':>7}", flush=True)
    out = {"invariant_ok": bool(ok)}
    for b in ("60s", "600s"):
        ids = [q for q in ids_all if id2bin[q] == b]
        for sel in ("topk", "omp", "aks"):
            obs = float(np.mean([len(idx[((sel, "sig"), q)] & idx[((sel, "lc"), q)])
                                 for q in ids]))
            rand = float(np.mean([min(K, len(times[q])) ** 2 / max(len(times[q]), 1)
                                  for q in ids]))
            out[f"{b}_{sel}"] = {"n": len(ids), "obs": round(obs, 3),
                                 "rand": round(rand, 3),
                                 "ratio": round(obs / rand, 2) if rand else None}
            print(f"{b:>6} {sel:>5} {len(ids):>5} {obs:>6.2f} {rand:>6.2f} "
                  f"{obs/rand:>6.2f}x", flush=True)

    print("\nwithin-scorer (topk vs omp, index-based):", flush=True)
    for sc in ("sig", "lc"):
        for b in ("60s", "600s"):
            ids = [q for q in ids_all if id2bin[q] == b]
            v = float(np.mean([len(idx[(("topk", sc), q)] & idx[(("omp", sc), q)])
                               for q in ids]))
            print(f"   {sc:>3} {b:>6}: {v:.2f} / {K}", flush=True)
            out[f"within_{sc}_{b}"] = round(v, 3)
    return out


# ============================================================ GPU: stage + eval

import shutil          # noqa: E402
import subprocess      # noqa: E402
import sys             # noqa: E402

AZ_MODEL_PREFIX = "hf/hub/models--Qwen--Qwen3-VL-8B-Instruct"
AZ_DS_PREFIX = "hf/hub/datasets--longvideobench--LongVideoBench"
AZ_VIDEOS = "slm-lab/data/videos"
AZ_LMMS = "lmms-eval"

HF_HOME = f"{ROOT}/hf"
MODEL_DIR = f"{HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct"
LVB_DIR = f"{HF_HOME}/datasets/longvideobench"
VIDEO_DIR = f"{LVB_DIR}/videos"
LMMS_DIR = f"{ROOT}/lmms-eval"
RESULTS_DIR = f"{ROOT}/results"

BIN_N = {"600s": 412, "60s": 172}

# Five arms. AKS is deliberately held back until we know whether the ranking moves.
# uniform is scorer-independent, so it appears once rather than per scorer.
ARMS = [
    ("uniform", None),
    ("topk", "lc"), ("omp", "lc"),
    ("topk", "sig"), ("omp", "sig"),
]

hf_secret = modal.Secret.from_name("hf-token")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "transformers==4.57.1",
        "accelerate==1.10.1", "qwen-vl-utils==0.0.14", "decord==0.6.0", "av",
        "pillow", "numpy", "azure-storage-blob==12.23.1", "datasets==4.0.0",
        "evaluate", "sacrebleu", "pytablewriter", "sqlitedict", "loguru",
        "hf_transfer", "openai", "pyyaml", "tenacity", "tqdm-multiprocess",
        "zstandard", "protobuf", "sentencepiece", "httpx",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "manifest.lvb.allbins.json"),
        "/root/manifest.json",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "lvb_id2bin.json"),
        "/root/lvb_id2bin.json",
    )
)

stage_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("azure-storage-blob==12.23.1", "tqdm", "pillow", "numpy")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "manifest.lvb.allbins.json"),
        "/root/manifest.json",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "lvb_id2bin.json"),
        "/root/lvb_id2bin.json",
    )
)


def _dl(cc, name, dest, skip_if_size=None):
    if skip_if_size is not None and os.path.exists(dest) and os.path.getsize(dest) == skip_if_size:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        cc.download_blob(name, max_concurrency=4).readinto(f)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def _dl_many(pairs, workers=16):
    from concurrent.futures import ThreadPoolExecutor

    cc = _client()
    done = {"n": 0, "b": 0}

    def one(t):
        n = _dl(cc, t[0], t[1], t[2])
        done["n"] += 1
        done["b"] += n
        if done["n"] % 50 == 0:
            print(f"    {done['n']}/{len(pairs)} files, {done['b']/2**30:.1f} GiB", flush=True)
        return n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, pairs))
    return done


def _materialize_snapshot(local_repo_dir, tree_json_path, rev):
    """Rebuild snapshots/<rev>/<file> as hardlinks into blobs/ from the saved tree map.

    Blob-storage upload drops HF's symlinks, so the snapshot dir has to be rebuilt
    deterministically from trees/<rev>.json rather than guessed. Hardlink so the
    16 GiB of weights is not duplicated on the Volume.
    """
    tree = json.load(open(tree_json_path))
    snap = os.path.join(local_repo_dir, "snapshots", rev)
    os.makedirs(snap, exist_ok=True)
    made, missing = 0, []
    for fname, meta in tree["files"].items():
        src = os.path.join(local_repo_dir, "blobs",
                           meta.get("lfs_sha256") or meta["blob_id"])
        dst = os.path.join(snap, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        made += 1
    return snap, made, missing


@app.function(image=stage_image, volumes={ROOT: vol}, secrets=[az_secret],
              cpu=8.0, memory=16384, timeout=60 * 90)
def stage():
    """Azure -> Volume. Idempotent: re-running skips anything already present at size."""
    cc = _client()

    def listing(prefix):
        return [(b.name, b.size) for b in cc.list_blobs(name_starts_with=prefix)]

    print("[1/4] Qwen3-VL-8B-Instruct", flush=True)
    pairs = [(n, os.path.join(HF_HOME, "hub", n.split("hf/hub/", 1)[1]), s)
             for n, s in listing(AZ_MODEL_PREFIX) if "/.no_exist/" not in n]
    d = _dl_many(pairs)
    print(f"    {d['n']} files, {d['b']/2**30:.2f} GiB", flush=True)
    rev = open(os.path.join(MODEL_DIR, "refs", "main")).read().strip()
    snap, made, missing = _materialize_snapshot(
        MODEL_DIR, os.path.join(MODEL_DIR, "trees", f"{rev}.json"), rev)
    assert not missing, f"model snapshot incomplete: {missing}"
    assert os.path.exists(os.path.join(snap, "config.json"))
    print(f"    snapshot {rev[:12]} -> {made} files", flush=True)

    print("[2/4] LongVideoBench dataset cache", flush=True)
    ds = [(n, os.path.join(HF_HOME, "hub", n.split("hf/hub/", 1)[1]), s)
          for n, s in listing(AZ_DS_PREFIX) if "/.no_exist/" not in n]
    _dl_many(ds, workers=8)
    print(f"    {len(ds)} files", flush=True)

    print("[3/4] videos for 600s + 60s", flush=True)
    id2bin = json.load(open("/root/lvb_id2bin.json"))
    man = json.load(open("/root/manifest.json"))
    need = sorted({r["video_file"] for r in man if id2bin.get(r["id"]) in BIN_N})
    sizes = {os.path.basename(n): s for n, s in listing(AZ_VIDEOS) if n.endswith(".mp4")}
    absent = [n for n in need if n not in sizes]
    assert not absent, f"{len(absent)} needed videos absent on Azure: {absent[:5]}"
    d = _dl_many([(f"{AZ_VIDEOS}/{n}", f"{VIDEO_DIR}/{n}", sizes[n]) for n in need],
                 workers=24)
    print(f"    {len(need)} needed, {d['n']} fetched, {d['b']/2**30:.2f} GiB", flush=True)

    print("[4/4] lmms-eval tree", flush=True)
    # Never stage backup/scratch dirs: lmms-eval scans task dirs recursively and will try
    # to load any *.yaml it finds. A `_bak_*` dir of copied yamls without utils.py took the
    # whole task registry down on 2026-07-28.
    lp = [(n, os.path.join(ROOT, n), s) for n, s in listing(AZ_LMMS + "/")
          if not n.endswith("/") and "_bak" not in n and "/backups/" not in n]
    d = _dl_many(lp, workers=24)
    print(f"    {d['n']} files, {d['b']/2**20:.0f} MiB", flush=True)

    pu = f"{LMMS_DIR}/lmms_eval/tasks/longvideobench/picks_utils.py"
    src = open(pu).read()
    assert "_video_path(" not in src, "STALE picks_utils: undefined _video_path helper"
    assert "total_valid_frames" in src, "picks_utils missing the stride sampling rule"
    vol.commit()
    print("stage OK", flush=True)


TASK = {("uniform", "600s"): "longvideobench_val_i_600s_k8",
        ("uniform", "60s"): "longvideobench_val_i_60s_k8",
        ("picks", "600s"): "longvideobench_val_picks_600s",
        ("picks", "60s"): "longvideobench_val_picks_60s"}


def _run_lmms(task, out_name, picks_file=None, limit=None):
    """One lmms-eval arm. `picks_file` is injected via $LVB_PICKS (None = uniform arm)."""
    rev = open(os.path.join(MODEL_DIR, "refs", "main")).read().strip()
    pretrained = os.path.join(MODEL_DIR, "snapshots", rev)

    env = dict(os.environ)
    env.update(HF_HOME=HF_HOME, HF_DATASETS_CACHE=f"{HF_HOME}/datasets",
               PYTHONPATH=LMMS_DIR, TOKENIZERS_PARALLELISM="false")
    # NSHARD=1 always. Doc-sharding silently corrupted four 3600s runs on 2026-07-19:
    # the shards ran the SAME subset, producing full line counts with half the unique ids.
    env.pop("LVB_DOC_NSHARD", None)
    env.pop("LVB_DOC_SHARD", None)
    if picks_file:
        env["LVB_PICKS"] = picks_file

    out = f"{RESULTS_DIR}/{out_name}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "lmms_eval",
           "--model", "qwen3_vl",
           "--model_args", f"pretrained={pretrained},device_map=auto",
           "--tasks", task,
           "--batch_size", "1",       # PROTOCOL AXIS: never change, never auto
           "--log_samples",
           "--output_path", out]
    if limit:
        cmd += ["--limit", str(limit)]
    print("RUN: " + " ".join(cmd), flush=True)
    print(f"     LVB_PICKS={picks_file}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=ROOT)
    # lmms-eval catches fatal errors and STILL exits 0 -- rc==0 is not success.
    # Gate on artifacts produced, never on the return code.
    print(f"rc={r.returncode}", flush=True)
    return out


def _coverage(out, expect_n):
    """Unique-qid gate. qid lives at lvb_acc.id -- doc.id silently yields 0."""
    import glob

    js = glob.glob(f"{out}/**/*samples*.jsonl", recursive=True)
    assert js, f"no samples jsonl under {out} -- run produced nothing"
    ids = set()
    for p in js:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            acc = r.get("lvb_acc")
            ids.add(acc["id"] if isinstance(acc, dict) else r.get("doc_id"))
    print(f"COVERAGE: {len(ids)} unique qids (expect {expect_n})", flush=True)
    if len(ids) != expect_n:
        raise RuntimeError(f"coverage gate FAILED: {len(ids)} != {expect_n}")
    return len(ids)


@app.function(image=gpu_image, gpu="L40S", volumes={ROOT: vol}, secrets=[hf_secret],
              timeout=60 * 60)
def smoke():
    """2 docs. Confirms the answerer module path and that the video dir resolves.

    Patching the wrong qwen3_vl module (simple/ when chat/ loads) once produced a fake
    0/0 McNemar tie, so the module path is asserted before any real spend.
    """
    sys.path.insert(0, LMMS_DIR)
    os.environ["HF_HOME"] = HF_HOME

    from lmms_eval.models import get_model

    m = get_model("qwen3_vl")
    print(f"ANSWERER MODULE: {m.__module__}", flush=True)

    import importlib

    pu = importlib.import_module("lmms_eval.tasks.longvideobench.picks_utils")
    cache_dir, _ = pu._resolve_dataset_dir("longvideobench_val_v.yaml", "video_subdir",
                                           "videos/")
    n_here = len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else -1
    print(f"RESOLVED VIDEO DIR: {cache_dir} (files={n_here}, staged={VIDEO_DIR})", flush=True)
    assert n_here > 0, f"resolved video dir {cache_dir} is empty -- HF_HOME mismatch"

    out = _run_lmms(TASK[("picks", "600s")], "smoke_600",
                    picks_file=f"{PICKS_DIR}/picks_omp_sig_k8.json", limit=2)
    vol.commit()
    import glob
    js = glob.glob(f"{out}/**/*samples*.jsonl", recursive=True)
    n = sum(1 for f in js for _ in open(f))
    print(f"SMOKE produced {len(js)} sample file(s), {n} lines", flush=True)
    assert n >= 2, "smoke produced no predictions"
    print("SMOKE OK", flush=True)


@app.function(image=gpu_image, gpu="L40S", volumes={ROOT: vol}, secrets=[hf_secret],
              timeout=60 * 60 * 6)
def run_arm(spec: str):
    """spec = '<bin>:<selector>:<scorer>', scorer '-' for the uniform arm."""
    b, sel, sc = spec.split(":")
    assert b in BIN_N, b
    tag = f"{sel}_{sc}_{b}" if sc != "-" else f"uniform_{b}"
    if sel == "uniform":
        out = _run_lmms(TASK[("uniform", b)], tag)
    else:
        pf = f"{PICKS_DIR}/picks_{sel}_{sc}_k8.json"
        assert os.path.exists(pf), f"missing picks file {pf}"
        out = _run_lmms(TASK[("picks", b)], tag, picks_file=pf)
    vol.commit()
    n = _coverage(out, BIN_N[b])
    return {"arm": tag, "out": out, "n": n}


@app.local_entrypoint()
def run_all():
    """All five arms x both bins, one container each.

    Arm-level parallelism, NOT doc-sharding: every container runs one complete
    (arm, bin) cell with NSHARD=1 and gates on its own unique-qid count. Containers
    write disjoint result dirs and only read the shared staged data.
    """
    specs = [f"{b}:{sel}:{sc or '-'}" for b in ("600s", "60s") for sel, sc in ARMS]
    print(f"launching {len(specs)} arms:\n  " + "\n  ".join(specs), flush=True)
    for r in run_arm.map(specs, order_outputs=True):
        print(f"DONE {r['arm']}: n={r['n']} -> {r['out']}", flush=True)


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=2.0, timeout=60 * 15)
def audit_results():
    """Unique-qid coverage per arm dir. Presence of a dir is not evidence of a run.

    A killed container leaves a populated results dir whose samples file is truncated,
    which is indistinguishable from success by listing alone.
    """
    import glob

    rows = []
    for b in ("600s", "60s"):
        for sel, sc in ARMS:
            tag = f"{sel}_{sc}_{b}" if sc else f"uniform_{b}"
            out = f"{RESULTS_DIR}/{tag}"
            js = glob.glob(f"{out}/**/*samples*.jsonl", recursive=True)
            ids, lines = set(), 0
            for p in js:
                for line in open(p):
                    if not line.strip():
                        continue
                    lines += 1
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # truncated final line of a killed run
                    acc = r.get("lvb_acc")
                    ids.add(acc["id"] if isinstance(acc, dict) else r.get("doc_id"))
            exp = BIN_N[b]
            rows.append((tag, len(ids), exp, lines, len(ids) == exp))
    print(f"{'arm':>18} {'uniq':>6} {'exp':>5} {'lines':>6}  status", flush=True)
    incomplete = []
    for tag, n, exp, lines, ok in rows:
        print(f"{tag:>18} {n:>6} {exp:>5} {lines:>6}  {'OK' if ok else 'INCOMPLETE'}",
              flush=True)
        if not ok:
            incomplete.append(tag)
    print(f"\ncomplete: {len(rows)-len(incomplete)}/{len(rows)}", flush=True)
    if incomplete:
        print(f"INCOMPLETE -> {incomplete}", flush=True)
    return {"incomplete": incomplete, "rows": [list(r) for r in rows]}


@app.local_entrypoint()
def run_missing():
    """Re-run only the arms whose unique-qid coverage is short, via spawn().

    run_all() drove the fan-out with run_arm.map() from a @local_entrypoint, i.e. on the
    CLIENT. `--detach` keeps the App alive but not that driver, so a client network drop
    on 2026-08-21 killed six in-flight arms. spawn() hands each arm to the server as an
    independent FunctionCall, which survives client disconnect.
    """
    todo = audit_results.remote()["incomplete"]
    if not todo:
        print("nothing incomplete -- all arms already at full coverage", flush=True)
        return
    specs = []
    for tag in todo:
        if tag.startswith("uniform_"):
            specs.append(f"{tag.split('_')[-1]}:uniform:-")
        else:
            sel, sc, b = tag.split("_")
            specs.append(f"{b}:{sel}:{sc}")
    print(f"spawning {len(specs)} arms server-side:\n  " + "\n  ".join(specs), flush=True)
    for s in specs:
        run_arm.spawn(s)
    print("spawned; these now run independently of this client", flush=True)


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=2.0, timeout=60 * 20)
def analyze():
    """Accuracy per arm, paired McNemars, and the scorer x selector interaction.

    The question is NOT whether SigLIP beats LongCLIP on absolute accuracy. It is
    whether the SELECTOR RANKING survives the scorer swap: if uniform < top-k < OMP
    under one scorer but the order changes under the other, then published selector
    tables are not comparable across papers that use different scorers.

    Every arm scores the same items in the same environment, so all tests are paired.
    """
    import glob
    from math import comb

    import numpy as np

    def load(tag):
        out = f"{RESULTS_DIR}/{tag}"
        d = {}
        for p in glob.glob(f"{out}/**/*samples*.jsonl", recursive=True):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("lvb_acc")
                if isinstance(a, dict):
                    d[a["id"]] = int(a["answer"] == a["parsed_pred"])
        return d

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

    def paired(A, B, ids):
        """B minus A, in percentage points, with an exact McNemar."""
        b = sum(1 for q in ids if B[q] > A[q])
        c = sum(1 for q in ids if A[q] > B[q])
        return (np.mean([B[q] - A[q] for q in ids]) * 100, b, c, mcnemar(b, c))

    report = {}
    for bin_ in ("600s", "60s"):
        arms = {}
        for sel, sc in ARMS:
            tag = f"{sel}_{sc}_{bin_}" if sc else f"uniform_{bin_}"
            arms[(sel, sc)] = load(tag)
        ids = sorted(set.intersection(*(set(v) for v in arms.values())))
        assert len(ids) == BIN_N[bin_], f"{bin_}: {len(ids)} common ids != {BIN_N[bin_]}"

        print(f"\n{'='*70}\n{bin_}  (n={len(ids)}, all arms paired on the same items)\n{'='*70}",
              flush=True)
        acc = {k: sum(v[q] for q in ids) / len(ids) for k, v in arms.items()}
        print(f"{'arm':>16} {'acc':>8}", flush=True)
        for k in [("uniform", None), ("topk", "lc"), ("omp", "lc"),
                  ("topk", "sig"), ("omp", "sig")]:
            print(f"{(k[0]+'_'+(k[1] or '-')):>16} {acc[k]:>8.4f}", flush=True)

        U = arms[("uniform", None)]
        print(f"\n  gains over uniform:", flush=True)
        gains = {}
        for sc in ("lc", "sig"):
            for sel in ("topk", "omp"):
                d, b, c, p = paired(U, arms[(sel, sc)], ids)
                gains[(sel, sc)] = d
                print(f"    {sel:>4}_{sc:<4} {d:+7.2f} pt  (b={b:3d} c={c:3d})  p={p:.4g}",
                      flush=True)

        print(f"\n  OMP vs top-k, within scorer:", flush=True)
        for sc in ("lc", "sig"):
            d, b, c, p = paired(arms[("topk", sc)], arms[("omp", sc)], ids)
            print(f"    {sc}: {d:+7.2f} pt  (b={b:3d} c={c:3d})  p={p:.4g}", flush=True)

        print(f"\n  RANKING (best last):", flush=True)
        for sc in ("lc", "sig"):
            order = sorted([("uniform", acc[("uniform", None)]),
                            ("topk", acc[("topk", sc)]),
                            ("omp", acc[("omp", sc)])], key=lambda t: t[1])
            print(f"    {sc}: " + "  <  ".join(f"{n}({a:.4f})" for n, a in order), flush=True)

        print(f"\n  scorer x selector interaction (does the gain depend on scorer?):",
              flush=True)
        inter = {}
        for sel in ("topk", "omp"):
            gl = [arms[(sel, "lc")][q] - U[q] for q in ids]
            gs = [arms[(sel, "sig")][q] - U[q] for q in ids]
            b = sum(1 for x, y in zip(gs, gl) if y < x)
            c = sum(1 for x, y in zip(gs, gl) if x < y)
            d = (np.mean(gs) - np.mean(gl)) * 100
            inter[sel] = (d, b, c, mcnemar(b, c))
            print(f"    {sel:>4}: sig-minus-lc {d:+7.2f} pt  (b={b:3d} c={c:3d})  "
                  f"p={mcnemar(b,c):.4g}", flush=True)

        report[bin_] = {"n": len(ids),
                        "acc": {f"{k[0]}_{k[1] or '-'}": round(v, 4) for k, v in acc.items()},
                        "gains_vs_uniform": {f"{k[0]}_{k[1]}": round(v, 3)
                                             for k, v in gains.items()},
                        "interaction": {k: [round(v[0], 3), v[1], v[2], round(v[3], 6)]
                                        for k, v in inter.items()}}
    return report


@app.function(image=pick_image, volumes={ROOT: vol}, secrets=[az_secret],
              cpu=4.0, memory=8192, timeout=60 * 25)
def topk_provenance():
    """Which top-k pick file is consistent with the banked LongCLIP scores?

    Measured top-k_lc on 600s is .6068 here vs .6141 banked, while uniform and OMP
    reproduce the bank exactly -- so the environment is fine and the disagreement is
    isolated to the top-k picks. Two candidate files exist; recompute top-k from the
    banked per-frame scores and see which one it matches.
    """
    import numpy as np

    cc = _client()
    os.makedirs("/tmp/prov", exist_ok=True)
    srcs = {
        "banked_600": "slm-lab/results/picks_lmmseval/picks_topk_lc_600_k8.json",
        "scores": "slm-lab/results/scores/scores_lc_600.jsonl",
    }
    for k, blob in srcs.items():
        _dl(cc, blob, f"/tmp/prov/{k}")
        print(f"fetched {k}: {os.path.getsize(f'/tmp/prov/{k}')/1e6:.2f} MB", flush=True)

    used = json.load(open(f"{PICKS_DIR}/picks_topk_lc_k8.json"))   # what THIS run used
    banked = json.load(open("/tmp/prov/banked_600"))

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    ids600 = sorted(q for q in used if id2bin[q] == "600s")
    print(f"\n600s ids: used={len(ids600)} banked_file={len(banked)}", flush=True)

    # recompute top-k straight from the banked scores
    recomputed, nscored = {}, 0
    for line in open("/tmp/prov/scores"):
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r.get("id")
        if qid not in set(ids600):
            continue
        sc = np.asarray(r["scores"], dtype=np.float64)
        tt = np.asarray(r["times"], dtype=np.float64)
        k = min(K, len(sc))
        recomputed[qid] = sorted(int(i) for i in np.argsort(-sc)[:k])
        nscored += 1
    print(f"recomputed top-k from banked scores for {nscored} items", flush=True)

    def idxs(picks_map, qid):
        t = np.load(os.path.join(EMBEDS_DIR, qid + ".npz"))["times"].astype(np.float64)
        return _to_idx(t, picks_map[qid])

    # score-derived indices are positions in the SCORES file's own times array; compare
    # by resolving both back to seconds on that array
    times_of = {}
    for line in open("/tmp/prov/scores"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") in recomputed:
            times_of[r["id"]] = np.asarray(r["times"], dtype=np.float64)

    agree_used = agree_banked = both = 0
    for q in ids600:
        if q not in recomputed:
            continue
        both += 1
        want = {round(float(times_of[q][i]), 3) for i in recomputed[q]}
        u = {round(float(x), 3) for x in used[q]}
        bk = {round(float(x), 3) for x in banked.get(q, [])}
        agree_used += (u == want)
        agree_banked += (bk == want)
    print(f"\nitems compared: {both}", flush=True)
    print(f"  file THIS RUN used  exactly matches score-derived top-k: "
          f"{agree_used}/{both} ({agree_used/both:.1%})", flush=True)
    print(f"  banked 600s file    exactly matches score-derived top-k: "
          f"{agree_banked}/{both} ({agree_banked/both:.1%})", flush=True)

    same = sum(1 for q in ids600
               if q in banked and {round(float(x), 3) for x in used[q]}
               == {round(float(x), 3) for x in banked[q]})
    print(f"  the two pick FILES agree with each other: {same}/{len(ids600)} "
          f"({same/len(ids600):.1%})", flush=True)

    diff = [q for q in ids600 if q in banked
            and {round(float(x), 3) for x in used[q]} != {round(float(x), 3) for x in banked[q]}]
    if diff:
        q = diff[0]
        print(f"\nexample disagreement {q}:", flush=True)
        print(f"   used  : {sorted(round(float(x),2) for x in used[q])}", flush=True)
        print(f"   banked: {sorted(round(float(x),2) for x in banked[q])}", flush=True)
        if q in recomputed:
            print(f"   from scores: "
                  f"{sorted(round(float(times_of[q][i]),2) for i in recomputed[q])}", flush=True)
    return {"n": both, "used_matches": agree_used, "banked_matches": agree_banked,
            "files_agree": same}


# ==================================================== ITEM 5: stem vs fused query

AZ_EMBEDS_LC = "slm-lab/results/embeds_lc"
EMBEDS_LC_DIR = f"{ROOT}/embeds_lc"
FUSED_BIN = "600s"          # where the "~40% of frames change" figure was measured

fused_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 extra_options="--index-url https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==4.51.3", "numpy==1.26.4", "sentencepiece==0.2.0",
                 "protobuf==5.28.3", "azure-storage-blob==12.22.0", "pillow==11.0.0",
                 "ftfy", "regex", "huggingface_hub")
    .run_commands("git clone --depth 1 https://github.com/beichenzbc/Long-CLIP "
                  "/root/Long-CLIP")
    .add_local_dir(os.path.join(os.path.dirname(__file__), "harness"), "/root/harness")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "manifest.lvb.allbins.json"),
        "/root/manifest.json")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data", "lvb_id2bin.json"),
        "/root/lvb_id2bin.json")
)


@app.function(image=fused_image, volumes={ROOT: vol}, secrets=[az_secret, hf_secret],
              cpu=8.0, memory=16384, timeout=60 * 60)
def picks_fused():
    """Picks from the FUSED question+options string, under both scorers.

    The paper asserts a prompt boundary: answer options must not reach the scorer,
    because the fused string moved ~40% of selected frames on LVB-600s. That is a
    pick-overlap statistic, not an accuracy consequence. This generates the fused-query
    counterpart of the stem-only picks so the consequence can be measured, paired,
    against arms already run in this same environment.

    LongCLIP 600s image embeds live in embeds_lc/ under key 'emb' (embeds/ carries
    siglip for this bin); harness.embeds documents that split.
    """
    import concurrent.futures as cf
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.replay_selectors import omp_indices
    from harness.text import question_stem

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    man = {r["id"]: r for r in json.load(open("/root/manifest.json"))}
    ids = sorted(q for q, b in id2bin.items() if b == FUSED_BIN)
    print(f"{FUSED_BIN}: {len(ids)} items", flush=True)

    # ---- pull LongCLIP image embeds for this bin -------------------------
    os.makedirs(EMBEDS_LC_DIR, exist_ok=True)
    cc = _client()

    def fetch(qid):
        dest = os.path.join(EMBEDS_LC_DIR, qid + ".npz")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return 0
        try:
            with open(dest, "wb") as fh:
                fh.write(cc.download_blob(f"{AZ_EMBEDS_LC}/{qid}.npz").readall())
            return 1
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            return -1

    with cf.ThreadPoolExecutor(32) as ex:
        got = list(ex.map(fetch, ids))
    missing = [q for q, r in zip(ids, got) if r < 0]
    print(f"embeds_lc: {sum(1 for r in got if r>0)} fetched, "
          f"{sum(1 for r in got if r==0)} cached, {len(missing)} MISSING", flush=True)
    assert not missing, f"embeds_lc missing for {len(missing)} ids: {missing[:5]}"

    stems = [question_stem(man[q]) for q in ids]
    fused = [man[q]["question"] for q in ids]      # answerer prompt: stem + options
    print(f"\nSTEM : {stems[0][:100]!r}", flush=True)
    print(f"FUSED: {fused[0][:160]!r}", flush=True)
    assert any(f != s for f, s in zip(fused, stems)), "fused == stem; wrong manifest field"
    n_opt = sum(1 for f in fused if "Options:" in f)
    print(f"fused strings containing 'Options:': {n_opt}/{len(fused)}", flush=True)
    assert n_opt == len(fused), "fused strings do not all carry the options block"
    print(f"FUSED TAIL: {fused[0][-220:]!r}", flush=True)

    out = {}

    # ---- SigLIP -----------------------------------------------------------
    from transformers import AutoModel, AutoProcessor

    sg = AutoModel.from_pretrained(SIGLIP_ID).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)

    def sig_encode(texts):
        vs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                tok = proc(text=texts[i:i + 32], padding="max_length", truncation=True,
                           max_length=64, return_tensors="pt")
                f = sg.get_text_features(**tok)
                vs.append((f / f.norm(dim=-1, keepdim=True)).float().numpy())
        return np.concatenate(vs, 0)

    qf_sig = sig_encode(fused)
    print(f"siglip fused text: {qf_sig.shape}", flush=True)
    del sg

    # ---- LongCLIP ---------------------------------------------------------
    sys.path.insert(0, "/root/Long-CLIP")
    from huggingface_hub import hf_hub_download
    from model import longclip

    ck = hf_hub_download("BeichenZhang/LongCLIP-L", "longclip-L.pt")
    lc_model, _ = longclip.load(ck, device="cpu")
    lc_model.eval()

    def lc_encode(texts):
        vs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                tok = longclip.tokenize(texts[i:i + 32], truncate=True)
                f = lc_model.encode_text(tok)
                vs.append((f / f.norm(dim=-1, keepdim=True)).float().numpy())
        return np.concatenate(vs, 0)

    qf_lc = lc_encode(fused)
    print(f"longclip fused text: {qf_lc.shape}", flush=True)

    # ---- build picks ------------------------------------------------------
    for sc in ("sig", "lc"):
        out[("topk", sc)] = {}
        out[("omp", sc)] = {}
    ov = {"sig": [], "lc": []}
    stem_picks = {sc: json.load(open(f"{PICKS_DIR}/picks_omp_{sc}_k8.json"))
                  for sc in ("sig", "lc")}

    for n, qid in enumerate(ids):
        for sc, Q in (("sig", qf_sig), ("lc", qf_lc)):
            if sc == "sig":
                z = np.load(os.path.join(EMBEDS_DIR, qid + ".npz"))
                E = np.asarray(z["siglip"], dtype=np.float32)
            else:
                z = np.load(os.path.join(EMBEDS_LC_DIR, qid + ".npz"))
                E = np.asarray(z["emb"], dtype=np.float32)
            t = np.asarray(z["times"], dtype=np.float32)
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            q = Q[n]
            s = E @ q
            k = min(K, len(t))
            tk = sorted(int(i) for i in np.argsort(-s)[:k])
            out[("topk", sc)][qid] = [round(float(t[i]), 4) for i in tk]
            om = omp_indices(q, E, k)
            out[("omp", sc)][qid] = [round(float(t[i]), 4) for i in om]
            ov[sc].append(len(_to_idx(t, stem_picks[sc][qid]) & set(om)))
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    os.makedirs(PICKS_DIR, exist_ok=True)
    for (sel, sc), d in out.items():
        assert len(d) == len(ids), f"{sel}_{sc}: {len(d)} != {len(ids)}"
        fp = f"{PICKS_DIR}/picks_{sel}_{sc}_fused_k8.json"
        json.dump(d, open(fp, "w"))
        print(f"  wrote {fp}: {len(d)} ids", flush=True)

    print("\nOMP pick overlap, fused-query vs stem-only (out of 8):", flush=True)
    res = {}
    for sc in ("sig", "lc"):
        m = float(np.mean(ov[sc]))
        res[sc] = round(m, 3)
        print(f"   {sc}: {m:.2f}/8 shared  ->  {(1-m/K):.0%} of selected frames CHANGE",
              flush=True)
    vol.commit()
    return res


@app.function(image=gpu_image, gpu="L40S", volumes={ROOT: vol}, secrets=[hf_secret],
              timeout=60 * 60 * 4)
def run_arm_fused(spec: str):
    """spec = '<selector>:<scorer>'. Fused-query arm on the 600s bin."""
    sel, sc = spec.split(":")
    pf = f"{PICKS_DIR}/picks_{sel}_{sc}_fused_k8.json"
    assert os.path.exists(pf), f"missing {pf}"
    tag = f"{sel}_{sc}_fused_{FUSED_BIN}"
    out = _run_lmms(TASK[("picks", FUSED_BIN)], tag, picks_file=pf)
    vol.commit()
    return {"arm": tag, "out": out, "n": _coverage(out, BIN_N[FUSED_BIN])}


@app.local_entrypoint()
def run_fused():
    """Four fused-query arms, spawned server-side (see run_missing for why not map())."""
    specs = [f"{sel}:{sc}" for sc in ("lc", "sig") for sel in ("topk", "omp")]
    for s in specs:
        run_arm_fused.spawn(s)
    print("spawned: " + ", ".join(specs), flush=True)


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=2.0, timeout=60 * 20)
def analyze_fused():
    """Accuracy cost of letting answer options reach the scorer.

    Paired against the stem-only arms already run in THIS environment, so the only
    thing differing between the two arms is what text the scorer saw.
    """
    import glob
    from math import comb

    import numpy as np

    def load(tag):
        d = {}
        for p in glob.glob(f"{RESULTS_DIR}/{tag}/**/*samples*.jsonl", recursive=True):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("lvb_acc")
                if isinstance(a, dict):
                    d[a["id"]] = int(a["answer"] == a["parsed_pred"])
        return d

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    print(f"{'scorer':>7} {'selector':>9} {'stem':>8} {'fused':>8} {'delta':>8} "
          f"{'b':>4} {'c':>4} {'p':>9}", flush=True)
    rep = {}
    for sc in ("lc", "sig"):
        for sel in ("topk", "omp"):
            S = load(f"{sel}_{sc}_{FUSED_BIN}")
            F = load(f"{sel}_{sc}_fused_{FUSED_BIN}")
            ids = sorted(set(S) & set(F))
            assert len(ids) == BIN_N[FUSED_BIN], \
                f"{sel}_{sc}: {len(ids)} paired ids != {BIN_N[FUSED_BIN]}"
            accs = sum(S[q] for q in ids) / len(ids)
            accf = sum(F[q] for q in ids) / len(ids)
            b = sum(1 for q in ids if F[q] > S[q])
            c = sum(1 for q in ids if S[q] > F[q])
            p = mcnemar(b, c)
            rep[f"{sel}_{sc}"] = [round(accs, 4), round(accf, 4),
                                  round((accf - accs) * 100, 2), b, c, round(p, 6)]
            print(f"{sc:>7} {sel:>9} {accs:>8.4f} {accf:>8.4f} "
                  f"{(accf-accs)*100:>+8.2f} {b:>4} {c:>4} {p:>9.4g}", flush=True)
    print("\nNegative delta = letting answer options into the scorer HURTS.", flush=True)

    # The paper also claims the fused query "could reverse the ordering between rules".
    # Test it directly: does OMP still beat top-k once options reach the scorer?
    print("\nSELECTOR ORDERING under each query (does contamination flip it?):",
          flush=True)
    for sc in ("lc", "sig"):
        for qkind, suf in (("stem", ""), ("fused", "_fused")):
            T = load(f"topk_{sc}{suf}_{FUSED_BIN}" if suf
                     else f"topk_{sc}_{FUSED_BIN}")
            O = load(f"omp_{sc}{suf}_{FUSED_BIN}" if suf
                     else f"omp_{sc}_{FUSED_BIN}")
            ids = sorted(set(T) & set(O))
            at = sum(T[q] for q in ids) / len(ids)
            ao = sum(O[q] for q in ids) / len(ids)
            b = sum(1 for q in ids if O[q] > T[q])
            c = sum(1 for q in ids if T[q] > O[q])
            win = "OMP" if ao > at else ("top-k" if at > ao else "tie")
            print(f"   {sc:>3} {qkind:>5}: top-k {at:.4f}  OMP {ao:.4f}  "
                  f"-> {win:>5} by {abs(ao-at)*100:.2f} pt  p={mcnemar(b,c):.4g}",
                  flush=True)
            rep[f"order_{sc}_{qkind}"] = [round(at, 4), round(ao, 4), win]
    return rep


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=4.0, memory=8192, timeout=60 * 20)
def fused_overlap_full():
    """Fused-vs-stem pick divergence for BOTH selectors, both scorers, index-based.

    The paper states "roughly 40%" of frames change on LVB-600s when the fused
    question+options string is used. This resolves which selector that figure
    describes -- OMP leans on the query vector harder than top-k, because every pick
    is chosen to explain a residual component of it, so the two need not agree.
    """
    import numpy as np

    ids = sorted(json.load(open(f"{PICKS_DIR}/picks_omp_lc_fused_k8.json")))
    print(f"{'scorer':>7} {'selector':>9} {'shared/8':>10} {'% changed':>11}", flush=True)
    out = {}
    for sc in ("lc", "sig"):
        for sel in ("topk", "omp"):
            stem = json.load(open(f"{PICKS_DIR}/picks_{sel}_{sc}_k8.json"))
            fus = json.load(open(f"{PICKS_DIR}/picks_{sel}_{sc}_fused_k8.json"))
            src = EMBEDS_LC_DIR if sc == "lc" else EMBEDS_DIR
            key = "times"
            sh = []
            for q in ids:
                t = np.load(os.path.join(src, q + ".npz"))[key].astype(np.float64)
                sh.append(len(_to_idx(t, stem[q]) & _to_idx(t, fus[q])))
            m = float(np.mean(sh))
            out[f"{sel}_{sc}"] = round(m, 3)
            print(f"{sc:>7} {sel:>9} {m:>10.2f} {(1-m/K)*100:>10.1f}%", flush=True)
    return out


@app.function(image=pick_image, cpu=2.0, timeout=60 * 15)
def fused_token_audit():
    """How much of the fused string does each scorer actually see?

    SigLIP-so400m's text tower is capped at 64 tokens. If the fused question+options
    string exceeds that, SigLIP's "fused" arm is a WEAKER intervention than LongCLIP's,
    and its near-null result cannot be read as scorer robustness.
    """
    import sys

    import numpy as np
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root")
    from harness.text import question_stem

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    man = {r["id"]: r for r in json.load(open("/root/manifest.json"))}
    ids = sorted(q for q, b in id2bin.items() if b == "600s")
    stems = [question_stem(man[q]) for q in ids]
    fused = [man[q]["question"] for q in ids]

    tok = AutoTokenizer.from_pretrained(SIGLIP_ID)
    LIMIT = 64
    for name, texts in (("stem", stems), ("fused", fused)):
        n = [len(tok(t)["input_ids"]) for t in texts]
        over = sum(1 for x in n if x > LIMIT)
        kept = [min(x, LIMIT) / x for x in n]
        print(f"{name:>6}: median {np.median(n):5.0f} tok | mean {np.mean(n):6.1f} | "
              f"max {max(n):4d} | >{LIMIT} tok: {over}/{len(n)} ({over/len(n):.0%}) | "
              f"mean fraction SigLIP sees: {np.mean(kept):.0%}", flush=True)
    print(f"\nLongCLIP context is 248 tokens, so it sees the fused string in full.",
          flush=True)
    return "done"


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=2.0, timeout=60 * 20)
def true_scorer_interaction():
    """The four-arm difference-in-differences the SS5.2 sentence actually claims.

    Two DIFFERENT quantities were being conflated:
      (A) does OMP's gain OVER UNIFORM depend on the scorer?
          = (omp_sig - unif) - (omp_lc - unif) = omp_sig - omp_lc.
          Uniform is scorer-independent, so it cancels and this collapses to a
          two-arm test. This is what was computed.
      (B) does the OMP-vs-TOP-K gap depend on the scorer?
          = (omp_sig - topk_sig) - (omp_lc - topk_lc).
          This is what the sentence about "+3.40 under SigLIP vs +2.43 under
          LongCLIP" is asserting, and it needs all four arms.

    Their POINT ESTIMATES coincide here only because top-k has identical marginal
    accuracy under both scorers (.6068). Their per-item discordance patterns, and
    therefore their p-values, need not coincide at all.
    """
    import glob
    from math import comb

    import numpy as np

    def load(tag):
        d = {}
        for p in glob.glob(f"{RESULTS_DIR}/{tag}/**/*samples*.jsonl", recursive=True):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("lvb_acc")
                if isinstance(a, dict):
                    d[a["id"]] = int(a["answer"] == a["parsed_pred"])
        return d

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    A = {k: load(f"{k}_600s") for k in ("uniform",)}
    for sel in ("topk", "omp"):
        for sc in ("lc", "sig"):
            A[f"{sel}_{sc}"] = load(f"{sel}_{sc}_600s")
    ids = sorted(set.intersection(*(set(v) for v in A.values())))
    print(f"n = {len(ids)} (all four arms + uniform, paired)\n", flush=True)

    # (A) what was computed and reported
    gl = [A["omp_lc"][q] - A["uniform"][q] for q in ids]
    gs = [A["omp_sig"][q] - A["uniform"][q] for q in ids]
    bA = sum(1 for x, y in zip(gs, gl) if y < x)
    cA = sum(1 for x, y in zip(gs, gl) if x < y)
    dA = (np.mean(gs) - np.mean(gl)) * 100
    print(f"(A) OMP gain-over-uniform, SigLIP vs LongCLIP  [what was reported]")
    print(f"    delta {dA:+.2f} pt   b={bA} c={cA}   p={mcnemar(bA,cA):.4g}\n", flush=True)

    # (B) the four-arm difference-in-differences the sentence claims
    dl = [A["omp_lc"][q] - A["topk_lc"][q] for q in ids]
    ds = [A["omp_sig"][q] - A["topk_sig"][q] for q in ids]
    bB = sum(1 for x, y in zip(ds, dl) if y < x)
    cB = sum(1 for x, y in zip(ds, dl) if x < y)
    dB = (np.mean(ds) - np.mean(dl)) * 100
    print(f"(B) OMP-minus-topk gap, SigLIP vs LongCLIP  [what the sentence needs]")
    print(f"    delta {dB:+.2f} pt   b={bB} c={cB}   p={mcnemar(bB,cB):.4g}\n", flush=True)

    print(f"point estimates equal? {abs(dA-dB) < 1e-9}   "
          f"p-values equal? {abs(mcnemar(bA,cA)-mcnemar(bB,cB)) < 1e-9}", flush=True)

    # discordant counts for every contrast the section reports (reviewer C1)
    print("\ndiscordant counts for SS5.2 contrasts:", flush=True)
    for lab, X, Y in (("uniform -> topk_lc", "uniform", "topk_lc"),
                      ("uniform -> omp_lc", "uniform", "omp_lc"),
                      ("uniform -> topk_sig", "uniform", "topk_sig"),
                      ("uniform -> omp_sig", "uniform", "omp_sig"),
                      ("topk_lc -> omp_lc", "topk_lc", "omp_lc"),
                      ("topk_sig -> omp_sig", "topk_sig", "omp_sig")):
        b = sum(1 for q in ids if A[Y][q] > A[X][q])
        c = sum(1 for q in ids if A[X][q] > A[Y][q])
        d = (sum(A[Y][q] for q in ids) - sum(A[X][q] for q in ids)) / len(ids) * 100
        print(f"   {lab:22s} {d:+6.2f} pt  b={b:3d} c={c:3d}  p={mcnemar(b,c):.4g}",
              flush=True)
    return {"A": [round(dA,3), bA, cA], "B": [round(dB,3), bB, cB]}


# ---------------------------------------------------------------------------
# Pre-registered analyses for the fused-query effect.
# Protocol: video-understanding/prereg/2026-08-22_fused_query.md
# Both are CPU-only and read artifacts already on the volume.
# ---------------------------------------------------------------------------


@app.function(image=fused_image, volumes={ROOT: vol}, secrets=[hf_secret],
              cpu=8.0, memory=16384, timeout=60 * 40)
def residual_trace_fused():
    """Analysis 1 -- does the fused query close any of the CLIP modality gap?

    Replays the omp_residual_trace.py instrumentation twice over the SAME 412
    LongVideoBench-600s items and the SAME LongCLIP image embeddings, changing only
    the text query (stem vs question+options).

    Baseline from 2026-07-19 (stem, LongCLIP): the text query is ~97% orthogonal to
    the image subspace -- residual fraction moves 0.972 -> 0.967 across 16 picks, so
    OMP explains ~3.3% of the query norm ever, and cos_orig maxes around 0.23.

    Pre-registered predictions, fused vs stem:
      P1.1  cos_orig at pick 1 HIGHER
      P1.2  resid_frac after 8 picks LOWER
      P1.3  total drained mass over picks 1-8 HIGHER
    Flat on all three => the accuracy gain is not an alignment effect, and the
    length/distribution confound becomes the leading alternative.
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.text import question_stem

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    man = {r["id"]: r for r in json.load(open("/root/manifest.json"))}
    ids = sorted(q for q, b in id2bin.items() if b == FUSED_BIN)
    assert len(ids) == BIN_N[FUSED_BIN], f"{len(ids)} != {BIN_N[FUSED_BIN]}"

    stems = [question_stem(man[q]) for q in ids]
    fused = [man[q]["question"] for q in ids]
    assert all(f != s for f, s in zip(fused, stems)), "fused == stem somewhere"

    sys.path.insert(0, "/root/Long-CLIP")
    from huggingface_hub import hf_hub_download
    from model import longclip

    ck = hf_hub_download("BeichenZhang/LongCLIP-L", "longclip-L.pt")
    lc_model, _ = longclip.load(ck, device="cpu")
    lc_model.eval()

    def lc_encode(texts):
        vs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                tok = longclip.tokenize(texts[i:i + 32], truncate=True)
                f = lc_model.encode_text(tok)
                vs.append((f / f.norm(dim=-1, keepdim=True)).float().numpy())
        return np.concatenate(vs, 0)

    Q = {"stem": lc_encode(stems), "fused": lc_encode(fused)}
    print(f"encoded stem {Q['stem'].shape} fused {Q['fused'].shape}", flush=True)

    KMAX = 16

    def omp_trace(query, emb, k):
        """Verbatim math from scripts/omp_residual_trace.py::omp_trace."""
        q0 = query.astype(np.float32).copy()
        q0n = float(np.linalg.norm(q0))
        q = q0.copy()
        E = emb.astype(np.float32)
        n = E.shape[0]
        basis, chosen = [], np.zeros(n, dtype=bool)
        steps = []
        for _ in range(min(k, n)):
            s = E @ q
            s[chosen] = -np.inf
            best = int(np.argmax(s))
            qn = float(np.linalg.norm(q))
            cos_resid = float(s[best] / qn) if qn > 1e-9 else 0.0
            cos_orig = float(E[best] @ q0 / q0n) if q0n > 1e-9 else 0.0
            chosen[best] = True
            v = E[best].copy()
            for b in basis:
                v -= (v @ b) * b
            norm = float(np.linalg.norm(v))
            drained = 0.0
            if norm > 1e-6:
                v /= norm
                drained = float(q @ v)
                basis.append(v)
                q = q - (q @ v) * v
            resid_frac = float(np.linalg.norm(q) / q0n) if q0n > 1e-9 else 0.0
            steps.append((best, resid_frac, abs(drained), cos_resid, cos_orig))
        return steps

    acc = {k: {"rf": [[] for _ in range(KMAX)],
               "dr": [[] for _ in range(KMAX)],
               "cr": [[] for _ in range(KMAX)],
               "co": [[] for _ in range(KMAX)],
               "maxcos": []} for k in ("stem", "fused")}
    per_item = {}

    for n, qid in enumerate(ids):
        z = np.load(os.path.join(EMBEDS_LC_DIR, qid + ".npz"))
        E = np.asarray(z["emb"], dtype=np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        rec = {}
        for kind in ("stem", "fused"):
            st = omp_trace(Q[kind][n], E, KMAX)
            for t, (_, rf, dr, cr, co) in enumerate(st):
                acc[kind]["rf"][t].append(rf)
                acc[kind]["dr"][t].append(dr)
                acc[kind]["cr"][t].append(cr)
                acc[kind]["co"][t].append(co)
            acc[kind]["maxcos"].append(float((E @ Q[kind][n]).max()))
            rec[kind] = {"co1": st[0][4], "rf8": st[7][1],
                         "drain8": float(sum(s[2] for s in st[:8]))}
        per_item[qid] = rec
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    def col(kind, key, t):
        return float(np.mean(acc[kind][key][t]))

    print("\npick   resid_frac        drained          cos_resid        cos_orig",
          flush=True)
    print("      stem   fused     stem   fused     stem   fused     stem   fused",
          flush=True)
    for t in range(KMAX):
        print(f"{t+1:>3}  {col('stem','rf',t):.4f} {col('fused','rf',t):.4f}   "
              f"{col('stem','dr',t):.4f} {col('fused','dr',t):.4f}   "
              f"{col('stem','cr',t):.4f} {col('fused','cr',t):.4f}   "
              f"{col('stem','co',t):.4f} {col('fused','co',t):.4f}", flush=True)

    # ---- the three pre-registered contrasts, paired over items ----------
    def paired(key):
        a = np.array([per_item[q]["stem"][key] for q in ids])
        b = np.array([per_item[q]["fused"][key] for q in ids])
        d = b - a
        # paired t on the mean difference; n=412 so the CLT is comfortable here
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        tstat = float(d.mean() / se) if se > 1e-12 else 0.0
        return (float(a.mean()), float(b.mean()), float(d.mean()), tstat,
                float((d > 0).mean()))

    print("\nPRE-REGISTERED CONTRASTS (paired over 412 items)", flush=True)
    print(f"{'quantity':>26} {'stem':>9} {'fused':>9} {'delta':>9} "
          f"{'t':>8} {'frac>0':>8} {'predicted':>10}", flush=True)
    spec = [("P1.1 cos_orig pick 1", "co1", "higher"),
            ("P1.2 resid_frac after 8", "rf8", "lower"),
            ("P1.3 drained mass 1-8", "drain8", "higher")]
    rep = {}
    for lab, key, want in spec:
        a, b, d, tt, fr = paired(key)
        ok = (d > 0) if want == "higher" else (d < 0)
        print(f"{lab:>26} {a:>9.4f} {b:>9.4f} {d:>+9.4f} {tt:>8.2f} {fr:>8.2%} "
              f"{want:>10} {'OK' if ok else 'FAILS'}", flush=True)
        rep[key] = [round(a, 5), round(b, 5), round(d, 5), round(tt, 2), round(fr, 4)]

    ms, mf = np.mean(acc["stem"]["maxcos"]), np.mean(acc["fused"]["maxcos"])
    print(f"\nmax cos(query, any frame): stem {ms:.4f}  fused {mf:.4f}  "
          f"delta {mf-ms:+.4f}", flush=True)
    print("Reference (2026-07-19, stem): resid 0.972->0.967 over 16 picks, "
          "cos_orig max ~0.23.", flush=True)
    rep["maxcos"] = [round(float(ms), 4), round(float(mf), 4)]

    os.makedirs(f"{RESULTS_DIR}/prereg", exist_ok=True)
    json.dump({"per_item": per_item, "summary": rep},
              open(f"{RESULTS_DIR}/prereg/residual_trace_fused.json", "w"))
    vol.commit()
    return rep


@app.function(image=fused_image, volumes={ROOT: vol}, secrets=[hf_secret],
              cpu=4.0, memory=8192, timeout=60 * 30)
def gain_split():
    """Analysis 2 -- where does the fused-query gain sit?

    Primary split, fixed in the pre-registration and computed from text alone:

        added_frac = (tok(fused) - tok(stem)) / tok(fused)

    at its median, tokenised with the LongCLIP tokenizer that produced the
    embeddings. High-added = the options carry most of the query's content.

    PRIMARY TEST IS THE INTERACTION, not the two within-half McNemar p-values. On
    2026-08-03 the question-type regime claim was withdrawn for exactly that error:
    a significant subgroup beside a non-significant one is not an interaction.
    """
    import glob
    import sys
    from math import comb

    import numpy as np

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/Long-CLIP")
    from harness.text import question_stem
    from model import longclip

    man = {r["id"]: r for r in json.load(open("/root/manifest.json"))}

    def load(tag):
        d = {}
        for p in glob.glob(f"{RESULTS_DIR}/{tag}/**/*samples*.jsonl", recursive=True):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("lvb_acc")
                if isinstance(a, dict):
                    d[a["id"]] = int(a["answer"] == a["parsed_pred"])
        return d

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    def fisher(a, b, c, d):
        """Two-sided Fisher exact on a 2x2, by summing tables no more likely."""
        rows = (a + b, c + d)
        cols = (a + c, b + d)
        n = a + b + c + d

        def pr(x):
            return (comb(rows[0], x) * comb(rows[1], cols[0] - x)
                    / comb(n, cols[0]))
        obs = pr(a)
        lo = max(0, cols[0] - rows[1])
        hi = min(rows[0], cols[0])
        return min(1.0, sum(pr(x) for x in range(lo, hi + 1)
                            if pr(x) <= obs * (1 + 1e-9)))

    S = load(f"omp_lc_{FUSED_BIN}")
    F = load(f"omp_lc_fused_{FUSED_BIN}")
    ids = sorted(set(S) & set(F))
    assert len(ids) == BIN_N[FUSED_BIN], f"{len(ids)} paired != {BIN_N[FUSED_BIN]}"

    # ---- the pre-registered split variable, outcome-blind -----------------
    def ntok(s):
        t = longclip.tokenize([s], truncate=True)[0]
        return int((t != 0).sum()) - 2          # drop SOT/EOT

    added, stem_tok = {}, {}
    for q in ids:
        st = ntok(question_stem(man[q]))
        fu = ntok(man[q]["question"])
        added[q] = (fu - st) / max(fu, 1)
        stem_tok[q] = st
    med = float(np.median([added[q] for q in ids]))
    hi = [q for q in ids if added[q] > med]
    lo = [q for q in ids if added[q] <= med]
    print(f"added_frac median {med:.4f} -> high {len(hi)}, low {len(lo)}", flush=True)
    print(f"stem tokens: median {np.median(list(stem_tok.values())):.0f}, "
          f"high-added half {np.median([stem_tok[q] for q in hi]):.0f}, "
          f"low-added half {np.median([stem_tok[q] for q in lo]):.0f}", flush=True)

    def cell(sub):
        b = sum(1 for q in sub if F[q] > S[q])     # fused rescues
        c = sum(1 for q in sub if S[q] > F[q])     # fused breaks
        ds = sum(S[q] for q in sub) / len(sub)
        df = sum(F[q] for q in sub) / len(sub)
        return b, c, ds, df

    print(f"\n{'half':>12} {'n':>5} {'stem':>8} {'fused':>8} {'delta':>8} "
          f"{'rescue':>7} {'break':>6} {'p':>9}", flush=True)
    out = {}
    for lab, sub in (("high-added", hi), ("low-added", lo)):
        b, c, ds, df = cell(sub)
        out[lab] = [len(sub), round(ds, 4), round(df, 4), round((df - ds) * 100, 2),
                    b, c, round(mcnemar(b, c), 6)]
        print(f"{lab:>12} {len(sub):>5} {ds:>8.4f} {df:>8.4f} {(df-ds)*100:>+8.2f} "
              f"{b:>7} {c:>6} {mcnemar(b,c):>9.4g}", flush=True)

    bh, ch, _, _ = cell(hi)
    bl, cl, _, _ = cell(lo)
    p_int = fisher(bh, ch, bl, cl)
    print(f"\nPRIMARY TEST -- interaction (discordants only)", flush=True)
    print(f"  high-added: {bh} rescued / {ch} broken", flush=True)
    print(f"  low-added : {bl} rescued / {cl} broken", flush=True)
    print(f"  Fisher exact p = {p_int:.4g}   (alpha .05, one pre-registered test)",
          flush=True)
    pred = out["high-added"][3] > out["low-added"][3]
    print(f"  P2.1 (gain larger in high-added half): "
          f"{'direction OK' if pred else 'direction WRONG'}", flush=True)
    out["interaction"] = {"high": [bh, ch], "low": [bl, cl], "p": round(p_int, 6),
                          "direction_as_predicted": bool(pred)}

    # ---- exploratory, explicitly not claimable ---------------------------
    print("\nEXPLORATORY (not claimable, no correction applied)", flush=True)
    cats = {}
    for q in ids:
        cats.setdefault(str(man[q].get("question_category", "?")), []).append(q)
    print(f"{'category':>28} {'n':>5} {'delta':>8} {'resc':>5} {'brk':>5} {'p':>8}",
          flush=True)
    expl = {}
    for cat, sub in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < 15:
            continue
        b, c, ds, df = cell(sub)
        expl[cat] = [len(sub), round((df - ds) * 100, 2), b, c,
                     round(mcnemar(b, c), 6)]
        print(f"{cat:>28} {len(sub):>5} {(df-ds)*100:>+8.2f} {b:>5} {c:>5} "
              f"{mcnemar(b,c):>8.4g}", flush=True)
    out["exploratory_categories"] = expl

    # continuous check on the same axis, Spearman of added_frac vs per-item gain
    g = np.array([F[q] - S[q] for q in ids], dtype=float)
    x = np.array([added[q] for q in ids])
    rx, rg = x.argsort().argsort().astype(float), g.argsort().argsort().astype(float)
    rho = float(np.corrcoef(rx, rg)[0, 1])
    print(f"\nSpearman(added_frac, per-item gain) = {rho:+.4f}  (n={len(ids)})",
          flush=True)
    out["spearman_added_vs_gain"] = round(rho, 4)

    os.makedirs(f"{RESULTS_DIR}/prereg", exist_ok=True)
    json.dump(out, open(f"{RESULTS_DIR}/prereg/gain_split.json", "w"))
    vol.commit()
    return out


# ---------------------------------------------------------------------------
# Foreign-options control arm.
# Protocol: video-understanding/prereg/2026-08-22_foreign_options.md
# ---------------------------------------------------------------------------

_OPT_MARK = "\n\nOptions:\n"
_ANS_MARK = "\n\nAnswer with"


def _split_question(q):
    """-> (stem, options_block, answer_instruction). Asserts the expected format."""
    i = q.find(_OPT_MARK)
    assert i != -1, f"no options marker: {q[:80]!r}"
    stem = q[:i]
    rest = q[i + len(_OPT_MARK):]
    j = rest.find(_ANS_MARK)
    assert j != -1, f"no answer marker: {q[-80:]!r}"
    return stem, rest[:j], rest[j:]


@app.function(image=fused_image, volumes={ROOT: vol}, secrets=[hf_secret],
              cpu=8.0, memory=16384, timeout=60 * 40)
def picks_foreign():
    """OMP/LongCLIP picks from stem + ANOTHER item's options block.

    Length- and structure-matched counterpart of the fused query. If the fused-query
    gain is structural (a longer, multi-direction query lets OMP keep draining its
    residual past pick 5) rather than about the answer set, these picks should be
    about as good as the real fused ones.

    Donor constraints, applied before any outcome is seen: different video, identical
    option count so the A)/B)/... letters line up, nearest fused-token length.
    """
    import re
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/Long-CLIP")
    from harness.replay_selectors import omp_indices
    from harness.text import question_stem

    id2bin = json.load(open("/root/lvb_id2bin.json"))
    man = {r["id"]: r for r in json.load(open("/root/manifest.json"))}
    ids = sorted(q for q, b in id2bin.items() if b == FUSED_BIN)
    assert len(ids) == BIN_N[FUSED_BIN], f"{len(ids)} != {BIN_N[FUSED_BIN]}"

    from model import longclip

    def ntok(s):
        t = longclip.tokenize([s], truncate=True)[0]
        return int((t != 0).sum()) - 2

    parts, meta = {}, {}
    for q in ids:
        stem, opts, ans = _split_question(man[q]["question"])
        assert question_stem(man[q]).strip() == stem.strip(), f"stem mismatch {q}"
        parts[q] = (stem, opts, ans)
        meta[q] = (len(opts.strip().split("\n")), str(man[q]["video_file"]),
                   ntok(man[q]["question"]))

    # ---- donor assignment, deterministic --------------------------------
    # Option letters are rewritten to run contiguously from A), so a donor with a
    # different option count still yields a well-formed block. Exactly one 600s item
    # has 3 options, so an exact-count-only rule has no donor for it; count is a
    # preference, not a hard constraint. The ANSWERER's prompt is untouched by this
    # -- only the scorer's query text changes -- so a count mismatch costs nothing.
    def reletter(block):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        body = [re.sub(r"^\s*[A-Za-z][)\.]\s*", "", ln) for ln in lines]
        return "\n".join(f"{chr(65+i)}) {t}" for i, t in enumerate(body))

    foreign, diffs, reuse, ncount_mismatch = {}, [], {}, 0
    for q in ids:
        nopt, vid, L = meta[q]
        cands = [d for d in ids if d != q and meta[d][1] != vid]
        assert cands, f"no donor for {q}"
        # same option count first, then nearest fused length, then least-used, then id
        d = min(cands, key=lambda c: (meta[c][0] != nopt, abs(meta[c][2] - L),
                                      reuse.get(c, 0), c))
        if meta[d][0] != nopt:
            ncount_mismatch += 1
        reuse[d] = reuse.get(d, 0) + 1
        stem, _, ans = parts[q]
        foreign[q] = stem + _OPT_MARK + reletter(parts[d][1]) + ans
        diffs.append(abs(ntok(foreign[q]) - L))
    print(f"donors with a different option count: {ncount_mismatch}/{len(ids)}",
          flush=True)

    print(f"donor length |delta tok|: mean {np.mean(diffs):.2f} "
          f"median {np.median(diffs):.1f} max {np.max(diffs)}", flush=True)
    print(f"distinct donors used: {len(reuse)}/{len(ids)}, "
          f"max reuse {max(reuse.values())}", flush=True)
    ex = ids[0]
    print(f"\nSELF  : {man[ex]['question']!r}", flush=True)
    print(f"\nFOREIGN: {foreign[ex]!r}", flush=True)
    for q in ids:
        assert foreign[q] != man[q]["question"], f"foreign == self for {q}"

    # ---- encode + OMP ----------------------------------------------------
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("BeichenZhang/LongCLIP-L", "longclip-L.pt")
    lc_model, _ = longclip.load(ck, device="cpu")
    lc_model.eval()

    texts = [foreign[q] for q in ids]
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            tok = longclip.tokenize(texts[i:i + 32], truncate=True)
            f = lc_model.encode_text(tok)
            vs.append((f / f.norm(dim=-1, keepdim=True)).float().numpy())
    Q = np.concatenate(vs, 0)
    print(f"\nlongclip foreign text: {Q.shape}", flush=True)

    stem_picks = json.load(open(f"{PICKS_DIR}/picks_omp_lc_k8.json"))
    fused_picks = json.load(open(f"{PICKS_DIR}/picks_omp_lc_fused_k8.json"))
    out, ov_s, ov_f = {}, [], []
    for n, qid in enumerate(ids):
        z = np.load(os.path.join(EMBEDS_LC_DIR, qid + ".npz"))
        E = np.asarray(z["emb"], dtype=np.float32)
        t = np.asarray(z["times"], dtype=np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        om = omp_indices(Q[n], E, min(K, len(t)))
        out[qid] = [round(float(t[i]), 4) for i in om]
        ov_s.append(len(_to_idx(t, stem_picks[qid]) & set(om)))
        ov_f.append(len(_to_idx(t, fused_picks[qid]) & set(om)))
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    fp = f"{PICKS_DIR}/picks_omp_lc_foreign_k8.json"
    json.dump(out, open(fp, "w"))
    print(f"\nwrote {fp}: {len(out)} ids", flush=True)
    print(f"pick overlap vs stem  : {np.mean(ov_s):.2f}/8 "
          f"({(1-np.mean(ov_s)/K):.0%} changed)", flush=True)
    print(f"pick overlap vs fused : {np.mean(ov_f):.2f}/8 "
          f"({(1-np.mean(ov_f)/K):.0%} changed)", flush=True)
    vol.commit()
    return {"n": len(out), "ov_stem": round(float(np.mean(ov_s)), 3),
            "ov_fused": round(float(np.mean(ov_f)), 3),
            "len_delta_mean": round(float(np.mean(diffs)), 2)}


@app.function(image=gpu_image, gpu="L40S", volumes={ROOT: vol}, secrets=[hf_secret],
              timeout=60 * 60 * 4)
def run_arm_foreign():
    """The single paid step: OMP/LongCLIP foreign-options arm, 600s, k=8."""
    pf = f"{PICKS_DIR}/picks_omp_lc_foreign_k8.json"
    assert os.path.exists(pf), f"missing {pf}"
    tag = f"omp_lc_foreign_{FUSED_BIN}"
    out = _run_lmms(TASK[("picks", FUSED_BIN)], tag, picks_file=pf)
    vol.commit()
    return {"arm": tag, "out": out, "n": _coverage(out, BIN_N[FUSED_BIN])}


@app.function(image=pick_image, volumes={ROOT: vol}, cpu=2.0, timeout=60 * 20)
def analyze_foreign():
    """Pre-registered contrasts C1 (foreign vs stem) and C2 (fused vs foreign)."""
    import glob
    from math import comb

    def load(tag):
        d = {}
        for p in glob.glob(f"{RESULTS_DIR}/{tag}/**/*samples*.jsonl", recursive=True):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("lvb_acc")
                if isinstance(a, dict):
                    d[a["id"]] = int(a["answer"] == a["parsed_pred"])
        return d

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    A = {"stem": load(f"omp_lc_{FUSED_BIN}"),
         "fused": load(f"omp_lc_fused_{FUSED_BIN}"),
         "foreign": load(f"omp_lc_foreign_{FUSED_BIN}")}
    ids = sorted(set(A["stem"]) & set(A["fused"]) & set(A["foreign"]))
    assert len(ids) == BIN_N[FUSED_BIN], f"{len(ids)} paired != {BIN_N[FUSED_BIN]}"

    print(f"{'arm':>9} {'acc':>8}", flush=True)
    for k in ("stem", "fused", "foreign"):
        print(f"{k:>9} {sum(A[k][q] for q in ids)/len(ids):>8.4f}", flush=True)

    print(f"\n{'contrast':>22} {'delta':>8} {'b':>4} {'c':>4} {'p':>9}", flush=True)
    rep = {}
    for lab, X, Y in (("C1 foreign vs stem", "stem", "foreign"),
                      ("C2 fused vs foreign", "foreign", "fused"),
                      ("(ref) fused vs stem", "stem", "fused")):
        b = sum(1 for q in ids if A[Y][q] > A[X][q])
        c = sum(1 for q in ids if A[X][q] > A[Y][q])
        d = (sum(A[Y][q] for q in ids) - sum(A[X][q] for q in ids)) / len(ids) * 100
        p = mcnemar(b, c)
        rep[lab] = [round(d, 2), b, c, round(p, 6)]
        print(f"{lab:>22} {d:>+8.2f} {b:>4} {c:>4} {p:>9.4g}", flush=True)

    os.makedirs(f"{RESULTS_DIR}/prereg", exist_ok=True)
    json.dump(rep, open(f"{RESULTS_DIR}/prereg/foreign.json", "w"))
    vol.commit()
    return rep
