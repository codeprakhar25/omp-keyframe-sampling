#!/usr/bin/env python3
"""Conditional k-DPP frame selection via fast greedy log-det MAP (Chen et al. 2018).

WHY THIS EXISTS (2026-07-18): the selection axis is saturated *within the OMP objective*
(max query-correlation + orthogonal-residual spread). Every variant we tried -- alpha-sweep,
MMR, rfloor, iteralpha -- is that same objective and all landed <= OMP. A k-DPP maximizes a
DIFFERENT quantity: log det(L) = the VOLUME spanned by the chosen frames. Our saturation
result does not cover it. This is also what our baseline papers actually use: LDDR is
"Linear-DPP-Based" sampling; MDP3 (2501.02885, training-free, LongVideoBench) is conditional
DPP and reports +4.8 over uniform, beating CLIP top-k.

Kernel (MDP3 CMGK, quality-diversity decomposition):
    L = diag(r) . K . diag(r)
    K_ij = cos(f_i, f_j)                 (frame-frame similarity, E is L2-normed)
    r_i  = exp(beta * z(sim_i))          (query relevance as DPP quality; z = per-video
                                          standardized query-frame cosine)
`beta` is the quality-vs-diversity temperature -- DPP's one knob, the clean-math analogue of
OMP's alpha. beta->inf collapses to top-k relevance; beta->0 is pure-diversity volume.

Same I/O + faithfulness discipline as gen_extra_picks.py. Emits lmms-eval {qid: sorted([secs])}.
CPU, O(n k^2). Refuses on unusable items unless --allow-skips.
"""
import argparse, json, os
import numpy as np
from harness.embeds import l2, load_image_embed
from harness.replay_selectors import omp_indices


def dpp_greedy(q0, E, k, beta, Ediv=None):
    """Fast greedy MAP for k-DPP with kernel L = diag(r) K diag(r).
    Returns sorted LOCAL indices. Chen et al. 2018 incremental-Cholesky greedy: O(n k^2).

    Ediv: optional alternative L2-normed matrix used for the DIVERSITY kernel K only
    (--centre passes the cone-removed embeddings here). Query relevance `sim` always
    comes from the original E, so centring changes the geometry and nothing else."""
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    if Ediv is None:
        Ediv = E
    sim = E @ q0                                   # query-frame cosine (E, q0 L2-normed)
    z = (sim - sim.mean()) / (sim.std() + 1e-8)    # per-video standardize -> scale-free beta
    logr = beta * z                                # r_i = exp(logr); log-quality = 2*logr
    # Incremental greedy log-det. d2[i] = current squared conditional "gain" of atom i.
    # For L = diag(r) K diag(r): L_ii = r_i^2 * K_ii = r_i^2 (K_ii=1). L_ij = r_i r_j K_ij.
    c = np.zeros((k, n), dtype=np.float64)         # Cholesky factors of chosen atoms
    d2 = np.exp(2.0 * logr).astype(np.float64)     # init = L_ii = r_i^2
    sel = []
    j = int(np.argmax(d2))
    sel.append(j)
    for t in range(1, k):
        rj = np.exp(logr[j])
        # e_i = (L_ji - c[:t,j].c[:t,i]) / sqrt(d2[j]) ; L_ji = r_j r_i K_ji
        Lji = rj * np.exp(logr) * (Ediv @ Ediv[j])  # vector over i (K from Ediv)
        ci = (Lji - c[:t, :].T @ c[:t, j]) / np.sqrt(d2[j])
        c[t, :] = ci
        d2 = d2 - ci * ci
        d2[sel] = -np.inf
        j = int(np.argmax(d2))
        if not np.isfinite(d2[j]) or d2[j] <= 1e-12:
            break                                  # rank-deficient: remaining add ~0 volume
        sel.append(j)
    # backfill if volume collapsed before k (pad with top-relevance not already chosen)
    if len(sel) < k:
        order = np.argsort(sim)[::-1]
        for i in order:
            if i not in sel:
                sel.append(int(i))
            if len(sel) == k:
                break
    return sorted(sel[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="optional scores jsonl for bin filtering; if omitted, "
                    "enumerate every qid in --text-embeds with a cached image embed")
    ap.add_argument("--scorer", default="lc", choices=["sig", "lc"])
    ap.add_argument("--text-embeds", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--beta", type=float, default=2.0)   # quality-vs-diversity temperature
    ap.add_argument("--out")
    ap.add_argument("--audit", action="store_true",
                    help="don't write picks; report frame-move stats vs OMP on local pool")
    ap.add_argument("--allow-skips", action="store_true")
    ap.add_argument("--centre", action="store_true",
                    help="build the diversity kernel K on per-video CENTRED embeddings "
                         "(remove the shared cone). Query relevance stays on the original "
                         "embeddings. Raw frame Gram has median effective rank 1.97 with "
                         "70.5%% of mass in one shared direction; centring lifts it to 11.2.")
    a = ap.parse_args()

    t = np.load(a.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))

    if a.scores:
        ids = []
        for line in open(a.scores):
            if line.strip():
                r = json.loads(line)
                if str(r["length_bin"]).rstrip("s") == str(a.bin).rstrip("s"):
                    ids.append(r["id"])
    else:
        ids = list(qmap.keys())
    print(f"qids in scope: {len(ids)} (bin={a.bin}) beta={a.beta} audit={a.audit} "
          f"centre={a.centre}")

    picks, reasons = {}, {"no_text": [], "no_img": []}
    moved = []   # per-video: #frames DPP picks that OMP did not
    for rid in ids:
        if rid not in qmap:
            reasons["no_text"].append(rid); continue
        times, emb = load_image_embed(rid, a.scorer)
        if times is None:
            reasons["no_img"].append(rid); continue
        E = l2(emb)
        q0 = l2(qmap[rid].astype(np.float32))
        Ediv = l2(E - E.mean(0, keepdims=True)) if a.centre else None
        idx = dpp_greedy(q0, E, a.k, a.beta, Ediv=Ediv)
        picks[rid] = [round(float(times[i]), 1) for i in idx]
        if a.audit:
            omp = set(omp_indices(q0, E, a.k))
            moved.append(len(set(idx) - omp))

    skipped = sum(len(v) for v in reasons.values())
    usable = len(picks)
    if a.audit:
        if moved:
            m = np.array(moved, float)
            print(f"AUDIT n={len(m)} k={a.k}: mean frames moved vs OMP = {m.mean():.2f}/{a.k} "
                  f"(median {np.median(m):.0f}, max {int(m.max())}, "
                  f"identical={int((m==0).sum())}/{len(m)})")
        print(f"({skipped} skipped: " + ", ".join(f"{k}={len(v)}" for k, v in reasons.items()) + ")")
        return

    if skipped:
        print(f"!! {skipped} unusable: " + ", ".join(f"{k}={len(v)}" for k, v in reasons.items()))
        if not a.allow_skips:
            raise SystemExit("REFUSING incomplete picks.")
    if not a.out:
        raise SystemExit("--out required when not --audit")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(picks, open(a.out, "w"))
    print(f"wrote {a.out}: {usable} items ({skipped} skipped)")


if __name__ == "__main__":
    main()
