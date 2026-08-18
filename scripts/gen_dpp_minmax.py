#!/usr/bin/env python3
"""Lever-1 Arm A: DPP frame selection with LDDR's exact quality normalization.

WHY (2026-07-19): banked DPP (gen_dpp_picks.py) ties OMP but used quality
r_i = exp(beta * z(sim)) (z-standardized cosine, temperature beta). LDDR (arXiv
2605.11477 sec 3.3.1) instead defines selection relevance as MinMax-normalized
cosine, NO temperature:

    r_i = (sim_i - min_j sim_j) / (max_j sim_j - min_j sim_j) ;   L = diag(r) K diag(r)

This isolates ONE variable: does the quality-weighting SHAPE (MinMax vs z/exp)
account for any of the -1.3/-3.2pt gap to LDDR-LD? Everything else identical
(LongCLIP scorer, 1fps pool, fixed tokens, greedy log-det MAP, k=8). ~zero cost:
CPU pick-gen on cached embeds. If this moves toward LD 66.3/55.9 the gap is
r-norm; if it ties banked DPP the gap is elsewhere (pool density -> Arm B).

Greedy MAP is Chen et al. 2018 incremental-Cholesky, O(n k^2), same as banked.
"""
import argparse, json, os
import numpy as np
from harness.embeds import l2, load_image_embed
from harness.replay_selectors import omp_indices


def dpp_greedy_minmax(q0, E, k, eps=1e-6):
    """Greedy k-DPP MAP, quality r_i = MinMax(query-frame cosine) (LDDR, no temp).
    Returns sorted LOCAL indices."""
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    sim = E @ q0                                   # query-frame cosine (both L2-normed)
    lo, hi = sim.min(), sim.max()
    r = (sim - lo) / (hi - lo + 1e-12)             # MinMax -> [0,1], LDDR relevance
    r = np.clip(r, eps, 1.0)                        # floor so log-quality finite
    logr = np.log(r).astype(np.float64)            # log-quality = 2*logr; L_ii = r_i^2
    c = np.zeros((k, n), dtype=np.float64)
    d2 = (r * r).astype(np.float64)                # init L_ii = r_i^2
    sel = []
    j = int(np.argmax(d2))
    sel.append(j)
    for t in range(1, k):
        rj = r[j]
        Lji = rj * r * (E @ E[j])                   # L_ji = r_j r_i K_ji, vector over i
        ci = (Lji - c[:t, :].T @ c[:t, j]) / np.sqrt(d2[j])
        c[t, :] = ci
        d2 = d2 - ci * ci
        d2[sel] = -np.inf
        j = int(np.argmax(d2))
        if not np.isfinite(d2[j]) or d2[j] <= 1e-12:
            break
        sel.append(j)
    if len(sel) < k:                               # backfill top-relevance if volume collapsed
        order = np.argsort(sim)[::-1]
        for i in order:
            if i not in sel:
                sel.append(int(i))
            if len(sel) == k:
                break
    return sorted(sel[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="lc", choices=["sig", "lc"])
    ap.add_argument("--text-embeds", required=True)
    ap.add_argument("--scores", default=None,
                    help="optional scores.jsonl to filter ids (+ length_bin via --bin)")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--audit", action="store_true",
                    help="don't write; report frames moved vs OMP and vs banked z-DPP intent")
    ap.add_argument("--allow-skips", action="store_true")
    a = ap.parse_args()

    t = np.load(a.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))
    if a.scores:
        import json as _json
        ids = []
        with open(a.scores) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = _json.loads(line)
                if a.bin not in (None, "", "all", "ALL"):
                    b = str(r.get("length_bin", "")).rstrip("s")
                    want = str(a.bin).rstrip("s")
                    # only filter when the scores row has a comparable length_bin
                    if b and b != want:
                        continue
                ids.append(r["id"])
        ids = [i for i in ids if i in qmap]
    else:
        ids = list(qmap.keys())
    print(f"qids in scope: {len(ids)} (bin={a.bin}) rnorm=MinMax audit={a.audit} embeds={a.embeds_dir}/{a.embeds_lc_dir}")

    picks, reasons = {}, {"no_text": [], "no_img": []}
    moved = []
    for rid in ids:
        if rid not in qmap:
            reasons["no_text"].append(rid); continue
        times, emb = load_image_embed(rid, a.scorer, a.embeds_dir, a.embeds_lc_dir)
        if times is None:
            reasons["no_img"].append(rid); continue
        E = l2(emb)
        q0 = l2(qmap[rid].astype(np.float32))
        idx = dpp_greedy_minmax(q0, E, a.k)
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
