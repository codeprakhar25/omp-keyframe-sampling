#!/usr/bin/env python3
"""Two anti-drift selectors (2026-07-18 visual analysis: OMP's #1 failure is drift to
irrelevant-but-distinctive frames). Same I/O + faithfulness discipline as gen_alpha_picks.py.
Emits lmms-eval {qid: sorted([secs])}. CPU. Refuses on unusable items unless --allow-skips.

  rfloor    : keep top-frac most query-relevant frames (min 3k candidates), run FULL OMP within
              that pool. Constrains OMP's spread to relevant frames -> can't wander off-topic.
              Isolates the relevance-floor effect (alpha=1 inside).
  iteralpha : gain-proportional deflation. alpha_i = clip(gain_i/gain_1, floor, 1). Late picks
              (which explain almost nothing) barely deflate -> residual stays near q0 -> less drift.
"""
import argparse, json, os
import numpy as np
from harness.embeds import l2, load_image_embed
from harness.replay_selectors import omp_indices


def omp_local(q0, E, k):
    """Full OMP (alpha=1), returns pick order-free sorted LOCAL indices. Faithful to omp_indices."""
    return omp_indices(q0, E, k)


def rfloor_indices(q0, E, k, frac):
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    rel = E @ q0
    m = min(n, max(int(np.ceil(frac * n)), 3 * k))     # candidate pool size
    cand = np.argsort(rel)[::-1][:m]                     # top-m most relevant (global idx)
    loc = omp_indices(q0, E[cand], k)                    # full OMP within the relevant pool
    return sorted(int(cand[i]) for i in loc)


def iteralpha_indices(q0, E, k, floor):
    n = E.shape[0]
    if k >= n:
        return list(range(n))
    q = q0.astype(np.float32).copy()
    basis, sel = [], []
    chosen = np.zeros(n, bool)
    g1 = None
    for _ in range(k):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        gain = float(s[best])
        if g1 is None:
            g1 = gain if gain > 1e-9 else 1.0
        a = float(np.clip(gain / g1, floor, 1.0))       # gain-proportional deflation
        sel.append(best); chosen[best] = True
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-6:
            v /= nrm; basis.append(v)
            q = q - a * (q @ v) * v
    return sorted(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["rfloor", "iteralpha"])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scorer", default="lc", choices=["sig", "lc"])
    ap.add_argument("--text-embeds", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.33)   # rfloor pool fraction
    ap.add_argument("--floor", type=float, default=0.30)  # iteralpha alpha floor
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-skips", action="store_true")
    a = ap.parse_args()

    t = np.load(a.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))
    rows = {}
    for line in open(a.scores):
        if line.strip():
            r = json.loads(line)
            if str(r["length_bin"]).rstrip("s") == str(a.bin).rstrip("s"):
                rows[r["id"]] = r
    print(f"scores rows in scope: {len(rows)} (bin={a.bin}) method={a.method} "
          f"frac={a.frac} floor={a.floor}")

    picks, reasons = {}, {"no_text": [], "no_img": [], "mismatch": []}
    for rid, r in rows.items():
        if rid not in qmap:
            reasons["no_text"].append(rid); continue
        times, emb = load_image_embed(rid, a.scorer)
        if times is None:
            reasons["no_img"].append(rid); continue
        E = l2(emb)
        if len(r["scores"]) != E.shape[0]:
            reasons["mismatch"].append(rid); continue
        q0 = l2(qmap[rid].astype(np.float32))
        idx = rfloor_indices(q0, E, a.k, a.frac) if a.method == "rfloor" \
            else iteralpha_indices(q0, E, a.k, a.floor)
        picks[rid] = {"idx": idx, "secs": [round(float(times[i]), 1) for i in idx]}

    skipped = sum(len(v) for v in reasons.values())
    if skipped:
        print(f"!! {skipped} unusable: " + ", ".join(f"{k}={len(v)}" for k, v in reasons.items()))
        if not a.allow_skips:
            raise SystemExit("REFUSING incomplete picks.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({q: sorted(v["secs"]) for q, v in picks.items()}, open(a.out, "w"))
    json.dump({"method": a.method, "bin": a.bin, "k": a.k, "frac": a.frac, "floor": a.floor,
               "n": len(picks), "picks": picks}, open(a.out.replace(".json", ".meta.json"), "w"))
    print(f"wrote {a.out}: {len(picks)} items ({skipped} skipped)")


if __name__ == "__main__":
    main()
