#!/usr/bin/env python3
"""HPD location-posterior vs peak-NMS for moment retrieval, replayed offline from cached curves.

Idea (research/keyframe_sampling_timeline.md follow-up): treat the SigLIP score curve as evidence
about WHERE the needle is. Smooth scores with a temporal kernel, exponentiate to a posterior over
time, return highest-posterior-density (HPD) frames/intervals. Replaces peak-NMS's three hard knobs
(tau floor / gmin / pad) with two interpretable ones (kernel sigma, temperature tau).

Offline replay on results/scores/scores.jsonl (200 LVB curves, all with gold_evidence_seconds).
Budget-matched comparison: every arm returns exactly B frames. Arms:
  topk : top-B raw scores (baseline)
  nms  : build_union_indices (existing peak-NMS, budget=B)
  hpd  : top-B frames by posterior mass, over a (tau, sigma) grid
Plus hpd mass-90 variant (adaptive frame count, reported separately, NOT budget-matched).

Pre-registered call (set before running): HPD wins iff budget-matched any-hit >= NMS AND mean
best-IoU > NMS on >= half the (tau, sigma) grid points. Single-best-gridpoint wins don't count
(that's tuning on test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.union_retrieval import build_union_indices, any_hit, full_recall  # noqa: E402


def gaussian_smooth(scores: np.ndarray, times: np.ndarray, sigma_sec: float) -> np.ndarray:
    """Kernel-smooth scores in time. Uniform 1/fps spacing assumed (true for our dumps)."""
    if sigma_sec <= 0:
        return scores.copy()
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    sig = max(sigma_sec / dt, 1e-6)
    half = int(np.ceil(3 * sig))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k /= k.sum()
    # reflect-pad so edges aren't dragged toward zero
    pad = np.concatenate([scores[half:0:-1], scores, scores[-2:-half - 2:-1]])
    return np.convolve(pad, k, mode="valid")


def posterior(scores: np.ndarray, times: np.ndarray, tau: float, sigma_sec: float) -> np.ndarray:
    s = gaussian_smooth(scores, times, sigma_sec)
    z = (s - s.max()) / tau
    p = np.exp(z)
    return p / p.sum()


def hpd_frames(p: np.ndarray, budget: int) -> list:
    return sorted(np.argsort(-p)[:budget].tolist())


def hpd_mass(p: np.ndarray, level: float) -> list:
    order = np.argsort(-p)
    c = np.cumsum(p[order])
    m = int(np.searchsorted(c, level)) + 1
    return sorted(order[:m].tolist())


def indices_to_intervals(idx: list, times: np.ndarray) -> list:
    """Contiguous index runs -> [lo, hi] second spans (half-frame pad each side)."""
    if not idx:
        return []
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    out, lo, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append([times[lo] - dt / 2, times[prev] + dt / 2])
        lo = prev = i
    out.append([times[lo] - dt / 2, times[prev] + dt / 2])
    return out


def span_iou(a: list, b: list) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def eval_arm(idx: list, times: np.ndarray, gold: list, tol: float = 0.5) -> dict:
    sel_t = [float(times[i]) for i in idx]
    ivs = indices_to_intervals(idx, times)
    best_ious = [max((span_iou(g, iv) for iv in ivs), default=0.0) for g in gold]
    in_gold = sum(1 for t in sel_t if any(lo - tol <= t <= hi + tol for lo, hi in gold))
    return {
        "any_hit": any_hit(sel_t, gold),
        "full_recall": full_recall(sel_t, gold),
        "frame_precision": in_gold / len(sel_t) if sel_t else 0.0,
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "recall_iou_0.5": float(np.mean([i >= 0.5 for i in best_ious])) if best_ious else 0.0,
        "n_frames": len(idx),
        "n_intervals": len(ivs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--budgets", default="8,16")
    ap.add_argument("--taus", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--sigmas", default="0,1,2,4", help="kernel sigma seconds; 0 = no smoothing")
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--out", default="results/hpd_vs_nms.json")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]

    recs = [json.loads(l) for l in open(args.scores)]
    recs = [r for r in recs if r.get("gold_evidence_seconds")]

    # agg[bin][budget][arm] -> list of per-question metric dicts
    agg: dict = {}
    for r in recs:
        s = np.asarray(r["scores"], dtype=np.float64)
        t = np.asarray(r["times"], dtype=np.float64)
        gold = r["gold_evidence_seconds"]
        b_in = r["length_bin"]
        for B in budgets:
            arms = {
                "topk": sorted(np.argsort(-s)[:B].tolist()),
                "nms": build_union_indices(s.tolist(), t.tolist(), budget=B),
            }
            for tau in taus:
                for sig in sigmas:
                    p = posterior(s, t, tau, sig)
                    arms[f"hpd_t{tau}_s{sig}"] = hpd_frames(p, B)
            # adaptive-count variant at a reference gridpoint (mid tau/sigma)
            p_ref = posterior(s, t, taus[len(taus) // 2], sigmas[len(sigmas) // 2])
            arms["hpd_mass"] = hpd_mass(p_ref, args.mass)
            for name, idx in arms.items():
                m = eval_arm(idx, t, gold)
                agg.setdefault(b_in, {}).setdefault(B, {}).setdefault(name, []).append(m)

    def summarize(ms):
        return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}

    summary = {b: {B: {a: summarize(v) for a, v in arms.items()}
                   for B, arms in bud.items()} for b, bud in agg.items()}

    # pre-registered verdict: budget-matched grid majority vs nms, per bin+budget
    verdict = {}
    for b, bud in summary.items():
        for B, arms in bud.items():
            nms = arms["nms"]
            wins = sum(1 for a, m in arms.items() if a.startswith("hpd_t")
                       and m["any_hit"] >= nms["any_hit"] and m["mean_best_iou"] > nms["mean_best_iou"])
            total = sum(1 for a in arms if a.startswith("hpd_t"))
            verdict[f"bin{b}_B{B}"] = {"hpd_grid_wins": wins, "grid_total": total,
                                       "call": "HPD" if wins >= total / 2 else "NMS"}

    out = {"n_questions": len(recs), "summary": summary, "verdict": verdict,
           "args": vars(args)}
    json.dump(out, open(args.out, "w"), indent=1)

    for b, bud in sorted(summary.items()):
        for B, arms in sorted(bud.items()):
            print(f"\n== bin {b}  budget {B}  (n={len(agg[b][B]['nms'])}) ==")
            print(f"{'arm':<16} {'any_hit':>7} {'f_prec':>7} {'mIoU':>7} {'R@.5':>6} {'nfrm':>5} {'nint':>5}")
            for a in sorted(arms, key=lambda x: -arms[x]["mean_best_iou"]):
                m = arms[a]
                print(f"{a:<16} {m['any_hit']:>7.3f} {m['frame_precision']:>7.3f} "
                      f"{m['mean_best_iou']:>7.3f} {m['recall_iou_0.5']:>6.3f} "
                      f"{m['n_frames']:>5.1f} {m['n_intervals']:>5.1f}")
    print("\nverdict:", json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
