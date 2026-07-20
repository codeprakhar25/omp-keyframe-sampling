"""Offline replay implementations of literature frame selectors (CPU, numpy only).

Both consume cached per-frame data (scores from scripts/dump_scores.py, embedding
vectors from scripts/dump_embeds.py) and return picked frame indices -- no GPU, no
frames. Scorer is held fixed at SigLIP for every method (the papers vary it; we
isolate the pick-math axis). Both are greedy maximizers of a submodular-style
objective; both seed with the argmax-relevance frame, so they differ from plain
top-k only in how the remaining budget is spent.

- adard_indices: AdaRD-Key (arXiv 2510.02778) -- relevance + log-det (Gram) diversity.
  Marginal gain of candidate i given selected F:
      delta(i|F) = R(i) + lam * log((1+eps) - r^T (G_F + eps I)^{-1} r)
  where r = cosine sims of i to F, G_F = Gram of F (SigLIP image embeddings).
  A near-duplicate of an already-picked frame drives the log term to ~log(eps),
  killing its gain. R = sigmoid(SigLIP logit) to match the paper's [0,1] ITM scale;
  paper's gate (diversity-only when max R < 0.4) is kept. Deviation from paper:
  their R is BLIP-2 ITM, ours SigLIP -- same-scorer comparison is the point.

- adaptive_greedy_indices: Adaptive Greedy (arXiv 2603.20180) -- relevance +
  facility-location coverage on a pure-vision encoder (DINOv2), greedy (1-1/e).
      F(S) = alpha * sum_{i in S} Rhat(i) + beta * (1/n) sum_j max_{s in S} sim(j, s)
  sim = (1+cos)/2 in [0,1] so coverage stays monotone submodular. Rhat = min-max
  normalized relevance. Paper adds question-type routing for alpha/beta; we expose
  them as knobs and default to 1/1.

- focus_indices: FOCUS (arXiv 2510.27280) -- coarse-to-fine bandit over 16s clips,
  simulated on the cached score array (an "arm pull" = revealing one frame's score).
  Faithful to Algorithm 2: coarse stage pulls q random frames per clip; ONE fine
  batch gives z extra pulls to the top ceil(alpha*M) clips by optimistic mean
  (empirical-Bernstein radius, Eq. 5: sqrt(2 var ln n / N_a) + 3 ln n / N_a with
  n = total pulls); final = top-m clips by UNBIASED empirical mean; the K output
  frames are drawn per clip (k_a ~= K/m each) by nearest-neighbor interpolating the
  observed rewards across the clip and sampling proportional to interpolated reward
  without replacement (Section 2.4) -- unobserved frames are pickable, which is
  exactly the clip-mean dilution the paper admits. Deviation from paper: reward is
  sigmoid(SigLIP logit), theirs BLIP ITM (both [0,1]; scorer held fixed on purpose).
  Needs scores only, no embeddings. Deterministic per-item (seeded rng).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def adard_indices(scores: Sequence[float], emb: np.ndarray, k: int,
                  lam: float = 1.0, eps: float = 0.01,
                  gate: float = 0.4) -> List[int]:
    """AdaRD-Key greedy: relevance + lam * log-det diversity over `emb` (L2-normalized).

    Returns sorted frame indices, len == min(k, n).
    """
    n = len(scores)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    emb = emb.astype(np.float32)
    rel = _sigmoid(np.asarray(scores, dtype=np.float32))
    use_rel = float(rel.max()) >= gate  # paper's low-confidence gate: diversity-only
    sims = emb @ emb.T  # cosine (normalized inputs); n<=few-thousand -> fits RAM

    selected: List[int] = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(selected)
    while len(selected) < k and remaining:
        F = np.array(selected)
        G = sims[np.ix_(F, F)] + eps * np.eye(len(F), dtype=np.float32)
        Ginv = np.linalg.inv(G)
        cand = np.array(sorted(remaining))
        R = sims[np.ix_(cand, F)]                      # (m, |F|) sims to selected
        q = np.einsum("ij,jk,ik->i", R, Ginv, R)       # r^T Ginv r per candidate
        div = np.log(np.clip((1.0 + eps) - q, 1e-9, None))
        gain = (rel[cand] if use_rel else 0.0) + lam * div
        best = int(cand[int(np.argmax(gain))])
        selected.append(best)
        remaining.discard(best)
    return sorted(selected)


def adaptive_greedy_indices(scores: Sequence[float], cov_emb: np.ndarray, k: int,
                            alpha: float = 1.0, beta: float = 1.0) -> List[int]:
    """Adaptive-Greedy: alpha*relevance + beta*facility-location coverage on `cov_emb`.

    cov_emb = L2-normalized pure-vision embeddings (DINOv2). Returns sorted indices.
    """
    n = len(scores)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    rel = np.asarray(scores, dtype=np.float32)
    span = float(rel.max() - rel.min())
    rhat = (rel - rel.min()) / span if span > 0 else np.zeros(n, dtype=np.float32)
    sims = (1.0 + cov_emb.astype(np.float32) @ cov_emb.T) / 2.0  # [0,1]

    selected: List[int] = [int(np.argmax(rhat))]
    covered = sims[selected[0]].copy()          # current max-sim of every frame to S
    chosen = np.zeros(n, dtype=bool)
    chosen[selected[0]] = True
    while len(selected) < k:
        marg = np.maximum(sims - covered[None, :], 0.0).mean(axis=1)  # coverage gain
        gain = alpha * rhat + beta * marg
        gain[chosen] = -np.inf
        best = int(np.argmax(gain))
        selected.append(best)
        chosen[best] = True
        covered = np.maximum(covered, sims[best])
    return sorted(selected)


def omp_indices(query: np.ndarray, emb: np.ndarray, k: int) -> List[int]:
    """Orthogonal Matching Pursuit frame selection ("question coverage").

    Pick argmax(q_res · f_i); orthogonalize the residual query against the span of
    picked frame embeddings (Gram-Schmidt, textbook OMP — no residual renorm); repeat.
    Each pick must explain a query component the previous picks did not.

    query: L2-normed SigLIP/LongCLIP text embedding [d]; emb: L2-normed SigLIP/LongCLIP image
    embeddings [n, d]. Returns sorted indices.
    """
    n = emb.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    q = query.astype(np.float32).copy()
    E = emb.astype(np.float32)
    basis: List[np.ndarray] = []          # orthonormal basis of picked-frame span
    selected: List[int] = []
    chosen = np.zeros(n, dtype=bool)
    for _ in range(k):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        selected.append(best)
        chosen[best] = True
        # Gram-Schmidt: orthonormalize the new frame against current basis, then
        # remove its component from the residual query
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v /= norm
            basis.append(v)
            q = q - (q @ v) * v
    return sorted(selected)


def focus_indices(scores: Sequence[float], times: Sequence[float], k: int,
                  clip_len: float = 16.0, q: int = 3, alpha: float = 0.25,
                  pull_budget: float = 0.5, m: int | None = None) -> List[int]:
    """FOCUS replay (Algorithm 2 + Section 2.4 of arXiv 2510.27280). See module doc.

    pull_budget = fraction of all frames the bandit may ever score; the fine-stage
    z (pulls per surviving clip) is derived from what the coarse stage left over.
    m = final clip count (paper hyperparameter); default ~k/4 -> 4 frames per clip.
    Returns sorted, de-duplicated frame indices (len == k when n >= k).
    """
    n = len(scores)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    rel = _sigmoid(np.asarray(scores, dtype=np.float32))  # [0,1] like BLIP ITM
    t = np.asarray(times, dtype=np.float32)
    clip_of = np.minimum((t / clip_len).astype(int), int(t[-1] / clip_len))
    clips = sorted(set(clip_of.tolist()))
    members = {c: np.where(clip_of == c)[0] for c in clips}
    rng = np.random.default_rng(n)  # deterministic per item

    pulled: dict = {c: [] for c in clips}

    def pull(c, count):
        avail = [i for i in members[c] if i not in pulled[c]]
        if avail and count > 0:
            take = rng.choice(len(avail), size=min(count, len(avail)), replace=False)
            pulled[c].extend(int(avail[j]) for j in take)

    # Stage I (coarse): q pulls per clip, optimistic means via empirical Bernstein
    for c in clips:
        pull(c, q)
    n_pulls = sum(len(v) for v in pulled.values())
    mu_opt = {}
    for c in clips:
        obs = rel[pulled[c]]
        cnt = max(1, len(obs))
        var = float(obs.var()) if len(obs) > 1 else 0.25
        beta = np.sqrt(2 * var * np.log(n_pulls) / cnt) + 3 * np.log(n_pulls) / cnt
        mu_opt[c] = float(obs.mean()) + beta if len(obs) else beta

    # Stage II (fine): ONE batch of z extra pulls to the top ceil(alpha*M) clips
    survivors = sorted(mu_opt, key=mu_opt.get, reverse=True)[
        : max(1, int(np.ceil(alpha * len(clips))))]
    z = max(0, int(pull_budget * n) - n_pulls) // len(survivors)
    for c in survivors:
        pull(c, z)

    # Final: top-m clips by UNBIASED empirical mean; k_a frames per clip sampled
    # proportional to nearest-neighbor-interpolated reward, without replacement
    m = m or max(1, round(k / 4))
    clip_mean = {c: float(rel[pulled[c]].mean()) for c in clips if pulled[c]}
    final = sorted(clip_mean, key=clip_mean.get, reverse=True)[:m]
    quota = {c: k // len(final) for c in final}
    for c in final[: k - sum(quota.values())]:   # distribute remainder
        quota[c] += 1

    selected: List[int] = []
    leftovers: List[int] = []
    for c in final:
        mem = members[c]
        obs = np.array(sorted(pulled[c]))
        # nearest observed frame's reward stands in for every unobserved frame
        nearest = obs[np.abs(mem[:, None] - obs[None, :]).argmin(axis=1)]
        w = rel[nearest].astype(np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(mem), 1 / len(mem))
        take = min(quota[c], len(mem))
        picks = rng.choice(len(mem), size=take, replace=False, p=w)
        selected.extend(int(mem[j]) for j in picks)
        leftovers.extend(int(i) for i in mem if int(i) not in set(selected))
    for i in leftovers:                          # short clips: fill quota elsewhere
        if len(selected) >= k:
            break
        if i not in selected:
            selected.append(i)
    return sorted(set(selected))
