"""Power + equivalence analysis for the context-injection correctness null (Codex arm).

Within-task design: each task seen under none/always_on/selective, n=3 repeats, binary
outcome. Task is the unit (repeats nested in task). We ask:
  1. Marginal pass-rate per strategy + Wilson 95% CI.
  2. Within-task paired strategy differences + task-clustered bootstrap CI.
  3. Omnibus permutation test (does strategy explain pass beyond task?).
  4. Equivalence (TOST) at delta = 0.10 / 0.15.
  5. Power: MDE at current n; tasks needed to detect delta=0.10 at 80% power.
"""
import sqlite3, sys
import numpy as np
from scipy import stats

DB = "results_from_pod_codex/experiment_merged.db"
STRATS = ["none", "always_on", "selective"]
rng = np.random.default_rng(42)


def load(db, agent="codex"):
    c = sqlite3.connect(db)
    rows = c.execute(
        "SELECT task_id, strategy, task_passed FROM runs "
        "WHERE agent=? AND eval_method IS NOT NULL AND (error IS NULL OR error='')",
        (agent,)).fetchall()
    tasks = sorted({t for t, _, _ in rows})
    # mat[task][strat] = list of 0/1
    M = {t: {s: [] for s in STRATS} for t in tasks}
    for t, s, p in rows:
        if s in STRATS:
            M[t][s].append(int(p or 0))
    return tasks, M


def rate_matrix(tasks, M):
    """per-task pass-rate under each strategy (NaN if no data)."""
    R = np.full((len(tasks), 3), np.nan)
    for i, t in enumerate(tasks):
        for j, s in enumerate(STRATS):
            v = M[t][s]
            if v:
                R[i, j] = np.mean(v)
    return R


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DB
    agent = sys.argv[2] if len(sys.argv) > 2 else "codex"
    tasks, M = load(db, agent)
    R = rate_matrix(tasks, M)
    n_tasks = len(tasks)
    print(f"=== {agent}: {n_tasks} tasks, repeats up to 3, strategies {STRATS} ===\n")

    # 1. marginal rates + Wilson CI
    print("1) MARGINAL pass-rate per strategy (Wilson 95% CI)")
    marg = {}
    for j, s in enumerate(STRATS):
        ks = sum(sum(M[t][s]) for t in tasks)
        ns = sum(len(M[t][s]) for t in tasks)
        lo, hi = wilson(ks, ns)
        marg[s] = ks / ns
        print(f"   {s:10s} {ks:3d}/{ns:3d} = {100*ks/ns:5.1f}%  CI[{100*lo:4.1f}, {100*hi:4.1f}]")

    # 2. within-task paired diffs + task-bootstrap CI
    print("\n2) WITHIN-TASK paired strategy differences (task-clustered bootstrap, 10k)")
    pairs = [("always_on", "none"), ("selective", "none"), ("selective", "always_on")]
    B = 10000
    diffs = {}
    for a, b in pairs:
        ja, jb = STRATS.index(a), STRATS.index(b)
        d = R[:, ja] - R[:, jb]
        d = d[~np.isnan(d)]
        obs = d.mean()
        boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # sign-flip permutation p for H0: mean diff = 0
        P = 20000
        flips = rng.choice([-1, 1], size=(P, len(d)))
        perm = (flips * d).mean(axis=1)
        pval = (np.abs(perm) >= abs(obs) - 1e-12).mean()
        diffs[(a, b)] = (obs, lo, hi, pval)
        print(f"   {a:10s} - {b:10s}  Δ={100*obs:+5.1f}pp  CI[{100*lo:+5.1f}, {100*hi:+5.1f}]  p_perm={pval:.3f}")

    # 3. omnibus permutation: does strategy label explain pass beyond task?
    print("\n3) OMNIBUS permutation test (strategy effect beyond task)")
    # statistic: variance of the 3 marginal strategy rates (spread). Permute strategy
    # labels WITHIN each task (keeps task difficulty + per-task n fixed).
    def marg_spread(Mp):
        rates = []
        for s in STRATS:
            k = sum(sum(Mp[t][s]) for t in tasks); n = sum(len(Mp[t][s]) for t in tasks)
            rates.append(k / n)
        return np.var(rates)
    obs_stat = marg_spread(M)
    P = 20000
    cnt = 0
    for _ in range(P):
        Mp = {t: {s: [] for s in STRATS} for t in tasks}
        for t in tasks:
            pool = [(s, v) for s in STRATS for v in M[t][s]]
            labels = [s for s, _ in pool]
            vals = [v for _, v in pool]
            rng.shuffle(labels)
            # reassign preserving per-strategy counts within task
            idx = {s: 0 for s in STRATS}
            # group vals by shuffled label position
            # simpler: shuffle vals, split by original per-strat counts
            rng.shuffle(vals)
            pos = 0
            for s in STRATS:
                c = len(M[t][s])
                Mp[t][s] = vals[pos:pos + c]; pos += c
        if marg_spread(Mp) >= obs_stat - 1e-15:
            cnt += 1
    p_omni = cnt / P
    print(f"   observed strategy-rate variance = {obs_stat:.5f}")
    print(f"   p_omnibus = {p_omni:.3f}  ({'NO' if p_omni>0.05 else 'YES'} detectable strategy effect)")

    # 4. TOST equivalence
    print("\n4) EQUIVALENCE (TOST) — is each pairwise effect bounded?")
    for delta in (0.10, 0.15):
        print(f"   delta = ±{int(delta*100)}pp:")
        for (a, b), (obs, lo, hi, _) in diffs.items():
            equiv = (lo > -delta) and (hi < delta)
            print(f"     {a:10s}-{b:10s}  CI[{100*lo:+5.1f},{100*hi:+5.1f}]  "
                  f"{'EQUIVALENT (no effect > delta)' if equiv else 'inconclusive'}")

    # 5. power / MDE via Monte Carlo
    print("\n5) POWER — Monte Carlo (paired sign-flip test, alpha=0.05, 80% target)")
    base = np.array([np.nanmean(M[t]["none"]) if M[t]["none"] else np.nan for t in tasks])
    base = base[~np.isnan(base)]
    base_p = float(np.mean(base))
    print(f"   baseline per-task pass dist: mean={base_p:.2f}, used as task random effect")

    def sim_power(n_tasks_sim, reps, true_delta, nsim=600, P=2000):
        # task baseline probs resampled from observed 'none' per-task rates
        rej = 0
        for _ in range(nsim):
            tb = rng.choice(base, n_tasks_sim, replace=True)
            pa = np.clip(tb + true_delta, 0, 1)   # treatment arm
            pb = np.clip(tb, 0, 1)                 # control arm
            ka = rng.binomial(reps, pa) / reps
            kb = rng.binomial(reps, pb) / reps
            d = ka - kb
            obs = d.mean()
            flips = rng.choice([-1, 1], size=(P, n_tasks_sim))
            perm = (flips * d).mean(axis=1)
            if (np.abs(perm) >= abs(obs) - 1e-12).mean() < 0.05:
                rej += 1
        return rej / nsim

    print(f"\n   (a) MDE at current n={n_tasks}, reps=3:")
    for dlt in (0.10, 0.15, 0.20, 0.25, 0.30):
        pw = sim_power(n_tasks, 3, dlt)
        print(f"        true Δ={int(dlt*100):2d}pp -> power={pw:.2f}{'  <-- ~80%' if 0.75<=pw<=0.85 else ''}")

    print(f"\n   (b) tasks needed to detect Δ=10pp at reps=3 (80% power):")
    for nt in (17, 30, 50, 80, 120, 200):
        pw = sim_power(nt, 3, 0.10, nsim=400)
        print(f"        n_tasks={nt:3d} -> power={pw:.2f}{'  <-- crosses 80%' if pw>=0.80 else ''}")

    print(f"\n   (c) effect of more repeats (n={n_tasks}, Δ=15pp):")
    for rp in (3, 5, 10):
        pw = sim_power(n_tasks, rp, 0.15, nsim=400)
        print(f"        reps={rp:2d} -> power={pw:.2f}")


if __name__ == "__main__":
    main()
