#!/usr/bin/env python3
"""Reliability study — pre-registered variance metrics.

Reads a runs DB (same schema as experiment_full.db) and computes the six
pre-registered metrics from reliability-study/SPEC.md §4. Stdlib only.

Usage:
    python3 analyze_variance.py <db> [--eval tests] [--agents codex claude_code]

By design we restrict to eval_method='tests' (gold-test eval); line_overlap runs
inflate apparent variance (SPEC §1).
"""
import argparse, sqlite3, random, math
from collections import defaultdict

random.seed(0)
N_BOOT = 10000


def load(db, eval_method, agents):
    c = sqlite3.connect(db)
    q = "SELECT agent,task_id,task_passed FROM runs WHERE 1=1"
    args = []
    if eval_method:
        q += " AND eval_method=?"; args.append(eval_method)
    rows = c.execute(q, args).fetchall()
    pool = defaultdict(list)          # (agent,task) -> [0/1,...]
    for ag, t, p in rows:
        if agents and ag not in agents:
            continue
        pool[(ag, t)].append(int(p))
    return pool


def per_agent(pool, ag):
    return {t: v for (a, t), v in pool.items() if a == ag}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)


def m1_per_task(tasks):
    """Metric 1+2: per-task p_i, %nondeterministic, mean Bernoulli var, flip rate."""
    ps = [sum(v)/len(v) for v in tasks.values()]
    nd = sum(1 for p in ps if 0 < p < 1)
    mean_var = sum(p*(1-p) for p in ps)/len(ps)
    flip = 2*mean_var                     # P(two random runs of same task disagree)
    return dict(n_tasks=len(ps), nondeterministic=nd,
                pct_nondet=100*nd/len(ps), mean_bernoulli_var=mean_var,
                flip_rate=flip, rates=ps)


def m3_benchmark_sd_vs_k(tasks, ks=(1, 3, 5, 10, 25)):
    """Metric 3+4: benchmark-score SD and 95% CI width as a function of K.

    For each task we resample K runs WITH replacement from its observed runs,
    score = mean over tasks of (that task's resampled pass-rate), repeat N_BOOT.
    SD across bootstrap = run-to-run wobble of a K-repeat benchmark.
    """
    items = list(tasks.values())
    out = {}
    for K in ks:
        scores = []
        for _ in range(N_BOOT):
            s = 0.0
            for runs in items:
                draws = [random.choice(runs) for _ in range(K)]
                s += sum(draws)/K
            scores.append(s/len(items))
        m = sum(scores)/len(scores)
        sd = math.sqrt(sum((x-m)**2 for x in scores)/len(scores))
        scores.sort()
        lo = scores[int(0.025*len(scores))]; hi = scores[int(0.975*len(scores))]
        out[K] = dict(mean=m, sd=sd, ci_lo=lo, ci_hi=hi, ci_width=hi-lo)
    return out


def m4_repeats_needed(sd_vs_k, targets=(0.05, 0.02)):
    """Smallest K whose 95% CI width <= target."""
    res = {}
    for tgt in targets:
        hit = next((K for K in sorted(sd_vs_k) if sd_vs_k[K]["ci_width"] <= tgt), None)
        res[tgt] = hit
    return res


def m5_ranking_flip(tasks_a, tasks_b, ks=(1, 3, 5, 10)):
    """Metric 5: P(single-run-style ranking disagrees with high-K truth) on shared tasks."""
    shared = sorted(set(tasks_a) & set(tasks_b))
    if not shared:
        return dict(shared=0)
    truth_a = sum(sum(tasks_a[t])/len(tasks_a[t]) for t in shared)/len(shared)
    truth_b = sum(sum(tasks_b[t])/len(tasks_b[t]) for t in shared)/len(shared)
    truth_sign = truth_a > truth_b
    out = dict(shared=len(shared), truth_a=truth_a, truth_b=truth_b)
    for K in ks:
        flips = 0
        for _ in range(N_BOOT):
            sa = sum(sum(random.choice(tasks_a[t]) for _ in range(K))/K for t in shared)/len(shared)
            sb = sum(sum(random.choice(tasks_b[t]) for _ in range(K))/K for t in shared)/len(shared)
            if (sa > sb) != truth_sign:
                flips += 1
        out[K] = flips/N_BOOT
    return out


def m6_agent_var_contrast(tasks_a, tasks_b, name_a, name_b):
    """Metric 6: flip-rate difference between agents w/ bootstrap CI over tasks."""
    def flip(tasks):
        ps = [sum(v)/len(v) for v in tasks.values()]
        return 2*sum(p*(1-p) for p in ps)/len(ps)
    fa, fb = flip(tasks_a), flip(tasks_b)
    # bootstrap over tasks (cluster) for the difference
    la, lb = list(tasks_a.values()), list(tasks_b.values())
    diffs = []
    for _ in range(N_BOOT):
        ra = [random.choice(la) for _ in la]
        rb = [random.choice(lb) for _ in lb]
        diffs.append(2*sum((sum(v)/len(v))*(1-sum(v)/len(v)) for v in ra)/len(ra)
                     - 2*sum((sum(v)/len(v))*(1-sum(v)/len(v)) for v in rb)/len(rb))
    diffs.sort()
    return dict(flip_a=fa, flip_b=fb, diff=fa-fb,
                ci=(diffs[int(0.025*N_BOOT)], diffs[int(0.975*N_BOOT)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--eval", default="tests", help="eval_method filter ('' for all)")
    ap.add_argument("--agents", nargs="*", default=["codex", "claude_code"])
    a = ap.parse_args()

    pool = load(a.db, a.eval or None, a.agents)
    present = [ag for ag in a.agents if any(k[0] == ag for k in pool)]
    print(f"# Reliability metrics — db={a.db} eval={a.eval or 'ALL'} agents={present}\n")

    per = {}
    for ag in present:
        tasks = per_agent(pool, ag)
        per[ag] = tasks
        ks = (sum(len(v) for v in tasks.values()) // max(1, len(tasks)))
        print(f"## {ag}  ({len(tasks)} tasks, ~{ks} runs/task)")
        m1 = m1_per_task(tasks)
        print(f"  [M1] nondeterministic tasks: {m1['nondeterministic']}/{m1['n_tasks']} "
              f"({m1['pct_nondet']:.0f}%) | mean Bernoulli var={m1['mean_bernoulli_var']:.3f}")
        print(f"  [M2] flip rate (P two runs disagree): {100*m1['flip_rate']:.1f}%")
        sd = m3_benchmark_sd_vs_k(tasks)
        print(f"  [M3] benchmark score SD / 95%-CI-width by K:")
        for K in sorted(sd):
            r = sd[K]
            print(f"        K={K:>2}: score={r['mean']*100:5.1f}%  SD={r['sd']*100:4.1f}pp  "
                  f"CIwidth={r['ci_width']*100:4.1f}pp")
        need = m4_repeats_needed(sd)
        print(f"  [M4] repeats needed: CI<=5pp -> K={need[0.05]} | CI<=2pp -> K={need[0.02]}")
        print()

    if len(present) >= 2:
        ag, bg = present[0], present[1]
        print(f"## Cross-agent ({ag} vs {bg}, shared tasks)")
        rf = m5_ranking_flip(per[ag], per[bg])
        if rf["shared"]:
            print(f"  [M5] shared={rf['shared']} | truth {ag}={rf['truth_a']*100:.1f}% "
                  f"{bg}={rf['truth_b']*100:.1f}% | P(ranking flip):")
            for K in (1, 3, 5, 10):
                print(f"        K={K:>2}: {100*rf[K]:.0f}%")
        else:
            print("  [M5] no shared tasks")
        c = m6_agent_var_contrast(per[ag], per[bg], ag, bg)
        print(f"  [M6] flip {ag}={100*c['flip_a']:.1f}% vs {bg}={100*c['flip_b']:.1f}% | "
              f"diff={100*c['diff']:+.1f}pp 95%CI[{100*c['ci'][0]:+.1f},{100*c['ci'][1]:+.1f}]")


if __name__ == "__main__":
    main()
