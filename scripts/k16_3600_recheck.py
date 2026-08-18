import json
from math import comb

SP = "/tmp/claude-1000/-home-prakh-ml-resarch-slm-lab/ddbad3fe-64e6-4e39-9994-e64b82be2ddf/scratchpad"


def load(path):
    d, n = {}, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        n += 1
        a = json.loads(line)["lvb_acc"]
        d[a["id"]] = int(a["answer"] == a["parsed_pred"])
    return d, n


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    b01 = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    b10 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0, len(ids)
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), len(ids)


tasks = {"uniform": "i", "topk": "picks_lc", "omp": "picks_omp_lc"}

print("shard-level coverage (the 2026-07-19 sharding bug = shards ran the SAME subset)")
for name, t in tasks.items():
    a, na = load(f"{SP}/az2/hA_{t}.jsonl")
    b, nb = load(f"{SP}/az2/hB_{t}.jsonl")
    ov = len(set(a) & set(b))
    print(f"  {name:8s} hA {na:3d}ln/{len(a):3d}uniq  hB {nb:3d}ln/{len(b):3d}uniq  "
          f"overlap={ov}  union={len(set(a) | set(b))}")

print()
print("hA+hB union vs the merged file already on Azure")
arms_union, arms_merged = {}, {}
mf = {"uniform": "val_i", "topk": "val_picks_lc", "omp": "val_picks_omp_lc"}
for name, t in tasks.items():
    a, _ = load(f"{SP}/az2/hA_{t}.jsonl")
    b, _ = load(f"{SP}/az2/hB_{t}.jsonl")
    u = dict(a)
    u.update(b)
    arms_union[name] = u
    m, _ = load(f"{SP}/az/lmmseval_matrix_clean_merged_samples_longvideobench_"
                f"{mf[name]}_3600s_k16.jsonl")
    arms_merged[name] = m
    agree = sum(1 for i in set(u) & set(m) if u[i] == m[i])
    print(f"  {name:8s} union n={len(u)} acc={sum(u.values())/len(u):.4f}   "
          f"merged n={len(m)} acc={sum(m.values())/len(m):.4f}   "
          f"per-item agree {agree}/{len(set(u) & set(m))}")

print()
for label, arms in [("UNION(hA,hB)", arms_union), ("MERGED-file", arms_merged)]:
    print(label)
    for x, y in [("uniform", "topk"), ("uniform", "omp"), ("topk", "omp")]:
        b01, b10, p, n = mcnemar(arms[x], arms[y])
        d = 100 * (sum(arms[y].values()) / len(arms[y]) - sum(arms[x].values()) / len(arms[x]))
        print(f"  {y:>4s} vs {x:<8s} d={d:+6.2f}pp  disc {b10}v{b01}  p={p:.4g}  n={n}")
    print()
