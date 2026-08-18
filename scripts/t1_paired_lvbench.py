import json
from math import comb

SP = "/tmp/claude-1000/-home-prakh-ml-resarch-slm-lab/ddbad3fe-64e6-4e39-9994-e64b82be2ddf/scratchpad/lvb"
ARMS = ["uniform", "topk", "omp", "aks", "focus", "dppmm", "omp_k16"]


def load(name):
    """doc_id -> (correct, target). doc_id is the lmms-eval index; every arm here is a
    full unsharded 1549-item run, so indices align. Target is carried so we can prove it."""
    d = {}
    n = 0
    for line in open(f"{SP}/{name}.jsonl"):
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        d[r["doc_id"]] = (int(bool(r["lvbench_score"])), r["target"])
    return d, n


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    b01 = sum(1 for i in ids if a[i][0] == 0 and b[i][0] == 1)
    b10 = sum(1 for i in ids if a[i][0] == 1 and b[i][0] == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0, len(ids)
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), len(ids)


A = {}
print("coverage gate (LVBench, n should be 1549 unique in every arm)")
for m in ARMS:
    d, n = load(m)
    A[m] = d
    acc = sum(v[0] for v in d.values()) / len(d)
    print(f"  {m:8s} lines={n:5d} unique={len(d):5d} acc={acc:.4f} "
          f"{'OK' if n == len(d) == 1549 else 'CHECK'}")

# prove doc_id alignment: the gold answer must agree across arms item-by-item
ref = A["uniform"]
print("\ndoc_id alignment (gold target must match uniform on every shared id)")
for m in ARMS[1:]:
    ids = set(ref) & set(A[m])
    bad = sum(1 for i in ids if ref[i][1] != A[m][i][1])
    print(f"  {m:8s} shared={len(ids)} target-mismatches={bad} "
          f"{'ALIGNED' if bad == 0 else 'MISALIGNED — do not pair'}")

print("\nT1 paired tests vs uniform (McNemar, exact two-sided binomial)")
for m in ["topk", "omp", "aks", "focus", "dppmm"]:
    b01, b10, p, n = mcnemar(A["uniform"], A[m])
    d = 100 * (sum(v[0] for v in A[m].values()) / len(A[m])
               - sum(v[0] for v in ref.values()) / len(ref))
    print(f"  {m:8s} vs uniform  d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  n={n}")

print("\nT1 paired tests vs OMP (the honest strong baseline)")
for m in ["topk", "aks", "focus", "dppmm"]:
    b01, b10, p, n = mcnemar(A["omp"], A[m])
    d = 100 * (sum(v[0] for v in A[m].values()) / len(A[m])
               - sum(v[0] for v in A["omp"].values()) / len(A["omp"]))
    print(f"  {m:8s} vs omp      d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  n={n}")

print("\nBudget: OMP-16 vs OMP-8")
b01, b10, p, n = mcnemar(A["omp"], A["omp_k16"])
d = 100 * (sum(v[0] for v in A["omp_k16"].values()) / len(A["omp_k16"])
           - sum(v[0] for v in A["omp"].values()) / len(A["omp"]))
print(f"  omp_k16 vs omp       d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  n={n}")
