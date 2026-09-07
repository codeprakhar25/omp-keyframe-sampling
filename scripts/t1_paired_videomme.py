import json, glob, os
from math import comb

# Directory of per-arm lmms-eval sample JSONLs. The uniform / topk / omp arms this
# script needs are in the release under data/predictions/, where filenames carry an
# lmms-eval timestamp prefix (20260722_155022_samples_videomme_uniform_k8_short.jsonl).
SP = os.environ.get("SLM_PREDS", "data/predictions")


def score(r):
    s = r["videomme_perception_score"]
    if isinstance(s, dict):                       # {'question_id':..,'answer':..,'pred_answer':..}
        return int(s.get("answer") == s.get("pred_answer"))
    return int(bool(s))


def load(*paths):
    """Merge one or more wave/shard files into id -> (correct, target).
    Keyed on doc_hash: Video-MME long was run in waves, so doc_id is wave-local."""
    d = {}
    total = 0
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            d[r["videomme_perception_score"]["question_id"]] = (score(r), json.dumps(r["target"]))
    return d, total


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    b01 = sum(1 for i in ids if a[i][0] == 0 and b[i][0] == 1)
    b10 = sum(1 for i in ids if a[i][0] == 1 and b[i][0] == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0, len(ids)
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), len(ids)


BINS = ["short", "medium", "long"]
SRC = {
    "uniform": {b: [f"{SP}/uniform_{b}.jsonl"] for b in BINS},
    "topk":    {b: [f"{SP}/topk_{b}.jsonl"] for b in BINS},
    "omp":     {"short": [f"{SP}/omp_short.jsonl"], "medium": [f"{SP}/omp_medium.jsonl"],
                "long": sorted(glob.glob(f"{SP}/omp_long_*.jsonl"))},
    "aks":     {b: [f"{SP}/aks_{b}.jsonl"] for b in BINS},
    "focus":   {"short": [f"{SP}/focus_short.jsonl"], "medium": [f"{SP}/focus_medium.jsonl"],
                "long": sorted(glob.glob(f"{SP}/focus_long_h*.jsonl"))},
    "dppmm":   {b: [f"{SP}/dppmm_{b}.jsonl"] for b in BINS},
}

A = {}
print("coverage gate (Video-MME: 900 unique per bin, 2700 total)")
for m, per_bin in SRC.items():
    merged, ok = {}, True
    row = []
    for b in BINS:
        d, total = load(*per_bin[b])
        row.append(f"{b}={len(d)}({total}ln)")
        if len(d) != 900:
            ok = False
        merged.update(d)
    A[m] = merged
    acc = sum(v[0] for v in merged.values()) / len(merged)
    print(f"  {m:8s} {' '.join(row):44s} total={len(merged):5d} acc={acc:.4f} "
          f"{'OK' if ok and len(merged) == 2700 else 'CHECK'}")

ref = A["uniform"]
print("\nquestion_id alignment (gold target must match uniform)")
for m in list(A)[1:]:
    ids = set(ref) & set(A[m])
    bad = sum(1 for i in ids if ref[i][1] != A[m][i][1])
    print(f"  {m:8s} shared={len(ids)} mismatches={bad} "
          f"{'ALIGNED' if bad == 0 and len(ids) == 2700 else 'CHECK'}")


def report(base, others, label):
    print(f"\n{label}")
    for m in others:
        b01, b10, p, n = mcnemar(A[base], A[m])
        d = 100 * (sum(v[0] for v in A[m].values()) / len(A[m])
                   - sum(v[0] for v in A[base].values()) / len(A[base]))
        print(f"  {m:8s} vs {base:8s} d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  n={n}")


report("uniform", ["topk", "omp", "aks", "focus", "dppmm"],
       "T1 paired tests vs uniform (McNemar, exact two-sided binomial)")
report("omp", ["topk", "aks", "focus", "dppmm"],
       "T1 paired tests vs OMP (honest strong baseline)")

print("\nper-bin, OMP vs uniform")
for b in BINS:
    u, _ = load(*SRC["uniform"][b])
    o, _ = load(*SRC["omp"][b])
    b01, b10, p, n = mcnemar(u, o)
    d = 100 * (sum(v[0] for v in o.values()) / len(o) - sum(v[0] for v in u.values()) / len(u))
    print(f"  {b:7s} d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  n={n}")
