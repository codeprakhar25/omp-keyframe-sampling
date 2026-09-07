"""TOST equivalence tests for Claim A (compression is free), plus paired CIs.

Claim A is an EQUIVALENCE claim. It was argued from non-significant McNemar tests, which is
"absence of evidence", not "evidence of absence" (PAPER_DRAFT 6). This script supplies the
missing inference: a two-one-sided-tests procedure and an interval on the paired difference.

Paired binary data. With b01 = (compressed right, full wrong) and b10 = (the reverse),
    diff = (b01 - b10) / n
    var  = (b01 + b10 - (b01 - b10)**2 / n) / n**2       # exact paired variance
Equivalence at alpha ⟺ the (1-2*alpha) CI lies entirely inside (-margin, +margin), which is
what TOST tests; we report the 90% CI alongside so the reader can apply their own margin.

MARGIN IS POST HOC. It was chosen after seeing the data (2026-07-29) and must be reported as
such -- a pre-registered margin would be stronger. We report several.
"""
import glob
import json
import math
import os
import sys

# Directory of the compression-arm lmms-eval sample JSONLs (full_*/d53_*/ivl8b_*).
# These arms are NOT part of the release; regenerate them with scripts/lmmseval_run.sh
# at the reduced spatial budgets, then point SLM_CLAIMA at the output directory.
S = os.environ.get("SLM_CLAIMA", "data/predictions")
MARGINS = (0.02, 0.03, 0.04)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lvb(path):
    out = {}
    for r in map(json.loads, open(path)):
        a = r["lvb_acc"]
        out[a["id"]] = int(a["answer"] == a["parsed_pred"])
    return out


def merge(*ds):
    out = {}
    for d in ds:
        out.update(d)
    return out


def tost(comp, full, label):
    """comp = compressed arm, full = full-res arm. Positive diff favours compression."""
    ks = sorted(set(comp) & set(full))
    n = len(ks)
    b01 = sum(1 for k in ks if comp[k] == 1 and full[k] == 0)
    b10 = sum(1 for k in ks if comp[k] == 0 and full[k] == 1)
    m = b01 + b10
    diff = (b01 - b10) / n
    var = (m - (b01 - b10) ** 2 / n) / n ** 2
    se = math.sqrt(var) if var > 0 else 0.0
    # exact two-sided McNemar, for continuity with the rest of the paper
    p_mc = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b01, b10) + 1)) / 2 ** m) if m else 1.0
    lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
    lo95, hi95 = diff - 1.960 * se, diff + 1.960 * se
    print(f"\n{label}  n={n}  disc={b01}/{b10}")
    print(f"  diff = {100*diff:+.2f}pt   McNemar p={p_mc:.4g}")
    print(f"  90% CI [{100*lo90:+.2f}, {100*hi90:+.2f}]   95% CI [{100*lo95:+.2f}, {100*hi95:+.2f}]")
    for d in MARGINS:
        if se == 0:
            verdict = "EQUIVALENT (degenerate: zero discordance)"
            p_t = 0.0
        else:
            p_t = max(1 - _phi((diff + d) / se), 1 - _phi((d - diff) / se))
            verdict = "EQUIVALENT" if p_t < 0.05 else "not shown"
        print(f"  TOST margin +/-{100*d:.0f}pt: p={p_t:.4g}  -> {verdict}")
    return dict(n=n, diff=diff, se=se, lo90=lo90, hi90=hi90)


print("=" * 78)
print("CLAIM A -- EQUIVALENCE TESTS (TOST) + PAIRED CIs")
print("Margins are POST HOC (chosen 2026-07-29, after the runs). Report as such.")
print("=" * 78)

print("\n### Qwen3-VL-8B -- residual-proportional D@53% vs full-res, fixed OMP-8 timestamps")
f36, d36 = lvb(f"{S}/full_3600.jsonl"), lvb(f"{S}/d53_3600.jsonl")
f6 = merge(lvb(f"{S}/full_600_hA.jsonl"), lvb(f"{S}/full_600_hB.jsonl"))
# resprop_600_k8_b53 wrote twice; take whichever shares ids with the full-res arm and is complete
cands = [lvb(f"{S}/d53_600_a.jsonl"), lvb(f"{S}/d53_600_b.jsonl")]
d6 = max(cands, key=lambda d: len(set(d) & set(f6)))
r1 = tost(d36, f36, "LVB 3600s  D@53 vs full")
r2 = tost(d6, f6, "LVB  600s  D@53 vs full")
r3 = tost(merge({f"3600:{k}": v for k, v in d36.items()}, {f"600:{k}": v for k, v in d6.items()}),
          merge({f"3600:{k}": v for k, v in f36.items()}, {f"600:{k}": v for k, v in f6.items()}),
          "LVB pooled  D@53 vs full")

print("\n### InternVL3-8B -- 3x tile cut (24 -> 8 tiles/item) on uniform-8")
i36f, i36c = lvb(f"{S}/ivl8b_unif8_t64_3600.jsonl"), lvb(f"{S}/ivl8b_unif8_t8_3600.jsonl")
i6f, i6c = lvb(f"{S}/ivl8b_unif8_t64_600.jsonl"), lvb(f"{S}/ivl8b_unif8_t8_600.jsonl")
tost(i36c, i36f, "IVL 3600s  8 tiles vs 24")
tost(i6c, i6f, "IVL  600s  8 tiles vs 24")
tost(merge({f"3600:{k}": v for k, v in i36c.items()}, {f"600:{k}": v for k, v in i6c.items()}),
     merge({f"3600:{k}": v for k, v in i36f.items()}, {f"600:{k}": v for k, v in i6f.items()}),
     "IVL pooled  8 tiles vs 24")
