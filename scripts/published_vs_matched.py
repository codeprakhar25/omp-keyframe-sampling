#!/usr/bin/env python3
"""Published selector gains vs the same rules measured under one frozen scorer.

Purpose. Every published selector reports a gain over uniform sampling, but each
does so with its own scorer, frame budget, and answerer. This script puts those
numbers beside our matched-condition measurements so the reader can see how many
axes move between any two published rows.

WHAT THIS IS NOT. It is not a reproduction study and no row here is evidence that a
published result "fails to replicate". A different scorer means a different method:
AKS and FOCUS score frames with BLIP-ITM, and running their subset rule on LongCLIP
scores measures the SUBSET RULE's contribution, not their pipeline. FOCUS in
particular is represented by FOCUS*, a replay of its clip schedule on dense LongCLIP
scores -- its published method treats scoring as a budgeted online process, which we
deliberately do not reproduce.

Every published figure below was read from the source paper (verbatim table pull),
not from secondary notes. Re-verify before editing: a previous claim about LDDR in
this project's notes turned out to be false.
"""

# ---------------------------------------------------------------- published rows
# gain = method minus the uniform-sampling row in the SAME paper, same answerer/budget.
PUBLISHED = [
    # (method, paper, arxiv, scorer, frames, answerer, benchmark, uniform, method_acc)
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 32, "Qwen2-VL-7B",
     "LongVideoBench", 55.5, 60.5),
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 32, "LLaVA-OV-7B",
     "LongVideoBench", 54.8, 59.3),
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 64, "LLaVA-Video-7B",
     "LongVideoBench", 58.9, 62.7),
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 32, "Qwen2-VL-7B",
     "Video-MME", 57.6, 59.9),
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 32, "LLaVA-OV-7B",
     "Video-MME", 56.5, 58.4),
    ("AKS", "Tang et al. CVPR 2025", "2502.21271", "BLIP-ITM", 64, "LLaVA-Video-7B",
     "Video-MME", 64.4, 65.3),

    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 32, "GPT-4o",
     "LongVideoBench", 51.6, 54.8),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 32, "Qwen2-VL-7B",
     "LongVideoBench", 55.6, 62.3),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 32, "LLaVA-OV-7B",
     "LongVideoBench", 54.8, 60.7),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 64, "LLaVA-Video-7B",
     "LongVideoBench", 58.9, 63.5),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 32, "Qwen2-VL-7B",
     "Video-MME", 57.6, 59.7),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 32, "LLaVA-OV-7B",
     "Video-MME", 56.5, 58.3),
    ("FOCUS", "Zhu et al. 2025", "2510.27280", "BLIP-ITM", 64, "LLaVA-Video-7B",
     "Video-MME", 64.4, 65.4),

    # LDDR already standardizes every baseline to LongCLIP, so its scorer matches ours.
    # Verified: 60.56/64.48 sit in the Video-MME "Overall" sub-column, Qwen2.5-VL-7B, 32F.
    ("LDDR", "Chen et al. 2026", "2605.11477", "LongCLIP", 32, "Qwen2.5-VL-7B",
     "Video-MME", 60.56, 64.48),
]

# ------------------------------------------------------- our matched-scorer rows
# One frozen LongCLIP stem-only scorer, k=8, Qwen3-VL-8B, subtitles off, bs=1.
OURS = {
    "LongVideoBench": {"uniform": .5654, "top-k": .6028, "AKS": .6021,
                       "FOCUS*": .5819, "OMP": .6223, "LDDR-select": .6320},
    "Video-MME": {"uniform": .5637, "top-k": .5704, "AKS": .5785,
                  "FOCUS*": .5578, "OMP": .6222, "LDDR-select": .6193},
    "LVBench": {"uniform": .3454, "top-k": .4319, "AKS": .4280,
                "FOCUS*": .3983, "OMP": .4635, "LDDR-select": .4693},
}
MATCH = {"AKS": "AKS", "FOCUS": "FOCUS*", "LDDR": "LDDR-select"}


def main():
    print("=" * 100)
    print("PUBLISHED gains over uniform (each paper's own scorer / budget / answerer)")
    print("=" * 100)
    print(f"{'method':<7} {'bench':<16} {'scorer':<10} {'#F':>3} {'answerer':<16} "
          f"{'unif':>6} {'method':>7} {'gain':>7}")
    for m, _, aid, sc, f, ans, b, u, a in PUBLISHED:
        print(f"{m:<7} {b:<16} {sc:<10} {f:>3} {ans:<16} {u:>6.2f} {a:>7.2f} {a-u:>+7.2f}")

    print()
    print("=" * 100)
    print("OURS -- one frozen LongCLIP stem scorer, k=8, Qwen3-VL-8B (gain over uniform)")
    print("=" * 100)
    print(f"{'method':<13} " + " ".join(f"{b:>16}" for b in OURS))
    for meth in ("top-k", "AKS", "FOCUS*", "OMP", "LDDR-select"):
        cells = []
        for b in OURS:
            cells.append(f"{(OURS[b][meth]-OURS[b]['uniform'])*100:>+16.2f}")
        print(f"{meth:<13} " + " ".join(cells))

    print()
    print("=" * 100)
    print("SIDE BY SIDE -- published range vs matched, and which axes moved")
    print("=" * 100)
    print(f"{'method':<7} {'bench':<16} {'published gain':>22} {'matched gain':>13}   axes changed")
    for meth in ("AKS", "FOCUS", "LDDR"):
        for b in ("LongVideoBench", "Video-MME"):
            rows = [(a - u, sc, f, ans) for m, _, _, sc, f, ans, bb, u, a in PUBLISHED
                    if m == meth and bb == b]
            if not rows:
                continue
            gains = [r[0] for r in rows]
            scorers = {r[1] for r in rows}
            budgets = {r[2] for r in rows}
            rng = (f"{min(gains):+.2f}" if len(gains) == 1
                   else f"{min(gains):+.2f} to {max(gains):+.2f}")
            ours = (OURS[b][MATCH[meth]] - OURS[b]["uniform"]) * 100
            axes = []
            if scorers != {"LongCLIP"}:
                axes.append(f"scorer {'/'.join(scorers)}->LongCLIP")
            axes.append(f"budget {'/'.join(str(x) for x in sorted(budgets))}F->8F")
            axes.append("answerer->Qwen3-VL-8B")
            print(f"{meth:<7} {b:<16} {rng:>22} {ours:>+13.2f}   " + "; ".join(axes))

    print()
    print("Reading:")
    print("  * AKS transfers: published +3.8..+5.0 (LVB) vs +3.67 matched, despite a")
    print("    BLIP-ITM -> LongCLIP swap AND a 32F -> 8F budget cut.")
    print("  * FOCUS* is the SCHEDULE ONLY. Its smaller number is not a failed")
    print("    reproduction -- it isolates how much the clip schedule contributes once")
    print("    the budgeted ITM scorer, which is the method's core, is removed.")
    print("  * LDDR-select EXCEEDS its published gain (+5.56 vs +3.92 on Video-MME)")
    print("    using stage 1 only at a quarter of the frames. LDDR is also the one")
    print("    baseline that already standardized on LongCLIP, so no scorer moved.")
    print("  * No two published rows share scorer, budget, AND answerer. That is the")
    print("    point: cross-paper selector numbers are not comparable, which is why the")
    print("    matched column exists.")
    print()
    print("LIMIT OF THIS TABLE -- state it before a reviewer does:")
    print("  The matched column differs from every published column on THREE axes at")
    print("  once (scorer, budget, answerer). So a published-vs-matched difference")
    print("  CANNOT be attributed to the scorer alone -- this comparison is exactly as")
    print("  confounded as the cross-paper comparisons it critiques. What is controlled")
    print("  is the matched column INTERNALLY: those five rows share one scorer, one")
    print("  budget, one answerer, one prompt boundary, so differences WITHIN that")
    print("  column are attributable to the selection rule. The published columns are")
    print("  context for how far apart the literature's conditions are, not a baseline")
    print("  we claim to beat.")
    print("  The scorer axis on its own is isolated separately, by the LongCLIP-vs-SigLIP")
    print("  ablation (RUN_SIGLIP_SCORER.md), where budget and answerer are held fixed.")


if __name__ == "__main__":
    main()
