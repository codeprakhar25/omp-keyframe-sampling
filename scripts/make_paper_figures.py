#!/usr/bin/env python3
"""Draw Fig. 1-4 to the specs in PAPER_DRAFT.md §5.9.

Every number here is transcribed from the draft's own tables (§5.4-§5.8) -- this
script plots banked results, it does not compute anything. Source table is named
in a comment above each block so any figure can be traced back and re-checked.

Fig 1a caveat, stated in the caption too: the residual trace was banked at picks
1, 8 and 16 only (§5.4 Finding 1). Those three are drawn as measured markers with
a dashed connector; the connector is not measured data. The pick-5 annotation
comes from Finding 2 ("cos_resid reaches ~0 by the fifth pick").

Usage: python3 make_paper_figures.py [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# One palette, colour-blind safe (Okabe-Ito), used across all four figures.
C = {
    "uniform": "#999999",
    "topk":    "#0072B2",
    "omp":     "#D55E00",
    "lddr":    "#009E73",
    "aks":     "#CC79A7",
    "focus":   "#E69F00",
    "accent":  "#56B4E9",
}
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
})


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------
# Fig. 1 -- why the residual is inert.   Source: §5.4 Findings 1 and 2.
# --------------------------------------------------------------------------
def fig1(outdir):
    picks = [1, 8, 16]
    resid_frac = [0.972, 0.967, 0.967]          # §5.4 Finding 1, row 1
    cos_resid = [0.233, 0.004, 0.000]           # §5.4 Finding 1, row 2

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.1))
    fig.subplots_adjust(wspace=0.62)

    # --- 1a ---
    ax.plot(picks, cos_resid, "o-", color=C["omp"], lw=1.6, ms=5,
            label=r"$\cos(q_{r-1},\, e_{b_r})$  (left)")
    ax.set_xlabel("pick index")
    ax.set_ylabel(r"$\cos(q_{\mathrm{resid}},\, e)$", color=C["omp"])
    ax.tick_params(axis="y", labelcolor=C["omp"])
    ax.set_ylim(-0.02, 0.26)
    ax.set_xticks([1, 5, 8, 12, 16])

    ax.axvline(5, color="0.4", ls=":", lw=1)
    ax.annotate("criterion inert\nby pick ~5", xy=(5, 0.055), xytext=(7.0, 0.085),
                fontsize=7.5, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    axr = ax.twinx()
    axr.spines["top"].set_visible(False)
    # Range fixed at 0.95-1.00 by the spec: the flatness IS the result.
    axr.plot(picks, resid_frac, "s--", color=C["topk"], lw=1.6, ms=5,
             label=r"$\|q_r\|/\|q_0\|$  (right)")
    axr.set_ylim(0.95, 1.00)
    axr.set_ylabel(r"residual fraction $\|q_r\|/\|q_0\|$", color=C["topk"])
    axr.tick_params(axis="y", labelcolor=C["topk"])
    axr.grid(False)
    axr.annotate("flat: OMP never removes\nmore than 3.3% of the query",
                 xy=(13.5, 0.9668), xytext=(7.6, 0.9525), fontsize=7.5, color=C["topk"],
                 arrowprops=dict(arrowstyle="->", color=C["topk"], lw=0.8))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper center",
              bbox_to_anchor=(0.55, 0.99), framealpha=0.95)
    ax.set_title("(a) criterion dies, residual never moves", fontsize=8.5, loc="left")

    # --- 1b ---  Source: §5.4 Finding 1b (embedding geometry), n=411 videos.
    # Two summarised distributions: only mean/std/max are banked, so these are
    # drawn as mean +/- std bars with the max marked -- NOT invented histograms.
    ax2.barh([1], [0.714], height=0.42, color=C["accent"], edgecolor="none",
             label=r"within-video $\cos(e_i,e_j)$  (mean 0.714)")
    ax2.barh([0], [0.1747], height=0.42, color=C["omp"], edgecolor="none",
             xerr=[[0.0215], [0.0215]], error_kw=dict(ecolor="0.3", lw=1, capsize=3),
             label=r"cross-modal $\cos(q,e)$  (mean 0.175, sd 0.022)")
    ax2.plot([0.2313], [0], "|", color="0.15", ms=12, mew=1.6)
    ax2.annotate("max 0.231\n← 0.06 = all selection signal",
                 xy=(0.2313, 0.0), xytext=(0.33, 0.42), fontsize=7.5, color="0.2",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["query\nvs frames", "frame\nvs frame"], fontsize=7.5)
    ax2.tick_params(axis="y", pad=6)
    ax2.set_xlim(0, 0.85)
    ax2.set_xlabel("cosine similarity")
    ax2.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.28),
               framealpha=0.9)
    ax2.set_title("(b) coherent dictionary, orthogonal query ($n{=}411$)",
                  fontsize=8.5, loc="left")

    fig.suptitle("The dictionary is coherent and the query is near-orthogonal to it, "
                 "so pick 1 removes 85.6% of all query mass OMP ever removes",
                 fontsize=8, y=1.10)
    save(fig, outdir, "fig1_residual_inert")


# --------------------------------------------------------------------------
# Fig. 2 -- selection is a long-video effect.  Source: §5.5 LongVideoBench table.
# --------------------------------------------------------------------------
def fig2(outdir):
    bins = ["15 s\n$n$=189", "60 s\n$n$=172", "600 s\n$n$=412", "3600 s\n$n$=564"]
    series = [
        ("uniform",     [.7249, .7267, .5534, .4716], C["uniform"]),
        ("top-$k$",     [.7249, .7442, .6141, .5106], C["topk"]),
        ("OMP",         [.7249, .7384, .6311, .5461], C["omp"]),
        ("LDDR-select", [.7196, .7558, .6286, .5674], C["lddr"]),
    ]
    # OMP-vs-uniform significance only (§5.6 T1 paired tests).
    sig = ["n.s.", "n.s.", "$p$=.0022", r"$p$=5.9$\times$10$^{-4}$"]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    w, x = 0.2, range(len(bins))
    for i, (lab, vals, col) in enumerate(series):
        off = (i - 1.5) * w
        ax.bar([xi + off for xi in x], vals, width=w, label=lab, color=col,
               edgecolor="none")

    for i, s in enumerate(sig):
        top = max(series[0][1][i], series[2][1][i])
        y = top + 0.022
        ax.plot([i - 1.5 * w, i - 1.5 * w, i + 0.5 * w, i + 0.5 * w],
                [y, y + 0.008, y + 0.008, y], color="0.3", lw=0.8)
        ax.text(i - 0.5 * w, y + 0.012, s, ha="center", fontsize=7,
                color="0.15" if "p" in s else "0.45")

    ax.annotate("all three identical (.7249)\nno selection problem to solve",
                xy=(0, 0.7249), xytext=(-0.34, 0.60), fontsize=7.5, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    ax.set_xticks(list(x))
    ax.set_xticklabels(bins)
    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel("LongVideoBench accuracy")
    ax.set_xlabel("duration bin")
    ax.legend(fontsize=7.5, ncol=4, loc="upper right", framealpha=0.9)
    ax.set_title("The selection gain is entirely a long-video effect "
                 "(brackets: OMP vs uniform)", fontsize=8.5, loc="left")
    save(fig, outdir, "fig2_long_video_effect")


# --------------------------------------------------------------------------
# Fig. 3 -- budget allocation.  Source: §5.6 flat-curve table + Claim B table.
# --------------------------------------------------------------------------
def fig3(outdir):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.2),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(wspace=0.34)

    # --- 3a: 3600 s, fixed OMP-8 picks, budget 32 -> 100% ---
    pts = [(100, .5461, "full-res"), (60, .5496, "D@60"), (53, .5532, "D@53"),
           (50, .5443, "D@50"), (40, .5479, "D@40"), (32, .5443, "E1")]
    # restier is also at 53% (.5426) -- plotted, but unlabelled to avoid collision.
    pts_extra = [(53, .5426)]
    unif50 = (50, .5479)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ax.plot([xs[i] for i in order], [ys[i] for i in order], "o-",
            color=C["omp"], lw=1.5, ms=5, zorder=3, label="D (residual-proportional)")
    ax.plot([p[0] for p in pts_extra], [p[1] for p in pts_extra], "o",
            color=C["omp"], ms=5, alpha=0.45, zorder=3)

    # 95% band for the full-res reference; every point falls inside it.
    lo, hi = .5461 - .0413, .5461 + .0413          # +/- 1.96*se, se=sqrt(p(1-p)/564)
    ax.axhspan(lo, hi, color=C["uniform"], alpha=0.18, zorder=0,
               label="full-res 95% interval")
    ax.axhline(.5461, color="0.35", ls="--", lw=1, zorder=1)

    ax.plot([unif50[0]], [unif50[1]], "D", color=C["lddr"], ms=7, zorder=4,
            label="uniform50 (flat split control)")
    ax.annotate("flat split lands on D@50 —\nour D rule ≈ a uniform half",
                xy=(50, .5479), xytext=(88, .5265), fontsize=7.5, color="0.2",
                ha="left", arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    for bx, by, lab in pts:
        dy = -13 if lab in ("D@50", "E1") else 7
        ax.annotate(lab, (bx, by), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=6.5, color="0.35")

    ax.set_xlabel("visual budget retained (%)")
    ax.set_ylabel("accuracy, 3600 s ($n$=564)")
    ax.set_xlim(26, 108)
    ax.invert_xaxis()
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
    ax.set_title("(a) fixed timestamps: curve is flat to a ~68% token cut", fontsize=8.5, loc="left")

    # --- 3b: isoquant, pooled n=976 ---
    labs = ["OMP-8\nfull res", "OMP-16\n@ 50% res"]
    vals = [.5820, .6055]
    ax2.bar(labs, vals, width=0.55, color=[C["uniform"], C["omp"]], edgecolor="none")
    ax2.set_ylim(0.50, 0.66)
    ax2.set_ylabel("pooled accuracy ($n$=976)")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    y = 0.625
    ax2.plot([0, 0, 1, 1], [y, y + 0.006, y + 0.006, y], color="0.3", lw=0.9)
    ax2.text(0.5, y + 0.010, "+2.36 pp,  $p$=.0346", ha="center", fontsize=7.5)
    ax2.set_xlabel("measured token ratio 0.984–0.996\n(the reinvest arm costs no more)",
                   fontsize=7, color="0.25", labelpad=6)
    ax2.set_title("(b) equal budget: more frames wins", fontsize=8.5, loc="left")

    fig.suptitle("Token cut on fixed frames (a); equal-token reinvest (b)",
                 fontsize=8, y=1.02)
    save(fig, outdir, "fig3_budget_allocation")

    # Claim A uses panel (a) alone so (b) does not spoil Claim B.
    fig_a, ax_a = plt.subplots(figsize=(5.6, 3.05))
    ax_a.plot([xs[i] for i in order], [ys[i] for i in order], "o-",
              color=C["omp"], lw=1.5, ms=5, zorder=3, label="D (residual-proportional)")
    ax_a.plot([p[0] for p in pts_extra], [p[1] for p in pts_extra], "o",
              color=C["omp"], ms=5, alpha=0.45, zorder=3)
    ax_a.axhspan(lo, hi, color=C["uniform"], alpha=0.18, zorder=0,
                 label="full-res 95% interval")
    ax_a.axhline(.5461, color="0.35", ls="--", lw=1, zorder=1)
    ax_a.plot([unif50[0]], [unif50[1]], "D", color=C["lddr"], ms=7, zorder=4,
              label="uniform50 (flat split)")
    ax_a.annotate("flat split ≈ D@50",
                  xy=(50, .5479), xytext=(82, .527), fontsize=7.5, color="0.2",
                  ha="left", arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    for bx, by, lab in pts:
        dy = -13 if lab in ("D@50", "E1") else 7
        ax_a.annotate(lab, (bx, by), textcoords="offset points", xytext=(0, dy),
                      ha="center", fontsize=6.5, color="0.35")
    ax_a.set_xlabel("visual budget retained (%)")
    ax_a.set_ylabel("accuracy, 3600 s ($n$=564)")
    ax_a.set_xlim(26, 108)
    ax_a.invert_xaxis()
    ax_a.legend(fontsize=6.6, loc="lower left", framealpha=0.9)
    ax_a.set_title("Fixed OMP-8 timestamps: accuracy stays in the full-res band",
                   fontsize=8.5, loc="left")
    save(fig_a, outdir, "fig_claim_a")

    fig_b, ax_b = plt.subplots(figsize=(3.6, 3.05))
    labs = ["OMP-8\nfull res", "OMP-16\n@ 50% res"]
    vals = [.5820, .6055]
    ax_b.bar(labs, vals, width=0.55, color=[C["uniform"], C["omp"]], edgecolor="none")
    ax_b.set_ylim(0.50, 0.66)
    ax_b.set_ylabel("pooled LongVideoBench ($n$=976)")
    for i, v in enumerate(vals):
        ax_b.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    y = 0.625
    ax_b.plot([0, 0, 1, 1], [y, y + 0.006, y + 0.006, y], color="0.3", lw=0.9)
    ax_b.text(0.5, y + 0.010, "+2.36 pp,  $p$=.0346", ha="center", fontsize=7.5)
    ax_b.set_xlabel("token ratio 0.984–0.996 (reinvest costs no more)",
                    fontsize=7, color="0.25", labelpad=6)
    ax_b.set_title("Equal token budget: more frames win", fontsize=8.5, loc="left")
    save(fig_b, outdir, "fig_claim_b")


# --------------------------------------------------------------------------
# Fig. 4 (appendix) -- the negative sweep.  Source: §5.7 table + §5.4 negatives.
# --------------------------------------------------------------------------
def fig4(outdir):
    # (label, delta_600, delta_3600); None = not run at that bin.
    rows = [
        ("uniform (floor)",        -7.77, -7.45),
        ("MMR (diversity-first)",  -3.64, -4.96),
        ("iter-$\\alpha$",         -2.43, -1.77),
        ("rfloor 0.15",            -1.22, -0.35),
        ("$\\alpha$-orth 0.5",     -0.97, +1.24),
        ("rfloor 0.33",            -0.73, +0.35),
        ("$\\alpha$-orth 0.75",    -0.49, +0.35),
        ("plateau-stop $p$2",       None, -3.37),
        ("plateau-stop $p$3",       None, -3.01),
        ("adaptive-$k$",            None, -2.13),
        ("DPP (log-det MAP)",      +1.94, +2.13),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ys = range(len(rows))
    for y, (lab, d6, d36) in zip(ys, rows):
        ax.plot([-8.4, 3.0], [y, y], color="0.92", lw=0.8, zorder=0)
        if d6 is not None:
            ax.plot(d6, y, "o", color=C["topk"], ms=6, zorder=3)
        if d36 is not None:
            ax.plot(d36, y, "s", color=C["omp"], ms=5.5, zorder=3)

    ax.axvline(0, color="0.25", lw=1.2, zorder=2)
    ax.text(0.12, len(rows) - 0.35, "OMP", fontsize=7.5, color="0.2")
    ax.axvspan(-1.5, 1.5, color=C["uniform"], alpha=0.13, zorder=1)
    ax.text(-8.5, -1.25, "shaded: ±1.5 pp band", ha="left", fontsize=7, color="0.35")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel(r"$\Delta$ accuracy vs OMP (pp)")
    ax.set_xlim(-8.6, 3.2)
    ax.set_ylim(-1.6, len(rows) - 0.2)
    ax.legend(handles=[
        Patch(color=C["topk"], label="600 s ($n$=412)"),
        Patch(color=C["omp"], label="3600 s ($n$=564)"),
    ], fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("Most orthogonalization / diversity knobs sit a few points from OMP;\n"
                 "MMR and query-blind DPP lose; no variant significantly beats OMP",
                 fontsize=8.5, loc="left")
    save(fig, outdir, "fig4_negative_sweep")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"drawing into {a.outdir}/")
    fig1(a.outdir)
    fig2(a.outdir)
    fig3(a.outdir)
    fig4(a.outdir)
    print("done -- 4 figures, pdf + png each")
