#!/usr/bin/env python3
"""Protocol schematic for the paper (Fig. 1, Method / ~page 2).

Not a residual plot. Two rows:
  (a) score once — 1 fps pool + LongCLIP stem; every selector shares those scores
  (b) spend the budget — k=8 full-res vs k=16 at ~50%, matched tokens

Palette matches slm-lab/scripts/make_paper_figures.py (Okabe–Ito).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

C = {
    "uniform": "#999999",
    "topk": "#0072B2",
    "omp": "#D55E00",
    "lddr": "#009E73",
    "aks": "#CC79A7",
    "focus": "#E69F00",
    "accent": "#56B4E9",
    "ink": "#222222",
    "mute": "#555555",
}
plt.rcParams.update({
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, x, y, w, h, fc, text, *, ec="#222222", lw=0.8, fs=8, tc="#222222"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.4,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)
    return p


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=9, lw=0.9, color=C["ink"],
        shrinkA=0, shrinkB=0,
    ))


def frames_row(ax, x0, y0, n, size, gap, color, alpha=0.9):
    """Draw n square frames; returns (right_x, center_y)."""
    x = x0
    for i in range(n):
        ax.add_patch(Rectangle(
            (x, y0), size, size,
            facecolor=color, edgecolor="#222222", linewidth=0.5, alpha=alpha,
        ))
        x += size + gap
    return x - gap, y0 + size / 2


def draw(outdir):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 3.55),
                                     gridspec_kw={"height_ratios": [1.05, 1.0]})
    fig.subplots_adjust(hspace=0.38, left=0.04, right=0.98, top=0.90, bottom=0.06)

    # ----- (a) score once -----
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 2.35)
    ax_a.axis("off")
    ax_a.set_title("(a)  Score once. Every rule reads the same LongCLIP stem vector.",
                   loc="left", fontsize=9, color=C["ink"], pad=2)

    box(ax_a, 0.15, 0.85, 1.35, 0.85, "#F4F4F4", "long video\n1 fps pool", fs=7.5)
    arrow(ax_a, 1.55, 1.27, 1.95, 1.27)
    box(ax_a, 2.00, 0.85, 1.70, 0.85, C["accent"] + "33", "LongCLIP\nstem only", fs=7.5)
    ax_a.text(2.85, 0.62, "options never scored", ha="center", fontsize=6.5, color=C["mute"])
    arrow(ax_a, 3.75, 1.27, 4.15, 1.27)
    box(ax_a, 4.20, 0.85, 1.55, 0.85, "#F4F4F4", "score vector\n(cached)", fs=7.5)
    arrow(ax_a, 5.80, 1.27, 6.20, 1.27)

    chips = [
        (6.30, "uniform", C["uniform"]),
        (7.10, "top-$k$", C["topk"]),
        (7.90, "AKS", C["aks"]),
        (8.55, r"FOCUS$^\star$", C["focus"]),
        (9.20, "OMP", C["omp"]),
        (9.85, "LDDR-sel.", C["lddr"]),
    ]
    # overlap-avoid: vertical stack of chips on the right
    labels = ["uniform", r"top-$k$", "AKS", r"FOCUS$^\star$", "OMP (ref.)", "LDDR-select"]
    colors = [C["uniform"], C["topk"], C["aks"], C["focus"], C["omp"], C["lddr"]]
    y0 = 0.18
    for i, (lab, col) in enumerate(zip(labels, colors)):
        yy = 1.95 - i * 0.32
        box(ax_a, 6.45, yy, 3.30, 0.28, col + "22", lab, fs=7.5, ec=col, lw=1.0)
    ax_a.text(6.45, 2.18, "Table T1  ($k{=}8$)", fontsize=7, color=C["mute"], ha="left")

    # ----- (b) spend budget -----
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 2.15)
    ax_b.axis("off")
    ax_b.set_title("(b)  Then spend the same tokens two ways (timestamps frozen).",
                   loc="left", fontsize=9, color=C["ink"], pad=2)

    # left: 8 full
    ax_b.text(0.15, 1.88, r"$k{=}8$  full resolution", fontsize=8, color=C["ink"])
    r_end, cy = frames_row(ax_b, 0.15, 0.55, 8, 0.42, 0.08, C["omp"], alpha=0.75)
    ax_b.text(0.15, 0.18, "fewer frames, sharp", fontsize=7, color=C["mute"])

    # brace / vs
    ax_b.annotate(
        "", xy=(4.55, 1.15), xytext=(4.05, 1.15),
        arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.1),
    )
    ax_b.text(4.30, 1.32, "matched\nvisual tokens", ha="center", va="bottom",
              fontsize=7, color=C["ink"])

    # right: 16 half (half linear size ≈ quarter area; we use ~0.7 linear so it still reads)
    ax_b.text(5.05, 1.88, r"$k{=}16$  at $\sim$50\% resolution", fontsize=8, color=C["ink"])
    frames_row(ax_b, 5.05, 0.62, 16, 0.22, 0.05, C["accent"], alpha=0.85)
    ax_b.text(5.05, 0.18, "more frames, cheaper each", fontsize=7, color=C["mute"])

    fpath_pdf = os.path.join(outdir, "fig_protocol.pdf")
    fpath_png = os.path.join(outdir, "fig_protocol.png")
    fig.savefig(fpath_pdf, bbox_inches="tight")
    fig.savefig(fpath_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {fpath_pdf}")
    print(f"wrote {fpath_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="video-understanding/figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    draw(args.outdir)


if __name__ == "__main__":
    main()
