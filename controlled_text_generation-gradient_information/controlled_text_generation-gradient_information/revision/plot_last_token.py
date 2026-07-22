#!/usr/bin/env python3
"""Regenerate the final-position figure (Doc Figure 5.8 / flat Figure 8).

Reads results_revision/last_token_figure.csv, which is produced by
diagnostics/run_revision.py --exp last_token, and writes
figures/fig_lasttoken.{png,pdf}.

Written in Phase 8. The original figure was produced ad hoc and its generator was
not kept in the repository; this script reproduces it from the same CSV and fixes
three print defects the author reported against the rendered thesis:

 1. the acceptance line and its "100%" annotation were drawn straight through the
    legend box. The legend now sits in reserved headroom above the plotted data,
    created by extending the left y-axis, so no artist can overlap it.
 2. the figure carried a verdict as its title ("At the final token the gradient is
    provably zero yet the energy is exact"). The title is now descriptive; the
    reading lives in the prose of Section 5.12, per the caption policy adopted in
    Phase 8.
 3. the interpretive footer line under the axes was a second caption. It is
    removed; the policy-minus-random result is stated in the text and in Table 5.6.

Figure size and font sizes are set for a 15cm text block at \\textwidth, so the
smallest text in the printed figure is about 9pt.

No number is recomputed here: every value is read from the CSV.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C_BAR = "#3C6E9F"
C_LINE = "#B5402F"

plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def main(csv="results_revision/last_token_figure.csv", fig_dir="figures"):
    df = pd.read_csv(csv)
    x = df["downstream_x"].to_numpy()
    labels = [f"{lab}\n({cond.replace('_', '-')})"
              for lab, cond in zip(df["downstream_label"], df["condition"])]
    grad = df["dls_policy_grad_norm_mean"].to_numpy()
    acc = df["indep_mh_accept_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax2 = ax.twinx()

    ax.bar(x, grad, width=0.45, color=C_BAR,
           label="mean LM gradient norm seen by DLS-policy")
    ax2.plot(x, acc, "o-", color=C_LINE, lw=2, ms=7,
             label="independence-MH acceptance (%)")

    # value labels, placed away from the reserved legend band
    for xi, g in zip(x, grad):
        if g > 0:
            ax.annotate(f"{g:.1f}", (xi, g), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=C_BAR)
    ax.annotate("exactly 0", (x[0], 0), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9, color=C_BAR)
    for xi, a in zip(x, acc):
        ax2.annotate(f"{a:.0f}%", (xi, a), textcoords="offset points",
                     xytext=(12, -14), ha="left", fontsize=9.5, color=C_LINE)

    # reserve the top third of the axes for the legend so nothing can cross it
    ax.set_ylim(0, max(grad) * 1.75)
    ax2.set_ylim(0, 155)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("number of downstream scored terms the masked position feeds",
                  fontsize=11)
    ax.set_ylabel("mean LM gradient norm seen by DLS-policy",
                  color=C_BAR, fontsize=11)
    ax2.set_ylabel("independence-MH acceptance (%)", color=C_LINE, fontsize=11)
    ax.tick_params(axis="y", labelcolor=C_BAR, labelsize=10)
    ax2.tick_params(axis="y", labelcolor=C_LINE, labelsize=10)
    ax.set_title("Gradient norm and energy-only acceptance by downstream context",
                 fontsize=11.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9.5,
              frameon=True, framealpha=0.95, edgecolor="0.8", ncol=1)

    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"fig_lasttoken.{ext}"))
    plt.close(fig)
    print("wrote fig_lasttoken.png / .pdf")


if __name__ == "__main__":
    main()
