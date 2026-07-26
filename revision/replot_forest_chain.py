#!/usr/bin/env python3
"""Redraw fig_forest_chain at a size legible in A4 print.

Plot only. Reads the CACHED contrasts in results/revision/rev_chain_stats.json, which
revision/analyze_chain_stats.py wrote, and redraws them. No result is recomputed and no
bootstrap is re-run, so every point and interval is bit-identical to the figure it replaces;
only the font sizes, the marker sizes and the figure aspect change.

Reason: at 0.72 textwidth the previous figure (7.6 x 11.9 in, 6.4pt tick labels) rendered its
35 configuration labels at roughly 4.5pt on the page, which is not readable in print.
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/revision/rev_chain_stats.json")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--statistic", default="chain_mean")
    args = ap.parse_args()

    d = json.load(open(args.stats))
    margin = d["margin"]
    rows = []
    for label, per_stat in d["paired_contrasts"].items():
        s = per_stat[args.statistic]
        rows.append({"label": label, "mean_diff": s["mean_diff"], "ci95": s["ci95"]})
    rows.sort(key=lambda r: r["label"])
    n = len(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sized backwards from the page slot. The figure occupies 0.90 textwidth = 384pt and must
    # leave room for a four-line caption inside the 591.5pt text height, so its on-page box is
    # about 384 x 470pt, an aspect of 1.22. A source figure of 8.0 x 9.8in has that aspect and
    # is reduced by 384/(72*8) = 0.667 on the page, so a 12pt source label prints at 8pt.
    fig, ax = plt.subplots(figsize=(8.0, 9.8))
    ax.axvspan(-margin, margin, color="0.88", zorder=0,
               label=f"equivalence margin $\\pm${margin:.3f} nats")
    ax.axvline(0.0, color="0.35", lw=1.2, zorder=1)
    for i, r in enumerate(rows):
        lo, hi = r["ci95"]
        crosses = lo <= 0.0 <= hi
        colour = "#2b6cb0" if crosses else "#c05621"
        ax.plot([lo, hi], [i, i], color=colour, lw=2.2, zorder=2)
        ax.plot([r["mean_diff"]], [i], "o", ms=6.0, color=colour, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("paired mean KL difference, policy minus random (nats)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, lw=0.7)
    fig.tight_layout(pad=0.6)
    os.makedirs(args.fig_dir, exist_ok=True)
    path = os.path.join(args.fig_dir, "fig_forest_chain")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {path}.pdf / .png ({n} configurations, statistic={args.statistic})")

if __name__ == "__main__":
    main()
