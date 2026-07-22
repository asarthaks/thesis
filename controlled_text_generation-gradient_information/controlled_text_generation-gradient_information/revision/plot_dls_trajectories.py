#!/usr/bin/env python3
"""Regenerate the two DLS trajectory figures for print (Doc Figures 5.1 and 5.2).

Written in Phase 8 in response to the author's report that the 100-step trajectory
figure is illegible in the printed thesis. The originals came from
notebook_plotting.plot_graphs, which lays the three metric panels out in a single
row at figsize (22, 6), an aspect ratio of 3.7:1, with the legend outside the axes.
Placed at \\textwidth on a 15cm text block that renders about 4cm tall, putting the
tick labels near 4pt.

This script reads exactly the same per-step CSVs and plots exactly the same three
metrics and the same method / correction split. The only changes are layout:

 * three panels stacked vertically instead of side by side, at 6.6 x 6.9 inches,
   so each panel is about 4.5cm tall at \\textwidth, no text falls below 9pt, and the
   float still fits on a page alongside text;
 * one shared legend inside the top panel rather than an outside column, so the
   figure uses the full text width for data;
 * a shared x axis, labelled once at the bottom;
 * the configuration string moved out of a two-line suptitle and into the caption.

notebook_plotting.py is deliberately left untouched: its docstring records that it
must stay byte-comparable with the original notebook figures.

Usage:
    python3 revision/plot_dls_trajectories.py
"""

import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "results_gpt2_v2"
OUT = "figures"

METHOD_LABEL = {
    "policy": "DLS policy",
    "grad_norm_preserved_random_dir": "Grad-norm-preserved random",
    "random": "Random noise",
}
METHOD_COLOR = {
    "DLS policy": "#2E7D77",
    "Grad-norm-preserved random": "#3C6E9F",
    "Random noise": "#E8A33D",
}
# sparse markers so curves that coincide exactly stay distinguishable. Under
# gradient normalization the grad-norm-preserved and fully random arms are the same
# object, so their curves lie on top of one another; that coincidence is the result of
# Section 5.5 and the markers make it visible rather than hiding one line under another.
METHOD_MARKER = {
    "DLS policy": "o",
    "Grad-norm-preserved random": "s",
    "Random noise": "^",
}
# short y-labels: the panels are about 4.5cm tall in print, so a long label does not
# fit inside one panel and runs into its neighbours (Phase 8 in-place render check).
METRICS = [
    ("L2 Distance", "Embedding\ndistance ($L2$)"),
    ("KL Divergence", "Contextual fit\n(KL divergence)"),
    ("Entropy", "Proposal\nentropy"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def load(steps):
    """Mean per-step curve for each (method, MH) cell, from the run CSVs."""
    rows = []
    for meth in METHOD_LABEL:
        for mh, mh_lab in ((True, "With MH"), (False, "No MH")):
            tag = "mh" if mh else "nomh"
            path = os.path.join(
                RES, f"gpt2-large.dls.{meth}.{tag}.gn.free.s{steps}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                traj = r.get("trajectory", "[]")
                traj = ast.literal_eval(traj) if isinstance(traj, str) else traj
                for sd in traj:
                    rows.append({
                        "Method": METHOD_LABEL[meth],
                        "Sampling": mh_lab,
                        "Step": sd.get("step", 0),
                        "L2 Distance": sd.get("avg_l2_distance", np.nan),
                        "KL Divergence": sd.get("avg_kl_divergence", np.nan),
                        "Entropy": sd.get("entropy", np.nan),
                    })
    if not rows:
        raise SystemExit(f"no per-step CSVs found for s{steps} under {RES}/")
    df = pd.DataFrame(rows)
    return df.groupby(["Method", "Sampling", "Step"]).mean().reset_index()


def draw(steps, out_name):
    g = load(steps)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.9), sharex=True)

    for ax, (col, title) in zip(axes, METRICS):
        for meth, sub_m in g.groupby("Method"):
            for samp, sub in sub_m.groupby("Sampling"):
                sub = sub.sort_values("Step")
                if sub[col].isna().all():
                    continue
                ax.plot(sub["Step"], sub[col],
                        color=METHOD_COLOR[meth], lw=2.0,
                        ls="-" if samp == "With MH" else "--",
                        marker=METHOD_MARKER[meth], markevery=max(1, steps // 10),
                        ms=5, markeredgecolor="white", markeredgewidth=0.6,
                        label=f"{meth}, {samp}")
        ax.set_ylabel(title, fontsize=11)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel("Annealing step", fontsize=11)
    fig.subplots_adjust(hspace=0.18, bottom=0.14)

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h, l = set(), [], []
    for hi, li in zip(handles, labels):
        if li not in seen:
            seen.add(li)
            h.append(hi)
            l.append(li)
    # legend below the axes, so it can never cover data (Phase 8 print fix)
    fig.legend(h, l, fontsize=9, frameon=True, framealpha=0.95, edgecolor="0.8",
               loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=2)

    fig.align_ylabels(axes)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{out_name}.{ext}"))
    plt.close(fig)
    print(f"wrote {out_name}.png / .pdf  (s{steps})")


if __name__ == "__main__":
    draw(50, "gpt2-large.dls.gn.free.s50_new_trajectories")
    draw(100, "gpt2-large.dls.gn.free.s100_new_trajectories")
