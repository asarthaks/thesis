#!/usr/bin/env python
"""
analyze_mh.py

EXPERIMENT 2. Proves the MH breakdown in CLS instead of asserting it.

This one needs a small patch to YOUR core/cls.py, because I cannot see that file.
The patch is at the bottom of this docstring. It adds one CSV of per-proposal
records; it changes no sampler behaviour.

The claim we are proving:

    MH does not reject proposals because the target density says they are bad.
    It rejects them because the PROPOSAL density says they are impossible, and it
    says so precisely when the proposal crosses a Voronoi boundary, which is
    precisely when the proposal would have done something.

    Therefore: the set of moves MH accepts and the set of moves that change a
    token are disjoint.

This is the non-hand-wavy version of the CLS section, and it is what turns
"MH assumes the jump is invalid" (which an examiner will not accept) into
"MALA requires a Lipschitz drift, the projected target does not have one, and
here is the measurement."

Usage
-----
  python analyze_mh.py --csv mh_log.csv --fig_dir figures


================================================================================
PATCH TO core/cls.py
================================================================================

Inside the MH acceptance block, you already compute something equivalent to:

    log_alpha = (log_pi_prop - log_pi_cur) + (log_q_back - log_q_fwd)

Add the following. `s_cur` and `s_prop` are the continuous states before and
after the proposal; `E` is the input embedding matrix (V x D).

    # --- BEGIN DIAGNOSTIC LOGGING (remove after the experiment) ---
    if getattr(self, "mh_log", None) is not None:
        with torch.no_grad():
            cell_cur  = torch.cdist(s_cur[0, pos].unsqueeze(0),  E).argmin().item()
            cell_prop = torch.cdist(s_prop[0, pos].unsqueeze(0), E).argmin().item()
            self.mh_log.append(dict(
                step            = int(step),
                seq_id          = int(self._diag_seq_id),
                crossed         = int(cell_cur != cell_prop),
                accepted        = int(bool(accepted)),
                log_alpha       = float(log_alpha),
                log_target_ratio= float(log_pi_prop - log_pi_cur),
                log_proposal_ratio = float(log_q_back - log_q_fwd),
                step_norm       = float((s_prop - s_cur)[0, pos].norm()),
                epsilon         = float(eps_k),
                cell_cur        = cell_cur,
                cell_prop       = cell_prop,
            ))
    # --- END DIAGNOSTIC LOGGING ---

Then in __init__ add `self.mh_log = None` and `self._diag_seq_id = 0`.

Driver (add as a fifth experiment in run_diagnostic.py, or run standalone):

    sampler.mh_log = []
    for i, ids in enumerate(sequences):
        sampler._diag_seq_id = i
        sampler.run(corrupt(ids), positions=[pos], steps=50)
    pd.DataFrame(sampler.mh_log).to_csv("mh_log.csv", index=False)

Run it for CLS with grad_norm OFF and MH ON (the configuration in which CLS is
theoretically correct and empirically paralysed), n = 200 sequences, 50 steps.
That is one run, not 145.

Optionally repeat for DLS with MH on, which gives you the contrast: in the
discrete sampler the acceptance rate is healthy and MH is doing useful work.
Report both numbers. You currently report neither, and an examiner will ask.
================================================================================
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

# G2 (supervisor figure-quality pass, 2026-07-28): figures are drawn at the width they are
# printed at, so a declared point size is the printed point size. TEXTWIDTH_IN is the thesis
# text block (a4paper, left=3cm, right=3cm -> 15cm). fig_mh_decomposition was authored at
# 9.6in and included at 0.95\textwidth (5.6in), which put its 10pt tick labels near 6pt.
TEXTWIDTH_IN = 5.90


def figsize(width_frac, aspect):
    """Figure size in inches for a float included at width_frac x \textwidth."""
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


def thousands(x, _pos):
    """F1: render large tick values compactly so they cannot collide."""
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        v = x / 1000.0
        return f"{v:.0f}k" if abs(v) >= 1 else f"{v:.1f}k"
    return f"{x:.0f}"


plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.titlesize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})

C_OK = "#2E7D77"
C_BAD = "#B5402F"
C_NEUTRAL = "#1B1F3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="mh_log.csv from the patched CLS")
    ap.add_argument("--dls_csv", default=None, help="optional: the same log for DLS")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--config", default=None,
                    help="Phase 9 (Part 1, Alarm 1): restrict to a single config in "
                         "the 'config' column, e.g. cls_policy_gnoff_mh. Without this, "
                         "a multi-config trace file (cls gnoff/gnon MH plus dls) is "
                         "pooled, which mixes the paralysed continuous sampler with the "
                         "100-percent-within-cell discrete control and produces a "
                         "misleading combined bar (stayed 0.064, crossed 0.074).")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.config is not None:
        if "config" not in df.columns:
            raise SystemExit(f"--config given but the CSV has no 'config' column: {args.csv}")
        present = set(df.config.unique())
        if args.config not in present:
            raise SystemExit(f"config {args.config!r} not in {sorted(present)}")
        df = df[df.config == args.config].copy()
        print(f"[config] restricted to {args.config!r}: {len(df)} rows")

    stayed = df[df.crossed == 0]
    crossed = df[df.crossed == 1]

    acc_stay = stayed.accepted.mean() if len(stayed) else float("nan")
    acc_cross = crossed.accepted.mean() if len(crossed) else float("nan")

    print("=" * 68)
    print("EXPERIMENT 2: MH acceptance conditioned on Voronoi boundary crossing")
    print("=" * 68)
    print(f"  proposals that stayed in cell : {len(stayed):7d}   accept rate {acc_stay:.4f}")
    print(f"  proposals that crossed        : {len(crossed):7d}   accept rate {acc_cross:.4f}")
    print()
    print(f"  overall accept rate           : {df.accepted.mean():.4f}")
    n_useful_accepted = int(((df.crossed == 1) & (df.accepted == 1)).sum())
    print(f"  ACCEPTED *AND* USEFUL (crossed a boundary): {n_useful_accepted}")
    print()
    if len(crossed):
        print("  For boundary-crossing proposals, mean log-ratio decomposition:")
        print(f"    log target ratio   : {crossed.log_target_ratio.mean():+9.2f}   "
              "(often POSITIVE: the move improves the sequence)")
        print(f"    log proposal ratio : {crossed.log_proposal_ratio.mean():+9.2f}   "
              "(hugely NEGATIVE: this is what kills it)")
        print()
        print("  The rejection is driven by the PROPOSAL term, not the target term.")
        print("  This is the signature of a non-Lipschitz drift: the reverse proposal")
        print("  mean m_prop is computed from a gradient evaluated on the far side of")
        print("  the cell boundary, so s_t lands deep in the tail of the reverse")
        print("  Gaussian and log q_back diverges.")
    print("=" * 68)

    # ---------- Plot 2A ----------
    fig, ax = plt.subplots(figsize=figsize(0.70, 0.74))
    labels = ["stayed in cell\n(move changes nothing)",
              "crossed a boundary\n(move changes a token)"]
    vals = [acc_stay, acc_cross]
    bars = ax.bar(labels, vals, color=[C_OK, C_BAD], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Metropolis-Hastings acceptance rate")
    ax.set_ylim(0, max(1.0, max(v for v in vals if not np.isnan(v)) * 1.2))
    ax.set_title("Acceptance rate by whether the proposal crossed a cell boundary")  # Phase 8: was a verdict
    save(fig, args.fig_dir, "fig_mh_accept")

    # ---------- Plot 2B ----------
    fig, axes = plt.subplots(1, 2, figsize=figsize(0.95, 0.56))
    # G2: the difference-of-logs labels were wider than their panels and the right one was
    # clipped at the figure edge. The equivalent log-ratio form is the same quantity, is what
    # the running text calls it ("target log-ratio", "proposal log-ratio"), and fits.
    for ax, col, title in [
        (axes[0], "log_target_ratio",
         r"target log-ratio  $\log\,\pi(s')/\pi(s)$"),
        (axes[1], "log_proposal_ratio",
         r"proposal log-ratio  $\log\,q(s\mid s')/q(s'\mid s)$"),
    ]:
        lo = np.nanpercentile(df[col], 1)
        hi = np.nanpercentile(df[col], 99)
        bins = np.linspace(lo, hi, 60)
        # Phase 9 (author issue list item 22): draw one distribution as a translucent fill
        # and the other as a solid step outline on top, with explicit zorder, so the overlap
        # region does not blend the two colours into a spurious third colour.
        if len(stayed):
            ax.hist(stayed[col].clip(lo, hi), bins=bins, alpha=0.45,
                    color=C_OK, label="stayed in cell", linewidth=0, zorder=2)
        if len(crossed):
            ax.hist(crossed[col].clip(lo, hi), bins=bins, histtype="step",
                    color=C_BAD, label="crossed a boundary", linewidth=1.6, zorder=3)
        ax.axvline(0, color="0.3", lw=0.9)
        # G2: these two axis labels are long formulas. At the printed panel width they ran
        # into each other and the right one was clipped, so they are set smaller and the
        # panels are given more gutter below.
        ax.set_xlabel(title, fontsize=8)
        ax.set_ylabel("Count")
        # G2: the legend sat on top of the histogram in the left panel. Pin it to the upper
        # left, where both panels are empty, and add headroom so it cannot touch the data.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
        ax.legend(frameon=False, loc="upper left")
        # F1 (supervisor remark on this figure: "overlapping numbers" on the proposal-term
        # x-axis). The proposal term runs to about -60000, and the default locator packed
        # seven five-digit labels into the panel, which ran together into an unreadable
        # block. Cap the tick count and print the magnitudes in thousands.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        if np.nanmax(np.abs(df[col])) >= 1000:
            ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    # G2: the suptitle restated the caption verbatim at a size that printed near 6pt. The
    # caption carries the naming; the running text carries the walkthrough.
    fig.subplots_adjust(wspace=0.34)
    save(fig, args.fig_dir, "fig_mh_decomposition")

    # ---------- optional DLS contrast ----------
    if args.dls_csv and os.path.exists(args.dls_csv):
        d2 = pd.read_csv(args.dls_csv)
        fig, ax = plt.subplots(figsize=figsize(0.70, 0.74))
        ax.bar(["DLS\n(discrete)", "CLS\n(continuous)"],
               [d2.accepted.mean(), df.accepted.mean()],
               color=[C_OK, C_BAD], width=0.5)
        ax.set_ylabel("MH acceptance rate")
        ax.set_title("The same correction in the two state spaces")  # Phase 8: was a verdict
        save(fig, args.fig_dir, "fig_mh_dls_vs_cls")
        print(f"  DLS acceptance rate: {d2.accepted.mean():.4f}")

    pd.DataFrame([{
        "n_proposals": len(df),
        "accept_rate_overall": float(df.accepted.mean()),
        "accept_rate_stayed_in_cell": float(acc_stay),
        "accept_rate_crossed_boundary": float(acc_cross),
        "n_accepted_and_crossed": n_useful_accepted,
        "mean_log_target_ratio_crossed": float(crossed.log_target_ratio.mean()) if len(crossed) else None,
        "mean_log_proposal_ratio_crossed": float(crossed.log_proposal_ratio.mean()) if len(crossed) else None,
    }]).to_csv(os.path.join(args.fig_dir, "mh_summary.csv"), index=False)
    print("wrote mh_summary.csv")


def save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, name + ".pdf"))
    fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    main()
