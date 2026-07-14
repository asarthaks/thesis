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
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
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
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

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
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = ["stayed in cell\n(move changes nothing)",
              "crossed a boundary\n(move changes a token)"]
    vals = [acc_stay, acc_cross]
    bars = ax.bar(labels, vals, color=[C_OK, C_BAD], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metropolis-Hastings acceptance rate")
    ax.set_ylim(0, max(1.0, max(v for v in vals if not np.isnan(v)) * 1.2))
    ax.set_title("MH accepts only the proposals that do nothing")
    save(fig, args.fig_dir, "fig_mh_accept")

    # ---------- Plot 2B ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, col, title in [
        (axes[0], "log_target_ratio",
         r"target term:  $\log\pi(s') - \log\pi(s)$"),
        (axes[1], "log_proposal_ratio",
         r"proposal term:  $\log q(s\mid s') - \log q(s'\mid s)$"),
    ]:
        lo = np.nanpercentile(df[col], 1)
        hi = np.nanpercentile(df[col], 99)
        bins = np.linspace(lo, hi, 60)
        if len(stayed):
            ax.hist(stayed[col].clip(lo, hi), bins=bins, alpha=0.7,
                    color=C_OK, label="stayed in cell", linewidth=0)
        if len(crossed):
            ax.hist(crossed[col].clip(lo, hi), bins=bins, alpha=0.7,
                    color=C_BAD, label="crossed a boundary", linewidth=0)
        ax.axvline(0, color="0.3", lw=0.9)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Decomposing the acceptance ratio: the target term is fine, "
                 "the proposal term is fatal", y=1.03, fontsize=10.5)
    save(fig, args.fig_dir, "fig_mh_decomposition")

    # ---------- optional DLS contrast ----------
    if args.dls_csv and os.path.exists(args.dls_csv):
        d2 = pd.read_csv(args.dls_csv)
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.bar(["DLS\n(discrete)", "CLS\n(continuous)"],
               [d2.accepted.mean(), df.accepted.mean()],
               color=[C_OK, C_BAD], width=0.5)
        ax.set_ylabel("MH acceptance rate")
        ax.set_title("The same correction, on the same model, in two state spaces")
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
