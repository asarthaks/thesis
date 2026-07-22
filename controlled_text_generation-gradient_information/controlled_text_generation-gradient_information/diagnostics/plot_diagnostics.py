#!/usr/bin/env python
"""
plot_diagnostics.py

Produces every figure for the mechanism chapter. Reads the outputs of
run_diagnostic.py and writes PDFs (vector, for LaTeX) plus PNGs (for slides)
into --fig_dir.

  python plot_diagnostics.py --res_dir results_diag --fig_dir figures

Figures produced
----------------
  fig_lin_scatter.pdf        1A  surrogate vs truth, coloured by distance
  fig_lin_radius.pdf         1B  THE MECHANISM FIGURE. correlation vs distance
  fig_lin_decomposition.pdf  1C  surrogate correlates with the FUTURE term only,
                                 and the SELF term is the one that dominates
  fig_lin_topk.pdf           1D  top-k recall vs a random ranker
  fig_traj_pca.pdf           3A  trajectories in PCA space, 2x2 config panel
  fig_traj_tsne.pdf          3A' same, t-SNE
  fig_traj_manifold.pdf      3B  distance from the token manifold over time
  fig_trap_scatter.pdf       4A  likelihood vs quality. the likelihood trap.
  fig_trap_length.pdf        4B  length vs total logp. the brevity incentive.
  fig_aniso_hist.pdf         5A  pairwise distance histograms, GPT-2 vs Llama

Style is deliberately plain: no seaborn defaults, no chart junk, axis labels on
everything (the IMS grading criteria list "correctness of visualization (axes
labels, ...)" as an explicit line item).
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

C_POLICY = "#2E7D77"
C_BAD = "#B5402F"
C_NEUTRAL = "#1B1F3B"
C_ACCENT = "#E8A33D"


def save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, name + ".pdf"))
    fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ==========================================================================
# EXPERIMENT 1
# ==========================================================================

def plot_linearization(res_dir, fig_dir, model_key="gpt2sft"):
    csv = os.path.join(res_dir, f"diag_linearization_{model_key}.csv")
    if not os.path.exists(csv):
        print("missing", csv); return
    df = pd.read_csv(csv)

    # ---------- 1A: the scatter ----------
    sub = df.sample(min(30000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    sc = ax.scatter(sub.surrogate, sub.true_delta, c=sub.dist,
                    s=3, alpha=0.35, cmap="viridis", linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\|e(v) - e(x_i)\|_2$  (embedding distance)")

    r_all = spearmanr(df.surrogate, df.true_delta)[0]
    ax.set_xlabel(r"Taylor surrogate  $\hat{\Delta}(v) = \nabla_{e_i}\log p^\top (e(v)-e(x_i))$")
    ax.set_ylabel(r"True energy change  $\Delta(v) = \log p(x_{i \to v}) - \log p(x)$")
    ax.set_title("Taylor surrogate against the true energy change")  # Phase 8: was a verdict
    ax.text(0.03, 0.95, rf"Spearman $\rho$ = {r_all:.3f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    save(fig, fig_dir, "fig_lin_scatter")

    # ---------- 1B: THE MECHANISM FIGURE ----------
    nb = 16
    qs = np.unique(np.quantile(df.dist, np.linspace(0, 1, nb + 1)))
    # adaptive floor: never demand more rows per bin than the data can supply
    min_per_bin = max(20, min(100, len(df) // (4 * max(1, len(qs) - 1))))
    centres, rhos, ns = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (df.dist >= lo) & (df.dist < hi)
        if m.sum() < min_per_bin:
            continue
        rhos.append(spearmanr(df.surrogate[m], df.true_delta[m])[0])
        centres.append(0.5 * (lo + hi))
        ns.append(int(m.sum()))

    if not centres:
        print("!! fig_lin_radius: no distance bin had enough rows "
              f"(min_per_bin={min_per_bin}, n={len(df)}). "
              "Run more sequences or lower --n_cand stratification.")
        return
    print(f"   fig_lin_radius: {len(centres)} bins, {min(ns)}-{max(ns)} rows each")

    # PHASE 8 (print legibility): the two vertical-line annotations used to be drawn as
    # floating ax.text labels, which collided with the legend entry at the lower left and
    # with the data line at the top. They are now legend entries, so nothing can overlap,
    # and the title is descriptive rather than a verdict.
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(centres, rhos, "o-", color=C_POLICY, lw=2, ms=5,
            label=r"Spearman $\rho$ within distance bin")
    ax.axhline(0, color="0.4", lw=0.8)

    ylo, yhi = ax.get_ylim()

    mean_d = float(df.dist.mean())
    ax.axvline(mean_d, ls="--", color=C_BAD, lw=1.4,
               label=f"mean candidate distance = {mean_d:.2f}")

    # the distance at which the binned correlation first falls below 0.1
    r_thresh = 0.1
    rad = None
    for c, r in zip(centres, rhos):
        if r < r_thresh:
            rad = c
            break
    if rad is not None:
        ax.axvline(rad, ls=":", color=C_ACCENT, lw=1.6,
                   label=rf"first bin with $\rho < {r_thresh}$, at {rad:.2f}")
    else:
        print("   note: rho never fell below 0.1, so no linearization radius was "
              "identified. That would be a POSITIVE result for gradient guidance "
              "and it would contradict the thesis. Check the data before writing.")

    ax.set_ylim(ylo, yhi + 0.30 * (yhi - ylo))
    ax.set_xlabel(r"Embedding distance of the candidate token, $\|e(v)-e(x_i)\|_2$",
                  fontsize=11)
    ax.set_ylabel(r"Spearman $\rho$ (surrogate vs. true energy change)", fontsize=11)
    ax.set_title("Surrogate-to-truth correlation by candidate embedding distance",
                 fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95,
              edgecolor="0.8")
    save(fig, fig_dir, "fig_lin_radius")

    # ---------- 1C: the decomposition ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, col, lab, col_c in [
        (axes[0], "true_delta_self",
         r"self term:  $\log p(x_i \mid x_{<i})$", C_BAD),
        (axes[1], "true_delta_future",
         r"future term:  $\sum_{t>i} \log p(x_t \mid x_{<t})$", C_POLICY),
    ]:
        s = df.sample(min(20000, len(df)), random_state=0)
        ax.scatter(s.surrogate, s[col], s=3, alpha=0.3, color=col_c, linewidths=0)
        rho = spearmanr(df.surrogate, df[col])[0]
        ax.set_xlabel("Taylor surrogate")
        ax.set_ylabel(lab)
        ax.set_title(rf"$\rho$ = {rho:.3f}")

    fig.suptitle("Surrogate against the self and future terms of the true change",
                 y=1.02, fontsize=11)  # Phase 8: was a verdict
    fig.text(0.5, -0.06,
             f"mean |self| = {df.true_delta_self.abs().mean():.2f} nats,   "
             f"mean |future| = {df.true_delta_future.abs().mean():.2f} nats.   "
             r"$\nabla_{e_i}\log p$ cannot see the self term at all, because "
             r"$\log p(x_i \mid x_{<i})$ depends on $x_i$ through a discrete index.",
             ha="center", fontsize=8.5)
    save(fig, fig_dir, "fig_lin_decomposition")

    # ---------- 1D: top-k recall ----------
    ks = [1, 5, 10, 20]
    grad_rec, rand_rec = [], []
    rng = np.random.RandomState(0)
    for k in ks:
        g, r = [], []
        for sid, grp in df.groupby("seq_id"):
            if len(grp) < 50:
                continue
            true_top = set(grp.nlargest(k, "true_delta").cand_id)
            grad_top = set(grp.nlargest(k, "surrogate").cand_id)
            rand_top = set(rng.choice(grp.cand_id.values, size=k, replace=False))
            g.append(len(true_top & grad_top) / k)
            r.append(len(true_top & rand_top) / k)
        grad_rec.append(np.mean(g)); rand_rec.append(np.mean(r))

    x = np.arange(len(ks)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x - w/2, grad_rec, w, color=C_POLICY, label="ranked by the LM gradient")
    ax.bar(x + w/2, rand_rec, w, color="0.65", label="ranked at random")
    ax.set_xticks(x); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("Recall of the true top-$k$ tokens")
    ax.set_xlabel("Cutoff")
    ax.set_title("Top-$k$ recall of gradient ranking against a random ranker")  # Phase 8: was a verdict
    ax.legend(frameon=False, fontsize=9)
    save(fig, fig_dir, "fig_lin_topk")


# ==========================================================================
# EXPERIMENT 3: trajectories, PCA and t-SNE
# ==========================================================================

def plot_trajectories(res_dir, fig_dir, model_key="gpt2sft"):
    npz_path = os.path.join(res_dir, f"diag_trajectory_{model_key}_traj.npz")
    if not os.path.exists(npz_path):
        print("missing", npz_path); return
    z = np.load(npz_path)
    vocab = z["vocab_embeddings"]                       # N x D

    configs = [
        ("dls_mh_gn", "DLS (MH on, grad-norm on)"),
        ("cls_gn_on", "CLS, gradient normalization ON"),
        ("cls_gn_off_nomh", "CLS, grad-norm OFF, no MH"),
        ("cls_gn_off_mh", "CLS, grad-norm OFF, MH ON"),
    ]

    # ---------- PCA: fit on the VOCABULARY, not the trajectory ----------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(vocab)
    V2 = pca.transform(vocab)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    for ax, (key, title) in zip(axes.ravel(), configs):
        k = f"{key}__states"
        if k not in z:
            ax.set_visible(False); continue
        S = z[k]                                        # nseq x T x D
        ax.scatter(V2[:, 0], V2[:, 1], s=2, c="0.85", linewidths=0, zorder=1)
        for si in range(S.shape[0]):
            T2 = pca.transform(S[si])
            ax.plot(T2[:, 0], T2[:, 1], "-", color="0.5", lw=0.6, zorder=2)
            ax.scatter(T2[:, 0], T2[:, 1], c=np.arange(len(T2)),
                       cmap="viridis", s=14, zorder=3, linewidths=0)
            ax.scatter(T2[0, 0], T2[0, 1], marker="X", s=70,
                       c=C_BAD, zorder=4, edgecolors="white", linewidths=0.6)
            ax.scatter(T2[-1, 0], T2[-1, 1], marker="*", s=130,
                       c=C_ACCENT, zorder=4, edgecolors="white", linewidths=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC 1 (fitted on the vocabulary)")
        ax.set_ylabel("PC 2")
    fig.suptitle("Sampler trajectories in embedding space (PCA). "
                 "Grey: token embeddings. X: start. Star: end. Colour: step index.",
                 y=0.995, fontsize=10.5)
    save(fig, fig_dir, "fig_traj_pca")

    # ---------- t-SNE: fit jointly on vocab subsample + all trajectory points ----------
    from sklearn.manifold import TSNE
    rng = np.random.RandomState(0)
    vsub = vocab[rng.choice(len(vocab), min(3000, len(vocab)), replace=False)]

    all_traj, spans = [], {}
    cur = len(vsub)
    for key, _ in configs:
        k = f"{key}__states"
        if k not in z:
            continue
        S = z[k].reshape(-1, z[k].shape[-1])
        spans[key] = (cur, cur + len(S), z[k].shape)
        all_traj.append(S)
        cur += len(S)

    X = np.concatenate([vsub] + all_traj, axis=0)
    print(f"t-SNE on {X.shape[0]} points, this takes a minute...")
    Y = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=0, max_iter=1000).fit_transform(X)
    Yv = Y[: len(vsub)]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    for ax, (key, title) in zip(axes.ravel(), configs):
        if key not in spans:
            ax.set_visible(False); continue
        a, b, shape = spans[key]
        Yt = Y[a:b].reshape(shape[0], shape[1], 2)
        ax.scatter(Yv[:, 0], Yv[:, 1], s=2, c="0.85", linewidths=0, zorder=1)
        for si in range(Yt.shape[0]):
            ax.plot(Yt[si, :, 0], Yt[si, :, 1], "-", color="0.5", lw=0.6, zorder=2)
            ax.scatter(Yt[si, :, 0], Yt[si, :, 1], c=np.arange(Yt.shape[1]),
                       cmap="viridis", s=14, zorder=3, linewidths=0)
            ax.scatter(Yt[si, 0, 0], Yt[si, 0, 1], marker="X", s=70,
                       c=C_BAD, zorder=4, edgecolors="white", linewidths=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    fig.suptitle("Sampler trajectories in embedding space (t-SNE, perplexity 30). "
                 "Fitted jointly on vocabulary and trajectory points.",
                 y=0.995, fontsize=10.5)
    save(fig, fig_dir, "fig_traj_tsne")

    # ---------- 3B: distance to the token manifold ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [C_POLICY, C_ACCENT, C_NEUTRAL, C_BAD]
    for (key, title), c in zip(configs, colors):
        k = f"{key}__dist_to_manifold"
        if k not in z:
            continue
        d = z[k]                                        # nseq x T
        m, s = d.mean(0), d.std(0)
        t = np.arange(len(m))
        ax.plot(t, m, color=c, lw=1.8, label=title)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.12, linewidth=0)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"$\min_v \|s_t - e(v)\|_2$   (distance to nearest token)")
    ax.set_title("How far the sampler state drifts from the token manifold")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, fig_dir, "fig_traj_manifold")


# ==========================================================================
# EXPERIMENT 4: the likelihood trap
# ==========================================================================

def plot_likelihood_trap(res_dir, fig_dir, model_key="gpt2sft"):
    csv = os.path.join(res_dir, f"diag_likelihood_trap_{model_key}.csv")
    if not os.path.exists(csv):
        print("missing", csv); return
    df = pd.read_csv(csv)

    order = ["beam20", "beam5", "greedy", "temp07", "topp90", "ancestral"]
    cmap = {"beam20": C_BAD, "beam5": "#D06A4C", "greedy": "#E08A5B",
            "temp07": "#7FA8A3", "topp90": C_POLICY, "ancestral": C_NEUTRAL}

    # ---------- 4A ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for name in order:
        s = df[df.strategy == name]
        if len(s) == 0:
            continue
        axes[0].scatter(s.mean_logp, s.rep4, s=10, alpha=0.5,
                        color=cmap[name], label=name, linewidths=0)
        axes[1].scatter(s.mean_logp, s.distinct2, s=10, alpha=0.5,
                        color=cmap[name], label=name, linewidths=0)

    for ax, ylab in [(axes[0], "4-gram repetition rate  (lower is better)"),
                     (axes[1], "distinct-2  (higher is better)")]:
        ax.set_xlabel(r"mean per-token $\log p$   (higher = lower energy)")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Per-token likelihood against repetition and diversity",
                 y=1.02, fontsize=11)  # Phase 8: was a verdict
    save(fig, fig_dir, "fig_trap_scatter")

    # ---------- 4B ----------
    from scipy.stats import linregress
    lr = linregress(df.gen_len, df.total_logp)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(df.gen_len, df.total_logp, s=6, alpha=0.25,
               color=C_NEUTRAL, linewidths=0)
    xs = np.linspace(df.gen_len.min(), df.gen_len.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=C_BAD, lw=2,
            label=rf"slope = {lr.slope:.2f} nats/token  ($r$ = {lr.rvalue:.2f})")
    ax.set_xlabel("Generated sequence length (tokens)")
    ax.set_ylabel(r"Total $\log p$   (the unnormalised GFlowNet reward)")
    ax.set_title("Total log-likelihood against generated length")  # Phase 8: was a verdict
    ax.legend(frameon=False, fontsize=9)
    save(fig, fig_dir, "fig_trap_length")


# ==========================================================================
# EXPERIMENT 5: embedding geometry
# ==========================================================================

def plot_anisotropy(res_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, name, c in [(axes[0], "gpt2sft", "GPT-2 Large (SFT)", C_POLICY),
                             (axes[1], "llama3-8b", "Llama-3 8B", C_ACCENT)]:
        p = os.path.join(res_dir, f"diag_anisotropy_{key}_dists.npz")
        if not os.path.exists(p):
            ax.set_visible(False); continue
        z = np.load(p)
        ax.hist(z["pairwise_l2"], bins=100, color=c, alpha=0.8, linewidth=0)
        mu = float(z["pairwise_l2"].mean())
        ax.axvline(mu, color=C_BAD, ls="--", lw=1.5)
        ax.text(mu, ax.get_ylim()[1] * 0.9, f"  mean = {mu:.2f}",
                color=C_BAD, fontsize=9)
        ax.set_title(name)
        ax.set_xlabel(r"Pairwise $L_2$ distance between token embeddings")
        ax.set_ylabel("Count")
    fig.suptitle("Pairwise token-embedding distance distributions, GPT-2 Large and Llama-3",
                 y=1.03, fontsize=10.5)  # Phase 8: was a verdict
    save(fig, fig_dir, "fig_aniso_hist")


# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results_diag")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--model_key", default="gpt2sft")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    plot_linearization(args.res_dir, args.fig_dir, args.model_key)
    plot_trajectories(args.res_dir, args.fig_dir, args.model_key)
    plot_likelihood_trap(args.res_dir, args.fig_dir, args.model_key)
    plot_anisotropy(args.res_dir, args.fig_dir)

    # dump every summary JSON into one table for the thesis appendix
    rows = []
    for p in sorted(glob.glob(os.path.join(args.res_dir, "diag_*.json"))):
        with open(p) as f:
            d = json.load(f)
        d.pop("argv", None)
        rows.append(d)
    if rows:
        pd.json_normalize(rows).to_csv(
            os.path.join(args.fig_dir, "diagnostic_summary.csv"), index=False)
        print("wrote diagnostic_summary.csv")


if __name__ == "__main__":
    main()
