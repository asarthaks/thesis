#!/usr/bin/env python
"""
plot_trajectories.py  -  Phase 5 Stage 1c: embedding-space trajectory figures.

Reproducible from the trace npz alone (collect_traces.py --run_name <r> writes
<r>_traj.npz with per-config __states (nseq x T x D), __gt_emb (nseq x D), __gt_tok,
and vocab_embeddings). Two figure pairs:

  fig_traj_pca_dls.png / fig_traj_tsne_dls.png   DLS policy vs random, MH on
  fig_traj_pca_cls.png / fig_traj_tsne_cls.png   CLS MH on vs off (gn off)

Method guards (the load-bearing methodological points):
  - PCA (2 components) is FIT ON THE VOCABULARY EMBEDDING MATRIX (the vocab_embeddings
    sample in the npz), never on the trajectory points; trajectories are PROJECTED into
    that fixed basis. This is what makes "the trajectory stays in a thin cone of the
    embedding space" a statement about the embedding geometry, not an artefact of
    fitting to the path.
  - Background: 2000 seeded-random vocabulary embeddings as a grey cloud, so the
    anisotropy cone (nearest-neighbour 1.82, mean pairwise 2.77 nats-of-space in the
    thesis) is visible around the trajectory.
  - Start state and the ground-truth token are marked.
  - t-SNE is the SECONDARY panel (seed 0, perplexity 30) and its caption notes that
    t-SNE distorts global distances; PCA is the primary evidence.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

BG_N = 2000
BG_SEED = 0


def project_pca(fit_X, comps=2):
    """Fit PCA on fit_X (n x D), return (mean, components) for projecting other points."""
    mu = fit_X.mean(0, keepdims=True)
    Xc = fit_X - mu
    # economy SVD; principal axes are the right-singular vectors
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:comps]                     # (1,D), (comps,D)


def apply_pca(X, mu, comps):
    return (X - mu) @ comps.T                 # (n, comps)


def _step_lines(ax, pts, cmap, lw=1.6, z=3):
    """Draw a trajectory as segments colored by step (0..T-1)."""
    T = pts.shape[0]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, zorder=z, linewidths=lw)
    lc.set_array(np.arange(T - 1))
    ax.add_collection(lc)


def panel(ax, npz, cfg, mu, comps, bg2d, title, n_show, cmap):
    ax.scatter(bg2d[:, 0], bg2d[:, 1], s=4, c="0.82", alpha=0.55, zorder=0,
               rasterized=True, label="vocabulary embeddings")
    states = npz[f"{cfg}__states"]                       # nseq x T x D
    gt_emb = npz.get(f"{cfg}__gt_emb")
    n = min(n_show, states.shape[0])
    traj_pts = []
    for s in range(n):
        p2 = apply_pca(states[s], mu, comps)             # T x 2
        traj_pts.append(p2)
        _step_lines(ax, p2, cmap)
        ax.scatter(p2[0, 0], p2[0, 1], s=55, marker="o", facecolor="white",
                   edgecolor="black", linewidths=1.3, zorder=5)   # start state
        ax.scatter(p2[-1, 0], p2[-1, 1], s=30, marker="o", c="black", zorder=5)  # end
    if gt_emb is not None:
        g2 = apply_pca(np.asarray(gt_emb)[:n], mu, comps)
        ax.scatter(g2[:, 0], g2[:, 1], s=150, marker="*", c="#118844",
                   edgecolor="black", linewidths=0.6, zorder=6, label="ground-truth token")
    # Robust axis limits: always frame the full vocabulary cone, and include the central
    # bulk (2-98 pct) of the trajectory. Continuous-Langevin states that escape far
    # off-manifold are clipped so the cone stays visible; the escape magnitude is
    # reported in the caption/prose. Discrete-Langevin trajectories stay in frame.
    allpts = np.concatenate(traj_pts, 0) if traj_pts else bg2d
    lo = np.minimum(bg2d.min(0), np.percentile(allpts, 2, axis=0))
    hi = np.maximum(bg2d.max(0), np.percentile(allpts, 98, axis=0))
    pad = 0.12 * (hi - lo + 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.tick_params(labelsize=8)


def pca_figure(npz, cfgA, cfgB, titleA, titleB, mu, comps, bg2d, out_path, n_show):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    panel(axes[0], npz, cfgA, mu, comps, bg2d, titleA, n_show, "viridis")
    panel(axes[1], npz, cfgB, mu, comps, bg2d, titleB, n_show, "plasma")
    # shared legend + a step colorbar
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                    markeredgecolor="black", label="start state", markersize=8))
    l.append("start state")
    axes[0].legend(h, l, fontsize=8, loc="best", framealpha=0.9)
    sm = plt.cm.ScalarMappable(cmap="viridis"); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("optimization step (normalized)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def tsne_figure(npz, cfgA, cfgB, titleA, titleB, bg_raw, out_path, n_show):
    from sklearn.manifold import TSNE
    blocks, meta = [], []   # meta: (kind, cfg, seq) ; kind in {bg, traj, start, gt}
    blocks.append(bg_raw); meta += [("bg", None, None)] * bg_raw.shape[0]
    for cfg in (cfgA, cfgB):
        states = npz[f"{cfg}__states"]; gt_emb = npz.get(f"{cfg}__gt_emb")
        n = min(n_show, states.shape[0])
        for s in range(n):
            T = states[s].shape[0]
            blocks.append(states[s]); meta += [("traj", cfg, s)] * T
        if gt_emb is not None:
            blocks.append(np.asarray(gt_emb)[:n]); meta += [("gt", cfg, s) for s in range(n)]
    X = np.concatenate(blocks, 0).astype(np.float32)
    emb = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(X)
    meta = np.array(meta, dtype=object)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for ax, cfg, title, cmap in ((axes[0], cfgA, titleA, "viridis"),
                                 (axes[1], cfgB, titleB, "plasma")):
        bg = emb[np.array([m[0] == "bg" for m in meta])]
        ax.scatter(bg[:, 0], bg[:, 1], s=4, c="0.82", alpha=0.5, zorder=0, rasterized=True)
        states = npz[f"{cfg}__states"]; n = min(n_show, states.shape[0])
        for s in range(n):
            sel = np.array([(m[0] == "traj" and m[1] == cfg and m[2] == s) for m in meta])
            p2 = emb[sel]
            if len(p2) < 2:
                continue
            _step_lines(ax, p2, cmap)
            ax.scatter(p2[0, 0], p2[0, 1], s=55, marker="o", facecolor="white",
                       edgecolor="black", linewidths=1.3, zorder=5)
        gsel = np.array([(m[0] == "gt" and m[1] == cfg) for m in meta])
        if gsel.any():
            g2 = emb[gsel]
            ax.scatter(g2[:, 0], g2[:, 1], s=150, marker="*", c="#118844",
                       edgecolor="black", linewidths=0.6, zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.tick_params(labelsize=8)
    fig.suptitle("t-SNE (secondary; distorts global distances, see PCA for the primary view)",
                 fontsize=9, y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results_diag/traces_gpt2sft_plot_traj.npz")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--n_show", type=int, default=6)
    ap.add_argument("--no_tsne", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    npz = dict(np.load(args.npz, allow_pickle=True))
    vocab = npz["vocab_embeddings"]                      # (Nv, D) random vocab sample
    print(f"loaded {args.npz}: vocab {vocab.shape}, configs "
          f"{[k[:-8] for k in npz if k.endswith('__states')]}")

    # PCA fit on the vocabulary embedding matrix (NOT the trajectory points)
    mu, comps = project_pca(vocab, comps=2)
    rng = np.random.RandomState(BG_SEED)
    bg_idx = rng.choice(vocab.shape[0], min(BG_N, vocab.shape[0]), replace=False)
    bg_raw = vocab[bg_idx]
    bg2d = apply_pca(bg_raw, mu, comps)

    pairs = [
        ("dls_policy_gn_mh", "dls_random_gn_mh",
         "DLS policy direction (MH on)", "DLS random direction (MH on)",
         "fig_traj_pca_dls.png", "fig_traj_tsne_dls.png"),
        ("cls_policy_gnoff_mh", "cls_policy_gnoff_nomh",
         "CLS policy, MH correction on", "CLS policy, MH correction off",
         "fig_traj_pca_cls.png", "fig_traj_tsne_cls.png"),
    ]
    for cfgA, cfgB, tA, tB, pca_name, tsne_name in pairs:
        if f"{cfgA}__states" not in npz or f"{cfgB}__states" not in npz:
            print(f"[skip] missing {cfgA} or {cfgB} in npz")
            continue
        pca_figure(npz, cfgA, cfgB, tA, tB, mu, comps, bg2d,
                   os.path.join(args.fig_dir, pca_name), args.n_show)
        if not args.no_tsne:
            tsne_figure(npz, cfgA, cfgB, tA, tB, bg_raw,
                        os.path.join(args.fig_dir, tsne_name), args.n_show)


if __name__ == "__main__":
    main()
