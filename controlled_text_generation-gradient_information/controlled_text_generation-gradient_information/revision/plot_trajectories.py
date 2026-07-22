#!/usr/bin/env python
"""
plot_trajectories.py  -  Phase 6 Part 3: trajectory figures, redesigned.

The author's objection is adopted: a 2D projection of a 1280-dimensional space cannot
carry the argument. So the PRIMARY figure is exact in the full space (L2 distances), and
the PCA projection is demoted to a single illustrative secondary panel with its explained
variance printed. t-SNE is dropped.

DECODING TRAP avoided: the npz `vocab_embeddings` array is a 6000-token SUBSAMPLE kept
only for the background cloud. ALL nearest-token decoding and ALL distance computation use
the FULL embedding matrix loaded from the gpt2sft checkpoint (transformer.wte.weight,
50257 x 1280). The subsample is decoration only.

Trace-index caveat (logged, and stated in the caption): collect_traces.py advances the
torch RNG across configs, so trace index i is the same source sentence but the randomly
masked position (hence the ground-truth token) differs per config. Each panel is therefore
self-consistent (its own exact gt_emb) and the cross-panel claim is about on- vs
off-manifold DYNAMICS, which does not require a shared token.

Primary figure fig_traj_distance.png: one panel per config, the SAME seeded trace index in
every panel (np.random.default_rng(0).integers(0,6) -> 5). X: step 0-50. Y: symlog,
linthresh 1, so DLS's exact zeros are visible. Solid line: L2 distance to the ground-truth
token embedding (full space). Dashed line (CLS only): L2 distance to the NEAREST token of
any kind (the off-manifold measure; identically 0 for DLS, stated in the caption). Token
strip: decoded token (DLS state itself; CLS nearest-token projection) at every change,
capped at ~15 labels. CLS rejected steps (identical consecutive states) marked as ticks.
Landing token and ground truth printed at the right edge with match/no-match.

Secondary figure fig_traj_pca.png: one PCA cone per config (PCA fit on the full wte,
trajectory projected in), explained-variance ratio printed (also in fig_traj_stats.json).

Writes figures/fig_traj_stats.json with per-config distance statistics for every number
the thesis quotes.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory

CKPT = ("/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/"
        "infill_subj_arithmetic/gpt2_large_sft_output")
CONFIGS = [
    ("dls_policy_gn_mh", "DLS, policy direction (MH on)", False),
    ("dls_random_gn_mh", "DLS, random direction (MH on)", False),
    ("cls_policy_gnoff_mh", "CLS, MH correction on", True),
    ("cls_policy_gnoff_nomh", "CLS, MH correction off", True),
]
BG_N, BG_SEED = 2000, 0


def load_wte(framework_path):
    from safetensors import safe_open
    with safe_open(os.path.join(framework_path, "model.safetensors"), framework="np") as f:
        key = next(k for k in f.keys() if k.endswith("wte.weight"))
        return np.asarray(f.get_tensor(key), dtype=np.float32)   # (V, D)


def load_tok(framework_path):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(framework_path)


def nearest_tokens(states, wte, wte_sq):
    """For each state row (n,D) return (nn_id, nn_dist) over the FULL vocab."""
    s_sq = (states ** 2).sum(1)                              # (n,)
    d2 = s_sq[:, None] - 2.0 * (states @ wte.T) + wte_sq[None, :]   # (n,V)
    nn_id = d2.argmin(1)
    nn_dist = np.sqrt(np.clip(d2[np.arange(len(states)), nn_id], 0, None))
    return nn_id, nn_dist


def token_strip(ax, steps, decoded, changes, rejected, cap=15):
    """Annotate the decoded token at each change step, capped at ~cap labels."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ch = list(changes)
    elided = False
    if len(ch) > cap:
        keep = np.linspace(0, len(ch) - 1, cap).round().astype(int)
        ch = [ch[i] for i in sorted(set(keep))]
        elided = True
    for t in ch:
        lab = decoded[t].strip() or "_"
        lab = lab.replace("\n", " ")
        if len(lab) > 10:
            lab = lab[:9] + "."
        ax.annotate(lab, xy=(t, 0.02), xycoords=tr, rotation=90, fontsize=6.0,
                    va="bottom", ha="center", color="#333333")
    if elided:
        ax.annotate("(token labels subsampled to %d changes)" % cap, xy=(0.5, 0.14),
                    xycoords="axes fraction", fontsize=6, ha="center", color="0.4",
                    style="italic")
    for t in rejected:
        ax.plot([t, t], [0.0, 0.03], transform=tr, color="#cc3311", lw=0.8, zorder=6)


def build_seq_stats(npz, wte, wte_sq):
    """Aggregate distance stats over ALL 6 seqs x 50 steps per config (for the JSON)."""
    stats = {}
    for cfg, _, is_cls in CONFIGS:
        states = npz[f"{cfg}__states"]                       # nseq x T x D
        gt_emb = npz[f"{cfg}__gt_emb"]                       # nseq x D
        cell_id = npz.get(f"{cfg}__cell_id")                 # nseq x T (sampler's cell)
        nseq, T, D = states.shape
        nn_all, gt_final, tok_changes = [], [], []
        end_nn, distinct_cells = [], []
        for s in range(nseq):
            nn_id, nnd = nearest_tokens(states[s], wte, wte_sq)
            nn_all.append(nnd)
            gt_final.append(float(np.linalg.norm(states[s, -1] - gt_emb[s])))
            tok_changes.append(int((nn_id[1:] != nn_id[:-1]).sum()))
            end_nn.append(float(nnd[-1]))
            if cell_id is not None:
                distinct_cells.append(int(len(np.unique(cell_id[s]))))
        nn_all = np.concatenate(nn_all)
        stats[cfg] = {
            "is_cls": is_cls,
            "nearest_token_dist_min": float(nn_all.min()),
            "nearest_token_dist_max": float(nn_all.max()),
            "nearest_token_dist_mean": float(nn_all.mean()),
            "end_nearest_token_dist_mean": float(np.mean(end_nn)),
            "final_dist_to_gt_mean": float(np.mean(gt_final)),
            "final_dist_to_gt_per_seq": [round(x, 3) for x in gt_final],
            "token_changes_per_seq": tok_changes,
            "distinct_cells_mean": (float(np.mean(distinct_cells))
                                    if distinct_cells else None),
        }
    return stats


def primary_figure(npz, wte, wte_sq, tok, seq_idx, out_path):
    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(8.4, 10.4), sharex=True)
    landing_gt = {}
    for ax, (cfg, title, is_cls) in zip(axes, CONFIGS):
        states = npz[f"{cfg}__states"][seq_idx]              # T x D
        gt_emb = npz[f"{cfg}__gt_emb"][seq_idx]              # D
        gt_tok = int(npz[f"{cfg}__gt_tok"][seq_idx])
        T = states.shape[0]
        steps = np.arange(T)
        dist_gt = np.linalg.norm(states - gt_emb[None, :], axis=1)
        nn_id, nn_dist = nearest_tokens(states, wte, wte_sq)
        decoded = [tok.decode([int(i)]) for i in nn_id]

        ax.plot(steps, dist_gt, "-", color="#1f4e8c", lw=1.8, zorder=4,
                label="dist. to ground-truth token embedding")
        if is_cls:
            ax.plot(steps, nn_dist, "--", color="#b8860b", lw=1.5, zorder=3,
                    label="dist. to nearest token (off-manifold)")
        ax.set_yscale("symlog", linthresh=1.0)
        # headroom above the highest curve so lines/legend clear the title
        top_val = float(max(dist_gt.max(), nn_dist.max() if is_cls else 0.0))
        ax.set_ylim(bottom=-0.2, top=top_val * 4.0 + 1.0)
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.set_ylabel("L2 distance", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)

        rejected = [t for t in range(1, T) if np.array_equal(states[t], states[t - 1])]
        changes = [0] + [t for t in range(1, T) if nn_id[t] != nn_id[t - 1]]
        token_strip(ax, steps, decoded, changes, rejected if is_cls else [])

        land = decoded[-1].strip() or "_"
        gtt = tok.decode([gt_tok]).strip() or "_"
        match = (nn_id[-1] == gt_tok)
        landing_gt[cfg] = {"landing_token": land, "gt_token": gtt, "match": bool(match)}
        ax.annotate("landing: %r  vs  GT: %r  [%s]" % (
                        land[:14], gtt[:14], "match" if match else "no match"),
                    xy=(0.99, 0.93), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.5, color=("#118844" if match else "#cc3311"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    axes[-1].set_xlabel("sampler step", fontsize=9)
    axes[-1].set_xlim(-0.5, npz[f"{CONFIGS[0][0]}__states"].shape[1] - 0.5)
    fig.suptitle("Full-space distances along the sampling trajectory (trace index %d; "
                 "red ticks = rejected CLS proposals)" % seq_idx, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)
    return landing_gt


def pca_secondary(npz, wte, tok, seq_idx, out_path):
    # PCA fit on the FULL vocab (explained variance is a real geometry statement)
    mu = wte.mean(0, keepdims=True)
    rng = np.random.RandomState(BG_SEED)
    # SVD on a large seeded sample of the full vocab (full 50k SVD is wasteful; 12k is ample)
    fit_idx = rng.choice(wte.shape[0], min(12000, wte.shape[0]), replace=False)
    Xc = wte[fit_idx] - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    evr = (S[:2] ** 2) / (S ** 2).sum()
    bg_idx = rng.choice(wte.shape[0], min(BG_N, wte.shape[0]), replace=False)
    bg2d = (wte[bg_idx] - mu) @ comps.T

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
    for ax, (cfg, title, is_cls) in zip(axes.ravel(), CONFIGS):
        states = npz[f"{cfg}__states"][seq_idx]
        gt_emb = npz[f"{cfg}__gt_emb"][seq_idx]
        p2 = (states - mu) @ comps.T
        g2 = (gt_emb[None, :] - mu) @ comps.T
        ax.scatter(bg2d[:, 0], bg2d[:, 1], s=4, c="0.82", alpha=0.5, zorder=0,
                   rasterized=True, label="vocabulary embeddings")
        segs = np.stack([p2[:-1], p2[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", zorder=3, linewidths=1.6)
        lc.set_array(np.arange(len(p2) - 1)); ax.add_collection(lc)
        ax.scatter(p2[0, 0], p2[0, 1], s=55, marker="o", facecolor="white",
                   edgecolor="black", linewidths=1.3, zorder=5)
        ax.scatter(p2[-1, 0], p2[-1, 1], s=30, marker="o", c="black", zorder=5)
        ax.scatter(g2[:, 0], g2[:, 1], s=150, marker="*", c="#118844",
                   edgecolor="black", linewidths=0.6, zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.tick_params(labelsize=8)
    fig.suptitle("PCA projection (illustration only; the two components capture %.1f%% of "
                 "the embedding variance. The full-space distances are the evidence)."
                 % (100 * evr.sum()), fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)
    return [float(x) for x in evr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results_diag/traces_gpt2sft_plot_traj.npz")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--numbers", default="results_revision/numbers.json")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    npz = dict(np.load(args.npz, allow_pickle=True))
    wte = load_wte(args.checkpoint)
    wte_sq = (wte ** 2).sum(1)
    tok = load_tok(args.checkpoint)
    seq_idx = int(np.random.default_rng(0).integers(0, 6))

    landing = primary_figure(npz, wte, wte_sq, tok, seq_idx,
                             os.path.join(args.fig_dir, "fig_traj_distance.png"))
    evr = pca_secondary(npz, wte, tok, seq_idx,
                        os.path.join(args.fig_dir, "fig_traj_pca.png"))
    stats = build_seq_stats(npz, wte, wte_sq)

    anis = {}
    if os.path.exists(args.numbers):
        nums = json.load(open(args.numbers))
        a = nums.get("diag_anisotropy_gpt2sft", {})
        anis = {"mean_nearest_neighbour_l2": a.get("mean_nearest_neighbour_l2"),
                "mean_pairwise_l2": a.get("mean_pairwise_l2"),
                "source": "results_revision/numbers.json:diag_anisotropy_gpt2sft"}

    out = {
        "experiment": "trajectory_stats",
        "npz": args.npz, "checkpoint": args.checkpoint,
        "seq_idx_seeded": seq_idx,
        "seq_idx_rule": "int(np.random.default_rng(0).integers(0,6))",
        "trace_index_caveat": ("collect_traces advances the torch RNG across configs, so "
                               "trace index i is the same source sentence but the masked "
                               "position (hence gt token) differs per config; each panel "
                               "uses its own exact gt_emb."),
        "token_spacing_reference": anis,
        "per_config": stats,
        "chosen_seq_landing": landing,
        "pca_explained_variance_ratio_2comp": evr,
        "pca_explained_variance_sum": float(sum(evr)),
    }
    # factor summary for the thesis prose (off-manifold vs token spacing)
    nn = anis.get("mean_nearest_neighbour_l2")
    pw = anis.get("mean_pairwise_l2")
    cls_mh = stats["cls_policy_gnoff_mh"]
    cls_nomh = stats["cls_policy_gnoff_nomh"]
    if nn and pw:
        out["offmanifold_factor_summary"] = {
            "cls_mh_on_nn_dist_range": [cls_mh["nearest_token_dist_min"],
                                        cls_mh["nearest_token_dist_max"]],
            "cls_mh_off_nn_dist_max": cls_nomh["nearest_token_dist_max"],
            "token_nn_spacing": nn, "token_mean_pairwise": pw,
            "factor_vs_nn_spacing_mh_on": [round(cls_mh["nearest_token_dist_min"] / nn, 1),
                                           round(cls_mh["nearest_token_dist_max"] / nn, 1)],
        }
    jpath = os.path.join(args.fig_dir, "fig_traj_stats.json")
    with open(jpath + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(jpath + ".tmp", jpath)
    print("wrote", jpath)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
