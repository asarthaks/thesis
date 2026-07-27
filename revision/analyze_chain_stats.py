"""Chain statistics for the whole sampling grid, plus the paired forest plot.

Two problems with the grid as originally reported.

1. Every headline number is the KL at the FINAL step of the chain. For an MCMC
   method that is the wrong summary: it discards every state the chain visited.
   This script recomputes, from the per-sample trajectories already stored in the
   grid CSVs, four statistics per sample:

       last        KL at the final step                       (what the thesis reported)
       chain_mean  mean KL over the second half of the schedule
       chain_min   minimum KL over the trajectory
       frac_below  fraction of steps with KL below --threshold

2. The paired policy-versus-random contrast is reported as point estimates for
   most of the grid. This script runs the paired comparison for EVERY configuration
   that has at least two arms, with a percentile bootstrap CI, a Wilcoxon signed-rank
   test, and a two one-sided tests (TOST) equivalence verdict at --margin, and draws
   the forest plot of those contrasts with the equivalence band.

Pairing is on sample_idx, which is safe because corruption is deterministic per
index (seed = data_seed + running index), identical across the arms of a config.

Usage:
    python revision/analyze_chain_stats.py \
        --results_dirs results/grid/gpt2_v2 results/grid/llama results/grid/gfn \
        --run_name rev_chain_stats --out_dir results/revision --fig_dir figures
"""
import argparse
import ast
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARMS = ["policy", "grad_norm_preserved_random_dir", "random", "policy_onehot"]
SHORT = {"policy": "policy", "grad_norm_preserved_random_dir": "gnp-random",
         "random": "random", "policy_onehot": "one-hot"}
STATS = ["last", "chain_mean", "chain_min", "frac_below"]


def parse_run_name(name):
    """{model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps} -> (family, method)."""
    parts = name.split(".")
    if len(parts) < 7:
        return None, None
    model, sampler, method = parts[0], parts[1], ".".join(parts[2:-4])
    family = ".".join([model, sampler] + parts[-4:])
    return family, method


def sample_stats(csv_path, threshold):
    df = pd.read_csv(csv_path).sort_values("sample_idx")
    kl = np.array([[d["avg_kl_divergence"] for d in ast.literal_eval(t)]
                   for t in df["trajectory"]], dtype=float)
    half = kl.shape[1] // 2
    return df["sample_idx"].to_numpy(), {
        "last": kl[:, -1],
        "chain_mean": kl[:, half:].mean(axis=1),
        "chain_min": kl.min(axis=1),
        "frac_below": (kl < threshold).mean(axis=1),
    }


def paired_test(d, margin, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        pval = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        pval = float("nan")
    return {
        "n": int(n),
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "wilcoxon_p": pval,
        # TOST at +-margin is equivalent to the 90% CI lying inside the band; the
        # 95% CI used here is the conservative reading and is what the plot draws.
        "tost_equivalent": bool(lo > -margin and hi < margin),
    }


def forest_plot(rows, margin, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["label"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.30 * n + 1.4)))
    ax.axvspan(-margin, margin, color="0.88", zorder=0,
               label=f"equivalence margin $\\pm${margin:.3f} nats")
    ax.axvline(0.0, color="0.35", lw=0.9, zorder=1)
    for i, r in enumerate(rows):
        lo, hi = r["ci95"]
        crosses = lo <= 0.0 <= hi
        colour = "#2b6cb0" if crosses else "#c05621"
        ax.plot([lo, hi], [i, i], color=colour, lw=1.6, zorder=2)
        ax.plot([r["mean_diff"]], [i], "o", ms=4.5, color=colour, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.4)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("paired mean KL difference, policy minus norm-matched random (nats)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=6.6, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {path}.pdf / .png ({n} configurations)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+",
                   default=["results/grid/gpt2_v2", "results/grid/llama", "results/grid/gfn"])
    p.add_argument("--run_name", default="rev_chain_stats")
    p.add_argument("--out_dir", default="results/revision")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.327,
                   help="pre-registered equivalence margin, 5 percent of the policy mean KL")
    p.add_argument("--reference_arm", default="grad_norm_preserved_random_dir")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    families = {}
    for d in args.results_dirs:
        for csv_path in sorted(glob.glob(os.path.join(d, "*.csv"))):
            run = os.path.basename(csv_path)[:-4]
            family, method = parse_run_name(run)
            if family is None or method not in ARMS:
                continue
            try:
                idx, st = sample_stats(csv_path, args.threshold)
            except Exception as exc:                       # noqa: BLE001
                print(f"[skip] {run}: {exc}")
                continue
            families.setdefault(family, {})[method] = {"idx": idx, "stats": st,
                                                       "run": run, "dir": d}

    per_config, contrasts = {}, {}
    for family, arms in sorted(families.items()):
        per_config[family] = {
            SHORT[m]: {s: float(np.mean(a["stats"][s])) for s in STATS}
            for m, a in arms.items()
        }
        if "policy" not in arms or args.reference_arm not in arms:
            continue
        a, b = arms["policy"], arms[args.reference_arm]
        common = np.intersect1d(a["idx"], b["idx"])
        if len(common) < 20:
            continue
        sel_a = np.isin(a["idx"], common)
        sel_b = np.isin(b["idx"], common)
        entry = {}
        for s in STATS:
            entry[s] = paired_test(a["stats"][s][sel_a] - b["stats"][s][sel_b],
                                   args.margin, args.n_boot, args.seed)
        contrasts[family] = entry

    rows = [{"label": f, "mean_diff": c["chain_mean"]["mean_diff"],
             "ci95": c["chain_mean"]["ci95"]} for f, c in contrasts.items()]
    if rows:
        forest_plot(rows, args.margin, os.path.join(args.fig_dir, "fig_forest_chain"),
                    "Paired policy minus norm-matched random direction, "
                    "chain mean KL over the second half of the schedule")
    rows_last = [{"label": f, "mean_diff": c["last"]["mean_diff"],
                  "ci95": c["last"]["ci95"]} for f, c in contrasts.items()]
    if rows_last:
        forest_plot(rows_last, args.margin, os.path.join(args.fig_dir, "fig_forest_last"),
                    "Paired policy minus norm-matched random direction, final-step KL")

    n_equiv = sum(1 for c in contrasts.values() if c["chain_mean"]["tost_equivalent"])
    n_sig = sum(1 for c in contrasts.values()
                if not (c["chain_mean"]["ci95"][0] <= 0.0 <= c["chain_mean"]["ci95"][1]))
    print(f"\n{len(contrasts)} paired configurations; chain-mean statistic: "
          f"{n_equiv} equivalent at +-{args.margin}, {n_sig} with a CI excluding zero")

    out = {
        "experiment": "chain_statistics_and_paired_contrasts",
        "run_name": args.run_name,
        "threshold": args.threshold,
        "margin": args.margin,
        "reference_arm": args.reference_arm,
        "n_boot": args.n_boot,
        "per_config_means": per_config,
        "paired_contrasts": contrasts,
        "summary": {"n_configs_paired": len(contrasts),
                    "n_tost_equivalent_chain_mean": n_equiv,
                    "n_ci_excludes_zero_chain_mean": n_sig},
    }
    dest = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
