#!/usr/bin/env python3
"""
compare_models.py

Overlays multiple models on the same axes for a fixed experimental configuration
(sampler, method, mh, gn, oracle, steps), so you can compare SFT vs the GFlowNet
variants directly. Reads the run JSONs (mean per-step curves), so it works on the
results you already have, no reruns.

IMPORTANT comparability note, enforced by the defaults:
  The GPT-2 SFT model and the three GFN variants share a tokenizer, corruptions
  (n=200, seed 0) and step-size schedule, so their curves are numerically
  comparable and overlaying them is valid. Llama uses a different tokenizer and a
  different step-size scale (its L2/KL live on a different axis), so it is NOT
  numerically comparable and is excluded from the default overlay. Add it with
  --include_llama only if you understand you are then eyeballing shape, not height.

Usage:
  # one figure per config, GPT-2 family overlaid
  python compare_models.py --results_dirs results_gpt2_v2 results_gfn --out_dir figures_compare

  # just the flagship config
  python compare_models.py --results_dirs results_gpt2_v2 results_gfn \
      --out_dir figures_compare --only dls.policy.mh.gn.free.s50
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# reference first, then GFN variants; controls colour + legend order
MODEL_ORDER = ["gpt2-large", "gfn-lb0-500", "gfn-lb0-2000", "gfn-lb1-500", "llama3-8b"]


def config_key(c):
    return (f"{c['sampler']}.{c['method']}."
            f"{'mh' if c['mh'] else 'nomh'}.{'gn' if c['grad_norm'] else 'nogn'}."
            f"{'oracle' if c['oracle'] else 'free'}.s{c['steps']}")


def load_all(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            j = json.load(open(f))
            c = j["config"]
            key = config_key(c)
            for k in range(len(j["mean_l2"])):
                rows.append({
                    "model": c["model_tag"], "config": key, "Step": k,
                    "L2 Distance": j["mean_l2"][k],
                    "KL Divergence": j["mean_kl"][k],
                    "Entropy": j["mean_entropy"][k],
                })
    return pd.DataFrame(rows)


def plot_config(df_cfg, cfg, out_dir, models):
    present = [m for m in models if m in set(df_cfg["model"])]
    if len(present) < 2:
        return False
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = {m: c for m, c in zip(MODEL_ORDER, sns.color_palette("tab10", len(MODEL_ORDER)))}

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Model comparison  |  {cfg}", fontsize=16, y=1.05)
    for ax, col, title in [(axes[0], "L2 Distance", "Embedding Distance (L2)"),
                           (axes[1], "KL Divergence", "Contextual Fit (KL)"),
                           (axes[2], "Entropy", "Prediction Uncertainty (Entropy)")]:
        sns.lineplot(data=df_cfg[df_cfg.model.isin(present)], x="Step", y=col, hue="model",
                     hue_order=present, palette=palette, ax=ax, linewidth=2.2, errorbar=None)
        ax.set_title(title)
        ax.set_ylabel(f"{col} (lower is better)" if col != "Entropy" else col)
        if ax is not axes[2]:
            leg = ax.get_legend()
            if leg:
                leg.remove()
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Model")
    plt.tight_layout()
    out = os.path.join(out_dir, f"compare_{cfg}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="figures_compare")
    ap.add_argument("--only", default=None, help="render just this config key (exact match)")
    ap.add_argument("--include_llama", action="store_true",
                    help="overlay Llama too. Off by default: Llama is on a different KL/L2 "
                         "scale (different tokenizer + schedule) and is not numerically comparable.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all(args.results_dirs)
    if df.empty:
        print("no JSONs found in", args.results_dirs)
        return

    models = list(MODEL_ORDER)
    if not args.include_llama:
        models = [m for m in models if m != "llama3-8b"]

    configs = sorted(df["config"].unique())
    if args.only:
        configs = [c for c in configs if c == args.only]

    made = 0
    summary = []
    for cfg in configs:
        df_cfg = df[df["config"] == cfg]
        if plot_config(df_cfg, cfg, args.out_dir, models):
            made += 1
        # summary row: final KL per model for this config
        finals = (df_cfg[df_cfg.Step == df_cfg.Step.max()]
                  .groupby("model")["KL Divergence"].first())
        summary.append({"config": cfg, **{m: round(finals.get(m, float("nan")), 3) for m in MODEL_ORDER}})

    sm = pd.DataFrame(summary).set_index("config")
    sm.to_csv(os.path.join(args.out_dir, "final_kl_by_model.csv"))
    print(f"wrote {made} comparison figures to {args.out_dir}/")
    print(f"summary table: {args.out_dir}/final_kl_by_model.csv")
    print("\nFinal KL by model (GPT-2 family is comparable; llama is different-scale):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm[[m for m in MODEL_ORDER if m in sm.columns]].to_string())


if __name__ == "__main__":
    main()
