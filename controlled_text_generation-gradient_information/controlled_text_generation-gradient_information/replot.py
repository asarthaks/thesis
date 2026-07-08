#!/usr/bin/env python3
"""
replot.py

Turns the per-run CSVs written by run_experiment.py into the thesis figure panels,
using the notebook's own prepare_dataset and plot_graphs (imported from
notebook_plotting.py) so the styling is identical to the figures already in the
thesis.

A "figure" is one trajectory panel set that overlays all methods and both MH
settings, exactly like the notebook. Runs are grouped by everything in the run
name except method and mh, i.e. (model, sampler, grad_norm, oracle, steps). For
each group this script:
  1. concatenates the group's per-run CSVs into one combined legacy-schema CSV
     (which you can also drop straight into the original notebook's filename list),
  2. renders it with plot_graphs, saving <figure>_new_trajectories.png and _bars.png,
  3. if --old_dir is given, finds the matching old CSV and renders it the same way
     as <figure>_old_*.png, so you get a like-for-like pair to compare.

Usage:
  python replot.py --new_dir results_gpt2 --out_dir figures_gpt2
  python replot.py --new_dir results_gpt2 --old_dir Experiments/IdealAlphaSchedule/GradientInformation/results --out_dir figures_gpt2
"""

import os
import re
import glob
import json
import argparse
import pandas as pd

from notebook_plotting import prepare_dataset, plot_graphs


METHOD_LABEL = {
    "policy": "{m} Policy",
    "grad_norm_preserved_random_dir": "Grad Norm Noisy",
    "random": "Random Noise",
}


def build_from_json(json_paths, config):
    """Build the same traj_df / summary_df the notebook produces, but from the
    per-step mean curves in the run JSONs. Used for runs made before per-sample
    CSVs were written. The notebook draws mean lines (errorbar=None), so a single
    mean 'sample' per config reproduces the trajectory panels exactly."""
    expanded, summary = [], []
    for p in json_paths:
        d = json.load(open(p))
        cfg = d.get("config", {})
        method = cfg.get("method", "policy")
        mh = cfg.get("mh", False)
        mname = METHOD_LABEL.get(method, "{m}").format(m=config["method"])
        mh_label = "With MH" if mh else "No MH"
        l2, kl, ent = d["mean_l2"], d["mean_kl"], d["mean_entropy"]
        for k in range(len(l2)):
            expanded.append({
                "Sample": 0, "Method": mname, "Sampling": mh_label, "Step": k,
                "L2 Distance": l2[k], "KL Divergence": kl[k], "Entropy": ent[k],
            })
        summary.append({
            "Sample": 0, "Method": mname, "Sampling": mh_label,
            "Final L2": l2[-1], "Final KL": kl[-1],
            "Accuracy %": d.get("accuracy", float("nan")),
        })
    return pd.DataFrame(expanded), pd.DataFrame(summary)


def parse_run_name(name):
    # model.sampler.method.(mh|nomh).(gn|nogn).(oracle|free).sSTEPS
    m = re.match(r"(?P<model>.+?)\.(?P<sampler>dls|cls)\.(?P<method>.+?)\."
                 r"(?P<mh>mh|nomh)\.(?P<gn>gn|nogn)\.(?P<oracle>oracle|free)\.s(?P<steps>\d+)$", name)
    if not m:
        return None
    d = m.groupdict()
    d["steps"] = int(d["steps"])
    return d


def figure_key(d):
    # everything except method and mh defines one overlaid figure
    return (d["model"], d["sampler"], d["gn"], d["oracle"], d["steps"])


def config_for(key):
    model, sampler, gn, oracle, steps = key
    return {
        "method": "CLS" if sampler == "cls" else "DLS",
        "tokens_masked": "multi",
        "peft": model if "gfn" in model else "OFF",
        "oracle": "on" if oracle == "oracle" else "off",
        "gradient_normalization": "on" if gn == "gn" else "off",
    }


def figure_label(key):
    model, sampler, gn, oracle, steps = key
    return f"{model}.{sampler}.{gn}.{oracle}.s{steps}"


def find_old_csv(old_dir, key):
    """Best-effort match of an original legacy CSV to a new figure by sampler, steps,
    grad-norm and oracle. Prefers a non-PEFT file when several match."""
    _, sampler, gn, oracle, steps = key
    gn_flag = "gn_True" if gn == "gn" else "gn_False"
    orc_flag = "oracle_True" if oracle == "oracle" else "oracle_False"
    cands = []
    for p in glob.glob(os.path.join(old_dir, "*.csv")):
        b = os.path.basename(p)
        if sampler in b and f"steps_{steps}" in b and gn_flag in b and orc_flag in b:
            cands.append(p)
    if not cands:
        return None
    non_peft = [c for c in cands if "peft" not in os.path.basename(c).lower()]
    return (non_peft or cands)[0]


def render(csv_path, config, save_prefix, markers=False):
    try:
        traj_df, summary_df = prepare_dataset(csv_path, config)
        plot_graphs(traj_df, summary_df, config, save_prefix=save_prefix, markers=markers)
        return True
    except Exception as e:
        print(f"  could not render {csv_path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_dir", required=True, help="dir of per-run CSVs from run_experiment.py")
    ap.add_argument("--old_dir", default=None, help="dir of original legacy CSVs (optional, for comparison)")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--markers", action="store_true",
                    help="overlay sparse per-method markers so coincident trajectory lines "
                         "(the three methods under grad-norm) stay distinguishable. Off by default "
                         "so figures match the notebook exactly for old-vs-new comparison.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined_dir = os.path.join(args.out_dir, "combined_csv")
    os.makedirs(combined_dir, exist_ok=True)

    # group per-run files into figures. Collect both CSVs (preferred, per-sample) and
    # JSONs (fallback, mean curves) so runs made before CSV output still plot.
    groups = {}
    for ext in ("csv", "json"):
        for p in glob.glob(os.path.join(args.new_dir, f"*.{ext}")):
            name = os.path.basename(p)[:-(len(ext) + 1)]
            d = parse_run_name(name)
            if d is None:
                continue
            g = groups.setdefault(figure_key(d), {"csv": [], "json": []})
            g[ext].append(p)

    if not groups:
        print(f"no parseable per-run files in {args.new_dir}")
        return

    print(f"found {len(groups)} figure group(s)\n")
    for key, files in sorted(groups.items()):
        label = figure_label(key)
        cfg = config_for(key)
        use = "csv" if files["csv"] else "json"
        print(f"=== {label}  ({len(files[use])} runs, source={use}) ===")

        if use == "csv":
            # concatenate into one legacy-schema CSV (notebook-ready) and render via
            # the notebook's own prepare_dataset path.
            combined = pd.concat([pd.read_csv(p) for p in files["csv"]], ignore_index=True)
            gn_flag = "gn_True" if key[2] == "gn" else "gn_False"
            orc_flag = "oracle_True" if key[3] == "oracle" else "oracle_False"
            combined_name = (f"experiment_results_{key[1]}_multi_steps_{key[4]}_"
                             f"{gn_flag}_{orc_flag}_model_{key[0]}.csv")
            combined_path = os.path.join(combined_dir, combined_name)
            combined.to_csv(combined_path, index=False)
            render(combined_path, cfg, os.path.join(args.out_dir, f"{label}_new"), markers=args.markers)
        else:
            # build the notebook dataframes straight from the JSON mean curves.
            try:
                traj_df, summary_df = build_from_json(files["json"], cfg)
                plot_graphs(traj_df, summary_df, cfg,
                            save_prefix=os.path.join(args.out_dir, f"{label}_new"),
                            markers=args.markers)
            except Exception as e:
                print(f"  could not render from json: {e}")

        # render matched OLD, if available
        if args.old_dir:
            old_csv = find_old_csv(args.old_dir, key)
            if old_csv:
                print(f"  matched old: {os.path.basename(old_csv)}")
                render(old_csv, cfg, os.path.join(args.out_dir, f"{label}_old"))
            else:
                print("  no matching old CSV found for this figure")
        print()

    print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
