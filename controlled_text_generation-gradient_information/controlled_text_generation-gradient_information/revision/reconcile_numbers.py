#!/usr/bin/env python
"""
reconcile_numbers.py

CONCERN 6 (internal numerical inconsistencies), with helpers for CONCERN 15
(config count) and CONCERN 17 (Spearman phrasing). Pure analysis.

What it does:

1. NUMBERS INDEX. Walks the result dirs, loads every *.json summary, and emits a
   flat numbers.json mapping run_name -> the headline metrics (final_kl,
   accept_rate, accuracy, spearman, ...). This is the table you diff the LaTeX
   against as the final pre-submission gate (concern 6 step 6).

2. LENGTH SLOPE, censored and uncensored (concern 6b). Recomputes the
   length-vs-total_logp regression from the likelihood_trap CSVs both including
   and excluding generations pinned at the length cap (hit_cap==1 or scored_len
   at the max). Reports both slopes so the -1.12 vs -0.11 discrepancy resolves to
   a stated censoring choice rather than a contradiction.

3. MH ACCEPTANCE by boundary crossing (concern 6a, concern 7). If the analyze_mh
   CSV is present (traces_*_mh.csv), reports within-cell (crossed==0) and
   boundary (crossed==1) acceptance per config, so each reported percentage is
   tied to the config it came from.

4. CONFIG COUNT (concern 15). Counts distinct configs present in the result dirs
   and prints the full-factorial arithmetic beside it, so the "145 configurations"
   claim has a formula.

5. SPEARMAN PHRASING (concern 17). Pulls the linearization spearman and n_pairs
   and emits the effect-size sentence ("|rho| < X across all models, n = ...").

Usage:
  python reconcile_numbers.py --results_dirs results_gpt2 results_llama results_diag \
      --run_name reconcile --out_dir results_revision
"""

import argparse
import glob
import json
import os

import numpy as np


def load_jsons(dirs):
    out = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(path) as f:
                    out[path] = json.load(f)
            except Exception:
                pass
    return out


def numbers_index(jsons):
    idx = {}
    for path, j in jsons.items():
        rn = j.get("run_name", os.path.basename(path)[:-5])
        keep = {}
        for k in ("accuracy", "accept_rate", "n",
                  "spearman_surrogate_vs_true_ALL", "pearson_surrogate_vs_true_ALL",
                  "spearman_surrogate_vs_true_SELF", "spearman_surrogate_vs_true_FUTURE",
                  "n_pairs", "mean_inter_token_distance",
                  "mean_pairwise_l2", "mean_nearest_neighbour_l2",
                  "mean_abs_loglik_diff", "pearson_base_tuned_loglik",
                  "frac_emitted_eos", "frac_hit_cap"):
            if k in j:
                keep[k] = j[k]
        if isinstance(j.get("mean_kl"), list) and j["mean_kl"]:
            keep["final_kl"] = j["mean_kl"][-1]
        if isinstance(j.get("mean_l2"), list) and j["mean_l2"]:
            keep["final_l2"] = j["mean_l2"][-1]
        idx[rn] = keep
    return idx


def length_slopes(dirs):
    import pandas as pd
    from scipy.stats import linregress
    res = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*likelihood_trap*.csv")):
            df = pd.read_csv(path)
            df = df[df.scored_len > 0]
            samp = df[df.strategy.isin(["ancestral", "topp90", "temp07"])].copy()
            if len(samp) < 30 or samp.scored_len.nunique() < 5:
                continue
            cap = samp.scored_len.max()
            uncensored = samp
            censored = samp[(samp.hit_cap == 0) & (samp.scored_len < cap)] \
                if "hit_cap" in samp.columns else samp[samp.scored_len < cap]
            entry = {}
            for name, frame in [("uncensored", uncensored), ("censored", censored)]:
                if len(frame) >= 30 and frame.scored_len.nunique() >= 5:
                    lr = linregress(frame.scored_len, frame.total_logp)
                    entry[name] = {"slope_nats_per_token": float(lr.slope),
                                   "r": float(lr.rvalue), "n": int(len(frame))}
                else:
                    entry[name] = None
            res[os.path.basename(path)] = entry
    return res


def mh_acceptance_by_boundary(dirs):
    import pandas as pd
    res = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*_mh.csv")):
            df = pd.read_csv(path)
            if "crossed" not in df.columns or "accepted" not in df.columns:
                continue
            keys = [c for c in ["sampler", "method", "grad_norm"] if c in df.columns]
            grouped = {}
            gb = df.groupby(keys) if keys else [("all", df)]
            for gkey, sub in (gb if keys else gb):
                within = sub[sub.crossed == 0]
                boundary = sub[sub.crossed == 1]
                grouped[str(gkey)] = {
                    "within_cell_accept_pct": float(100 * within.accepted.mean()) if len(within) else None,
                    "within_cell_n": int(len(within)),
                    "boundary_accept_pct": float(100 * boundary.accepted.mean()) if len(boundary) else None,
                    "boundary_n": int(len(boundary)),
                }
            res[os.path.basename(path)] = grouped
    return res


def config_count(jsons):
    configs = set()
    for path, j in jsons.items():
        rn = j.get("run_name", "")
        if rn:
            configs.add(rn)
    # full factorial the thesis quotes: 2 samplers x 3 proposals x 2 MH x 2 gn
    # x 2 schedules x 5 models = 240
    factorial = 2 * 3 * 2 * 2 * 2 * 5
    return {"distinct_run_names_present": len(configs),
            "full_factorial_2x3x2x2x2x5": factorial,
            "note": ("Enumerate the pruned cells (oracle counted separately, Llama "
                     "on a subset, CLS proposals reduced to policy/random) in an "
                     "appendix so the count lands on the reported number.")}


def spearman_phrasing(idx):
    rhos = {rn: v["spearman_surrogate_vs_true_ALL"]
            for rn, v in idx.items() if "spearman_surrogate_vs_true_ALL" in v}
    if not rhos:
        return {}
    maxabs = max(abs(x) for x in rhos.values())
    ns = [v.get("n_pairs") for v in idx.values() if v.get("n_pairs")]
    return {"per_run_spearman": rhos,
            "max_abs_spearman": float(maxabs),
            "suggested_sentence": (f"The surrogate-vs-truth rank correlation is "
                                   f"negligible in magnitude (|rho| < {maxabs:.2f}) "
                                   f"across all models"
                                   + (f", n = {min(ns):,}-{max(ns):,} candidate pairs." if ns else "."))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+", required=True)
    p.add_argument("--run_name", default="reconcile")
    p.add_argument("--out_dir", default="results_revision")
    args = p.parse_args()

    jsons = load_jsons(args.results_dirs)
    idx = numbers_index(jsons)
    report = {
        "experiment": "reconcile_numbers",
        "numbers_index": idx,
        "length_slopes_censored_vs_uncensored": length_slopes(args.results_dirs),
        "mh_acceptance_by_boundary": mh_acceptance_by_boundary(args.results_dirs),
        "config_count": config_count(jsons),
        "spearman_phrasing": spearman_phrasing(idx),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, dst)
    # also drop the flat numbers.json for the LaTeX diff
    with open(os.path.join(args.out_dir, "numbers.json"), "w") as f:
        json.dump(idx, f, indent=2)
    print(f"[reconcile] indexed {len(idx)} runs -> {dst}")
    ls = report["length_slopes_censored_vs_uncensored"]
    for k, v in ls.items():
        u = v.get("uncensored", {}) or {}
        c = v.get("censored", {}) or {}
        print(f"  {k}: uncensored slope={u.get('slope_nats_per_token')}, "
              f"censored slope={c.get('slope_nats_per_token')}")


if __name__ == "__main__":
    main()
