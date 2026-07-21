#!/usr/bin/env python
"""
analyze_likelihood_trap.py

CONCERN 10: the +0.65 likelihood-vs-repetition correlation pools generations
across decoding strategies, so strategy and likelihood are confounded. This
re-computes the correlation WITHIN each decoding strategy (ancestral alone,
top-p alone, temperature alone), where strategy is held fixed, and also reports
the between-strategy pattern (greedy/beam vs sampling), which is the Holtzman et
al. framing.

Reads the CSVs run_diagnostic.py's likelihood_trap experiment wrote:
    seq_id, strategy, gen_len, scored_len, emitted_eos, hit_cap,
    total_logp, mean_logp, rep4, distinct2, text
One CSV per model, e.g. diag_likelihood_trap_gpt2sft.csv. Point --csv at one or
pass --results_dir to sweep all diag_likelihood_trap_*.csv (gpt2 and llama both,
so the trap is measured on both architectures).

Pure analysis, no GPU.
"""

import argparse
import glob
import json
import os

import numpy as np


def corr(a, b):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return dict(pearson=float("nan"), spearman=float("nan"), n=int(len(a)))
    return dict(pearson=float(pearsonr(a, b)[0]),
                spearman=float(spearmanr(a, b)[0]), n=int(len(a)))


def analyze_one(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df.scored_len > 0]
    res = {"csv": os.path.basename(csv_path), "n_rows": int(len(df))}

    sampling = ["ancestral", "topp90", "temp07"]

    res["within_strategy"] = {}
    for s in df.strategy.unique():
        sub = df[df.strategy == s]
        res["within_strategy"][s] = {
            "logp_vs_rep4": corr(sub.mean_logp, sub.rep4),
            "logp_vs_distinct2": corr(sub.mean_logp, sub.distinct2),
        }

    # pooled over sampling strategies (the number the thesis currently reports)
    pool = df[df.strategy.isin(sampling)]
    res["pooled_sampling_logp_vs_rep4"] = corr(pool.mean_logp, pool.rep4)

    # between-strategy: mean rep and mean logp per strategy, so the reader can see
    # greedy/beam sit at the high-likelihood, high-repetition corner
    bs = (df.groupby("strategy")
            .agg(mean_logp=("mean_logp", "mean"),
                 mean_rep4=("rep4", "mean"),
                 mean_distinct2=("distinct2", "mean"),
                 n=("seq_id", "count"))
            .reset_index())
    res["between_strategy"] = bs.to_dict(orient="records")

    # honest verdict for the write-up
    within = [v["logp_vs_rep4"]["pearson"] for v in res["within_strategy"].values()
              if not np.isnan(v["logp_vs_rep4"]["pearson"])]
    res["max_abs_within_strategy_pearson"] = float(np.nanmax(np.abs(within))) if within else float("nan")
    res["note"] = ("If within-strategy correlations are weak, present the "
                   "between-strategy pattern (greedy/beam worst) as the primary "
                   "evidence; that is what Holtzman et al. did and it is not a confound.")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--run_name", default="ltrap_within")
    p.add_argument("--out_dir", default="results_revision")
    args = p.parse_args()

    paths = []
    if args.csv:
        paths = [args.csv]
    elif args.results_dir:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "*likelihood_trap*.csv")))
    assert paths, "pass --csv or --results_dir with likelihood_trap CSVs"

    out = {"experiment": "likelihood_trap_within_strategy", "per_model": {}}
    for path in paths:
        key = os.path.basename(path)[:-4]
        out["per_model"][key] = analyze_one(path)

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dst)
    print(f"[likelihood_trap] wrote {dst}")
    for k, v in out["per_model"].items():
        print(f"  {k}: pooled r={v['pooled_sampling_logp_vs_rep4']['pearson']:.3f}, "
              f"max within-strategy |r|={v['max_abs_within_strategy_pearson']:.3f}")


if __name__ == "__main__":
    main()
