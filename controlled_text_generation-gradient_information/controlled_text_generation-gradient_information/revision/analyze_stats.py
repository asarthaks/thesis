#!/usr/bin/env python
"""
analyze_stats.py

CONCERN 1: put statistical machinery behind the null result. Pure analysis on the
per-sample CSVs run_experiment.py already wrote (no reruns, no GPU).

The grid CSV schema (from run_experiment.py) is:
    sample_idx, method, mh_enabled, trajectory, avg_l2_dist, avg_kl_div, accuracy_pct
and run files are named <run_name>.csv where
    run_name = {model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps}

Corruption is deterministic per sample_idx (seed = data_seed + ti), and identical
across method runs of the same config, so sample_idx pairs policy against
grad-norm-preserved and against random within a config, sequence by sequence.

For every config group we report, per comparator (policy - comparator):
    mean paired difference in final KL
    95% bootstrap CI (10k paired resamples)
    Wilcoxon signed-rank p
    TOST equivalence at a declared margin, with the equivalence verdict
    the retrospective 80%-power minimum detectable difference

The equivalence margin is declared ONCE here and should be quoted in the
methodology chapter, not results, so it does not read as post hoc. Default:
5% of the policy mean final KL for that config. Override with --margin.

Run:
  python analyze_stats.py --results_dir results_gpt2 --run_name stats_gpt2 \
      --out_dir results_revision
  python analyze_stats.py --selftest      # validates the math on synthetic data
"""

import argparse
import glob
import json
import os

import numpy as np


def parse_run_name(rn):
    # {model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps}
    parts = rn.split(".")
    if len(parts) < 7:
        return None
    # method can itself contain no dots (it is one of the three fixed tokens)
    # find the method token by matching known values
    methods = {"policy", "grad_norm_preserved_random_dir", "random"}
    try:
        mi = next(i for i, p in enumerate(parts) if p in methods)
    except StopIteration:
        return None
    model = ".".join(parts[:mi])
    method = parts[mi]
    rest = parts[mi + 1:]
    # rest = [sampler?...] actually sampler comes BEFORE method in the schema:
    # {model}.{sampler}.{method}.{mh}.{gn}.{oracle}.sN
    sampler = parts[mi - 1]
    model = ".".join(parts[:mi - 1])
    tail = parts[mi + 1:]
    keys = ["mh", "gn", "oracle", "steps"]
    d = {"model": model, "sampler": sampler, "method": method}
    for k, v in zip(keys, tail):
        d[k] = v
    d["group"] = (model, sampler, d.get("mh"), d.get("gn"), d.get("oracle"), d.get("steps"))
    return d


def paired_bootstrap(diff, n_boot=10000, alpha=0.05, seed=0):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=len(diff))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi), n=int(len(diff)))


def wilcoxon_p(diff):
    from scipy.stats import wilcoxon
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    nz = diff[diff != 0]
    if len(nz) < 1:
        return float("nan")
    try:
        return float(wilcoxon(nz)[1])
    except Exception:
        return float("nan")


def tost(diff, margin):
    """
    Two one-sided tests for equivalence within +/- margin.
    Equivalence is declared if the 90% CI of the mean lies inside (-margin, margin),
    equivalently both one-sided p < 0.05.
    """
    from scipy import stats
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 3 or margin <= 0:
        return dict(p_lower=float("nan"), p_upper=float("nan"),
                    equivalent=False, margin=float(margin))
    mbar = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        eq = abs(mbar) < margin
        return dict(p_lower=0.0 if eq else 1.0, p_upper=0.0 if eq else 1.0,
                    equivalent=bool(eq), margin=float(margin), mean=float(mbar), se=0.0)
    df = n - 1
    t_lower = (mbar + margin) / se     # H0: mu <= -margin
    t_upper = (mbar - margin) / se     # H0: mu >=  margin
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    equivalent = (p_lower < 0.05) and (p_upper < 0.05)
    return dict(p_lower=float(p_lower), p_upper=float(p_upper),
                equivalent=bool(equivalent), margin=float(margin),
                mean=float(mbar), se=float(se))


def min_detectable_diff(diff, power=0.8, alpha=0.05):
    """Smallest true paired difference detectable at `power` given observed sd and n."""
    from scipy import stats
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 3:
        return float("nan")
    sd = diff.std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd / np.sqrt(n))


def load_run_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    # tolerate either avg_kl_div or avg_kl_divergence naming
    if "avg_kl_div" not in df.columns and "avg_kl_divergence" in df.columns:
        df = df.rename(columns={"avg_kl_divergence": "avg_kl_div"})
    return df[["sample_idx", "avg_kl_div"]].dropna()


def analyze(results_dir, margin_frac, margin_abs, seed):
    import pandas as pd
    runs = {}
    for path in glob.glob(os.path.join(results_dir, "*.csv")):
        rn = os.path.basename(path)[:-4]
        meta = parse_run_name(rn)
        if meta is None:
            continue
        meta["path"] = path
        runs[rn] = meta

    # group by config, collect the method -> csv path within each
    groups = {}
    for rn, meta in runs.items():
        groups.setdefault(meta["group"], {})[meta["method"]] = meta["path"]

    report = {"n_groups": 0, "comparisons": []}
    for grp, bym in groups.items():
        if "policy" not in bym:
            continue
        pol = load_run_csv(bym["policy"]).rename(columns={"avg_kl_div": "kl_policy"})
        for comp in ("grad_norm_preserved_random_dir", "random"):
            if comp not in bym:
                continue
            other = load_run_csv(bym[comp]).rename(columns={"avg_kl_div": "kl_other"})
            merged = pol.merge(other, on="sample_idx", how="inner")
            if len(merged) < 3:
                continue
            diff = (merged.kl_policy - merged.kl_other).values   # policy - comparator
            base = float(merged.kl_policy.mean())
            margin = margin_abs if margin_abs is not None else margin_frac * base
            boot = paired_bootstrap(diff, seed=seed)
            entry = {
                "model": grp[0], "sampler": grp[1], "mh": grp[2], "gn": grp[3],
                "oracle": grp[4], "steps": grp[5],
                "comparator": comp, "n_paired": int(len(merged)),
                "policy_mean_kl": base,
                "comparator_mean_kl": float(merged.kl_other.mean()),
                "mean_diff_policy_minus_comp": boot["mean"],
                "diff_ci95": [boot["lo"], boot["hi"]],
                "wilcoxon_p": wilcoxon_p(diff),
                "tost": tost(diff, margin),
                "min_detectable_diff_80pow": min_detectable_diff(diff),
            }
            # concern 1 step 6: flag len_beta=1 variants for a separate reading
            entry["is_len_beta1_variant"] = ("lb1" in str(grp[0]))
            report["comparisons"].append(entry)
    report["n_groups"] = len(groups)
    report["margin_rule"] = (f"absolute {margin_abs}" if margin_abs is not None
                             else f"{margin_frac:.3f} x policy_mean_kl (per config)")
    return report


def selftest():
    rng = np.random.RandomState(0)
    # truly equivalent: diffs centered at 0 with small noise
    eq = rng.normal(0.0, 0.2, size=200)
    # not equivalent: diffs centered at 0.67 (the len_beta=1 case)
    neq = rng.normal(0.67, 0.3, size=200)
    print("equivalent case:")
    print("  boot", paired_bootstrap(eq))
    print("  tost(margin=0.3)", tost(eq, 0.3))
    print("  mdd", min_detectable_diff(eq))
    print("non-equivalent case (len_beta=1 style, diff=0.67):")
    print("  boot", paired_bootstrap(neq))
    print("  tost(margin=0.3)", tost(neq, 0.3))
    print("  wilcoxon_p", wilcoxon_p(neq))
    assert tost(eq, 0.3)["equivalent"] is True
    assert tost(neq, 0.3)["equivalent"] is False
    print("OK: TOST flags equivalence for the null case and rejects it for 0.67.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", help="dir of grid *.csv files")
    p.add_argument("--run_name", default="stats")
    p.add_argument("--out_dir", default="results_revision")
    p.add_argument("--margin_frac", type=float, default=0.05,
                   help="equivalence margin as a fraction of policy mean KL")
    p.add_argument("--margin_abs", type=float, default=None,
                   help="absolute equivalence margin in KL nats; overrides --margin_frac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    assert args.results_dir, "--results_dir required (or use --selftest)"
    os.makedirs(args.out_dir, exist_ok=True)
    report = analyze(args.results_dir, args.margin_frac, args.margin_abs, args.seed)

    out = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, out)
    print(f"[analyze_stats] {len(report['comparisons'])} comparisons -> {out}")
    for c in report["comparisons"][:20]:
        eq = "EQUIV" if c["tost"]["equivalent"] else "not-equiv"
        print(f"  {c['model']}.{c['sampler']} mh={c['mh']} gn={c['gn']} "
              f"vs {c['comparator']}: diff={c['mean_diff_policy_minus_comp']:.3f} "
              f"CI{c['diff_ci95']} [{eq}] mdd={c['min_detectable_diff_80pow']:.3f}")


if __name__ == "__main__":
    main()
