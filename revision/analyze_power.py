"""Pool the flagship configuration over independent corruption seeds for an n~1000 paired test.

WikiText-2 validation yields only about 282 sentences under the 10-to-40-word filter, so
extra sample size cannot come from new sentences. It comes instead from independent
corruptions of the same sentences: `--data_seed S` offsets the per-sentence corruption seed,
so seed 0 and seed 1000 mask different positions in the same text. Pairs are formed on
(seed, sample_idx), which is exact, because corruption is deterministic in that key and
identical across the three proposal arms.

This is disclosed rather than presented as new sentences: the sequences repeat across seeds,
so the effective sample size for sentence-level variation is still 282, and the CI below is
correspondingly a statement about corruption-level rather than corpus-level variation.

    python revision/analyze_power.py --run_name rev_power --out_dir results/revision
"""
import argparse
import ast
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARMS = {"policy": "policy", "gnp": "grad_norm_preserved_random_dir", "random": "random"}


def traj_stats(csv_path):
    df = pd.read_csv(csv_path).sort_values("sample_idx")
    kl = np.array([[d["avg_kl_divergence"] for d in ast.literal_eval(t)]
                   for t in df["trajectory"]], dtype=float)
    half = kl.shape[1] // 2
    return df["sample_idx"].to_numpy(), {
        "last": kl[:, -1],
        "chain_mean": kl[:, half:].mean(axis=1),
        "chain_min": kl.min(axis=1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rev3_dir", default="results/grid/rev3")
    p.add_argument("--seed0_dir", default="results/grid/gpt2_v2")
    p.add_argument("--seed0_prefix", default="gpt2-large.dls.{arm}.mh.gn.free.s50")
    p.add_argument("--run_name", default="rev_power")
    p.add_argument("--out_dir", default="results/revision")
    p.add_argument("--margin", type=float, default=0.327)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pooled = {a: {} for a in ARMS}          # arm -> {(seed, idx): stats}
    for short, full in ARMS.items():
        f0 = os.path.join(args.seed0_dir, args.seed0_prefix.format(arm=full) + ".csv")
        if os.path.exists(f0):
            idx, st = traj_stats(f0)
            for k, i in enumerate(idx):
                pooled[short][(0, int(i))] = {s: st[s][k] for s in st}
        for f in sorted(glob.glob(os.path.join(args.rev3_dir, f"power_{short}_seed*.csv"))):
            sd = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            idx, st = traj_stats(f)
            for k, i in enumerate(idx):
                pooled[short][(sd, int(i))] = {s: st[s][k] for s in st}

    rng = np.random.default_rng(args.seed)
    out = {"experiment": "pooled_power_flagship", "run_name": args.run_name,
           "margin": args.margin, "n_boot": args.n_boot,
           "note": ("n comes from independent corruptions of the same 282 WikiText-2 sentences; "
                    "pairs are on (data_seed, sample_idx)."),
           "n_per_arm": {a: len(pooled[a]) for a in pooled}, "contrasts": {}}

    for comp in ("gnp", "random"):
        keys = sorted(set(pooled["policy"]) & set(pooled[comp]))
        if not keys:
            continue
        entry = {"n_paired": len(keys),
                 "n_seeds": len(sorted({k[0] for k in keys}))}
        for stat in ("last", "chain_mean", "chain_min"):
            d = np.array([pooled["policy"][k][stat] - pooled[comp][k][stat] for k in keys])
            n = len(d)
            boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry[stat] = {
                "policy_mean": float(np.mean([pooled["policy"][k][stat] for k in keys])),
                "comparator_mean": float(np.mean([pooled[comp][k][stat] for k in keys])),
                "mean_diff": float(d.mean()),
                "ci95": [float(lo), float(hi)],
                "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                "tost_equivalent": bool(lo > -args.margin and hi < args.margin),
            }
        out["contrasts"][comp] = entry
        print(f"policy vs {comp}: n={entry['n_paired']} over {entry['n_seeds']} seeds")
        for stat in ("last", "chain_mean", "chain_min"):
            e = entry[stat]
            print(f"   {stat:11s} {e['policy_mean']:.3f} vs {e['comparator_mean']:.3f}  "
                  f"diff {e['mean_diff']:+.4f}  CI [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]  "
                  f"p={e['wilcoxon_p']:.3f}  TOST {'PASS' if e['tost_equivalent'] else 'fail'}")

    dest = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
