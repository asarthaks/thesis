"""Concern OH: the one-hot / simplex gradient surrogate against the input-embedding one.

The thesis measures the linearization quality of

    Dhat_emb(v) = g^T ( e(v) - e(x_i) ),          g = d log p(x) / d e_i

which is the surrogate THIS repository's energy induces: it differentiates with
respect to the INPUT EMBEDDING while the target token enters as a discrete index
(see core/prep.py:joint_log_prob_from_inputs_embeds and core/base_sampler.py:50-53).

CORRECTION 2026-07-28: an earlier version of this docstring attributed that
surrogate to MuCoLa and COLD. That is wrong, verified against both papers.
MuCoLa (Kumar et al. 2022, sec. 3) substitutes the continuous vector into the
output softmax numerator, so its self term is differentiable; COLD (Qin et al.
2022, eq. 3) carries per-position vocabulary logits and a soft cross-entropy,
likewise. Grathwohl et al. (2021) and Zhang et al. (2022) differentiate the
ONE-HOT (or simplex-relaxed) input, which also keeps the self term. The
self-term-blind object is this repository's energy, not theirs. The measured
results below are unaffected; only the attribution was.

For an autoregressive language model the token index
enters the energy twice, through the embedding lookup and through the output
softmax that scores it as a target, so the one-hot gradient's v-th coordinate is

    d log p(x) / d x_i[v] = log p(v | x_<i)  +  g^T e(v)

and the corresponding candidate surrogate is

    Dhat_onehot(v) = [ log p(v | x_<i) - log p(x_i | x_<i) ]  +  g^T ( e(v) - e(x_i) )
                   = true_delta_self(v)                       +  Dhat_emb(v).

The first bracket is EXACTLY the self term the linearization CSV already stores as
`true_delta_self`, because substituting v at position i changes the log-likelihood
of position i by precisely that conditional log-ratio and nothing else. So the
comparison needs no new forward passes: it is a re-analysis of the existing
diagnostic output.

Reports, per energy and per distance stratum, the Spearman correlation against the
true energy change of: the embedding-gradient surrogate, the self term alone, and
the one-hot surrogate.

Usage:
    python revision/analyze_onehot_surrogate.py \
        --res_dir results/diagnostics/diag \
        --run_name rev_onehot_surrogate --out_dir results/revision
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

COLS = ["surrogate", "true_delta", "true_delta_self", "true_delta_future", "dist", "stratum"]


def boot_ci(x, y, n_boot, seed):
    """Percentile bootstrap CI for a Spearman correlation, resampling candidate pairs."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = spearmanr(x[idx], y[idx]).correlation
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyse_one(df, n_boot, seed):
    true = df["true_delta"].to_numpy()
    variants = {
        "embedding_gradient": df["surrogate"].to_numpy(),
        "self_term_only": df["true_delta_self"].to_numpy(),
        "onehot_gradient": (df["true_delta_self"] + df["surrogate"]).to_numpy(),
    }
    out = {}
    for name, hat in variants.items():
        out[name] = {
            "spearman": float(spearmanr(hat, true).correlation),
            "pearson": float(pearsonr(hat, true)[0]),
        }
        if n_boot:
            lo, hi = boot_ci(hat, true, n_boot, seed)
            out[name]["spearman_ci95"] = [lo, hi]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res_dir", default="results/diagnostics/diag")
    p.add_argument("--run_name", default="rev_onehot_surrogate")
    p.add_argument("--out_dir", default="results/revision")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.res_dir, "diag_linearization_*.csv")))
    if not paths:
        raise SystemExit(f"no diag_linearization_*.csv under {args.res_dir}")

    results = {}
    for path in paths:
        tag = re.sub(r"^diag_linearization_|\.csv$", "", os.path.basename(path))
        df = pd.read_csv(path)
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            print(f"[skip] {tag}: missing columns {missing}")
            continue
        df = df.dropna(subset=[c for c in COLS if c != "stratum"])
        if df.empty:
            print(f"[skip] {tag}: empty after dropna")
            continue

        entry = {
            "n_pairs": int(len(df)),
            "mean_abs_self": float(df["true_delta_self"].abs().mean()),
            "mean_abs_future": float(df["true_delta_future"].abs().mean()),
            "ALL": analyse_one(df, args.n_boot, args.seed),
            "by_stratum": {},
        }
        for stratum, sub in df.groupby("stratum"):
            if len(sub) < 100:
                continue
            entry["by_stratum"][str(stratum)] = {
                "n_pairs": int(len(sub)),
                "mean_dist": float(sub["dist"].mean()),
                **analyse_one(sub, 0, args.seed),
            }
        results[tag] = entry

        a = entry["ALL"]
        print(f"{tag:16s} n={entry['n_pairs']:>7d}  "
              f"emb rho {a['embedding_gradient']['spearman']:+.4f}  "
              f"self rho {a['self_term_only']['spearman']:+.4f}  "
              f"onehot rho {a['onehot_gradient']['spearman']:+.4f}")

    out = {
        "experiment": "onehot_vs_embedding_surrogate",
        "run_name": args.run_name,
        "res_dir": args.res_dir,
        "note": ("Dhat_onehot = true_delta_self + Dhat_emb, an exact algebraic identity "
                 "for a causal LM; no new forward passes were run."),
        "by_model": results,
    }
    dest = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
