#!/usr/bin/env python
"""aggregate_guided.py - merge Part G guided-generation shards and score steering.

Steering is measured by the concern-11 judge (hit_target already computed with the
frozen-GPT-2 sentiment head). Per target label: unguided vs guided target-hit rate,
the paired guided-minus-unguided gain with a bootstrap CI (paired per prompt/sample),
and the gpt2sft span NLL/token (fluency) for each arm.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from diagnostics.run_revision import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--shard_glob", default="rev_sedd_guided.shard*")
    ap.add_argument("--final_run_name", default="rev_sedd_guided")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.out_dir, args.shard_glob + ".csv")))
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df.to_csv(os.path.join(args.out_dir, args.final_run_name + ".csv"), index=False)

    summary = {"experiment": "guided", "final_run_name": args.final_run_name,
               "n_shards_merged": len(csvs),
               "n_generations": int(len(df)), "by_label": {}}
    for lbl in sorted(df.target_label.unique()):
        d = df[df.target_label == lbl]
        g = d[d.arm == "guided"].sort_values(["prompt_idx", "sample_idx"])
        u = d[d.arm == "unguided"].sort_values(["prompt_idx", "sample_idx"])
        gh, uh = g.hit_target.values.astype(float), u.hit_target.values.astype(float)
        n = min(len(gh), len(uh))
        paired = gh[:n] - uh[:n]
        gain_m, gain_lo, gain_hi = bootstrap_ci(paired, seed=args.seed)
        entry = {
            "n_pairs": int(n),
            "unguided_hit_pct": float(100 * uh.mean()),
            "guided_hit_pct": float(100 * gh.mean()),
            "gain_pts": float(100 * gain_m),
            "gain_ci95": [float(100 * gain_lo), float(100 * gain_hi)],
            "unguided_span_nll": float(u.span_nll.mean()),
            "guided_span_nll": float(g.span_nll.mean()),
        }
        # DIAGNOSTIC: did guidance move the guiding classifier's OWN judgment?
        if "clf_self_label" in df.columns:
            gs = (g.clf_self_label.values == lbl).astype(float)
            us = (u.clf_self_label.values == lbl).astype(float)
            ns = min(len(gs), len(us))
            sm, slo, shi = bootstrap_ci(gs[:ns] - us[:ns], seed=args.seed)
            entry["self_unguided_hit_pct"] = float(100 * us.mean())
            entry["self_guided_hit_pct"] = float(100 * gs.mean())
            entry["self_gain_pts"] = float(100 * sm)
            entry["self_gain_ci95"] = [float(100 * slo), float(100 * shi)]
        # judge-vs-clf agreement on unguided text (off-distribution mismatch check)
        if "clf_self_label" in df.columns:
            entry["judge_clf_agree_pct_unguided"] = float(
                100 * (u.judged_label.values == u.clf_self_label.values).mean())
        summary["by_label"][int(lbl)] = entry
    with open(os.path.join(args.out_dir, args.final_run_name + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
