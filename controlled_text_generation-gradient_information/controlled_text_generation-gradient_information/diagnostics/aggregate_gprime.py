#!/usr/bin/env python
"""aggregate_gprime.py - merge Phase 5 Stage 1a (G-prime) shards and score steering.

Headline: judge hit% per (target label, gamma), unguided vs guided, with paired
bootstrap CIs (paired per prompt). Also: gpt2sft span NLL per arm (fluency), SELF gain
(the guiding classifier's own verdict, mechanism check), trust-region eligibility, and
THE DIAGNOSIS TEST -> guide-judge agreement on the UNGUIDED on-domain generations (with
a bootstrap CI), plus agreement on the fully on-domain real held-out sentences and each
instrument's accuracy there (instrument-calibration context). Steering is scored only by
the concern-11 judge; the noisy classifier only guided.
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
    ap.add_argument("--shard_glob", default="rev_gprime.shard*")
    ap.add_argument("--final_run_name", default="rev_gprime")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.out_dir, args.shard_glob + ".csv")))
    if not csvs:
        raise SystemExit(f"no shard csvs match {args.shard_glob}")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df.to_csv(os.path.join(args.out_dir, args.final_run_name + ".csv"), index=False)

    gen = df[df.arm.isin(["guided", "unguided"])].copy()
    summary = {"experiment": "gprime", "final_run_name": args.final_run_name,
               "n_shards_merged": len(csvs), "n_rows": int(len(df)), "by_cell": {}}

    # ---- DIAGNOSIS TEST: guide vs judge agreement on UNGUIDED on-domain generations ----
    u = gen[gen.arm == "unguided"].drop_duplicates("prompt_idx")
    u_agree = (u.judged_label.values == u.clf_self_label.values).astype(float)
    am, alo, ahi = bootstrap_ci(u_agree, seed=args.seed)
    summary["diagnosis_unguided_gen_agreement_pct"] = float(100 * am)
    summary["diagnosis_unguided_gen_agreement_ci95"] = [float(100 * alo), float(100 * ahi)]
    summary["diagnosis_n_unguided"] = int(len(u))
    summary["phase4_offdomain_agreement_band_pct"] = [56.0, 64.0]

    # ---- instrument calibration on the fully on-domain real held-out sentences ----
    rt = df[df.arm == "realtext"].drop_duplicates("prompt_idx")
    if len(rt):
        rt_agree = (rt.judged_label.values == rt.clf_self_label.values).astype(float)
        rm, rlo, rhi = bootstrap_ci(rt_agree, seed=args.seed)
        summary["realtext_agreement_pct"] = float(100 * rm)
        summary["realtext_agreement_ci95"] = [float(100 * rlo), float(100 * rhi)]
        summary["realtext_judge_acc_pct"] = float(100 * (rt.judged_label.values == rt.target_label.values).mean())
        summary["realtext_clf_acc_pct"] = float(100 * (rt.clf_self_label.values == rt.target_label.values).mean())
        summary["realtext_n"] = int(len(rt))

    # ---- trust-region eligibility (over guided rows that carry the stats) ----
    gtr = gen[(gen.arm == "guided") & gen.mean_eligible.notna()]
    if len(gtr):
        summary["trust_region_mean_eligible"] = float(gtr.mean_eligible.mean())
        summary["trust_region_frac_le1_eligible"] = float(gtr.frac_le1_eligible.mean())
        summary["delta_nats"] = float(df.attrs.get("delta", 5.0))

    # ---- headline: per (target_label, gamma) ----
    for gamma in sorted(gen.gamma.unique()):
        for lbl in sorted(gen.target_label.unique()):
            d = gen[(gen.gamma == gamma) & (gen.target_label == lbl)]
            g = d[d.arm == "guided"].sort_values("prompt_idx")
            uu = d[d.arm == "unguided"].sort_values("prompt_idx")
            m = min(len(g), len(uu))
            if m == 0:
                continue
            gh, uh = g.hit_target.values.astype(float)[:m], uu.hit_target.values.astype(float)[:m]
            gm, glo, ghi = bootstrap_ci(gh - uh, seed=args.seed)
            gs = (g.clf_self_label.values == lbl).astype(float)[:m]
            us = (uu.clf_self_label.values == lbl).astype(float)[:m]
            sm, slo, shi = bootstrap_ci(gs - us, seed=args.seed)
            key = f"gamma{gamma:g}_label{int(lbl)}"
            summary["by_cell"][key] = {
                "gamma": float(gamma), "target_label": int(lbl), "n_pairs": int(m),
                "unguided_hit_pct": float(100 * uh.mean()),
                "guided_hit_pct": float(100 * gh.mean()),
                "gain_pts": float(100 * gm), "gain_ci95": [float(100 * glo), float(100 * ghi)],
                "unguided_span_nll": float(uu.span_nll.mean()),
                "guided_span_nll": float(g.span_nll.mean()),
                "nll_cost": float(g.span_nll.mean() - uu.span_nll.mean()),
                "self_unguided_hit_pct": float(100 * us.mean()),
                "self_guided_hit_pct": float(100 * gs.mean()),
                "self_gain_pts": float(100 * sm), "self_gain_ci95": [float(100 * slo), float(100 * shi)],
            }
    with open(os.path.join(args.out_dir, args.final_run_name + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
