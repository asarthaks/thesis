#!/usr/bin/env python3
"""
analyze_constrained.py  (concern 11)

Reads the aggregate constrained-generation JSONs (the ones summarize_constrained.py
prints) and turns them into the two contrasts the examiner asked to see, plus the
artifact check, as a machine-readable JSON for the thesis.

JSON schema per file (one cell = one arm at one target label):
    config: {task, setup, target_label, constraint_mode, model_tag?}
    sentiment_acc_before, sentiment_acc, steering_gain, final_kl

THE TWO NUMBERS THIS EXISTS TO REPORT, per (model, task, setup, label):
  1. constraint-direction value  =  gain(cons_only) - gain(cons_random)
     Same magnitude of classifier gradient, true direction vs random direction.
     Large => the constraint gradient's DIRECTION carries signal.
  2. lm_only gain  =  the LM-likelihood-only steering, i.e. our null result
     reproduced on a sentiment task. Near zero is the expected reproduction.

ARTIFACT CHECK (concern 11's anomaly): steering is run at BOTH target labels. If
an arm only ever moves sentiment one way (positive gain at one label, zero/negative
at the other), the "gain" is a classifier or task bias, not real control. Flagged.

Note on confidence intervals: these JSONs store one aggregate gain per cell, not
per-example gains, so there is nothing to bootstrap over here. The evidence for an
effect is the cons_only-vs-cons_random gap and its consistency across the two
labels, not a per-cell CI. If you want CIs, run_constrained has to dump per-example
gains; point --per_sample_dir at that and this will add bootstrap CIs.

Usage:
  python revision/analyze_constrained.py \
      --results_dirs results_constrained results_probe \
      --run_name rev_constrained --out_dir results_revision
"""

import os
import json
import glob
import argparse
import numpy as np

ARMS = ["lm_only", "full", "cons_only", "cons_random", "random"]


def load_cells(dirs):
    cells = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            try:
                j = json.load(open(path))
                c = j.get("config", {})
                model = c.get("model_tag") or os.path.basename(path).split(".")[0]
                key = (model, c["task"], c["setup"], int(c["target_label"]),
                       c["constraint_mode"])
                cells[key] = {
                    "gain": float(j["steering_gain"]),
                    "acc_before": float(j.get("sentiment_acc_before", float("nan"))),
                    "acc_after": float(j.get("sentiment_acc", float("nan"))),
                    "final_kl": (float(j["final_kl"]) if j.get("final_kl") is not None
                                 else float("nan")),
                }
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: {e}")
    return cells


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def load_per_sample(per_dir):
    """Optional: {model,task,setup,label,arm} -> array of per-example gains,
    if run_constrained dumped them as <...>_samples.csv with a 'gain' column."""
    if not per_dir:
        return {}
    import pandas as pd
    out = {}
    for path in glob.glob(os.path.join(per_dir, "*_samples.csv")):
        try:
            df = pd.read_csv(path)
            gc = next((col for col in df.columns
                       if col.lower() in ("gain", "steer_gain", "delta")), None)
            if gc:
                out[os.path.basename(path)] = df[gc].to_numpy()
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dirs", nargs="+", default=["results_constrained"])
    ap.add_argument("--per_sample_dir", default=None,
                    help="dir with optional <...>_samples.csv per-example gains for CIs")
    ap.add_argument("--run_name", default="rev_constrained")
    ap.add_argument("--out_dir", default="results_revision")
    args = ap.parse_args()

    cells = load_cells(args.results_dirs)
    if not cells:
        print("no constrained JSONs found in", args.results_dirs)
        return
    per_sample = load_per_sample(args.per_sample_dir)

    groups = sorted({k[:4] for k in cells})  # (model, task, setup, label)
    report = {"run_name": args.run_name, "groups": [], "contrasts": []}

    # group-level table
    for (model, task, setup, lbl) in groups:
        arms = {a: cells[(model, task, setup, lbl, a)]
                for a in ARMS if (model, task, setup, lbl, a) in cells}
        g = {"model": model, "task": task, "setup": setup, "target_label": lbl,
             "target": "positive" if lbl == 1 else "negative",
             "arms": {a: v["gain"] for a, v in arms.items()},
             "final_kl": {a: v["final_kl"] for a, v in arms.items()}}
        if "cons_only" in arms and "cons_random" in arms:
            g["constraint_direction_value"] = round(
                arms["cons_only"]["gain"] - arms["cons_random"]["gain"], 3)
        if "lm_only" in arms:
            g["lm_only_gain"] = round(arms["lm_only"]["gain"], 3)
        report["groups"].append(g)

    # artifact check: per (model, task, setup, arm), is the gain sign-consistent
    # across the two labels? if not, the gain is a bias artifact
    combos = sorted({(k[0], k[1], k[2], k[4]) for k in cells})
    for (model, task, setup, arm) in combos:
        g0 = cells.get((model, task, setup, 0, arm))
        g1 = cells.get((model, task, setup, 1, arm))
        if g0 and g1:
            both_positive = (g0["gain"] > 0) and (g1["gain"] > 0)
            # one-sided means materially large on exactly one label. BIG guards
            # against flagging two near-zero gains that merely straddle a threshold.
            BIG = 3.0
            one_sided = (g0["gain"] > BIG) != (g1["gain"] > BIG)
            report["contrasts"].append({
                "model": model, "task": task, "setup": setup, "arm": arm,
                "gain_label0": round(g0["gain"], 2), "gain_label1": round(g1["gain"], 2),
                "steers_both_directions": bool(both_positive),
                "one_sided_artifact_flag": bool(one_sided),
            })

    # optional CIs from per-sample dumps
    if per_sample:
        cis = {}
        for name, arr in per_sample.items():
            cis[name] = {"mean": float(np.nanmean(arr)),
                         "ci95": boot_ci(arr),
                         "n": int(np.sum(~np.isnan(arr)))}
        report["per_sample_bootstrap_ci"] = cis

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, args.run_name + ".json")
    with open(dst, "w") as f:
        json.dump(report, f, indent=2)

    # readable print
    print(f"[constrained] {len(groups)} cells -> {dst}\n")
    for g in report["groups"]:
        head = f"{g['model']} | {g['task']} | {g['setup']} | target={g['target']}"
        print(head)
        for a in ARMS:
            if a in g["arms"]:
                print(f"    {a:<13} gain={g['arms'][a]:+6.1f}")
        if "constraint_direction_value" in g:
            print(f"    >>> constraint DIRECTION value (cons_only - cons_random) "
                  f"= {g['constraint_direction_value']:+.1f}")
        if "lm_only_gain" in g:
            print(f"    >>> lm_only steering (our null, reproduced) "
                  f"= {g['lm_only_gain']:+.1f}")
        print()
    flags = [c for c in report["contrasts"] if c["one_sided_artifact_flag"]]
    if flags:
        print("ONE-SIDED ARTIFACT FLAGS (gain moves only one label, treat as bias):")
        for c in flags:
            print(f"    {c['model']}|{c['task']}|{c['setup']}|{c['arm']}: "
                  f"lbl0={c['gain_label0']:+.1f} lbl1={c['gain_label1']:+.1f}")


if __name__ == "__main__":
    main()
