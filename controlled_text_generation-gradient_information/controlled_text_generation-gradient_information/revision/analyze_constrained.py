#!/usr/bin/env python
"""
analyze_constrained.py

CONCERN 11: the constraint experiment tests only positive sentiment, reports no
error bars, and leaves the cons_only anomaly (steering -8.0, the wrong way)
unexplained. This adds:

  - bootstrap CIs on the steering gain per arm, for BOTH target labels (the
    negative-target runs come from gen_manifest_constrained.py, which already
    emits label 0 and label 1)
  - a symmetric positive/negative summary so a gain is not read as real when it
    is within noise
  - a dump of the lowest-scoring cons_only generations for the manual read the
    plan asks for (the likely story: constraint-only optimization leaves the
    fluent-text manifold and the classifier's outputs on gibberish are meaningless)

run_constrained.py is not in this bundle, so this reads its per-sample CSVs
defensively: it autodetects the gain/probability column, the arm/label columns,
and a text column. Point --results_dir at results_constrained (it globs *.csv),
or pass --csv for one file. Override column names if your schema differs.

Pure analysis, no GPU.
"""

import argparse
import glob
import json
import os

import numpy as np


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else float("nan"), float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(x.mean()), float(lo), float(hi))


def autodetect(df, prefer, kind):
    for c in prefer:
        if c in df.columns:
            return c
    # fall back to a fuzzy match
    for c in df.columns:
        lc = c.lower()
        if kind == "gain" and ("gain" in lc or "steer" in lc or "delta" in lc):
            return c
        if kind == "prob" and ("prob" in lc or "score" in lc or "conf" in lc):
            return c
        if kind == "arm" and ("arm" in lc or "mode" in lc or "constraint" in lc):
            return c
        if kind == "label" and ("label" in lc or "target" in lc):
            return c
        if kind == "text" and ("text" in lc or "gen" in lc or "output" in lc):
            return c
    return None


def load_all(paths):
    import pandas as pd
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = os.path.basename(p)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--run_name", default="constrained_ci")
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--gain_col", default=None)
    ap.add_argument("--prob_col", default=None)
    ap.add_argument("--arm_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [args.csv] if args.csv else sorted(glob.glob(os.path.join(args.results_dir or ".", "*.csv")))
    df = load_all(paths)
    assert df is not None and len(df), f"no readable CSVs in {paths}"

    gain = args.gain_col or autodetect(df, ["steer_gain", "gain", "delta_prob"], "gain")
    prob = args.prob_col or autodetect(df, ["classifier_prob", "prob", "score"], "prob")
    arm = args.arm_col or autodetect(df, ["arm", "constraint_mode", "mode"], "arm")
    label = args.label_col or autodetect(df, ["target_label", "label"], "label")
    text = args.text_col or autodetect(df, ["text", "generation", "output"], "text")

    metric = gain or prob
    assert metric and arm, (f"could not find a gain/prob column ({gain}/{prob}) or arm "
                            f"column ({arm}). Pass --gain_col/--arm_col. Columns: {list(df.columns)}")

    report = {"experiment": "constrained_ci", "metric_col": metric, "arm_col": arm,
              "label_col": label, "by_arm_label": {}}

    group_cols = [arm] + ([label] if label else [])
    for keys, sub in df.groupby(group_cols):
        m, lo, hi = bootstrap_ci(sub[metric].values, seed=args.seed)
        key = "|".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        report["by_arm_label"][key] = {
            "n": int(len(sub)), "mean": m, "ci95_lo": lo, "ci95_hi": hi,
            "within_noise": bool(lo <= 0 <= hi),
        }

    # cons_only anomaly dump
    if text is not None:
        cons = df[df[arm].astype(str).str.contains("cons_only", case=False, na=False)]
        if len(cons):
            worst = cons.sort_values(metric).head(20)
            report["cons_only_worst20"] = worst[[c for c in [metric, prob, label, text] if c]].to_dict(orient="records")
            report["cons_only_note"] = ("Manual read target: if these are off-manifold "
                                        "gibberish, the classifier scores on them are "
                                        "meaningless and the -8.0 is a symptom of leaving "
                                        "the fluent-text region, not real reverse steering.")

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, dst)
    print(f"[constrained] {len(report['by_arm_label'])} arm/label cells -> {dst}")
    for k, v in report["by_arm_label"].items():
        flag = "within-noise" if v["within_noise"] else "SIGNIFICANT"
        print(f"  {k}: mean={v['mean']:.3f} CI[{v['ci95_lo']:.3f},{v['ci95_hi']:.3f}] {flag}")


if __name__ == "__main__":
    main()
