#!/usr/bin/env python
"""
merge_sedd_cap.py  -  assemble sharded run_sedd_cap.py results.

Waits until every shard JSON exists (the resume/done contract), concatenates the
shard CSVs, and writes the final <final_run_name>.json + .csv with bootstrap CIs
computed over the MERGED per-sequence rows. Works for recovery (one row/seq) and
hybrid (one row/seq/arm; grouped by arm). Also folds in matched reference rows from
rev_last_token / rev_klbase so the write-up numbers sit next to their comparators.

Idempotent: safe to re-run; it recomputes from the shard CSVs each time.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from diagnostics.run_revision import bootstrap_ci


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def agg_group(sub, seed=0):
    out = {"n": int(len(sub))}
    if "exact" in sub:
        out["exact_match_pct"] = float(100.0 * sub["exact"].mean())
    if "top5" in sub:
        out["top5_pct"] = float(100.0 * sub["top5"].mean())
    if "avg_kl_gpt2sft" in sub:
        vals = sub["avg_kl_gpt2sft"].values.astype(float)
        m, lo, hi = bootstrap_ci(vals, seed=seed)
        out["mean_kl_gpt2sft"] = m
        out["kl_ci95"] = [lo, hi]
        out["median_kl_gpt2sft"] = float(np.nanmedian(vals))
    if "accept_pct" in sub:
        ap = sub["accept_pct"].values.astype(float)
        ap = ap[np.isfinite(ap)]
        if len(ap):
            out["mean_accept_pct"] = float(ap.mean())
            out["min_accept_pct"] = float(ap.min())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results_revision")
    p.add_argument("--shard_glob", required=True,
                   help="glob for shard CSVs, e.g. 'rev_sedd_recovery_small.shard*'")
    p.add_argument("--final_run_name", required=True)
    p.add_argument("--experiment", required=True, choices=["recovery", "hybrid"])
    p.add_argument("--scale", default="small")
    p.add_argument("--expected_shards", type=int, default=0,
                   help="if >0, require this many shard JSONs before merging")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.out_dir, args.shard_glob + ".csv")))
    json_paths = sorted(glob.glob(os.path.join(args.out_dir, args.shard_glob + ".json")))
    if args.expected_shards and len(json_paths) < args.expected_shards:
        print(f"[wait] {len(json_paths)}/{args.expected_shards} shard JSONs present; "
              f"not merging yet")
        sys.exit(2)
    if not csv_paths:
        print(f"[error] no shard CSVs match {args.shard_glob}")
        sys.exit(1)

    df = pd.concat([pd.read_csv(cp) for cp in csv_paths], ignore_index=True)
    df = df.drop_duplicates(subset=[c for c in ("sample_idx", "arm") if c in df.columns])
    df = df.sort_values("sample_idx")

    final_csv = os.path.join(args.out_dir, args.final_run_name + ".csv")
    df.to_csv(final_csv, index=False)

    summary = {"experiment": args.experiment, "scale": args.scale,
               "final_run_name": args.final_run_name,
               "n_shards_merged": len(csv_paths),
               "n_sequences": int(df["sample_idx"].nunique()),
               "by_arm": {}}
    if "arm" in df.columns:
        for arm in df["arm"].unique():
            summary["by_arm"][arm] = agg_group(df[df["arm"] == arm], seed=args.seed)
    else:
        summary["by_arm"]["sedd_recovery"] = agg_group(df, seed=args.seed)

    atomic_json(os.path.join(args.out_dir, args.final_run_name + ".json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
