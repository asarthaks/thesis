"""Merge sharded run_experiment.py outputs back into one run-level JSON and CSV.

`run_experiment.py --num_shards K --shard_idx i` splits the SAME sample set an unsharded run
would use, so the union of the shards is exactly that run. This script reassembles them:
per-step mean curves are recombined as a weighted mean over the shards' sample counts, and
scalars are recomputed from the totals rather than averaged, so the merged file is what the
unsharded run would have written.

    python revision/merge_shards.py --out_dir results/grid/rev3 --run_name xm_onehot_llama_sharp
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CURVES = ["mean_l2", "mean_kl", "mean_entropy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--keep_shards", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.out_dir, f"{args.run_name}.shard*of*.json")))
    if not paths:
        raise SystemExit(f"no shards found for {args.run_name} in {args.out_dir}")
    shards = [json.load(open(q)) for q in paths]
    expected = shards[0]["config"]["num_shards"]
    if len(shards) != expected:
        raise SystemExit(f"found {len(shards)} shards but the runs declare {expected}; "
                         "refusing to merge an incomplete set")

    n = [s["n"] for s in shards]
    total = sum(n)
    w = np.array(n, dtype=float) / total

    merged = dict(shards[0])
    merged["run_name"] = args.run_name
    merged["n"] = total
    merged["n_shards_merged"] = len(shards)
    merged["shard_sample_counts"] = n
    for key in CURVES:
        arr = np.array([s[key] for s in shards], dtype=float)
        merged[key] = (arr * w[:, None]).sum(axis=0).tolist()
    if shards[0].get("proposal_stats"):
        ps = {}
        for key in shards[0]["proposal_stats"]:
            arr = np.array([s["proposal_stats"][key] for s in shards], dtype=float)
            ps[key] = (arr * w[:, None]).sum(axis=0).tolist()
        merged["proposal_stats"] = ps
    # scalars are per-mask rates: recombine from totals, do not average the averages
    for key in ("accuracy", "ever_accuracy", "accept_rate"):
        if key in shards[0] and shards[0][key] is not None:
            vals = np.array([s.get(key, float("nan")) for s in shards], dtype=float)
            merged[key] = float(np.nansum(vals * w)) if not np.all(np.isnan(vals)) else float("nan")
    merged["examples"] = [e for s in shards for e in s.get("examples", [])][:8]
    merged["config"] = dict(shards[0]["config"], shard_idx=None, num_shards=expected)

    dest = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, dest)
    print(f"merged {len(shards)} shards, n={total} -> {dest}")

    csvs = [q[:-5] + ".csv" for q in paths]
    if all(os.path.exists(c) for c in csvs):
        df = pd.concat([pd.read_csv(c) for c in csvs]).sort_values("sample_idx")
        cdest = os.path.join(args.out_dir, args.run_name + ".csv")
        df.to_csv(cdest + ".tmp", index=False)
        os.replace(cdest + ".tmp", cdest)
        print(f"merged {len(csvs)} shard CSVs, {len(df)} rows -> {cdest}")

    if not args.keep_shards:
        for q in paths + [c for c in csvs if os.path.exists(c)]:
            os.rename(q, q + ".merged")
        print("shard files suffixed .merged (not deleted)")


if __name__ == "__main__":
    main()
