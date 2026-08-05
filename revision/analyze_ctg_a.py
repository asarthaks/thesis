#!/usr/bin/env python3
"""Study A analysis: quality against cost, paired on sample_idx.

Reads the per-shard JSONs (for the cost counters, which merge_shards.py does not
recombine) and the merged per-sample CSVs (for the paired quality contrasts), and
writes one flat index.

Pairing is exact and needs no rerun: the corruption is deterministic per sample_idx,
identical across arms, so arm A's sample 7 and arm B's sample 7 are the same corrupted
sentence.

    python revision/analyze_ctg_a.py --out_dir results/ctg/studyA \
        --out results/ctg/studyA_summary.json
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


def boot_ci(x, n_boot=10000, seed=0):
    r = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    bs = x[r.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/ctg/studyA")
    p.add_argument("--out", default="results/ctg/studyA_summary.json")
    p.add_argument("--baseline", default="policy_self",
                   help="arm each other arm is paired against within a family")
    a = p.parse_args()

    # ---- cost: sum the SHARD files. merge_shards.py copies shard 0's cost block, which
    # would under-report the run's total by the number of shards.
    cost = collections.defaultdict(lambda: collections.Counter())
    walls = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.out_dir, "*.shard*of*.json*")):
        d = json.load(open(f))
        rn = d["run_name"].split(".shard")[0]
        c = d.get("cost") or {}
        for k in ("n_forward", "n_backward", "n_exact_candidate_evals",
                  "n_sequences", "n_mh_steps", "n_accepted_moves"):
            if c.get(k):
                cost[rn][k] += int(c[k])
        if c.get("wall_time_sec"):
            walls[rn].append(float(c["wall_time_sec"]))

    runs = {}
    for f in sorted(glob.glob(os.path.join(a.out_dir, "*.json"))):
        if ".shard" in os.path.basename(f):
            continue
        d = json.load(open(f))
        rn = os.path.basename(f)[:-5]
        csvp = os.path.join(a.out_dir, rn + ".csv")
        df = pd.read_csv(csvp) if os.path.exists(csvp) else None
        c = cost[rn]
        nseq = max(c.get("n_sequences", d.get("n", 1)), 1)
        # Shards run CONCURRENTLY on different cards, so the wall-clock a user waits is
        # the slowest shard, while the GPU-time the run consumes is the sum. Both are
        # reported; a "matched compute" claim must say which one it means.
        runs[rn] = dict(
            run_name=rn, family=rn.split(".")[1] if "." in rn else "",
            arm=rn.split(".")[-1], n=int(d["n"]),
            exact_pct=float(d["accuracy"]), ever_pct=float(d.get("ever_accuracy", float("nan"))),
            final_kl=float(d["mean_kl"][-1]), accept_pct=float(d["accept_rate"]),
            n_forward=c.get("n_forward", 0), n_backward=c.get("n_backward", 0),
            n_exact_candidate_evals=c.get("n_exact_candidate_evals", 0),
            fwd_equiv_per_seq=(c.get("n_forward", 0) + 2.0 * c.get("n_backward", 0)) / nseq,
            gpu_sec_total=float(sum(walls.get(rn, [0.0]))),
            wall_sec_slowest_shard=float(max(walls.get(rn, [0.0]))),
            gpu_sec_per_seq=float(sum(walls.get(rn, [0.0]))) / nseq,
            gpu_sec_per_accepted_move=(float(sum(walls.get(rn, [0.0])))
                                        / max(c.get("n_accepted_moves", 1), 1)),
            _df=df,
        )

    # ---- paired contrasts within each family
    fams = collections.defaultdict(list)
    for rn, r in runs.items():
        fams[r["family"]].append(rn)
    contrasts = []
    for fam, names in sorted(fams.items()):
        base = next((n for n in names if runs[n]["arm"] == a.baseline), None)
        if base is None or runs[base]["_df"] is None:
            continue
        b = runs[base]["_df"].set_index("sample_idx")
        for n in sorted(names):
            if n == base or runs[n]["_df"] is None:
                continue
            o = runs[n]["_df"].set_index("sample_idx")
            idx = b.index.intersection(o.index)
            dk = (o.loc[idx, "avg_kl_div"] - b.loc[idx, "avg_kl_div"]).values
            da = (o.loc[idx, "accuracy_pct"] - b.loc[idx, "accuracy_pct"]).values
            m, lo, hi = boot_ci(dk)
            ma, loa, hia = boot_ci(da)
            contrasts.append(dict(
                family=fam, arm=runs[n]["arm"], baseline=a.baseline, n_paired=int(len(idx)),
                d_kl_mean=m, d_kl_ci=[lo, hi],
                d_exact_pts_mean=ma, d_exact_pts_ci=[loa, hia],
                cost_ratio_fwd_equiv=(runs[n]["fwd_equiv_per_seq"]
                                       / max(runs[base]["fwd_equiv_per_seq"], 1e-9)),
                cost_ratio_gpu_sec=(runs[n]["gpu_sec_per_seq"]
                                     / max(runs[base]["gpu_sec_per_seq"], 1e-9)),
            ))

    for r in runs.values():
        r.pop("_df", None)
    out = dict(experiment="ctg_studyA", runs=runs, contrasts=contrasts)
    tmp = a.out + ".tmp"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, a.out)

    print(f"{'run':34s} {'n':>4s} {'exact%':>7s} {'KL':>7s} {'acc%':>6s} "
          f"{'fe/seq':>8s} {'gpu_s/seq':>10s} {'gpu_s/acc':>10s}")
    for rn in sorted(runs):
        r = runs[rn]
        print(f"{rn:34s} {r['n']:4d} {r['exact_pct']:7.1f} {r['final_kl']:7.3f} "
              f"{r['accept_pct']:6.1f} {r['fwd_equiv_per_seq']:8.1f} "
              f"{r['gpu_sec_per_seq']:10.2f} {r['gpu_sec_per_accepted_move']:10.3f}")
    print()
    for c in contrasts:
        print(f"[{c['family']:>9s}] {c['arm']:>16s} - {c['baseline']:<12s} "
              f"dKL {c['d_kl_mean']:+.3f} [{c['d_kl_ci'][0]:+.3f},{c['d_kl_ci'][1]:+.3f}]  "
              f"dExact {c['d_exact_pts_mean']:+.1f} pts "
              f"[{c['d_exact_pts_ci'][0]:+.1f},{c['d_exact_pts_ci'][1]:+.1f}]  "
              f"cost x{c['cost_ratio_fwd_equiv']:.2f} fe / x{c['cost_ratio_gpu_sec']:.2f} s")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
