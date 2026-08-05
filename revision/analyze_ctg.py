#!/usr/bin/env python3
"""Merge and analyse the constrained-generation runs written by diagnostics/run_ctg.py.

Two things this does that a generic merger cannot:

  - RATE SCALARS ARE RECOMPUTED FROM TOTALS, never averaged over shards. Acceptance,
    shortlist coverage and the linearization correlation all have different denominators
    in different shards, and averaging the averages would silently reweight them.
  - The held-out judge files are joined in, so every row carries the ADHERENCE and the
    FLUENCY that a model with no part in the generation assigned, alongside the
    steering head's own opinion, which is reported separately and never mixed in.

    python revision/analyze_ctg.py --out_dir results/ctg/studyC \
        --judge_dir results/ctg/judged --out results/ctg/studyC_summary.json
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


def merge_run(paths):
    shards = [json.load(open(q)) for q in paths]
    exp = shards[0]["config"]["num_shards"]
    if len(shards) != exp:
        return None, f"{len(shards)} of {exp} shards"
    m = dict(shards[0])
    m["run_name"] = shards[0]["run_name"].split(".shard")[0]
    m["n"] = sum(s["n"] for s in shards)
    m["n_shards_merged"] = len(shards)

    c = collections.Counter()
    for s in shards:
        for k, v in (s.get("cost") or {}).items():
            if isinstance(v, (int, float)) and k.startswith("n_"):
                c[k] += v
    wall = [(s.get("cost") or {}).get("wall_time_sec", 0.0) for s in shards]
    m["cost"] = dict(
        c, wall_time_sec_sum=float(sum(wall)), wall_time_sec_max=float(max(wall)),
        sec_per_accepted_move=float(sum(wall)) / max(c["n_accepted_moves"], 1),
        forward_equivalents_per_sequence=(c["n_forward"] + 2.0 * c["n_backward"])
                                          / max(m["n"], 1))
    # rates from totals
    m["accept_rate_pct"] = 100.0 * c["n_accepted_moves"] / max(c["n_mh_steps"], 1)
    # The continuous-optimizer arms (run_mucola_faithful.py) have no shortlist at all
    # and write None here. Coerce to zero counts so they merge, and leave the reported
    # coverage as None for them rather than a misleading 0.0 percent.
    def _num(x, d=0):
        return d if x is None else x
    cov_n = sum(_num(s.get("shortlist_coverage_n")) for s in shards)
    cov_h = sum(round(_num(s.get("shortlist_coverage_pct")) / 100.0
                      * _num(s.get("shortlist_coverage_n"))) for s in shards)
    m["shortlist_coverage_pct"] = (100.0 * cov_h / cov_n) if cov_n else None
    m["shortlist_coverage_n"] = cov_n
    m["shortlist_rank_median"] = float(np.median(
        [s["shortlist_rank_median"] for s in shards
         if np.isfinite(_num(s.get("shortlist_rank_median"), np.nan))] or [np.nan]))
    # weighted mean of the per-shard Spearman, by the number of pairs behind each
    lw = [(s.get("constraint_linearization_spearman"), s.get("constraint_linearization_n", 0))
          for s in shards]
    lw = [(r, n) for r, n in lw if n and r is not None and np.isfinite(r)]
    m["constraint_linearization_spearman"] = (
        float(sum(r * n for r, n in lw) / sum(n for _, n in lw)) if lw else float("nan"))
    m["constraint_linearization_n"] = sum(n for _, n in lw)
    for k in ("mean_final_lm", "mean_final_cons", "steer_head_acc_pct"):
        vals = [(s.get(k), s["n"]) for s in shards if s.get(k) is not None]
        m[k] = float(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else None
    m["config"] = dict(shards[0]["config"], shard_idx=None)
    return m, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--judge_dir", default="results/ctg/judged")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    groups = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.out_dir, "*.shard*of*.json")):
        groups[os.path.basename(f).split(".shard")[0]].append(f)

    runs, pending = {}, []
    for run, paths in sorted(groups.items()):
        m, why = merge_run(sorted(paths))
        if m is None:
            pending.append(f"{run}: {why}")
            continue
        dest = os.path.join(a.out_dir, run + ".json")
        with open(dest + ".tmp", "w") as f:
            json.dump(m, f, indent=2)
        os.replace(dest + ".tmp", dest)
        csvs = [q[:-5] + ".csv" for q in sorted(paths)]
        if all(os.path.exists(c) for c in csvs):
            df = pd.concat([pd.read_csv(c) for c in csvs]).sort_values("global_idx")
            df.to_csv(os.path.join(a.out_dir, run + ".csv"), index=False)
        runs[run] = m

    # join the held-out judges
    for run, m in runs.items():
        for stage in ("sentiment", "fluency"):
            # ONLY the merged run's judge file. Per-shard judge files exist when the
            # judge was pointed at a whole directory, and mixing them in would both
            # double-count the sequences and drag in the per-shard diversity figures,
            # which are nan by construction: a shard holds too few sequences per prompt
            # for self-BLEU or semantic spread to be defined.
            hits = glob.glob(os.path.join(a.judge_dir, f"{run}.{stage}.json"))
            if not hits:
                continue
            ds = [json.load(open(h)) for h in hits]
            w = np.array([d["n"] for d in ds], dtype=float)
            w = w / w.sum()
            for k in ds[0]:
                if isinstance(ds[0][k], (int, float)) and k != "n":
                    m[f"judge_{k}"] = float(sum(d[k] * wi for d, wi in zip(ds, w)))
            m["judge_n"] = int(sum(d["n"] for d in ds))

    out = dict(experiment=os.path.basename(a.out_dir), runs=runs, pending=pending)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(a.out + ".tmp", a.out)

    hdr = (f"{'run':26s} {'n':>4s} {'cover%':>7s} {'rank~':>8s} {'linRho':>7s} "
           f"{'acc%':>6s} {'head%':>6s} {'adher%':>7s} {'ppl':>7s} {'d3':>5s} "
           f"{'sBLEU':>6s} {'fe/seq':>8s} {'gpu_s':>8s}")
    print(hdr)
    for run in sorted(runs):
        m = runs[run]
        g = lambda k, d=float("nan"): m.get(k) if m.get(k) is not None else d
        print(f"{run:26s} {m['n']:4d} {g('shortlist_coverage_pct'):7.1f} "
              f"{g('shortlist_rank_median'):8.0f} {g('constraint_linearization_spearman'):7.3f} "
              f"{m['accept_rate_pct']:6.1f} {g('steer_head_acc_pct'):6.1f} "
              f"{g('judge_adherence_pct'):7.1f} {g('judge_judge_perplexity'):7.1f} "
              f"{g('judge_distinct_3'):5.2f} {g('judge_self_bleu_4'):6.3f} "
              f"{m['cost']['forward_equivalents_per_sequence']:8.1f} "
              f"{m['cost']['wall_time_sec_sum']:8.0f}")
    if pending:
        print("\nincomplete (not merged):")
        for q in pending:
            print("  ", q)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
