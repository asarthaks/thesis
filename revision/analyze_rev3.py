"""Consolidated analysis of the evaluation-3 slate in results/grid/rev3.

Emits one JSON holding every number the thesis quotes from this slate, so the write-up has a
single traceable source:

  sweep          embedding-gradient proposal over the 5x5 (eps, temperature) grid
  ohsweep        the one-hot proposal over the SAME grid
  mhfixsweep     the three sharpest cells with the reverse-proposal term exact for all arms
  nomhsweep      the three sharpest cells with the correction DISABLED (the Llama regime)
  mhfix          calibrated and Table-2 configurations, correction exact for all arms
  onehot         the one-hot proposal at the calibrated configuration
  xm_onehot      the one-hot proposal on Llama-3 and the three GFlowNet variants

    python revision/analyze_rev3.py --run_name rev_rev3_summary --out_dir results/revision
"""
import argparse
import glob
import json
import os
import re

import numpy as np

CELL = re.compile(r"_(policy|gnp|random)_e([\dp]+)_t([\dp]+)$")
# ohsweep_* has a single arm and therefore no arm token in its run name
CELL_NOARM = re.compile(r"_e([\dp]+)_t([\dp]+)$")


def num(tok):
    return float(tok.replace("p", "."))


def load(path):
    with open(path) as f:
        d = json.load(f)
    ps = d.get("proposal_stats") or {}
    ent = ps.get("entropy") or [float("nan")]
    t2t1 = ps.get("t2_over_t1") or [float("nan")]
    lstd = ps.get("logit_std") or [float("nan")]
    return {
        "final_kl": float(d["mean_kl"][-1]),
        "exact_pct": float(d.get("accuracy", float("nan"))),
        "ever_pct": float(d.get("ever_accuracy", float("nan"))),
        "accept_pct": float(d.get("accept_rate", float("nan"))),
        "n": int(d.get("n", 0)),
        "min_entropy": float(np.nanmin(ent)),
        "max_logit_std": float(np.nanmax(lstd)),
        "mean_t2_over_t1": float(np.nanmean(t2t1)),
    }


def collect_grid(res_dir, prefix):
    """prefix_{arm}_e{eps}_t{temp} -> {(eps, temp): {arm: stats}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(res_dir, f"{prefix}_*.json"))):
        base = os.path.basename(path)[:-5]
        m = CELL.search(base)
        if m:
            arm, eps, temp = m.group(1), num(m.group(2)), num(m.group(3))
        else:
            m = CELL_NOARM.search(base)
            if not m:
                continue
            arm, eps, temp = "policy", num(m.group(1)), num(m.group(2))
        out.setdefault((eps, temp), {})[arm] = load(path)
    return out


def grid_table(cells, label):
    """policy-minus-random contrast per cell, plus the coverage the sweep achieved."""
    rows, diffs, ents, ratios, exacts = [], [], [], [], []
    for (eps, temp), arms in sorted(cells.items()):
        if "policy" not in arms:
            continue
        p = arms["policy"]
        ents.append(p["min_entropy"])
        ratios.append(p["mean_t2_over_t1"])
        exacts.extend(a["exact_pct"] for a in arms.values())
        row = {"eps": eps, "temperature": temp,
               "policy_final_kl": p["final_kl"],
               "min_entropy": p["min_entropy"],
               "mean_t2_over_t1": p["mean_t2_over_t1"],
               "max_exact_pct": max(a["exact_pct"] for a in arms.values()),
               "max_ever_pct": max(a["ever_pct"] for a in arms.values())}
        if "gnp" in arms:
            row["policy_minus_random"] = p["final_kl"] - arms["gnp"]["final_kl"]
            diffs.append(row["policy_minus_random"])
        rows.append(row)
    summary = {
        "label": label,
        "n_cells": len(rows),
        "entropy_range": [float(np.nanmin(ents)), float(np.nanmax(ents))] if ents else None,
        "t2_over_t1_range": [float(np.nanmin(ratios)), float(np.nanmax(ratios))] if ratios else None,
        "max_exact_pct_any_cell": float(np.nanmax(exacts)) if exacts else None,
        "mean_exact_pct": float(np.nanmean(exacts)) if exacts else None,
        "n_cells_policy_better": int(sum(d < 0 for d in diffs)),
        "n_cells_policy_worse": int(sum(d > 0 for d in diffs)),
        "policy_minus_random_range": [float(min(diffs)), float(max(diffs))] if diffs else None,
        "cells": rows,
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res_dir", default="results/grid/rev3")
    p.add_argument("--grid_dir", default="results/grid/gpt2_v2")
    p.add_argument("--run_name", default="rev_rev3_summary")
    p.add_argument("--out_dir", default="results/revision")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out = {"experiment": "evaluation3_slate_summary", "run_name": args.run_name,
           "res_dir": args.res_dir}

    for prefix, label in [("sweep", "embedding-gradient proposal, 5x5 sweep"),
                          ("ohsweep", "one-hot proposal, same 5x5 sweep"),
                          ("mhfixsweep", "sharp cells, correction exact for all arms"),
                          ("nomhsweep", "sharp cells, correction disabled")]:
        cells = collect_grid(args.res_dir, prefix)
        if cells:
            out[prefix] = grid_table(cells, label)
            s = out[prefix]
            print(f"{label}: {s['n_cells']} cells | entropy "
                  f"{s['entropy_range'][0]:.3f}-{s['entropy_range'][1]:.3f} | "
                  f"t2/t1 {s['t2_over_t1_range'][0]:.2f}-{s['t2_over_t1_range'][1]:.0f} | "
                  f"max exact {s['max_exact_pct_any_cell']:.1f}% | "
                  f"policy better in {s['n_cells_policy_better']}, worse in "
                  f"{s['n_cells_policy_worse']}")

    # paired comparison of the two sweeps, cell by cell
    emb, oh = collect_grid(args.res_dir, "sweep"), collect_grid(args.res_dir, "ohsweep")
    shared = sorted(set(emb) & set(oh))
    if shared:
        rows = []
        for k in shared:
            if "policy" not in emb[k]:
                continue
            e, o = emb[k]["policy"], oh[k].get("policy")
            if o is None:
                continue
            rows.append({"eps": k[0], "temperature": k[1],
                         "embedding_kl": e["final_kl"], "onehot_kl": o["final_kl"],
                         "onehot_minus_embedding": o["final_kl"] - e["final_kl"],
                         "embedding_exact": e["exact_pct"], "onehot_exact": o["exact_pct"],
                         "onehot_min_entropy": o["min_entropy"]})
        d = [r["onehot_minus_embedding"] for r in rows]
        out["onehot_vs_embedding_sweep"] = {
            "n_cells": len(rows),
            "mean_onehot_minus_embedding": float(np.mean(d)) if d else None,
            "n_cells_onehot_better": int(sum(x < 0 for x in d)),
            "best_onehot_exact_pct": max((r["onehot_exact"] for r in rows), default=None),
            "best_embedding_exact_pct": max((r["embedding_exact"] for r in rows), default=None),
            "cells": rows,
        }
        s = out["onehot_vs_embedding_sweep"]
        print(f"one-hot vs embedding over {s['n_cells']} shared cells: "
              f"one-hot better in {s['n_cells_onehot_better']}, "
              f"best exact one-hot {s['best_onehot_exact_pct']}% vs "
              f"embedding {s['best_embedding_exact_pct']}%")

    # flat families
    flat = {}
    for path in sorted(glob.glob(os.path.join(args.res_dir, "*.json"))):
        b = os.path.basename(path)[:-5]
        if b.startswith(("mhfix_", "onehot_mh", "onehot_nomh", "xm_onehot", "power_")):
            flat[b] = load(path)
    out["flat_runs"] = flat
    for k in sorted(flat):
        if k.startswith(("xm_onehot", "onehot_")):
            v = flat[k]
            print(f"  {k:34s} KL {v['final_kl']:.3f}  exact {v['exact_pct']:.1f}%  "
                  f"ever {v['ever_pct']:.1f}%  minH {v['min_entropy']:.3f}")

    dest = os.path.join(args.out_dir, args.run_name + ".json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
