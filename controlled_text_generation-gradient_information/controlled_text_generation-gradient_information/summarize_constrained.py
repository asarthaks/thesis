#!/usr/bin/env python3
"""
summarize_constrained.py

Reads results_constrained/*.json and prints the attribution table plus the one
contrast the whole experiment exists to measure.

THE CONTRAST:  cons_only  vs  cons_random
Both use ONLY the classifier gradient, at the same magnitude. The first uses its
true direction, the second a random direction. The gap between them is the value
of the constraint gradient's DIRECTION.

Compare that to the LM gradient's direction, which our main experiments showed is
worth nothing (policy vs random: statistically indistinguishable across 5 models).

If the constraint direction is worth several points and the LM direction is worth
zero, that explains the entire literature: MuCoLa/COLD work because the CLASSIFIER
steers them, not because the LM likelihood gradient does.

Usage:
  python summarize_constrained.py --dir results_constrained
"""

import os
import json
import glob
import argparse


ARMS = ["lm_only", "full", "cons_only", "cons_random", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_constrained")
    args = ap.parse_args()

    rows = {}
    for f in glob.glob(os.path.join(args.dir, "*.json")):
        j = json.load(open(f))
        c = j["config"]
        key = (c["task"], c["setup"], c["target_label"], c["constraint_mode"])
        rows[key] = j

    if not rows:
        print(f"no results in {args.dir}")
        return

    tasks = sorted({k[0] for k in rows})
    setups = sorted({k[1] for k in rows})
    labels = sorted({k[2] for k in rows})

    for task in tasks:
        for setup in setups:
            for lbl in labels:
                present = [a for a in ARMS if (task, setup, lbl, a) in rows]
                if not present:
                    continue
                tgt = "positive" if lbl == 1 else "negative"
                print(f"\n{'='*70}")
                print(f"task={task}  setup={setup}  target={tgt}")
                print(f"{'='*70}")
                print(f"  {'arm':<14}{'before':>8}{'after':>8}{'gain':>8}{'final_kl':>10}")
                for a in present:
                    j = rows[(task, setup, lbl, a)]
                    kl = j.get("final_kl")
                    kls = f"{kl:.2f}" if kl is not None else "n/a"
                    print(f"  {a:<14}{j['sentiment_acc_before']:>7.1f}%"
                          f"{j['sentiment_acc']:>7.1f}%"
                          f"{j['steering_gain']:>+8.1f}{kls:>10}")

                co = rows.get((task, setup, lbl, "cons_only"))
                cr = rows.get((task, setup, lbl, "cons_random"))
                lo = rows.get((task, setup, lbl, "lm_only"))
                if co and cr:
                    gap = co["steering_gain"] - cr["steering_gain"]
                    print(f"\n  >>> CONSTRAINT DIRECTION VALUE (cons_only - cons_random) "
                          f"= {gap:+.1f} pts")
                if lo:
                    print(f"  >>> LM-only steering                              "
                          f"= {lo['steering_gain']:+.1f} pts")

    print(f"\n{'='*70}")
    print("READING IT")
    print("""
  cons_only - cons_random  LARGE
     The constraint gradient's DIRECTION carries real signal. Contrast with our
     main result, where the LM gradient's direction was worth statistically
     nothing (policy vs random, indistinguishable on all 5 models). Conclusion:
     MuCoLa/COLD work because the CLASSIFIER steers them, not the LM gradient.

  cons_only - cons_random  ~ 0
     Even the classifier gradient does not steer via direction. The method would
     be driven by noise plus projection, and gradient guidance is questionable
     across the board.

  lm_only  ~ 0 everywhere
     Our existing null result, reproduced on a sentiment task.

  Sanity: run both target labels. If steering only ever moves one direction, the
  classifier or the task is biased and the 'gain' is an artifact.
""")


if __name__ == "__main__":
    main()
