#!/usr/bin/env python3
"""
gen_jobs.py

Emits one run_experiment.py command per line for the whole thesis matrix.
The tmux launcher reads these lines and spreads them across GPUs. Edit the
MODELS block (paths, dtypes, and especially the Llama step-size schedule) and
the GROUPS block to add or drop runs.

Usage:
  python gen_jobs.py                # print all jobs
  python gen_jobs.py --models gpt2-large   # only one model
  python gen_jobs.py --tier1        # only the runs the MH bug actually invalidated
"""

import argparse

# ---------------------------------------------------------------------------
# Per-model settings. CONFIRM the Llama schedule against your original thesis
# runs; the notes had Llama alpha ~ 0.1 while GPT-2 Large needed ~ 10.5 to cross
# its larger Voronoi cells. eps_start/eps_end feed np.linspace(start, end, steps).
# ---------------------------------------------------------------------------
MODELS = {
    "gpt2-large": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        dtype="float32",
        eps_start=10.5, eps_end=0.1,
    ),
    "llama3-8b": dict(
        path="meta-llama/Meta-Llama-3-8B",   # or your local path
        dtype="bfloat16",
        eps_start=0.1, eps_end=0.001,          # <-- CONFIRM against your thesis runs
    ),
}

N_SAMPLES = 200
WANDB_PROJECT = "ctg-langevin-thesis"

METHODS = ["policy", "grad_norm_preserved_random_dir", "random"]


def cmd(model, group, sampler, method, mh, grad_norm, steps, oracle):
    m = MODELS[model]
    parts = [
        "python run_experiment.py",
        f"--sampler {sampler}",
        f"--method {method}",
        "--mh" if mh else "",
        "--grad_norm" if grad_norm else "",
        "--oracle" if oracle else "",
        f"--steps {steps}",
        f"--model_path {m['path']}",
        f"--model_tag {model}",
        f"--dtype {m['dtype']}",
        f"--eps_start {m['eps_start']}",
        f"--eps_end {m['eps_end']}",
        f"--n_samples {N_SAMPLES}",
        f"--wandb_project {WANDB_PROJECT}",
        f"--wandb_group {model}.{group}",
    ]
    return " ".join(pp for pp in parts if pp)


def jobs_for_model(model, tier1_only):
    out = []
    # TIER 1: the runs the DLS MH inconsistency actually invalidated.
    # These are DLS with MH on (policy+grad_norm reverse bug, and random+MH).
    for method in METHODS:
        out.append(("tier1", cmd(model, "dls_traj_50", "dls", method, True, True, 50, False)))
        out.append(("tier1", cmd(model, "dls_traj_100", "dls", method, True, True, 100, False)))

    if tier1_only:
        return out

    # TIER 2: full cross-validation. No-MH DLS, CLS, and oracle. Expected to
    # reproduce the qualitative findings; run to confirm nothing else shifted.
    for method in METHODS:
        out.append(("tier2", cmd(model, "dls_traj_50", "dls", method, False, True, 50, False)))
        out.append(("tier2", cmd(model, "dls_traj_100", "dls", method, False, True, 100, False)))
        out.append(("tier2", cmd(model, "dls_oracle", "dls", method, False, True, 50, True)))

    for method in ["policy", "random"]:
        for mh in [False, True]:
            out.append(("tier2", cmd(model, "cls_gn_on", "cls", method, mh, True, 50, False)))
            out.append(("tier2", cmd(model, "cls_gn_off", "cls", method, mh, False, 50, False)))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--tier1", action="store_true", help="only the runs invalidated by the MH bug")
    args = ap.parse_args()

    for model in args.models:
        for _tier, c in jobs_for_model(model, args.tier1):
            print(c)


if __name__ == "__main__":
    main()
