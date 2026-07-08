#!/usr/bin/env python3
"""
gen_manifest.py

Emits the experiment matrix as a tab-separated manifest that the queue worker
consumes. Each line is:

    run_name <TAB> min_vram_gb <TAB> full_command

run_name is the stable job id. run_experiment.py writes <out_dir>/<run_name>.json
on success, so "done" == that file exists, which is how the queue resumes.

Partition the work across your servers with --models and --max_vram:

  # 3090 box (nine 24 GB cards): GPT-2 Large only
  python gen_manifest.py --models gpt2-large --max_vram 24 \
      --n_samples 200 --out_dir results_gpt2 > manifest_gpt2.tsv

  # A6000 box (one 48 GB card): Llama only, fewer samples to stay tractable
  python gen_manifest.py --models llama3-8b --tier1 --max_vram 48 \
      --n_samples 100 --out_dir results_llama > manifest_llama.tsv

Llama-3 8B is 32 GB of weights in fp32, so it is tagged 40 GB and only a >=40 GB
card will pick it up. If you want the 3090s to help with Llama too, pass
--llama_dtype bf16 (tagged 22 GB, fits a 24 GB card, but note it is a different
dtype than a fp32 thesis run).
"""

import argparse

MODELS = {
    "gpt2-large": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        dtype="float32", eps_start=10.5, eps_end=0.1, vram=6,
    ),
    "llama3-8b": dict(
        path="/mount/arbeitsdaten31/studenten1/singhsk/models/llama3-8b",   # or your local path
        dtype="float32", eps_start=0.1, eps_end=0.001, vram=40,   # <-- CONFIRM eps vs thesis
    ),

    # GFlowNet variants: the SFT GPT-2 Large base with a LoRA adapter merged on top.
    # Same base path as gpt2-large; only the adapter differs. VRAM is the same as
    # gpt2-large since the merged model is still 774M. Fill in the three adapter dirs.
    "gfn-lb0-500": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        peft="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-26/avid-leaf-24_490.pt",     # <-- CONFIRM (light_jazz?)
        dtype="float32", eps_start=10.5, eps_end=0.1, vram=6,
    ),
    "gfn-lb0-2000": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        peft="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-22/13-24-52/daily-resonance-15_1990.pt",    # <-- CONFIRM (least-degenerate)
        dtype="float32", eps_start=10.5, eps_end=0.1, vram=6,
    ),
    "gfn-lb1-500": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        peft="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-08/dutiful-durian-23_490.pt",     # <-- CONFIRM (hopefull_yogurt?)
        dtype="float32", eps_start=10.5, eps_end=0.1, vram=6,
    ),
}

METHODS = ["policy", "grad_norm_preserved_random_dir", "random"]
WANDB_PROJECT = "ctg-langevin-thesis"


def run_name(model, sampler, method, mh, gn, oracle, steps):
    return (f"{model}.{sampler}.{method}."
            f"{'mh' if mh else 'nomh'}.{'gn' if gn else 'nogn'}."
            f"{'oracle' if oracle else 'free'}.s{steps}")


def make_cmd(model, group, sampler, method, mh, gn, steps, oracle, args):
    m = MODELS[model]
    rn = run_name(model, sampler, method, mh, gn, oracle, steps)
    parts = [
        "python run_experiment.py",
        f"--sampler {sampler}",
        f"--method {method}",
        "--mh" if mh else "",
        "--grad_norm" if gn else "",
        "--oracle" if oracle else "",
        f"--steps {steps}",
        f"--model_path {m['path']}",
        f"--model_tag {model}",
        f"--dtype {m['dtype']}",
        f"--peft_checkpoint {m['peft']}" if m.get("peft") else "",
        f"--eps_start {m['eps_start']}",
        f"--eps_end {m['eps_end']}",
        f"--n_samples {args.n_samples}",
        f"--wandb_project {WANDB_PROJECT}",
        f"--wandb_group {model}.{group}",
        f"--run_name {rn}",
        f"--out_dir {args.out_dir}",
        "--overwrite" if args.overwrite else "",
    ]
    return rn, m["vram"], " ".join(pp for pp in parts if pp)


def jobs_for_model(model, args):
    out = []
    # TIER 1: the runs the DLS MH inconsistency actually invalidated (DLS + MH on).
    for method in METHODS:
        out.append(make_cmd(model, "dls_traj_50", "dls", method, True, True, 50, False, args))
        out.append(make_cmd(model, "dls_traj_100", "dls", method, True, True, 100, False, args))
    if args.tier1:
        return out

    # TIER 2: cross-validation. No-MH DLS, oracle, and all CLS.
    for method in METHODS:
        out.append(make_cmd(model, "dls_traj_50", "dls", method, False, True, 50, False, args))
        out.append(make_cmd(model, "dls_traj_100", "dls", method, False, True, 100, False, args))
        out.append(make_cmd(model, "dls_oracle", "dls", method, False, True, 50, True, args))
    for method in ["policy", "random"]:
        for mh in [False, True]:
            out.append(make_cmd(model, "cls_gn_on", "cls", method, mh, True, 50, False, args))
            out.append(make_cmd(model, "cls_gn_off", "cls", method, mh, False, 50, False, args))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--tier1", action="store_true", help="only the runs invalidated by the MH bug")
    ap.add_argument("--max_vram", type=int, default=999, help="only emit jobs needing <= this GB")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--out_dir", default="results_rerun")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--llama_dtype", choices=["float32", "bfloat16"], default="float32",
                    help="bfloat16 lowers Llama to 22 GB so a 24 GB card can run it")
    args = ap.parse_args()

    if args.llama_dtype == "bfloat16":
        MODELS["llama3-8b"]["dtype"] = "bfloat16"
        MODELS["llama3-8b"]["vram"] = 22

    for model in args.models:
        for rn, vram, cmd in jobs_for_model(model, args):
            if vram <= args.max_vram:
                print(f"{rn}\t{vram}\t{cmd}")


if __name__ == "__main__":
    main()
