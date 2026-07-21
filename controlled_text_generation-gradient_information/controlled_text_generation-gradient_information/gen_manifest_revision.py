#!/usr/bin/env python3
"""
gen_manifest_revision.py

One manifest generator for the whole examiner-revision plan. Emits the same TSV
schema every other generator here uses, so run_queue.sh / worker.sh consume it
unchanged and resume-by-existence works:

    run_name <TAB> min_vram_gb <TAB> command

A job is DONE when <out_dir>/<run_name>.json exists. Analysis-only jobs are
tagged 1 GB so any worker grabs them and they run on CPU inside the worker window.

Phases follow the plan's own weekly ordering:
  analysis     no GPU. concern 1 (stats), 6/15/17 (reconcile), 10-analysis, 11-analysis.
  light        cheap GPU. concern 2 (kl baselines), 5 (model divergence),
               10-gpu (llama likelihood trap), 3-generate, 12 (per-method oracle).
  experiments  targeted GPU. concern 3-judge, 8 (SEDD), 9 (continuation), 16 (seeds).

Examples
--------
  # everything, let the queue sort by vram
  python gen_manifest_revision.py --phase all \
      --grid_dirs results_gpt2 results_llama --diag_dir results_diag \
      --constrained_dir results_constrained \
      --out_dir results_revision > manifest_revision.tsv

  # only the no-GPU analysis you can run today
  python gen_manifest_revision.py --phase analysis \
      --grid_dirs results_gpt2 --diag_dir results_diag > manifest_analysis.tsv

  ./run_queue.sh --manifest manifest_revision.tsv --gpus "0 1 2 3" --per_gpu 2 \
        --vram 24 --out_dir results_revision --status status_rev --env gfn
"""

import argparse
import os

# Paths reused from gen_manifest.py / gen_diag_manifest.py. Confirm against your layout.
GPT2_SFT = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"
LLAMA = "/mount/arbeitsdaten31/studenten1/singhsk/models/llama3-8b"
ADAPTERS = {
    "gfn-lb0-500":  "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-26/avid-leaf-24_490.pt",
    "gfn-lb0-2000": "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-22/13-24-52/daily-resonance-15_1990.pt",
    "gfn-lb1-500":  "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-08/dutiful-durian-23_490.pt",
}

PY = "python"
REV = "diagnostics/run_revision.py"
DIAG = "diagnostics/run_diagnostic.py"
JUDGE = "diagnostics/run_external_judge.py"
SEDD = "diagnostics/run_sedd_linearization.py"
EXP = "run_experiment.py"


def emit(rows, rn, vram, cmd):
    rows.append((rn, vram, cmd))


def phase_analysis(a, rows):
    # concern 1: paired stats on every grid results dir
    for d in a.grid_dirs:
        tag = os.path.basename(d.rstrip("/"))
        rn = f"rev_stats_{tag}"
        emit(rows, rn, 1,
             f"{PY} revision/analyze_stats.py --results_dir {d} "
             f"--run_name {rn} --out_dir {a.out_dir} --margin_frac {a.margin_frac}")

    # concern 6/15/17: reconcile everything
    all_dirs = " ".join(a.grid_dirs + ([a.diag_dir] if a.diag_dir else []))
    if all_dirs.strip():
        emit(rows, "rev_reconcile", 1,
             f"{PY} revision/reconcile_numbers.py --results_dirs {all_dirs} "
             f"--run_name rev_reconcile --out_dir {a.out_dir}")

    # concern 10 (analysis): within-strategy likelihood trap
    if a.diag_dir:
        emit(rows, "rev_ltrap_within", 1,
             f"{PY} revision/analyze_likelihood_trap.py --results_dir {a.diag_dir} "
             f"--run_name rev_ltrap_within --out_dir {a.out_dir}")

    # concern 11 (analysis): constrained CIs
    if a.constrained_dir:
        emit(rows, "rev_constrained_ci", 1,
             f"{PY} revision/analyze_constrained.py --results_dir {a.constrained_dir} "
             f"--run_name rev_constrained_ci --out_dir {a.out_dir}")


def phase_light(a, rows):
    # concern 2: KL baselines on the SFT base (and optionally the gfn variants)
    models = [("gpt2sft", GPT2_SFT, None)]
    if a.gfn_baselines:
        for tag, adp in ADAPTERS.items():
            models.append((tag, GPT2_SFT, adp))
    for tag, path, adp in models:
        rn = f"rev_klbase_{tag}"
        cmd = (f"{PY} {REV} --exp kl_baselines --run_name {rn} --out_dir {a.out_dir} "
               f"--model_path {path} --model_tag {tag} --n_samples {a.n_samples} "
               f"--num_masks {a.num_masks} --gibbs_sweeps {a.gibbs_sweeps}")
        if adp:
            cmd += f" --adapter_path {adp}"
        emit(rows, rn, 8, cmd)

    # concern 5: model divergence, one job per gfn variant
    for tag, adp in ADAPTERS.items():
        rn = f"rev_divergence_{tag}"
        emit(rows, rn, 10,
             f"{PY} {REV} --exp model_divergence --run_name {rn} --out_dir {a.out_dir} "
             f"--model_path {GPT2_SFT} --adapter_path {adp} --model_tag {tag} "
             f"--n_seqs {a.n_div_seqs}")

    # concern 10 (gpu): likelihood trap on Llama, via the existing diagnostic runner
    llama_vram = 22 if a.llama_bf16 else 40
    dt = "bfloat16" if a.llama_bf16 else "float32"
    emit(rows, "diag_likelihood_trap_llama3-8b", llama_vram,
         f"{PY} {DIAG} --exp likelihood_trap --run_name diag_likelihood_trap_llama3-8b "
         f"--out_dir {a.diag_dir or a.out_dir} --model_path {LLAMA} --dtype {dt} "
         f"--n_seqs 500 --max_new_tokens 40")

    # concern 3 (generate stage): recover on GPT-2 for the three methods
    emit(rows, "rev_judge_gen_gpt2sft", 8,
         f"{PY} {JUDGE} --stage generate --run_name rev_judge_gen_gpt2sft "
         f"--out_dir {a.judge_dir} --model_path {GPT2_SFT} --n_samples {a.n_samples} "
         f"--num_masks {a.num_masks}")

    # concern 12: per-method oracle sweep on GPT-2 base, isolated in its own dir so
    # the schedule-selection asymmetry can be checked method by method
    for method in ["policy", "grad_norm_preserved_random_dir", "random"]:
        rn = f"gpt2-large.dls.{method}.nomh.gn.oracle.s50"
        emit(rows, rn, 8,
             f"{PY} {EXP} --sampler dls --method {method} --grad_norm --oracle --steps 50 "
             f"--model_path {GPT2_SFT} --model_tag gpt2-large --dtype float32 "
             f"--eps_start 10.5 --eps_end 0.1 --n_samples {a.n_samples} "
             f"--no_wandb --run_name {rn} --out_dir {a.oracle_dir}")


def phase_experiments(a, rows):
    # concern 3 (judge stage): score the recovered text under Llama
    llama_vram = 22 if a.llama_bf16 else 40
    dt = "bfloat16" if a.llama_bf16 else "float32"
    emit(rows, "rev_judge_score_gpt2sft", llama_vram,
         f"{PY} {JUDGE} --stage judge --run_name rev_judge_score_gpt2sft "
         f"--out_dir {a.judge_dir} --judge_path {LLAMA} --dtype {dt} "
         f"--source_run rev_judge_gen_gpt2sft")

    # concern 8: SEDD positive control. If no checkpoint given, emit the dry-run so
    # the loop is validated; then flip to the real --model_dir.
    if a.sedd_dir:
        emit(rows, "rev_sedd_linearization", 10,
             f"{PY} {SEDD} --run_name rev_sedd_linearization --out_dir {a.out_dir} "
             f"--model_dir {a.sedd_dir} --n_seqs 200")
    else:
        emit(rows, "rev_sedd_dryrun", 2,
             f"{PY} {SEDD} --run_name rev_sedd_dryrun --out_dir {a.out_dir} "
             f"--dry_run --n_seqs 20")

    # concern 9: continuation task, core ablation on GPT-2 SFT
    emit(rows, "rev_continuation_gpt2sft", 8,
         f"{PY} {REV} --exp continuation --run_name rev_continuation_gpt2sft "
         f"--out_dir {a.out_dir} --model_path {GPT2_SFT} --model_tag gpt2sft "
         f"--span 20 --n_samples 100 --steps 50 --eps_start 10.5 --eps_end 0.1")

    # concern 16: seed reruns of one representative config (vary data_seed)
    for s in range(a.n_seeds):
        rn = f"gpt2-large.dls.policy.mh.gn.free.s50.seed{s}"
        emit(rows, rn, 8,
             f"{PY} {EXP} --sampler dls --method policy --mh --grad_norm --steps 50 "
             f"--model_path {GPT2_SFT} --model_tag gpt2-large --dtype float32 "
             f"--eps_start 10.5 --eps_end 0.1 --n_samples {a.n_samples} "
             f"--data_seed {1000 + s} --no_wandb --run_name {rn} --out_dir {a.seed_dir}")


PHASES = {"analysis": phase_analysis, "light": phase_light, "experiments": phase_experiments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "analysis", "light", "experiments"])
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--judge_dir", default="results_judge")
    ap.add_argument("--oracle_dir", default="results_oracle_percheck")
    ap.add_argument("--seed_dir", default="results_seeds")
    ap.add_argument("--diag_dir", default="results_diag",
                    help="where the run_diagnostic outputs live (for reconcile / ltrap)")
    ap.add_argument("--grid_dirs", nargs="*", default=["results_gpt2_v2"],
                    help="dirs holding the main grid *.csv files (for concern 1 stats)")
    ap.add_argument("--constrained_dir", default=None,
                    help="dir with run_constrained.py outputs (for concern 11)")
    ap.add_argument("--sedd_dir", default=None,
                    help="a downloaded SEDD checkpoint dir; if omitted, a dry-run job is emitted")

    ap.add_argument("--max_vram", type=int, default=999)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--num_masks", type=int, default=1)
    ap.add_argument("--gibbs_sweeps", type=int, default=3)
    ap.add_argument("--n_div_seqs", type=int, default=1000)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--margin_frac", type=float, default=0.05)
    ap.add_argument("--gfn_baselines", action="store_true",
                    help="also run KL baselines on the three GFlowNet variants")
    ap.add_argument("--llama_bf16", action="store_true",
                    help="tag Llama jobs at 22 GB (bf16) so a 24 GB card runs them")
    args = ap.parse_args()

    rows = []
    phases = PHASES.keys() if args.phase == "all" else [args.phase]
    for ph in phases:
        PHASES[ph](args, rows)

    for rn, vram, cmd in rows:
        if vram <= args.max_vram:
            print(f"{rn}\t{vram}\t{cmd}")


if __name__ == "__main__":
    main()
