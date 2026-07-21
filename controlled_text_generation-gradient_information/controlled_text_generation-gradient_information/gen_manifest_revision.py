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
  analysis     no GPU. concern 1 (stats), 6/15/17 (reconcile), 10-analysis, 11.
  diagnostics  GPU. GENERATES results_diagnosis (linearization, likelihood_trap,
               anisotropy) which is currently empty, feeding concern 6/10/17.
  light        cheap GPU. concern 2 (kl baselines), 5 (model divergence),
               3-generate, 12 (per-method oracle).
  experiments  targeted GPU. concern 3-judge, 8 (SEDD), 9 (continuation), 16 (seeds).

The real result folders (confirmed by inspection) are the defaults below.
"""

import argparse
import os

# Paths reused from gen_manifest.py. Confirm against your layout.
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


def gpt2_family():
    yield ("gpt2sft", GPT2_SFT, None, 8)
    for tag, adp in ADAPTERS.items():
        yield (tag, GPT2_SFT, adp, 10)


def llama_vram(a):
    return 22 if a.llama_bf16 else 40


def llama_dtype(a):
    return "bfloat16" if a.llama_bf16 else "float32"


def phase_analysis(a, rows):
    # concern 1: paired stats, one job per grid results dir
    for d in a.grid_dirs:
        tag = os.path.basename(d.rstrip("/"))
        rn = f"rev_stats_{tag}"
        emit(rows, rn, 1,
             f"{PY} revision/analyze_stats.py --results_dir {d} "
             f"--run_name {rn} --out_dir {a.out_dir} --margin_frac {a.margin_frac}")

    # concern 6/15/17: reconcile across grid dirs + diagnostics
    all_dirs = " ".join(a.grid_dirs + ([a.diag_dir] if a.diag_dir else []))
    if all_dirs.strip():
        emit(rows, "rev_reconcile", 1,
             f"{PY} revision/reconcile_numbers.py --results_dirs {all_dirs} "
             f"--run_name rev_reconcile --out_dir {a.out_dir}")

    # concern 10 (analysis): within-strategy likelihood trap (needs the diag CSVs)
    if a.diag_dir:
        emit(rows, "rev_ltrap_within", 1,
             f"{PY} revision/analyze_likelihood_trap.py --results_dir {a.diag_dir} "
             f"--run_name rev_ltrap_within --out_dir {a.out_dir}")

    # concern 11: constrained contrasts across both constrained folders
    if a.constrained_dirs:
        cds = " ".join(a.constrained_dirs)
        emit(rows, "rev_constrained", 1,
             f"{PY} revision/analyze_constrained.py --results_dirs {cds} "
             f"--run_name rev_constrained --out_dir {a.out_dir}")


def phase_diagnostics(a, rows):
    # results_diagnosis is empty. Generate the standalone diagnostics that
    # reconcile (spearman, anisotropy) and analyze_likelihood_trap consume.
    dd = a.diag_dir or a.out_dir

    # linearization + likelihood_trap on the whole GPT-2 family
    for tag, path, adp, vram in gpt2_family():
        adp_flag = f" --adapter_path {adp}" if adp else ""
        for exp in ["linearization", "likelihood_trap"]:
            rn = f"diag_{exp}_{tag}"
            emit(rows, rn, vram,
                 f"{PY} {DIAG} --exp {exp} --run_name {rn} --out_dir {dd} "
                 f"--model_path {path}{adp_flag} --dtype float32 --n_seqs {a.n_diag_seqs}")

    # the same two on Llama (its own tokenizer/scale; kept separate in reconcile)
    for exp in ["linearization", "likelihood_trap"]:
        rn = f"diag_{exp}_llama3-8b"
        emit(rows, rn, llama_vram(a),
             f"{PY} {DIAG} --exp {exp} --run_name {rn} --out_dir {dd} "
             f"--model_path {LLAMA} --dtype {llama_dtype(a)} --n_seqs {a.n_diag_seqs}")

    # anisotropy only where the 2.77 vs 0.84 inter-token distance numbers come from
    emit(rows, "diag_anisotropy_gpt2sft", 8,
         f"{PY} {DIAG} --exp anisotropy --run_name diag_anisotropy_gpt2sft "
         f"--out_dir {dd} --model_path {GPT2_SFT} --dtype float32")
    emit(rows, "diag_anisotropy_llama3-8b", llama_vram(a),
         f"{PY} {DIAG} --exp anisotropy --run_name diag_anisotropy_llama3-8b "
         f"--out_dir {dd} --model_path {LLAMA} --dtype {llama_dtype(a)}")


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

    # concern 5: model divergence (base vs tuned energy), one per gfn variant
    for tag, adp in ADAPTERS.items():
        rn = f"rev_divergence_{tag}"
        emit(rows, rn, 10,
             f"{PY} {REV} --exp model_divergence --run_name {rn} --out_dir {a.out_dir} "
             f"--model_path {GPT2_SFT} --adapter_path {adp} --model_tag {tag} "
             f"--n_seqs {a.n_div_seqs}")

    # concern 3 (generate stage): recover on GPT-2 for the three methods
    emit(rows, "rev_judge_gen_gpt2sft", 8,
         f"{PY} {JUDGE} --stage generate --run_name rev_judge_gen_gpt2sft "
         f"--out_dir {a.judge_dir} --model_path {GPT2_SFT} --n_samples {a.n_samples} "
         f"--num_masks {a.num_masks}")

    # concern 12: per-method oracle sweep on GPT-2 base, isolated dir
    for method in ["policy", "grad_norm_preserved_random_dir", "random"]:
        rn = f"gpt2-large.dls.{method}.nomh.gn.oracle.s50"
        emit(rows, rn, 8,
             f"{PY} {EXP} --sampler dls --method {method} --grad_norm --oracle --steps 50 "
             f"--model_path {GPT2_SFT} --model_tag gpt2-large --dtype float32 "
             f"--eps_start 10.5 --eps_end 0.1 --n_samples {a.n_samples} "
             f"--no_wandb --run_name {rn} --out_dir {a.oracle_dir}")


def phase_experiments(a, rows):
    # concern 3 (judge stage): score recovered text under Llama
    emit(rows, "rev_judge_score_gpt2sft", llama_vram(a),
         f"{PY} {JUDGE} --stage judge --run_name rev_judge_score_gpt2sft "
         f"--out_dir {a.judge_dir} --judge_path {LLAMA} --dtype {llama_dtype(a)} "
         f"--source_run rev_judge_gen_gpt2sft")

    # concern 8: SEDD positive control. Dry-run if no checkpoint given.
    if a.sedd_dir:
        emit(rows, "rev_sedd_linearization", 10,
             f"{PY} {SEDD} --run_name rev_sedd_linearization --out_dir {a.out_dir} "
             f"--model_dir {a.sedd_dir} --n_seqs 200")
    else:
        emit(rows, "rev_sedd_dryrun", 2,
             f"{PY} {SEDD} --run_name rev_sedd_dryrun --out_dir {a.out_dir} "
             f"--dry_run --n_seqs 20")

    # concern 9: continuation task on GPT-2 SFT
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


PHASES = {"analysis": phase_analysis, "diagnostics": phase_diagnostics,
          "light": phase_light, "experiments": phase_experiments}
ORDER = ["analysis", "diagnostics", "light", "experiments"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all"] + ORDER)
    ap.add_argument("--out_dir", default="results_revision")
    # judge/oracle/seed default to out_dir so a SINGLE run_queue --out_dir covers a
    # whole phase (worker.sh checks doneness against that one dir). Override only if
    # you deliberately run them under their own queue with a matching --out_dir.
    ap.add_argument("--judge_dir", default="results_revision")
    ap.add_argument("--oracle_dir", default="results_revision")
    ap.add_argument("--seed_dir", default="results_revision")
    ap.add_argument("--diag_dir", default="results_diagnosis",
                    help="where run_diagnostic writes / reads (linearization, ltrap, anisotropy)")
    ap.add_argument("--grid_dirs", nargs="*",
                    default=["results_gpt2_v2", "results_llama", "results_gfn"],
                    help="dirs holding the main grid *.csv files (for concern 1 stats)")
    ap.add_argument("--constrained_dirs", nargs="*",
                    default=["results_constrained", "results_probe"],
                    help="dirs with the constrained-generation JSONs (for concern 11)")
    ap.add_argument("--sedd_dir", default=None,
                    help="a downloaded SEDD checkpoint dir; if omitted, a dry-run is emitted")

    ap.add_argument("--max_vram", type=int, default=999)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--num_masks", type=int, default=1)
    ap.add_argument("--gibbs_sweeps", type=int, default=3)
    ap.add_argument("--n_div_seqs", type=int, default=1000)
    ap.add_argument("--n_diag_seqs", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--margin_frac", type=float, default=0.05)
    ap.add_argument("--gfn_baselines", action="store_true",
                    help="also run KL baselines on the three GFlowNet variants")
    ap.add_argument("--llama_bf16", action="store_true",
                    help="tag Llama jobs at 22 GB (bf16) so a 24 GB card runs them")
    args = ap.parse_args()

    rows = []
    phases = ORDER if args.phase == "all" else [args.phase]
    for ph in phases:
        PHASES[ph](args, rows)

    for rn, vram, cmd in rows:
        if vram <= args.max_vram:
            print(f"{rn}\t{vram}\t{cmd}")


if __name__ == "__main__":
    main()
