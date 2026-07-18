#!/usr/bin/env python
"""
gen_diag_manifest.py

Emits a TSV manifest for the diagnostic experiments, in the same schema
gen_manifest.py uses, so run_queue.sh / worker.sh pick these up unchanged:

    run_name <TAB> vram_gb <TAB> command

A job is DONE when <out_dir>/<run_name>.json exists, so resume-by-existence
works exactly as before, and reset_incomplete.sh needs no changes.

Usage
-----
  python gen_diag_manifest.py --out_dir results_diag > manifest_diag.tsv
  ./run_queue.sh --manifest manifest_diag.tsv --gpus "0 1 2 3" --per_gpu 2 \
        --vram 24 --out_dir results_diag --status status_diag --env gfn

Only the linearization experiment is heavy. anisotropy and likelihood_trap are
minutes. trajectory needs your core/ samplers on the path.
"""

import argparse

# Match these to your local layout. VRAM is the gate used by run_queue.sh.
MODELS = {
    "gpt2sft": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        adapter=None,
        dtype="float32",
        vram=8,
        eps_start=10.5, eps_end=0.1,
    ),
    "gfn-lb0-500": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        adapter="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-26/avid-leaf-24_490.pt",
        dtype="float32",
        vram=8,
        eps_start=10.5, eps_end=0.1,
    ),
    "gfn-lb0-2000": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        adapter="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-22/13-24-52/daily-resonance-15_1990.pt",
        dtype="float32",
        vram=8,
        eps_start=10.5, eps_end=0.1,
    ),
    "gfn-lb1-500": dict(
        path="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output",
        adapter="/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-06-27/19-13-08/dutiful-durian-23_490.pt",
        dtype="float32",
        vram=8,
        eps_start=10.5, eps_end=0.1,
    ),
    "llama3-8b": dict(
        path="/mount/arbeitsdaten31/studenten1/singhsk/models/llama3-8b",
        adapter=None,
        dtype="bfloat16",
        vram=22,
        eps_start=0.1, eps_end=0.001,
    ),
}



# (experiment, [models], extra flags, vram surcharge)
JOBS = [
    # EXP 1. The mechanism figure. Run on GPT-2 SFT and Llama at minimum;
    # the GFlowNet variants make it a five-model claim, matching the rest of
    # the thesis, and they are cheap.
    ("linearization", ["gpt2sft", "gfn-lb0-500", "gfn-lb0-2000", "gfn-lb1-500", "llama3-8b"],
     "--n_seqs 200 --n_cand 2000 --n_near 500 --n_mid 500 --batch_size 64", 4),

    # EXP 4. Likelihood trap. Only needs the SFT base to make the point, but
    # running it on the GFlowNet variants gives you the reward-hacking evidence
    # in the same table.
    ("likelihood_trap", ["gpt2sft", "gfn-lb0-500", "gfn-lb0-2000", "gfn-lb1-500"],
     "--n_seqs 500 --max_new_tokens 40", 4),

    # EXP 5. Embedding geometry. Verifies the 4.5 vs 0.5 numbers you cite in
    # three places. Minutes.
    ("anisotropy", ["gpt2sft", "llama3-8b"],
     "--n_pairs 200000 --n_seqs 50", 2),

    # EXP 3 and EXP 2 both come from collect_traces.py, which needs the patched
    # core/ samplers. It is scheduled separately; see the README.
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results_diag")
    ap.add_argument("--max_vram", type=int, default=999)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model keys")
    ap.add_argument("--exps", nargs="*", default=None,
                    help="restrict to these experiments")
    ap.add_argument("--python", default="python")
    ap.add_argument("--script", default="diagnostics/run_diagnostic.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for exp, model_keys, flags, surcharge in JOBS:
        if args.exps and exp not in args.exps:
            continue
        for mk in model_keys:
            if args.models and mk not in args.models:
                continue
            m = MODELS[mk]
            vram = m["vram"] + surcharge
            if vram > args.max_vram:
                continue

            run_name = f"diag_{exp}_{mk}"
            cmd = [
                args.python, args.script,
                "--exp", exp,
                "--run_name", run_name,
                "--out_dir", args.out_dir,
                "--model_path", m["path"],
                "--dtype", m["dtype"],
                "--seed", str(args.seed),
            ]
            if m["adapter"]:
                cmd += ["--adapter_path", m["adapter"]]
            if exp == "trajectory":
                cmd += ["--eps_start", str(m["eps_start"]),
                        "--eps_end", str(m["eps_end"])]
            cmd_str = " ".join(cmd) + " " + flags

            print(f"{run_name}\t{vram}\t{cmd_str}")


if __name__ == "__main__":
    main()
