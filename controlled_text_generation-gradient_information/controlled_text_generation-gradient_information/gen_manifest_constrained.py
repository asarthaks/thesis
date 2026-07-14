#!/usr/bin/env python3
"""
gen_manifest_constrained.py

Emits the constrained-sentiment ablation as a TSV manifest for run_queue.sh.
Same format as gen_manifest.py: run_name <TAB> min_vram_gb <TAB> command

Matrix:
  task  in {continuation, infill}
  setup in {mucola, ours}
  arm   in {lm_only, full, cons_only, cons_random, random}
  label in {1 (positive), 0 (negative)}   [both, so steering is not a one-sided fluke]

= 2 x 2 x 5 x 2 = 40 runs. Each is GPT-2 Large (~6 GB), so they pack several to a card.

NOTE grad_norm is OFF in both setups, and that is forced. beta_lm/beta_c set the
relative magnitudes of the two energy terms, and gradient normalization discards
magnitude, which makes the constraint a no-op (we verified `full` became bitwise
identical to `lm_only`). MuCoLa does not normalize either.

Usage:
  python gen_manifest_constrained.py --model_path <sft> --head <head.pt> \
      --out_dir results_constrained > manifest_constrained.tsv
"""

import argparse

TASKS = ["continuation", "infill"]
SETUPS = ["mucola", "ours"]
ARMS = ["lm_only", "full", "cons_only", "cons_random", "random"]
LABELS = [1, 0]
VRAM = 6   # gpt2-large fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_tag", default="gpt2-large")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out_dir", default="results_constrained")
    ap.add_argument("--samples_per_prompt", type=int, default=10)
    ap.add_argument("--num_masks", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--wandb_project", default="ctg-langevin-thesis")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--labels", nargs="+", type=int, default=LABELS)
    args = ap.parse_args()

    for task in args.tasks:
        for setup in SETUPS:
            sampler = "cls" if setup == "mucola" else "dls"
            init = "centroid" if setup == "mucola" else "random_token"
            for arm in ARMS:
                for lbl in args.labels:
                    rn = (f"{args.model_tag}.{task}.{setup}.{sampler}."
                          f"init-{init}.{arm}.lbl{lbl}")
                    cmd = " ".join([
                        "python run_constrained.py",
                        f"--model_path {args.model_path}",
                        f"--model_tag {args.model_tag}",
                        f"--head {args.head}",
                        f"--task {task}",
                        f"--setup {setup}",
                        f"--constraint_mode {arm}",
                        f"--target_label {lbl}",
                        f"--samples_per_prompt {args.samples_per_prompt}",
                        f"--num_masks {args.num_masks}",
                        f"--n_samples {args.n_samples}",
                        f"--wandb_project {args.wandb_project}",
                        f"--out_dir {args.out_dir}",
                    ])
                    print(f"{rn}\t{VRAM}\t{cmd}")


if __name__ == "__main__":
    main()
