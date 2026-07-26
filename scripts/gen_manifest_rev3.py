"""Manifest generator for the evaluation-3 slate.

Four job families, all on the frozen GPT-2 Large SFT base, all writing into one
out_dir with distinct run_names so a single queue invocation balances them:

  sweep   (epsilon, temperature) grid with per-step proposal statistics logged.
          Answers the objection that the reported null was measured at a
          configuration whose proposal is numerically uniform over the vocabulary,
          so no gradient however informative could have shown up. Small n, many
          cells.
  power   the flagship configuration replicated over independent corruption seeds
          so the paired contrast reaches n of about 1000 and the pre-registered
          equivalence margin can actually be certified rather than merely not
          rejected. WikiText-2 validation yields only ~282 sentences under the
          10-to-40-word filter, so extra n comes from independent corruptions of
          the same sentences, not from new sentences.
  mhfix   the same arms with the reverse-proposal term computed exactly for EVERY
          arm, not only the policy arm. Bounds the effect of the asymmetry in the
          archived grid.
  onehot  the one-hot / simplex gradient proposal, the surrogate the discrete
          samplers this thesis follows actually use, against the input-embedding
          proposal the thesis measured.

Usage:
    python scripts/gen_manifest_rev3.py --families sweep power mhfix onehot > manifest_rev3.tsv
"""
import argparse
import os

PY = "gfn-lm-tuning/gfn/bin/python"
BASE = ("{py} scripts/run_experiment.py --sampler dls --model_path {model} "
        "--model_tag gpt2-large --dtype float32 --num_masks 1 "
        "--min_words 10 --max_words 40 --noise_scale 0.01 --no_wandb "
        "--out_dir {out} --run_name {run}")

ARMS = ["policy", "grad_norm_preserved_random_dir", "random"]
SHORT = {"policy": "policy", "grad_norm_preserved_random_dir": "gnp", "random": "random"}

EPS_START = [1.05, 10.5, 105.0, 1050.0, 10500.0]
TEMPS = [0.05, 0.25, 1.0, 5.0, 25.0]
POWER_SEEDS = [1000, 2000, 3000, 4000]


def tag(x):
    return str(x).replace(".", "p").replace("-", "m")


def emit(rows, run, vram, cmd):
    rows.append(f"{run}\t{vram}\t{cmd}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("GPT2SFT", ""))
    p.add_argument("--out_dir", default="results/grid/rev3")
    p.add_argument("--families", nargs="+",
                   default=["sweep", "power", "mhfix", "onehot"])
    p.add_argument("--sweep_n", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--vram", type=int, default=6)
    args = p.parse_args()
    if not args.model:
        raise SystemExit("--model or $GPT2SFT required (source env.sh first)")

    rows = []

    if "sweep" in args.families:
        # gradient normalization is DISABLED here on purpose: with it enabled the
        # gradient is rescaled to unit norm, which fixes the gradient term's
        # contribution at a level the distance term dominates by construction, so
        # the sweep could not answer its own question.
        for eps0 in EPS_START:
            for temp in TEMPS:
                for arm in ARMS:
                    run = f"sweep_{SHORT[arm]}_e{tag(eps0)}_t{tag(temp)}"
                    cmd = BASE.format(py=PY, model=args.model, out=args.out_dir, run=run) + (
                        f" --method {arm} --mh --steps 50 --n_samples {args.sweep_n}"
                        f" --data_seed 0 --eps_start {eps0} --eps_end {eps0 / 105.0:.6g}"
                        f" --temperature {temp} --log_proposal_stats")
                    emit(rows, run, args.vram, cmd)

    if "power" in args.families:
        for seed in POWER_SEEDS:
            for arm in ARMS:
                run = f"power_{SHORT[arm]}_seed{seed}"
                cmd = BASE.format(py=PY, model=args.model, out=args.out_dir, run=run) + (
                    f" --method {arm} --mh --grad_norm --steps 50"
                    f" --n_samples {args.n_samples} --data_seed {seed}"
                    f" --eps_start 10.5 --eps_end 0.1 --temperature 5.0 --log_proposal_stats")
                emit(rows, run, args.vram, cmd)

    if "mhfix" in args.families:
        for gn in (True, False):
            for arm in ARMS:
                run = f"mhfix_{SHORT[arm]}_{'gn' if gn else 'nogn'}"
                cmd = BASE.format(py=PY, model=args.model, out=args.out_dir, run=run) + (
                    f" --method {arm} --mh --mh_exact_all_arms"
                    f"{' --grad_norm' if gn else ''} --steps 50"
                    f" --n_samples {args.n_samples} --data_seed 0"
                    f" --eps_start 10.5 --eps_end 0.1 --temperature 5.0 --log_proposal_stats")
                emit(rows, run, args.vram, cmd)

    if "onehot" in args.families:
        for mh in (True, False):
            for gn in (True, False):
                run = f"onehot_{'mh' if mh else 'nomh'}_{'gn' if gn else 'nogn'}"
                cmd = BASE.format(py=PY, model=args.model, out=args.out_dir, run=run) + (
                    f" --method policy_onehot{' --mh' if mh else ''}"
                    f"{' --grad_norm' if gn else ''} --steps 50"
                    f" --n_samples {args.n_samples} --data_seed 0"
                    f" --eps_start 10.5 --eps_end 0.1 --temperature 5.0 --log_proposal_stats")
                emit(rows, run, args.vram, cmd)
        # The one-hot proposal only becomes load-bearing where the distance term
        # does not swamp it, so it also gets the two most gradient-favourable
        # cells of the sweep geometry.
        for eps0, temp in [(1050.0, 1.0), (10500.0, 0.25)]:
            run = f"onehot_sweep_e{tag(eps0)}_t{tag(temp)}"
            cmd = BASE.format(py=PY, model=args.model, out=args.out_dir, run=run) + (
                f" --method policy_onehot --mh --steps 50"
                f" --n_samples {args.n_samples} --data_seed 0"
                f" --eps_start {eps0} --eps_end {eps0 / 105.0:.6g}"
                f" --temperature {temp} --log_proposal_stats")
            emit(rows, run, args.vram, cmd)

    print("\n".join(rows))


if __name__ == "__main__":
    main()
