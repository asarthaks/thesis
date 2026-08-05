#!/usr/bin/env python3
"""Study A manifest: is the future term worth its cost, on the RECOVERY task?

Two framings, because they answer different objections.

FRAMING 1, full vocabulary. Each arm proposes over all 50,257 tokens with the
Langevin distance term present, exactly as the thesis grid does, in the cell that
produced the thesis's headline one-hot result (eps 10.5 -> 0.1, T = 1.0, MH on,
grad-norm off; REVISION_LOG C9). Directly comparable to the published 40.0 percent
one-hot and 2.0 percent input-embedding numbers. The exact-future arm cannot appear
here: scoring 50,257 candidates exactly would cost 50,257 forward passes per step.

FRAMING 2, matched shortlist. Every arm proposes over the SAME frozen top-k
candidate set (k in 16, 64, 256), scored only by its own surrogate, with the
distance term dropped. This is the clean comparison for Q1, because the exact arm
can only ever see a shortlist and the restriction is then charged to every arm
equally rather than to the exact one alone.

    python scripts/gen_manifest_ctg_a.py > manifest_ctg_a.tsv
"""
import argparse
import os

FULL_VOCAB_ARMS = ["policy_self", "policy_onehot", "policy", "uniform"]
SHORTLIST_ARMS = ["policy_self", "policy_onehot", "policy_exact_k", "policy", "uniform"]
KS = [16, 64, 256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/ctg/studyA")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--vram", type=int, default=16)
    a = p.parse_args()

    gpt2 = os.environ.get("GPT2SFT", "$GPT2SFT")
    base = (f"gfn-lm-tuning/gfn/bin/python scripts/run_experiment.py --sampler dls "
            f"--model_path {gpt2} --model_tag gpt2-large --dtype float32 "
            f"--n_samples {a.n_samples} --num_masks 1 --data_seed 0 --steps {a.steps} "
            f"--eps_start 10.5 --eps_end 0.1 --temperature 1.0 --mh --mh_exact_all_arms "
            f"--log_proposal_stats --no_wandb --out_dir {a.out_dir}")

    lines = []
    for arm in FULL_VOCAB_ARMS:
        for sh in range(a.shards):
            rn = f"A.fullvocab.{arm}"
            lines.append((f"{rn}.shard{sh}of{a.shards}", a.vram,
                          f"{base} --method {arm} --run_name {rn} "
                          f"--shard_idx {sh} --num_shards {a.shards}"))

    for k in KS:
        for arm in SHORTLIST_ARMS:
            for sh in range(a.shards):
                rn = f"A.k{k}.{arm}"
                # k=256 exact scoring holds a bigger activation peak; declare more VRAM
                vram = a.vram + (8 if (arm == "policy_exact_k" and k >= 256) else 0)
                lines.append((f"{rn}.shard{sh}of{a.shards}", vram,
                              f"{base} --method {arm} --run_name {rn} "
                              f"--proposal_topk {k} --exact_k {k} --drop_distance_term "
                              f"--shard_idx {sh} --num_shards {a.shards}"))

    for name, vram, cmd in lines:
        print(f"{name}\t{vram}\t{cmd}")


if __name__ == "__main__":
    main()
