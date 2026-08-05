#!/usr/bin/env python3
"""Manifest generator for the constrained-generation studies (A-constrained, B, C, D, E).

    python scripts/gen_manifest_ctg.py --study Aq > manifest_ctg_aq.tsv
    python scripts/gen_manifest_ctg.py --study Cprobe > manifest_ctg_cprobe.tsv
    python scripts/gen_manifest_ctg.py --study C --proposal self > manifest_ctg_c.tsv
    python scripts/gen_manifest_ctg.py --study B --mech accept_only > manifest_ctg_b.tsv
    python scripts/gen_manifest_ctg.py --study D --proposal self --mech rescore_k > ...
    python scripts/gen_manifest_ctg.py --study E --proposal self --mech rescore_k > ...

Shard counts are per family and chosen so no single shard runs much beyond half an hour;
the exact-scoring arms get more shards because they cost roughly twenty times the
self-term arm per step.
"""
import argparse
import os

# VRAM declarations. GPT-2 Large fp32 is ~4 GB; the rest is activations, which grow with
# the candidate batch. RoBERTa and SEDD add a second model on the same card.
VRAM = {"self": 12, "onehot": 14, "embgrad": 14, "uniform": 12,
        "exact_k": 20, "roberta": 16, "sedd": 18}
# Arms that evaluate a batch of k candidate sequences per step, and are therefore
# roughly k times the cost of the self-term arm.
EXPENSIVE_PROPOSALS = {"exact_k"}
EXPENSIVE_MECHS = {"rescore_k", "both"}


def shards_for(proposal, mech, base):
    n = base
    if proposal in EXPENSIVE_PROPOSALS or mech in EXPENSIVE_MECHS:
        n = base * 3
    elif proposal in ("onehot", "embgrad") or mech in ("grad_prop", "mix_shortlist"):
        n = base * 2
    return n


def emit(lines, run, proposal, mech, a, extra="", vram=None, shards=None):
    sh = shards or shards_for(proposal, mech, a.shards)
    v = vram or VRAM.get(proposal, 14)
    if mech in EXPENSIVE_MECHS:
        v = max(v, 20)
    cmd = (f"gfn-lm-tuning/gfn/bin/python diagnostics/run_ctg.py "
           f"--model_path {a.model} --head {a.head} "
           f"--proposal {proposal} --constraint_mech {mech} "
           f"--target_label {a.target_label} --steps {a.steps} --span_len {a.span_len} "
           f"--samples_per_prompt {a.samples_per_prompt} --topk {a.topk} "
           f"--temperature {a.temperature} --out_dir {a.out_dir} {extra}".strip())
    for i in range(sh):
        lines.append((f"{run}.shard{i}of{sh}", v,
                      f"{cmd} --run_name {run} --shard_idx {i} --num_shards {sh}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True,
                   choices=["Aq", "Cprobe", "Ccoverage", "C", "B", "D", "E"])
    p.add_argument("--proposal", default="self", help="fixed proposal for C, D, E")
    p.add_argument("--mech", default="accept_only", help="fixed mechanism for B, D, E")
    p.add_argument("--model", default=os.environ.get("GPT2SFT", "$GPT2SFT"))
    p.add_argument("--head", default=os.environ.get("SENTIMENT_HEAD", "$SENTIMENT_HEAD"))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--span_len", type=int, default=20)
    p.add_argument("--samples_per_prompt", type=int, default=4)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--mucola_steps", type=int, default=300,
                   help="MuCoLa's own optimizer step count, read from their repo "
                        "(examples/prompt/constrained_sampling_mucola.sh, OPTIMSTEPS=300)")
    a = p.parse_args()
    if a.out_dir is None:
        a.out_dir = f"results/ctg/study{a.study}"

    L = []
    if a.study == "Aq":
        # Study A on the constrained task: the same five arms, constraint in the accept
        # step only, so the comparison is about the PROPOSAL exactly as on recovery.
        for pr in ["self", "onehot", "exact_k", "embgrad", "uniform"]:
            emit(L, f"Aq.{pr}", pr, "accept_only", a, extra="--coverage_every 25")

    elif a.study == "Cprobe":
        # Cheap early probe: which mechanism to fix for Study B. Short chains, fewer
        # sequences; it decides a configuration, it does not report a headline.
        a.steps, a.samples_per_prompt = 200, 2
        for m in ["accept_only", "rescore_k", "grad_prop", "both", "mix_shortlist"]:
            emit(L, f"Cprobe.{m}", a.proposal, m, a,
                 extra="--coverage_every 10 --lin_check 20", shards=2)

    elif a.study == "Ccoverage":
        # Shortlist coverage at k in {16, 64, 256}, reported BEFORE any adherence number.
        # Short chains: coverage is a property of the shortlist and the classifier, not
        # of how long the chain runs.
        a.steps, a.samples_per_prompt = 150, 2
        for k in [16, 64, 256]:
            a.topk = k
            emit(L, f"Ccov.k{k}", a.proposal, "accept_only", a,
                 extra="--coverage_every 5 --lin_check 10", shards=2)

    elif a.study == "C":
        for m in ["accept_only", "rescore_k", "grad_prop", "both", "mix_shortlist"]:
            emit(L, f"C.{m}", a.proposal, m, a,
                 extra="--coverage_every 25 --lin_check 50")

    elif a.study == "B":
        for pr in ["self", "onehot", "roberta", "sedd", "embgrad", "uniform"]:
            emit(L, f"B.{pr}", pr, a.mech, a, extra="--coverage_every 25")

    elif a.study == "D":
        # The head-to-head. Every method gets the SAME constraint-weight sweep, so the
        # comparison is a curve (adherence against fluency) and not a single point that
        # could have been picked to flatter one arm.
        gains = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        for g in gains:
            emit(L, f"D.ours.g{g}", a.proposal, a.mech, a,
                 extra=f"--cons_gain {g} --coverage_every 50")
        # the two continuous embedding-space optimizers, run through identical code so
        # the only difference between them is where the target token enters the energy
        for variant, vram in (("mucola_faithful", 14), ("cls_thesis", 14)):
            for g in gains:
                run = f"D.{variant}.g{g}"
                sh = a.shards
                cmd = (f"gfn-lm-tuning/gfn/bin/python diagnostics/run_mucola_faithful.py "
                       f"--model_path {a.model} --head {a.head} --variant {variant} "
                       f"--target_label {a.target_label} --steps {a.mucola_steps} "
                       f"--span_len {a.span_len} "
                       f"--samples_per_prompt {a.samples_per_prompt} "
                       f"--cons_gain {g} --out_dir {a.out_dir}")
                for i in range(sh):
                    L.append((f"{run}.shard{i}of{sh}", vram,
                              f"{cmd} --run_name {run} --shard_idx {i} --num_shards {sh}"))

    elif a.study == "E":
        # Study E: many independent chains per prompt so within-prompt diversity is
        # measurable at all. Different data_seed per replicate keeps the chains
        # independent rather than re-running the same one.
        a.samples_per_prompt = 8
        for r in range(4):
            emit(L, f"E.chain.r{r}", a.proposal, a.mech, a,
                 extra=f"--cons_gain {a.temperature and 1.0} --data_seed {r * 7919} "
                       f"--coverage_every 100")

    for name, vram, cmd in L:
        print(f"{name}\t{vram}\t{cmd}")


if __name__ == "__main__":
    main()
