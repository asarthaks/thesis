#!/usr/bin/env python3
"""
run_constrained.py

The constraint ablation, on a clean 2x2 so any difference is attributable.

TWO TASKS (--task)
  continuation : MuCoLa's ACTUAL sentiment task. A short fixed prompt ("The movie",
                 "Once upon a time", ...) followed by L NEW tokens optimized jointly
                 from an uninformative init. Nothing is repaired; the whole span is
                 conjured. This is where the constraint gradient has the degrees of
                 freedom to actually steer sentiment, so it is the fair test.
  infill       : OUR task. Real sentence, corrupt M tokens, recover them. Keeps
                 things commensurate with our existing 145 runs.
                 WARNING: with few masks the fixed real context dominates the
                 sentence-level classifier and the sampler cannot flip sentiment no
                 matter how good the gradient is. Use --num_masks 8+ or this test is
                 rigged to find nothing for reasons unrelated to the gradient.

TWO SETUPS (--setup)
  mucola : their hyperparameters, read from their repo.
             steps=300, span=20, noise 5.0 -> 0.05, betas 0.8/0.2, no MH, no gradnorm
  ours   : ours. steps=50, eps 10.5 -> 0.1, MH on, grad-norm on.

Crossing task x setup separates "the task matters" from "the hyperparameters
matter" instead of confounding the two.

FIVE ARMS (--constraint_mode)
  lm_only     : LM gradient only. Our null result, restated on this task.
  full        : 0.8*LM + 0.2*classifier. The real MuCoLa-style energy.
  cons_only   : classifier gradient alone.
  cons_random : classifier gradient's MAGNITUDE, RANDOM direction.   <<< THE TEST
  random      : pure random direction control.

If `full` steers sentiment and `cons_random` does not, then the CONSTRAINT
gradient's DIRECTION carries real information, exactly where the LM gradient's
direction did not. That single contrast explains why MuCoLa/COLD work despite our
null result on the LM gradient.
"""

import os
import json
import argparse
import random

import numpy as np
import torch

from core.prep import load_tokenizer_and_model
from core.dls import DiscreteLangevinSampler
from core.cls import ContinuousLangevinSampler
from core.constraint import load_sentiment_head, ConstrainedEnergyMixin


class ConstrainedDLS(ConstrainedEnergyMixin, DiscreteLangevinSampler):
    pass


class ConstrainedCLS(ConstrainedEnergyMixin, ContinuousLangevinSampler):
    pass


# MuCoLa's 15 PPLM discriminator prompts, verbatim from their repo
# (data/control-prompts/pplm-discrim-prompts/prompts.txt)
MUCOLA_PROMPTS = [
    "Once upon a time", "The book", "The chicken", "The city", "The country",
    "The horse", "The lake", "The last time", "The movie", "The painting",
    "The pizza", "The potato", "The president of the country", "The road",
    "The year is 1910.",
]

# Read from their repo:
#   examples/prompt/sentiment-all/mucola-disc.sh   -> LENGTH=20, noise 5.0 -> 0.05
#   examples/prompt/constrained_sampling_mucola.sh -> OPTIMSTEPS=300, betas 0.8:0.2
# Note: their optimizer (EmbedGD) has NO Metropolis-Hastings anywhere in the repo,
# takes an argmin rather than a sample, and does NOT normalize the gradient.
#
# grad_norm is FALSE in BOTH setups, and that is forced, not a choice. The weights
# beta_lm/beta_c set the RELATIVE MAGNITUDES of the two energy terms; gradient
# normalization discards magnitude by construction. With grad_norm=True the combined
# gradient is normalized to a unit vector whose direction is dominated by the (much
# larger) LM gradient, so the 0.2-weighted constraint has literally zero effect and
# `full` becomes bitwise identical to `lm_only` (we verified this empirically).
# A weighted multi-term energy and unit-norm gradients cannot both be honoured.
# `ours` therefore keeps its identity (DLS, 50 steps, MH, our eps schedule) but must
# run unnormalized. This deviation is required for the constraint to exist at all.
SETUPS = {
    "mucola": dict(steps=300, span_len=20, eps_start=5.0, eps_end=0.05,
                   beta_lm=0.8, beta_c=0.2, mh=False, grad_norm=False),
    "ours":   dict(steps=50, span_len=20, eps_start=10.5, eps_end=0.1,
                   beta_lm=0.8, beta_c=0.2, mh=True, grad_norm=False),
}


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def classify(model, head, ids):
    with torch.no_grad():
        emb = model.get_input_embeddings()(ids)
        out = model(inputs_embeds=emb, output_hidden_states=True, return_dict=True)
        return int(head(out.hidden_states[-1]).argmax(-1).item())


def build_continuation_case(tok, prompt, span_len, seed, device, vocab):
    """MuCoLa-style: fixed prompt + span_len NEW tokens to optimize.

    The token ids of the span are placeholders (uniform random). What actually
    determines the starting point of the sampler is `init_s` (see --init):
      random_token : s starts at those random token embeddings (discrete analogue)
      centroid     : s starts at the embedding-table centroid, which IS the
                     embedding of MuCoLa's near-uniform simplex init ("zeros").
    """
    rng = np.random.RandomState(seed)
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    span = torch.tensor(rng.randint(0, vocab, size=(1, span_len)),
                        device=device, dtype=p_ids.dtype)
    ids = torch.cat([p_ids, span], dim=1)
    mask_indices = list(range(p_ids.shape[1], ids.shape[1]))  # only the new span moves
    return ids, mask_indices, ids.clone()   # no ground truth; ref is a copy


def make_init_s(init, sampler, n_masks, device, seed=0):
    """Starting continuous state (M, D) for the sampler, or None for the default.

    MuCoLa's optimizable variable is a SIMPLEX over the vocabulary: the embedding it
    feeds the model is a convex combination sum_v p_v * e_v. Their "zeros" init makes
    p near-uniform, so the embedding they start from is (approximately) the CENTROID
    of the embedding table, i.e. a point deep inside the convex hull that corresponds
    to no particular token.

    Our CLS is also continuous, but it parameterizes a FREE vector in R^D and projects
    to the nearest token to evaluate the energy. Starting it at a random token embedding
    is therefore NOT the same starting point as theirs. `centroid` is the faithful
    analogue and is what --setup mucola should use.
    """
    if init == "default":
        return None
    E = sampler.emb_matrix                       # (V, D)
    if init == "centroid":
        c = E.mean(dim=0, keepdim=True)          # (1, D)
        return c.repeat(n_masks, 1)
    if init == "random_token":
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randint(0, E.shape[0], (n_masks,), generator=g).to(device)
        return E[idx].clone()
    raise ValueError(f"unknown init: {init}")


def build_infill_case(tok, text, num_masks, seed, device, vocab):
    rng = np.random.RandomState(seed)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    L = ids.shape[1]
    if L < num_masks + 3:
        return None
    idxs = sorted(rng.choice(range(1, L - 1), size=num_masks, replace=False).tolist())
    orig = ids.clone()
    corrupted = ids.clone()
    for i in idxs:
        r = int(rng.randint(0, vocab))
        while r == ids[0, i].item():
            r = int(rng.randint(0, vocab))
        corrupted[0, i] = r
    return corrupted, idxs, orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_tag", default="gpt2-large")
    p.add_argument("--head", required=True, help="sentiment head .pt")
    p.add_argument("--task", choices=["continuation", "infill"], required=True)
    p.add_argument("--setup", choices=["mucola", "ours"], required=True)
    p.add_argument("--constraint_mode", required=True,
                   choices=["lm_only", "full", "cons_only", "cons_random", "random"])
    p.add_argument("--sampler", choices=["dls", "cls"], default=None,
                   help="default: cls for --setup mucola (they optimize a continuous "
                        "simplex), dls for --setup ours")
    p.add_argument("--init", choices=["default", "centroid", "random_token"], default=None,
                   help="starting continuous state. default: centroid for --setup mucola "
                        "(matches their near-uniform simplex 'zeros' init), random_token "
                        "for --setup ours on the continuation task. Ignored for infill, "
                        "which always starts from the corrupted sentence.")
    p.add_argument("--target_label", type=int, default=1, help="1=positive, 0=negative")

    p.add_argument("--samples_per_prompt", type=int, default=20,
                   help="continuation task: MuCoLa uses 20 per prompt x 15 prompts")
    p.add_argument("--num_masks", type=int, default=8,
                   help="infill task: needs MANY masks or the real context dominates")
    p.add_argument("--n_samples", type=int, default=100, help="infill task")

    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=0.01)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--out_dir", default="results_constrained")
    p.add_argument("--wandb_project", default="ctg-langevin-thesis")
    p.add_argument("--no_wandb", action="store_true")

    p.add_argument("--steps", type=int, default=None, help="override the setup preset")
    p.add_argument("--span_len", type=int, default=None, help="override the setup preset")
    args = p.parse_args()

    S = dict(SETUPS[args.setup])
    if args.steps is not None:
        S["steps"] = args.steps
    if args.span_len is not None:
        S["span_len"] = args.span_len

    # Setup-aware defaults. MuCoLa optimizes a CONTINUOUS simplex over the vocabulary,
    # so the faithful reproduction is CLS started at the embedding centroid (their
    # near-uniform "zeros" init). Our own setup is the discrete sampler from a random
    # token. Both are reported; the difference between them is itself informative.
    if args.sampler is None:
        args.sampler = "cls" if args.setup == "mucola" else "dls"
    if args.init is None:
        args.init = "centroid" if args.setup == "mucola" else "random_token"

    run_name = (f"{args.model_tag}.{args.task}.{args.setup}.{args.sampler}."
                f"init-{args.init}.{args.constraint_mode}.lbl{args.target_label}")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, run_name + ".json")
    if os.path.exists(out_path):
        print(f"[{run_name}] already done; skipping")
        return

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=run_name,
                       group=f"constrained.{args.task}.{args.setup}",
                       config={**vars(args), **S})
        except Exception as e:
            print("wandb off:", e)
            use_wandb = False

    seed_all(1234)
    tok, model = load_tokenizer_and_model(args.model_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    device = next(model.parameters()).device
    head = load_sentiment_head(args.head, device)
    vocab = tok.vocab_size

    Cls = ConstrainedDLS if args.sampler == "dls" else ConstrainedCLS
    sampler = Cls(
        model=model, tokenizer=tok, steps=S["steps"], temperature=args.temperature,
        oracle=False, method="policy", mh_sampling=S["mh"],
        grad_normalization=S["grad_norm"], noise_scale=args.noise_scale,
        epsilon_schedule=np.linspace(S["eps_start"], S["eps_end"], S["steps"]),
    )
    sampler.head = head
    sampler.target_label = args.target_label
    sampler.beta_lm = S["beta_lm"]
    sampler.beta_c = S["beta_c"]
    sampler.constraint_mode = args.constraint_mode

    # ---- build cases --------------------------------------------------------
    cases = []
    if args.task == "continuation":
        for pi, prompt in enumerate(MUCOLA_PROMPTS):
            for k in range(args.samples_per_prompt):
                cases.append((prompt,
                              build_continuation_case(tok, prompt, S["span_len"],
                                                      args.data_seed + 1000 * pi + k,
                                                      device, vocab)))
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x["text"].strip() for x in ds if 12 < len(x["text"].strip().split()) < 40]
        ti = 0
        while len(cases) < args.n_samples and ti < len(texts):
            c = build_infill_case(tok, texts[ti], args.num_masks,
                                  args.data_seed + ti, device, vocab)
            if c is not None:
                cases.append((texts[ti], c))
            ti += 1

    print(f"[{run_name}] task={args.task} setup={args.setup} arm={args.constraint_mode} "
          f"steps={S['steps']} eps={S['eps_start']}->{S['eps_end']} "
          f"mh={S['mh']} gn={S['grad_norm']} | {len(cases)} cases", flush=True)

    n_before = n_after = 0
    kls, examples = [], []

    for j, (src, case) in enumerate(cases):
        start_ids, mask_indices, ref_ids = case

        seed_all(args.data_seed + j)

        # Only the continuation task gets a custom init. Infill must start from the
        # corrupted sentence, since that IS the task.
        init_s = None
        if args.task == "continuation":
            init_s = make_init_s(args.init, sampler, len(mask_indices), device,
                                 seed=args.data_seed + j)

        # The "before" baseline must be measured at the ACTUAL starting state. With a
        # centroid init the placeholder token ids are meaningless, so we project the
        # starting continuous state to tokens first. Otherwise steering_gain would be
        # computed against a state the sampler never occupied.
        baseline_ids = start_ids.clone()
        if init_s is not None:
            from core.prep import project_to_vocab_by_l2
            baseline_ids[0, mask_indices] = project_to_vocab_by_l2(init_s, sampler.emb_matrix)
        if classify(model, head, baseline_ids) == args.target_label:
            n_before += 1

        _, metrics = sampler.optimize(start_ids.clone(), mask_indices, ref_ids.clone(),
                                      init_s=init_s)

        final_ids = start_ids.clone()
        final_ids[0, mask_indices] = torch.tensor(metrics[-1]["token_ids"], device=device)
        if classify(model, head, final_ids) == args.target_label:
            n_after += 1

        kl = metrics[-1].get("avg_kl_divergence")
        if kl is not None and np.isfinite(kl):
            kls.append(kl)
        if len(examples) < 12:
            examples.append([str(src)[:50],
                             tok.decode(baseline_ids[0], skip_special_tokens=True)[:100],
                             tok.decode(final_ids[0], skip_special_tokens=True)[:100]])

        if (j + 1) % 20 == 0:
            print(f"[{run_name}] {j+1}/{len(cases)} sent_acc={100*n_after/(j+1):.1f}%",
                  flush=True)

    n = len(cases)
    before = 100.0 * n_before / max(n, 1)
    after = 100.0 * n_after / max(n, 1)
    res = dict(run_name=run_name, config={**vars(args), **S}, n=n,
               sentiment_acc_before=before, sentiment_acc=after,
               steering_gain=after - before,
               final_kl=float(np.mean(kls)) if kls else None,
               examples=examples)

    print(f"\n[{run_name}] DONE  n={n}")
    print(f"  sentiment BEFORE : {before:.1f}%")
    print(f"  sentiment AFTER  : {after:.1f}%")
    print(f"  STEERING GAIN    : {after - before:+.1f} points")
    if kls:
        print(f"  final KL         : {np.mean(kls):.3f}")

    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, out_path)
    print(f"  wrote {out_path}")

    if use_wandb:
        import wandb
        wandb.run.summary.update({k: v for k, v in res.items()
                                  if k not in ("config", "examples", "run_name")})
        wandb.log({"examples": wandb.Table(columns=["source", "start", "final"],
                                           data=examples)})
        wandb.finish()


if __name__ == "__main__":
    main()
