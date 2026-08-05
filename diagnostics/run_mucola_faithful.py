#!/usr/bin/env python3
"""
run_mucola_faithful.py

Two continuous embedding-space optimizers run through IDENTICAL code, differing in
NOTHING but how the target token enters the language-model term. That is the whole
design: the difference between the two arms is the measurement.

    --variant mucola_faithful
        MuCoLa's energy (Kumar, Paria and Tsvetkov 2022, section 3, "Energy as a
        function of embeddings"):

            P(e~_{n+1} | e~_{1:n}) = exp(h_n . e~_{n+1} + b_{n+1})
                                     / sum_j exp(h_n . e_j + b_j)

        The NUMERATOR takes the continuous state e~ in place of the looked-up target
        embedding; the denominator sums over the real embedding table. GPT-2's lm_head
        is tied to the input embedding and carries no bias, so b = 0. The state
        therefore receives gradient by BOTH of the paths the paper names: (a) directly
        from -log P, whose gradient with respect to e~_{n+1} is h_n, and (b) through
        the following hidden states by back-propagation.

    --variant cls_thesis
        The energy this repository's continuous sampler actually implements
        (core/prep.py:joint_log_prob_from_inputs_embeds, driven from
        core/base_sampler.py:50-53): a hard-target cross-entropy in which the target
        token enters as a DISCRETE INDEX, namely the nearest-neighbour projection of
        the current state. Path (a) is absent by construction.

NEITHER ARM IS CALLED "MuCoLa" LOOSELY. The thesis's sampler reproduces MuCoLa's state
geometry, a continuous embedding state with a projection onto the embedding table after
every update, and this script reproduces that geometry for both arms. What it varies is
the ENERGY. See MUCOLA_CORRECTION_PROPOSED.md.

Method details taken from the paper rather than from the thesis's setup:
  - plain gradient ASCENT on the continuous state, no Metropolis-Hastings anywhere
    (their EmbedGD optimizer has none and takes an argmin, not a sample);
  - projection to the nearest embedding after every gradient step;
  - a FIXED constraint weight (their betas 0.8 / 0.2). The paper's Lagrangian with
    epsilon thresholds is deliberately not reproduced, the same deviation the existing
    core/constraint.py documents, because the question here is about the energy's
    coordinates and not about constraint-satisfaction guarantees.
  - the near-uniform simplex "zeros" initialization, whose embedding is the centroid of
    the embedding table (scripts/run_constrained.py:make_init_s, imported).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.prep import (load_tokenizer_and_model, project_to_vocab_by_l2,
                       joint_log_prob_from_inputs_embeds)
from core.constraint import load_sentiment_head
from scripts.run_constrained import MUCOLA_PROMPTS, build_continuation_case
from scripts.run_experiment import seed_all


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def project_l2_lowmem(s, E, sq_norm):
    """Nearest-embedding projection without the (M, V, D) intermediate.

    core/prep.py:project_to_vocab_by_l2 forms `s[:, None, :] - E[None, :, :]` explicitly.
    At one masked position that is 0.26 GB and harmless, which is why the whole recovery
    grid never hit it; at the 20 positions of the continuation task it is 20 x 50257 x
    1280 x 4 bytes = 4.8 GB in a single allocation, and it OOM'd three jobs here.

    This computes the same argmin through the standard expansion
        ||s - e||^2 = ||s||^2 + ||e||^2 - 2 s.e
    dropping ||s||^2, which is constant across candidates and cannot change the argmin.
    That is the same form core/dls.py already uses for its distance term. Verified
    against the original to give IDENTICAL indices before being used (see the equivalence
    check in the run log); the shared helper is deliberately left untouched so nothing in
    the archived CLS path changes.
    """
    return torch.argmin(sq_norm.unsqueeze(0) - 2.0 * (s @ E.T), dim=-1)


def mucola_lm_term(model, inputs_embeds):
    """log P(e~_{2:L} | e~_{1:L-1}) with the CONTINUOUS state in the softmax numerator.

    This is the term whose gradient reaches the token's own score directly. The
    denominator is the ordinary normalizer over the real embedding table, which is
    exactly what the model's logits already are, so no second matmul is needed.
    """
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1][0]                      # (L, D) final hidden states
    logits = out.logits[0].float()                    # (L, V) = h @ W^T (tied, no bias)
    logZ = torch.logsumexp(logits, dim=-1)            # (L,)
    # numerator: h_t . e~_{t+1}, the continuous vector standing in for the target
    num = (h[:-1].float() * inputs_embeds[0, 1:].float()).sum(dim=-1)   # (L-1,)
    return (num - logZ[:-1]).sum(), out


def cls_lm_term(model, inputs_embeds, target_ids):
    """The repository's energy: hard target, entering as a discrete index."""
    lp = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids)
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
    return lp, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--variant", choices=["mucola_faithful", "cls_thesis"], required=True)
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--beta_lm", type=float, default=0.8)
    p.add_argument("--beta_c", type=float, default=0.2)
    p.add_argument("--cons_gain", type=float, default=1.0,
                   help="multiplies beta_c, so Study D can sweep the constraint weight "
                        "without changing the paper's default ratio at gain 1.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--normalize_grad", action="store_true",
                   help="unit-norm the combined gradient per position before stepping. "
                        "Leaves the direction, and hence the beta_lm/beta_c weighting, "
                        "untouched; fixes the step LENGTH so a constraint-weight sweep is "
                        "interpretable. Without it the step length scales with the weight "
                        "and the sweep is dominated by overshoot.")
    p.add_argument("--lr_start", type=float, default=5.0)
    p.add_argument("--lr_end", type=float, default=0.05)
    p.add_argument("--span_len", type=int, default=20)
    p.add_argument("--samples_per_prompt", type=int, default=4)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results/ctg/studyD")
    a = p.parse_args()

    run_name = a.run_name
    if a.num_shards > 1:
        run_name = f"{run_name}.shard{a.shard_idx}of{a.num_shards}"
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, run_name + ".json")
    if os.path.exists(out_path):
        print(f"[{run_name}] already done; skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(1234)
    t0 = time.time()
    tok, model = load_tokenizer_and_model(a.model_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    head = load_sentiment_head(a.head, device)
    E = model.get_input_embeddings().weight.detach()
    E_sq = (E * E).sum(dim=1)
    centroid = E.mean(dim=0, keepdim=True)
    lrs = np.linspace(a.lr_start, a.lr_end, a.steps)
    beta_c = a.beta_c * a.cons_gain

    plan = []
    for pi, prompt in enumerate(MUCOLA_PROMPTS):
        for k in range(a.samples_per_prompt):
            plan.append((len(plan), pi, prompt, a.data_seed + 1000 * pi + k))
    if a.num_shards > 1:
        plan = [q for q in plan if q[0] % a.num_shards == a.shard_idx]
    print(f"[{run_name}] variant={a.variant} {len(plan)} sequences, {a.steps} steps",
          flush=True)

    n_fwd = n_bwd = 0
    rows = []
    for gi, pi, prompt, seed in plan:
        seed_all(seed)
        ids, span_idx, _ = build_continuation_case(tok, prompt, a.span_len, seed,
                                                   device, tok.vocab_size)
        span = torch.tensor(span_idx, device=device)
        with torch.no_grad():
            base = model.get_input_embeddings()(ids)
        # MuCoLa's near-uniform simplex init: its embedding is the table centroid.
        s = centroid.repeat(len(span_idx), 1).clone().detach().requires_grad_(True)

        for k in range(a.steps):
            emb = base.clone()
            emb[0, span, :] = s
            if a.variant == "mucola_faithful":
                lm, out = mucola_lm_term(model, emb)
                n_fwd += 1
            else:
                tgt = ids.clone()
                tgt[0, span] = project_l2_lowmem(s.detach(), E, E_sq)
                lm, out = cls_lm_term(model, emb, tgt)
                n_fwd += 2
            cons = torch.log_softmax(head(out.hidden_states[-1]), dim=-1)[0, a.target_label]
            obj = a.beta_lm * lm + beta_c * cons
            g = torch.autograd.grad(obj, s)[0]
            n_bwd += 1
            with torch.no_grad():
                if a.normalize_grad:
                    # Unit-norm the COMBINED gradient, per position. This does NOT change
                    # the direction, so the beta_lm / beta_c weighting still decides where
                    # the step points; it fixes the step LENGTH, which otherwise scales
                    # with the constraint weight and makes a weight sweep uninterpretable.
                    # Measured without it: raising the constraint weight made the
                    # constraint score WORSE (mean log p(target) -1.240 -> -1.268 -> -1.614
                    # at gains 0.5, 2.0, 4.0), because the larger gradient overshot at
                    # lr = 5.0 and projected to unrelated tokens.
                    g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
                s_new = s + lrs[k] * g
                # projection onto the embedding table after EVERY gradient step,
                # which is what the paper does and what the thesis's CLS also does
                s_new = E[project_l2_lowmem(s_new, E, E_sq)]
            s = s_new.clone().detach().requires_grad_(True)

        with torch.no_grad():
            final = ids.clone()
            final[0, span] = project_l2_lowmem(s.detach(), E, E_sq)
            o = model(final, output_hidden_states=True, return_dict=True)
            lp = torch.log_softmax(o.logits[0, :-1].float(), dim=-1)
            lm_f = float(lp.gather(1, final[0, 1:].unsqueeze(1)).sum())
            c_f = float(torch.log_softmax(head(o.hidden_states[-1]), dim=-1)[0, a.target_label])
        rows.append(dict(global_idx=gi, prompt_idx=pi, prompt=prompt, seed=seed,
                         start_text=prompt,
                         text=tok.decode(final[0], skip_special_tokens=True),
                         final_lm=lm_f, final_cons=c_f, accepts=0, steps=a.steps))
        if len(rows) % 5 == 0:
            print(f"[{run_name}] {len(rows)}/{len(plan)}", flush=True)

    csv_path = os.path.join(a.out_dir, run_name + ".csv")
    with open(csv_path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(csv_path + ".tmp", csv_path)

    wall = time.time() - t0
    res = dict(experiment="ctg_continuous_optimizer", run_name=run_name, config=vars(a),
               n=len(rows), variant=a.variant,
               accept_rate_pct=float("nan"),   # no MH in this method, by design
               mean_final_lm=float(np.mean([r["final_lm"] for r in rows])),
               mean_final_cons=float(np.mean([r["final_cons"] for r in rows])),
               steer_head_acc_pct=100.0 * float(np.mean(
                   [np.exp(r["final_cons"]) > 0.5 for r in rows])),
               shortlist_coverage_pct=None, shortlist_coverage_n=0,
               shortlist_rank_median=float("nan"),
               constraint_linearization_spearman=float("nan"),
               constraint_linearization_n=0,
               cost=dict(n_forward=n_fwd, n_backward=n_bwd, n_candidate_evals=0,
                         wall_time_sec=wall, n_sequences=len(rows),
                         n_mh_steps=len(rows) * a.steps, n_accepted_moves=len(rows) * a.steps,
                         sec_per_accepted_move=wall / max(len(rows) * a.steps, 1),
                         forward_equivalents_per_sequence=
                             (n_fwd + 2.0 * n_bwd) / max(len(rows), 1)),
               csv=csv_path)
    atomic_json(out_path, res)
    print(f"[{run_name}] DONE n={res['n']} steer_head={res['steer_head_acc_pct']:.1f}% "
          f"lm={res['mean_final_lm']:.1f} wall={wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
