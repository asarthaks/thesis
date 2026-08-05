#!/usr/bin/env python3
"""
run_ctg.py

The controllable-generation half of the CTG phase: an exact-energy discrete
Metropolis-Hastings chain over the SAME constrained task `scripts/run_constrained.py`
already defines, with the proposal source and the constraint mechanism as independent
pluggable choices.

The task is IMPORTED, not rebuilt: the 15 PPLM discriminator prompts, the 20-token
continuation span, and the sentiment head all come from the existing code.

WHAT IS NEW HERE, and why the Langevin runner could not do it:
  - the state is discrete token ids, so a proposal may be any distribution over the
    vocabulary at a position (RoBERTa, SEDD, an exact rescore), not only a Gaussian
    step in embedding space;
  - the constraint can enter at four different places, which is Study C's question;
  - per-sequence TEXT is written out, so a held-out judge can score it in a separate
    pass (`run_ctg_judge.py`) and never sit in VRAM next to a generator.

ENERGY (exact, evaluated on real tokens, never on a continuous relaxation):

    U(x) = beta_lm * log p_LM(x) + beta_c * log p_head(target | x)

Both terms come out of ONE forward pass with output_hidden_states, so the energy costs
one forward call however the proposal is built.

PROPOSALS (--proposal), all scoring candidate v at the position being resampled:
    self       0.5 * log p(v | x_<i)                  1 fwd (the energy's own), 0 bwd
    onehot     self + 0.5 * g^T(e(v) - e(x_i))        + 1 bwd
    exact_k    0.5 * (U(v) - U(x))  on the shortlist  + k fwd
    embgrad    0.5 * g^T(e(v) - e(x_i))               + 1 bwd   (the thesis's control)
    uniform    constant                               0 extra
    roberta    a bidirectional masked-LM conditional  + 1 MLM fwd
    sedd       the SEDD concrete score                + 1 SEDD fwd

CONSTRAINT MECHANISMS (--constraint_mech), Study C:
    accept_only    the constraint is in U and therefore in the accept step, nowhere else
    rescore_k      candidates on the shortlist are rescored by the EXACT classifier
    grad_prop      the classifier's first-order term enters the proposal
    both           rescore_k and grad_prop together
    mix_shortlist  half the shortlist by likelihood rank, half by the classifier's own
                   preference (its first-order ranking over the vocabulary)

SHORTLIST COVERAGE is recorded for every arm and reported BEFORE any adherence number,
because a weak rescoring arm whose preferred candidate never appears on the shortlist is
a proposal-support limitation and not evidence about the mechanism.

REVERSIBILITY. The shortlist support is frozen per (sequence, position, step) and always
contains the incumbent, and the reverse proposal is evaluated under the same rule at the
proposed state. Study A's build log records what happens when either of those is skipped:
acceptance collapses to exactly zero while the proposal itself is healthy.
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

from core.prep import load_tokenizer_and_model
from core.constraint import load_sentiment_head
from scripts.run_constrained import MUCOLA_PROMPTS, build_continuation_case
from scripts.run_experiment import seed_all

PROPOSALS = ["self", "onehot", "exact_k", "embgrad", "uniform", "roberta", "sedd"]
MECHS = ["accept_only", "rescore_k", "grad_prop", "both", "mix_shortlist"]


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


class CtgChain:
    """One exact-energy MH chain over a continuation span."""

    def __init__(self, model, tok, head, args, device):
        self.model, self.tok, self.head, self.a, self.device = model, tok, head, args, device
        self.emb = model.get_input_embeddings().weight.detach()
        self.V = self.emb.shape[0]
        self.n_forward = 0
        self.n_backward = 0
        self.n_candidate_evals = 0
        self.mlm = None
        self.sedd = None
        # cons_gain is the SINGLE knob that controls constraint strength, and it must
        # act on the energy as well as on the proposal. With it applied only inside the
        # proposal, the accept_only arm ignores it entirely and a constraint-weight
        # sweep would produce six identical runs. Study D's curve depends on this.
        self.beta_c_eff = args.beta_c * args.cons_gain

    # ---------------- energy ----------------
    def energy(self, ids, want_self=False):
        """Exact U(x) on real tokens. One forward pass yields the LM term, the
        classifier term and (optionally) every position's next-token conditional."""
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True, return_dict=True)
            self.n_forward += 1
            lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            lm = lp.gather(1, ids[0, 1:].unsqueeze(1)).sum()
            cons = None
            if self.head is not None:
                logits_c = self.head(out.hidden_states[-1])
                cons = torch.log_softmax(logits_c, dim=-1)[0, self.a.target_label]
            total = self.a.beta_lm * lm + (self.beta_c_eff * cons if cons is not None else 0.0)
        return (total, lm, cons, lp) if want_self else (total, lm, cons)

    def energy_batch(self, ids_batch):
        """U(x) for a batch of candidate sequences, chunked. Returns (total, lm, cons)."""
        tot, lms, cons = [], [], []
        chunk = self.a.exact_batch
        with torch.no_grad():
            for i0 in range(0, ids_batch.shape[0], chunk):
                sub = ids_batch[i0:i0 + chunk]
                out = self.model(sub, output_hidden_states=True, return_dict=True)
                self.n_forward += sub.shape[0]
                self.n_candidate_evals += sub.shape[0]
                lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
                lm = lp.gather(2, sub[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1)
                if self.head is not None:
                    c = torch.log_softmax(self.head(out.hidden_states[-1]), dim=-1)[:, self.a.target_label]
                else:
                    c = torch.zeros_like(lm)
                tot.append(self.a.beta_lm * lm + self.beta_c_eff * c)
                lms.append(lm)
                cons.append(c)
        return torch.cat(tot), torch.cat(lms), torch.cat(cons)

    def grads_at(self, ids, which):
        """d/d(inputs_embeds) of the LM term ('lm') or the classifier term ('cons'),
        as an (L, D) tensor. One backward pass."""
        emb = self.model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
        out = self.model(inputs_embeds=emb, output_hidden_states=True, return_dict=True)
        self.n_forward += 1
        if which == "lm":
            lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            obj = lp.gather(1, ids[0, 1:].unsqueeze(1)).sum()
        else:
            obj = torch.log_softmax(self.head(out.hidden_states[-1]), dim=-1)[0, self.a.target_label]
        g = torch.autograd.grad(obj, emb)[0][0]
        self.n_backward += 1
        return g.detach()

    # ---------------- proposal ----------------
    def _mlm_logits(self, ids, pos):
        from diagnostics.run_mlm_control import build_bridge
        if self.mlm is None:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self.rob_tok = AutoTokenizer.from_pretrained(self.a.mlm_path)
            self.mlm = AutoModelForMaskedLM.from_pretrained(
                self.a.mlm_path, torch_dtype=torch.float32).to(self.device).eval()
            self.rob_idx, self.gpt_idx = build_bridge(self.rob_tok, self.tok, self.device)
        surface = self.tok.decode(ids[0].tolist())
        prefix = self.tok.decode(ids[0, :pos].tolist())
        r_ids = self.rob_tok(surface, return_tensors="pt", truncation=True,
                             max_length=512).input_ids.to(self.device)
        r_prefix = self.rob_tok(prefix, add_special_tokens=False).input_ids
        r_pos = min(len(r_prefix) + 1, r_ids.shape[1] - 2)
        masked = r_ids.clone()
        masked[0, r_pos] = self.rob_tok.mask_token_id
        with torch.no_grad():
            lg = self.mlm(masked).logits[0, r_pos].float()
            self.n_forward += 1
        # scatter the bridged subset back onto the GPT-2 vocabulary
        full = torch.full((self.V,), -float("inf"), device=self.device)
        full[self.gpt_idx] = torch.log_softmax(lg[self.rob_idx], dim=-1)
        return full

    def _sedd_logits(self, ids, pos):
        import diagnostics.sedd_lib as sl
        if self.sedd is None:
            self.sedd = sl.load_sedd(self.a.sedd_scale, device=self.device)
        pref = sl.logpref_at(self.sedd, ids, pos, self.a.sedd_sigma)
        full = torch.full((self.V,), -float("inf"), device=self.device)
        n = min(self.V, pref.numel())
        full[:n] = pref[:n].float()
        return full

    def shortlist(self, self_lp_i, incumbent, cons_pref=None):
        """The frozen candidate support at this position. Always contains the incumbent
        (without it the reverse move has probability zero and MH rejects everything)."""
        k = self.a.topk
        if self.a.constraint_mech == "mix_shortlist" and cons_pref is not None:
            half = max(k // 2, 1)
            a = torch.topk(self_lp_i, half).indices
            b = torch.topk(cons_pref, k - half).indices
            idx = torch.unique(torch.cat([a, b]))
        else:
            idx = torch.topk(self_lp_i, k).indices
        return torch.unique(torch.cat([idx, incumbent.view(1)]))

    def proposal_logits(self, ids, pos, cand, self_lp_i, g_lm, g_cons, cur_U):
        """Score every candidate in `cand`. Returns (scores, diagnostics)."""
        e_cand = self.emb[cand]
        e_cur = self.emb[ids[0, pos]]
        d = e_cand - e_cur.unsqueeze(0)
        p = self.a.proposal
        diag = {}

        if p == "uniform":
            score = torch.zeros(cand.numel(), device=self.device)
        elif p == "self":
            score = 0.5 * self_lp_i[cand]
        elif p == "embgrad":
            score = 0.5 * (d @ g_lm[pos].float())
        elif p == "onehot":
            score = 0.5 * self_lp_i[cand] + 0.5 * (d @ g_lm[pos].float())
        elif p == "exact_k":
            batch = ids.repeat(cand.numel(), 1)
            batch[:, pos] = cand
            tot, lm_b, c_b = self.energy_batch(batch)
            score = 0.5 * (tot - cur_U)
            diag["exact_lm"] = lm_b
            diag["exact_cons"] = c_b
            diag["exact_total"] = tot
        elif p == "roberta":
            score = 0.5 * self._mlm_logits(ids, pos)[cand]
        elif p == "sedd":
            score = 0.5 * self._sedd_logits(ids, pos)[cand]
        else:
            raise ValueError(p)

        # ---- the constraint mechanism acts here (Study C) ----
        mech = self.a.constraint_mech
        if mech in ("grad_prop", "both") and g_cons is not None:
            score = score + self.a.cons_gain * 0.5 * (d @ g_cons[pos].float())
        if mech in ("rescore_k", "both") and self.head is not None:
            if "exact_cons" in diag:
                c_b = diag["exact_cons"]
            else:
                batch = ids.repeat(cand.numel(), 1)
                batch[:, pos] = cand
                _, _, c_b = self.energy_batch(batch)
                diag["exact_cons"] = c_b
            score = score + self.a.cons_gain * c_b
        return score, diag


def run_sequence(chain, ids, span_idx, args, rng, recorder):
    """One chain over one sequence. Returns the final ids and per-sequence stats."""
    dev = chain.device
    cur = ids.clone()
    cur_U, cur_lm, cur_cons, cur_lp = chain.energy(cur, want_self=True)
    accepts, steps = 0, 0
    cov_hits, cov_ranks = 0, []
    lin_true, lin_pred = [], []

    for t in range(args.steps):
        pos = int(span_idx[rng.randint(len(span_idx))])
        self_lp_i = cur_lp[pos - 1].float()          # p(. | x_<pos)

        g_lm = g_cons = None
        if args.proposal in ("onehot", "embgrad"):
            g_lm = chain.grads_at(cur, "lm")
        if args.constraint_mech in ("grad_prop", "both", "mix_shortlist") and chain.head is not None:
            g_cons = chain.grads_at(cur, "cons")

        cons_pref = None
        if g_cons is not None:
            # the classifier's own first-order preference over the whole vocabulary
            cons_pref = (chain.emb - chain.emb[cur[0, pos]].unsqueeze(0)) @ g_cons[pos].float()

        cand = chain.shortlist(self_lp_i, cur[0, pos], cons_pref)
        score, diag = chain.proposal_logits(cur, pos, cand, self_lp_i, g_lm, g_cons, cur_U)
        logq = torch.log_softmax(score / args.temperature, dim=-1)

        # ---- SHORTLIST COVERAGE, recorded before anything is accepted ----
        if chain.head is not None and recorder is not None and t % args.coverage_every == 0:
            # which candidate does the constraint most prefer, over the FULL vocab, and
            # where does it fall in the likelihood ranking? grads_at must run OUTSIDE
            # any no_grad context, or autograd has no graph to differentiate.
            cp = cons_pref
            if cp is None:
                gc = chain.grads_at(cur, "cons")
                cp = (chain.emb - chain.emb[cur[0, pos]].unsqueeze(0)) @ gc[pos].float()
            with torch.no_grad():
                best = int(cp.argmax().item())
                rank = int((self_lp_i > self_lp_i[best]).sum().item())
                cov_ranks.append(rank)
                cov_hits += int(bool((cand == best).any().item()))

        # ---- constraint-side linearization check, on the same candidates ----
        if (recorder is not None and chain.head is not None
                and args.lin_check and t % args.lin_check == 0):
            gc = g_cons if g_cons is not None else chain.grads_at(cur, "cons")
            with torch.no_grad():
                sub = cand[:min(args.lin_n, cand.numel())]
                batch = cur.repeat(sub.numel(), 1)
                batch[:, pos] = sub
                _, _, c_true = chain.energy_batch(batch)
                dd = chain.emb[sub] - chain.emb[cur[0, pos]].unsqueeze(0)
                lin_pred.extend((dd @ gc[pos].float()).tolist())
                lin_true.extend((c_true - cur_cons).tolist())

        j = int(torch.multinomial(logq.exp(), 1).item())
        prop_tok = int(cand[j].item())
        steps += 1
        if prop_tok == int(cur[0, pos].item()):
            continue

        prop = cur.clone()
        prop[0, pos] = prop_tok
        prop_U, prop_lm, prop_cons, prop_lp = chain.energy(prop, want_self=True)

        # reverse proposal, same rule evaluated at the proposed state
        self_lp_b = prop_lp[pos - 1].float()
        g_lm_b = g_cons_b = None
        if args.proposal in ("onehot", "embgrad"):
            g_lm_b = chain.grads_at(prop, "lm")
        if args.constraint_mech in ("grad_prop", "both", "mix_shortlist") and chain.head is not None:
            g_cons_b = chain.grads_at(prop, "cons")
        cons_pref_b = None
        if g_cons_b is not None:
            cons_pref_b = (chain.emb - chain.emb[prop[0, pos]].unsqueeze(0)) @ g_cons_b[pos].float()
        cand_b = chain.shortlist(self_lp_b, prop[0, pos], cons_pref_b)
        # The incumbent must be reachable from the proposed state, or the move is not
        # reversible and MH is obliged to reject it. Union guarantees it and keeps the
        # two supports identical, which is what makes the ratio below exact.
        cand_b = torch.unique(torch.cat([cand_b, cand]))
        score_b, _ = chain.proposal_logits(prop, pos, cand_b, self_lp_b, g_lm_b,
                                            g_cons_b, prop_U)
        logq_b = torch.log_softmax(score_b / args.temperature, dim=-1)

        lf = float(logq[j])
        back = (cand_b == int(cur[0, pos].item())).nonzero()
        lb = float(logq_b[back[0, 0]]) if back.numel() else -float("inf")

        log_alpha = float(prop_U - cur_U) + (lb - lf)
        if float(np.log(max(float(torch.rand(1)), 1e-30))) < log_alpha:
            cur, cur_U, cur_lm, cur_cons, cur_lp = prop, prop_U, prop_lm, prop_cons, prop_lp
            accepts += 1

    return cur, dict(
        accepts=accepts, steps=steps,
        final_lm=float(cur_lm), final_cons=float(cur_cons) if cur_cons is not None else None,
        coverage_hits=cov_hits, coverage_n=len(cov_ranks),
        coverage_rank_median=float(np.median(cov_ranks)) if cov_ranks else float("nan"),
        lin_true=lin_true, lin_pred=lin_pred,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--head", default=None, help="sentiment head .pt; omit for an unconstrained chain")
    p.add_argument("--proposal", choices=PROPOSALS, required=True)
    p.add_argument("--constraint_mech", choices=MECHS, default="accept_only")
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--beta_lm", type=float, default=0.8)
    p.add_argument("--beta_c", type=float, default=0.2)
    p.add_argument("--cons_gain", type=float, default=1.0,
                   help="weight on the constraint INSIDE the proposal. Swept in Study D.")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--span_len", type=int, default=20)
    p.add_argument("--samples_per_prompt", type=int, default=4)
    p.add_argument("--n_per_prompt_out", type=int, default=1,
                   help="Study E: independent chains per (prompt, sample) to keep")
    p.add_argument("--exact_batch", type=int, default=32)
    p.add_argument("--coverage_every", type=int, default=10)
    p.add_argument("--lin_check", type=int, default=0,
                   help="record the constraint linearization check every N steps; 0 = off")
    p.add_argument("--lin_n", type=int, default=32)
    p.add_argument("--mlm_path", default="roberta-large")
    p.add_argument("--sedd_scale", default="small")
    p.add_argument("--sedd_sigma", type=float, default=0.5)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results/ctg/constrained")
    args = p.parse_args()

    run_name = args.run_name
    if args.num_shards > 1:
        run_name = f"{run_name}.shard{args.shard_idx}of{args.num_shards}"
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, run_name + ".json")
    if os.path.exists(out_path):
        print(f"[{run_name}] already done; skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(1234)
    t0 = time.time()
    print(f"[{run_name}] loading {args.model_path}", flush=True)
    tok, model = load_tokenizer_and_model(args.model_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    head = load_sentiment_head(args.head, device) if args.head else None

    chain = CtgChain(model, tok, head, args, device)

    # resolve the sample plan FIRST, exactly as an unsharded run would, then partition.
    # The global index is kept so shards and arms pair on the same sequences.
    plan = []
    for pi, prompt in enumerate(MUCOLA_PROMPTS):
        for k in range(args.samples_per_prompt):
            plan.append((len(plan), pi, prompt, args.data_seed + 1000 * pi + k))
    if args.num_shards > 1:
        plan = [q for q in plan if q[0] % args.num_shards == args.shard_idx]
    print(f"[{run_name}] {len(plan)} sequences, proposal={args.proposal} "
          f"mech={args.constraint_mech} k={args.topk}", flush=True)

    rows, stats = [], []
    lin_true_all, lin_pred_all = [], []
    for gi, pi, prompt, seed in plan:
        seed_all(seed)
        rng = np.random.RandomState(seed)
        ids, span_idx, _ = build_continuation_case(tok, prompt, args.span_len, seed,
                                                   device, tok.vocab_size)
        start_txt = tok.decode(ids[0], skip_special_tokens=True)
        final, st = run_sequence(chain, ids, span_idx, args, rng, recorder=True)
        txt = tok.decode(final[0], skip_special_tokens=True)
        rows.append(dict(global_idx=gi, prompt_idx=pi, prompt=prompt, seed=seed,
                         start_text=start_txt, text=txt,
                         final_lm=st["final_lm"], final_cons=st["final_cons"],
                         accepts=st["accepts"], steps=st["steps"]))
        stats.append(st)
        lin_true_all.extend(st["lin_true"])
        lin_pred_all.extend(st["lin_pred"])
        if len(rows) % 5 == 0:
            print(f"[{run_name}] {len(rows)}/{len(plan)}  "
                  f"accept={100.0*sum(s['accepts'] for s in stats)/max(sum(s['steps'] for s in stats),1):.1f}%",
                  flush=True)

    csv_path = os.path.join(args.out_dir, run_name + ".csv")
    with open(csv_path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(csv_path + ".tmp", csv_path)

    tot_steps = sum(s["steps"] for s in stats)
    tot_acc = sum(s["accepts"] for s in stats)
    cov_n = sum(s["coverage_n"] for s in stats)
    cov_h = sum(s["coverage_hits"] for s in stats)
    wall = time.time() - t0
    res = dict(
        experiment="ctg_constrained_chain", run_name=run_name, config=vars(args),
        n=len(rows),
        accept_rate_pct=100.0 * tot_acc / max(tot_steps, 1),
        mean_final_lm=float(np.mean([s["final_lm"] for s in stats])),
        mean_final_cons=(float(np.mean([s["final_cons"] for s in stats]))
                         if stats and stats[0]["final_cons"] is not None else None),
        steer_head_acc_pct=(100.0 * float(np.mean(
            [np.exp(s["final_cons"]) > 0.5 for s in stats]))
            if stats and stats[0]["final_cons"] is not None else None),
        # coverage FIRST: a weak mechanism with poor coverage is a support limitation
        shortlist_coverage_pct=100.0 * cov_h / max(cov_n, 1),
        shortlist_coverage_n=cov_n,
        shortlist_rank_median=float(np.median(
            [s["coverage_rank_median"] for s in stats
             if np.isfinite(s["coverage_rank_median"])])) if stats else float("nan"),
        constraint_linearization_spearman=spearman(lin_pred_all, lin_true_all),
        constraint_linearization_n=len(lin_true_all),
        cost=dict(n_forward=chain.n_forward, n_backward=chain.n_backward,
                  n_candidate_evals=chain.n_candidate_evals,
                  wall_time_sec=wall, n_sequences=len(rows),
                  n_mh_steps=tot_steps, n_accepted_moves=tot_acc,
                  sec_per_accepted_move=wall / max(tot_acc, 1),
                  forward_equivalents_per_sequence=
                      (chain.n_forward + 2.0 * chain.n_backward) / max(len(rows), 1)),
        csv=csv_path,
    )
    atomic_json(out_path, res)
    print(f"[{run_name}] DONE n={res['n']} accept={res['accept_rate_pct']:.1f}% "
          f"steer_head={res['steer_head_acc_pct']} "
          f"coverage={res['shortlist_coverage_pct']:.1f}% wall={wall/60:.1f}m", flush=True)
    print(f"[{run_name}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
