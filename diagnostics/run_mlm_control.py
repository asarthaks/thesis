"""Bidirectional masked-LM control: separates bidirectional conditioning from score training.

The diffusion positive control shows that swapping the proposal in the exact-energy chain
from the autoregressive gradient to the SEDD concrete score lifts exact recovery from 0 to
39 percent. But SEDD differs from GPT-2 in objective, scale, corpus AND conditioning
direction, so that experiment cannot say which of them is operative.

A bidirectional masked language model is the discriminating control: RoBERTa is NOT
score-trained, but it IS bidirectional and it answers exactly the query the revision
operation needs, p(v | left context, right context). If an MLM proposal also works in this
chain, the operative variable is bidirectional conditioning rather than score training.

Everything except the proposal is held fixed against the autoregressive experiments: the same
WikiText-2 sequences, the same deterministic per-index corruption, the same exact GPT-2
energy, the same Metropolis-Hastings accept/reject, the same KL metric.

The proposal is an INDEPENDENCE sampler: q(v) is the MLM conditional at the masked position
and does not depend on the current token, so the Hastings ratio is exactly
q(x_cur) / q(x_prop) and is computed in closed form.

Tokenizer bridge. RoBERTa and GPT-2 both use byte-level BPE but their vocabularies are
indexed differently. A once-off map is built from RoBERTa ids to GPT-2 ids, keeping only
RoBERTa tokens whose decoded surface string re-encodes to exactly one GPT-2 token. The
proposal is restricted to that subset and renormalized; the coverage is reported so the
restriction is auditable rather than hidden. Ground-truth tokens outside the map are counted
separately, since they are unreachable by construction and would otherwise silently deflate
the recovery rate.

    python diagnostics/run_mlm_control.py --gpt2_path $GPT2SFT --run_name rev_mlm_control \
        --out_dir results/revision --n_samples 200 --steps 50
"""
import argparse
import json
import os
import time

import numpy as np
import torch

sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path_root not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_root)
from scripts.run_experiment import build_corruption, load_texts as _grid_load_texts


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_bridge(rob_tok, gpt_tok, device):
    """roberta_id -> gpt2_id for tokens that map to exactly one GPT-2 token."""
    rob_ids, gpt_ids = [], []
    for rid in range(rob_tok.vocab_size):
        s = rob_tok.convert_tokens_to_string([rob_tok.convert_ids_to_tokens(rid)])
        if not s:
            continue
        enc = gpt_tok.encode(s)
        if len(enc) == 1:
            rob_ids.append(rid)
            gpt_ids.append(enc[0])
    return (torch.tensor(rob_ids, device=device),
            torch.tensor(gpt_ids, device=device))


def gpt2_joint_logprob(model, ids):
    """log p(x) under the frozen autoregressive model. The same energy the grid uses."""
    with torch.no_grad():
        out = model(ids)
        lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
        return lp.gather(1, ids[0, 1:].unsqueeze(1)).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt2_path", required=True)
    p.add_argument("--mlm_path", default="roberta-large")
    p.add_argument("--run_name", default="rev_mlm_control")
    p.add_argument("--out_dir", default="results/revision")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=40)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--uniform_proposal", action="store_true",
                   help="PIPELINE CONTROL: replace the MLM conditional with a uniform draw "
                        "over the same bridged vocabulary. Everything else is identical, so "
                        "a high recovery rate here would indicate leakage rather than signal.")
    p.add_argument("--top_k", type=int, default=0,
                   help="restrict the MLM proposal to its top-k; 0 uses the full bridged vocabulary")
    args = p.parse_args()

    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForMaskedLM)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.run_name + ".json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(1234)
    t0 = time.time()

    print(f"[{args.run_name}] loading GPT-2 energy {args.gpt2_path}", flush=True)
    gpt_tok = AutoTokenizer.from_pretrained(args.gpt2_path)
    gpt = AutoModelForCausalLM.from_pretrained(
        args.gpt2_path, torch_dtype=torch.float32).to(device).eval()
    print(f"[{args.run_name}] loading MLM proposal {args.mlm_path}", flush=True)
    rob_tok = AutoTokenizer.from_pretrained(args.mlm_path)
    rob = AutoModelForMaskedLM.from_pretrained(
        args.mlm_path, torch_dtype=torch.float32).to(device).eval()

    rob_idx, gpt_idx = build_bridge(rob_tok, gpt_tok, device)
    print(f"[{args.run_name}] tokenizer bridge: {len(rob_idx)} of {rob_tok.vocab_size} "
          f"RoBERTa tokens map to a single GPT-2 token "
          f"({100.0 * len(rob_idx) / rob_tok.vocab_size:.1f}%)", flush=True)

    class _A:
        data_file = None
        min_words = args.min_words
        max_words = args.max_words
    texts = _grid_load_texts(_A())
    print(f"[{args.run_name}] {len(texts)} candidate sentences", flush=True)

    done = 0
    exact, ever, unreachable = 0, 0, 0
    kl_last, kl_chain, accepts, steps_taken = [], [], 0, 0

    for ti, text in enumerate(texts):
        if done >= args.n_samples:
            break
        # The grid's own corruption routine, imported rather than reimplemented, so the
        # masked position and the corrupting token are bit-identical to the Langevin runs
        # and the recovery rates are directly comparable.
        case = build_corruption(gpt_tok, text, 1, args.data_seed + ti, device)
        if case is None:
            continue
        cur, mask_indices, ids = case
        pos = mask_indices[0]
        gt = int(ids[0, pos].item())
        if pos >= ids.shape[1] - 1:
            continue
        with torch.no_grad():
            p_ref = torch.softmax(gpt(ids).logits[0, pos].float(), dim=-1)

        def kl_now(state):
            with torch.no_grad():
                lp = torch.log_softmax(gpt(state).logits[0, pos].float(), dim=-1)
                return float(torch.nn.functional.kl_div(
                    lp.unsqueeze(0), p_ref.unsqueeze(0),
                    reduction="batchmean", log_target=False))

        # MLM proposal at this position, computed once: it is an independence sampler.
        surface = gpt_tok.decode(cur[0].tolist())
        r_ids = rob_tok(surface, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)
        # locate the RoBERTa token to mask by aligning on the decoded prefix length
        prefix = gpt_tok.decode(cur[0, :pos].tolist())
        r_prefix = rob_tok(prefix, add_special_tokens=False).input_ids
        r_pos = min(len(r_prefix) + 1, r_ids.shape[1] - 2)   # +1 for <s>
        masked = r_ids.clone()
        masked[0, r_pos] = rob_tok.mask_token_id
        with torch.no_grad():
            logits = rob(masked).logits[0, r_pos].float()
        q_full = torch.log_softmax(logits[rob_idx], dim=-1)      # over bridged subset
        if args.uniform_proposal:
            q_full = torch.full_like(q_full, -float(np.log(q_full.numel())))
        if args.top_k:
            kth = torch.topk(q_full, min(args.top_k, q_full.numel())).values[-1]
            q_full = torch.where(q_full >= kth, q_full,
                                 torch.full_like(q_full, -float("inf")))
            q_full = torch.log_softmax(q_full, dim=-1)
        q_prob = q_full.exp()

        gt_reachable = bool((gpt_idx == gt).any().item())
        if not gt_reachable:
            unreachable += 1

        cur_lp = gpt2_joint_logprob(gpt, cur)
        visited_gt = False
        kls = []
        for _ in range(args.steps):
            j = int(torch.multinomial(q_prob, 1).item())
            prop_tok = int(gpt_idx[j].item())
            prop = cur.clone()
            prop[0, pos] = prop_tok
            prop_lp = gpt2_joint_logprob(gpt, prop)

            # Independence-sampler Hastings ratio: q(current) / q(proposed).
            cur_slot = (gpt_idx == cur[0, pos]).nonzero()
            log_q_cur = float(q_full[cur_slot[0, 0]]) if cur_slot.numel() else -60.0
            log_q_prop = float(q_full[j])
            log_alpha = float(prop_lp - cur_lp) + (log_q_cur - log_q_prop)
            steps_taken += 1
            if float(torch.rand(1)) < min(1.0, float(np.exp(min(0.0, log_alpha)))):
                cur, cur_lp = prop, prop_lp
                accepts += 1
            if int(cur[0, pos].item()) == gt:
                visited_gt = True
            kls.append(kl_now(cur))

        exact += int(int(cur[0, pos].item()) == gt)
        ever += int(visited_gt)
        kl_last.append(kls[-1])
        kl_chain.append(float(np.mean(kls[len(kls) // 2:])))
        done += 1
        if done % 25 == 0:
            print(f"[{args.run_name}] {done}/{args.n_samples}  "
                  f"exact={100.0 * exact / done:.1f}%", flush=True)

    res = {
        "experiment": "bidirectional_mlm_control",
        "run_name": args.run_name,
        "mlm": args.mlm_path,
        "uniform_proposal_control": bool(args.uniform_proposal),
        "energy": args.gpt2_path,
        "n": done,
        "steps": args.steps,
        "top_k": args.top_k,
        "bridge_tokens": int(len(rob_idx)),
        "bridge_coverage_pct": float(100.0 * len(rob_idx) / rob_tok.vocab_size),
        "gt_unreachable_n": unreachable,
        "exact_match_pct": 100.0 * exact / max(done, 1),
        "ever_visited_pct": 100.0 * ever / max(done, 1),
        "exact_match_pct_reachable_only":
            100.0 * exact / max(done - unreachable, 1),
        "mean_kl_last": float(np.mean(kl_last)) if kl_last else float("nan"),
        "mean_kl_chain_second_half": float(np.mean(kl_chain)) if kl_chain else float("nan"),
        "accept_rate_pct": 100.0 * accepts / max(steps_taken, 1),
        "wall_time_sec": time.time() - t0,
        "argv": vars(args),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, out_path)
    print(f"[{args.run_name}] DONE n={done} exact={res['exact_match_pct']:.1f}% "
          f"ever={res['ever_visited_pct']:.1f}% kl_last={res['mean_kl_last']:.3f} "
          f"accept={res['accept_rate_pct']:.1f}%", flush=True)
    print(f"[{args.run_name}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
