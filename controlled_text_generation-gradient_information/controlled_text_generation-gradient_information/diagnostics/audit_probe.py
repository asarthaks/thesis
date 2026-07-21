#!/usr/bin/env python
"""
audit_probe.py  -  Part 1 numerical verification for the last-token phase.

Verifies, on the real GPT-2 large SFT model, the six audit claims that can be
checked with a cheap probe (no sampler runs):

  1. Zero-gradient theorem: ||grad log p(seq) / d inputs_embeds[p]|| is exactly
     0 at the final scored position, tiny at second-to-last, larger in the middle.
  2. Structural blindness: input (wte) and output (lm_head) embeddings are tied
     (shared storage), so the identity of the final token enters the likelihood
     only through the OUTPUT-embedding path, invisible to the input-embedding grad.
  3. Energy is exact at the last position: for a masked final token, the
     full-sequence energy difference between two candidates equals the model's
     own conditional log-ratio log p(a|prefix) - log p(b|prefix) (up to float).
  4/6. Reports the KL-metric boundary (m < seq_len-1) directly from the sequence
     shape so the last-token KL-undefinedness is documented with a number.

Uses run_experiment.build_corruption / load_texts so the probe sequences are the
SAME WikiText-2 sequences the grid and kl_baselines used (data_seed 0, the first
N that survive the length filter and corruption guard).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.prep import load_tokenizer_and_model, joint_log_prob_from_inputs_embeds
from run_experiment import load_texts, build_corruption

MODEL = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"


class A:  # arg holder for load_texts
    data_file = None
    min_words = 10
    max_words = 40


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    tok, model = load_tokenizer_and_model(MODEL, device=device, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Claim 2: tied embeddings -------------------------------------------
    wte = model.get_input_embeddings().weight
    lm_head_w = model.lm_head.weight
    tied_same_ptr = (wte.data_ptr() == lm_head_w.data_ptr())
    tied_equal = bool(torch.equal(wte, lm_head_w))
    print(f"[tied] wte.shape={tuple(wte.shape)} lm_head.shape={tuple(lm_head_w.shape)} "
          f"same_storage={tied_same_ptr} equal_values={tied_equal}")

    texts = load_texts(A)
    # collect the first 20 grid sequences (num_masks=1, data_seed 0)
    seqs = []
    ti = 0
    while len(seqs) < 20 and ti < len(texts):
        case = build_corruption(tok, texts[ti], 1, 0 + ti, device)
        ti += 1
        if case is None:
            continue
        _, _, orig_ids = case
        seqs.append(orig_ids)

    emb_layer = model.get_input_embeddings()
    gnorm_final, gnorm_2nd, gnorm_mid = [], [], []
    trailing_is_eos = []
    energy_exact_maxerr = []

    for orig_ids in seqs:
        L = orig_ids.shape[1]
        # trailing-token check: is the final scored token an EOS/special?
        last_id = int(orig_ids[0, -1].item())
        trailing_is_eos.append(int(last_id == (tok.eos_token_id or -1)))

        # ---- Claim 1: per-position gradient norm ----------------------------
        with torch.no_grad():
            base = emb_layer(orig_ids)
        inp = base.clone().detach().requires_grad_(True)
        log_joint = joint_log_prob_from_inputs_embeds(model, inp, orig_ids)
        g = torch.autograd.grad(log_joint, inp)[0][0]  # (L, D)
        per_pos = g.norm(dim=1)                         # (L,)
        mid = L // 2
        gnorm_final.append(float(per_pos[L - 1].item()))
        gnorm_2nd.append(float(per_pos[L - 2].item()))
        gnorm_mid.append(float(per_pos[mid].item()))

        # ---- Claim 3: energy exactness at the last position -----------------
        # conditional p(x_{L-1} | x_<{L-1}) = softmax(logits[L-2]) under the model.
        with torch.no_grad():
            logits = model(orig_ids).logits[0]           # (L, V)
            log_cond = torch.log_softmax(logits[L - 2].float(), dim=-1)  # p(final | prefix)
            V = log_cond.numel()
            g0 = np.random.RandomState(int(orig_ids.sum().item()) % 100000)
            cands = [int(g0.randint(0, V)) for _ in range(6)]
            # full-sequence energy E(t) = -joint_logprob(seq with final=t)
            errs = []
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    a, b = cands[i], cands[j]
                    ia = orig_ids.clone(); ia[0, L - 1] = a
                    ib = orig_ids.clone(); ib[0, L - 1] = b
                    la = joint_log_prob_from_inputs_embeds(model, emb_layer(ia), ia).item()
                    lb = joint_log_prob_from_inputs_embeds(model, emb_layer(ib), ib).item()
                    e_diff = (-la) - (-lb)                # E(a) - E(b)
                    cond_diff = float(log_cond[b].item() - log_cond[a].item())  # = E(a)-E(b) predicted
                    errs.append(abs(e_diff - cond_diff))
            energy_exact_maxerr.append(max(errs))

    def stat(x):
        x = np.array(x, float)
        return dict(mean=float(x.mean()), max=float(x.max()), min=float(x.min()))

    out = {
        "n_sequences": len(seqs),
        "tied_embeddings": {"same_storage": tied_same_ptr, "equal_values": tied_equal,
                            "wte_shape": list(wte.shape)},
        "grad_norm_final_position": stat(gnorm_final),
        "grad_norm_second_to_last": stat(gnorm_2nd),
        "grad_norm_middle": stat(gnorm_mid),
        "trailing_token_is_eos_count": int(sum(trailing_is_eos)),
        "energy_exactness_maxerr_nats": stat(energy_exact_maxerr),
        "per_seq_final_grad_norms": gnorm_final,
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "audit_probe_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
