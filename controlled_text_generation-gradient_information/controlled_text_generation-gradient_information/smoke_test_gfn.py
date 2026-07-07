#!/usr/bin/env python3
"""
smoke_test_gfn.py

Ten-minute de-risking check before you spend GPU time running the Langevin matrix
on a GFlowNet checkpoint. It verifies three things, in order of how likely each is
to bite:

  1. the LoRA adapter loads and merges onto the SFT base without erroring,
  2. the merged GFlowNet model actually differs from the plain SFT model
     (i.e. the adapter is really in effect, not silently dropped),
  3. a real DLS run produces finite, changing metrics and finite gradients through
     the merged model.

If all three pass, the full experiment is cheap and safe to launch. If (2) says the
log-probs are identical, the adapter did not apply and you must fix loading before
running anything, otherwise you would just be re-running the SFT experiment under a
GFlowNet label.

Usage:
  python smoke_test_gfn.py \
    --base /path/to/gpt2_large_sft_output \
    --adapter /path/to/gfn_lb0_2000_checkpoint \
    --eps_start 10.5 --eps_end 0.1
"""

import argparse
import numpy as np
import torch

from core.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft
from core.dls import DiscreteLangevinSampler


SENTENCE = "She poured a cup of coffee and stared out the window at the rain."


def seq_logprob(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids)
        logp = torch.log_softmax(out.logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        ll = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return ll


def build_case(tokenizer, text, device, seed=0):
    rng = np.random.RandomState(seed)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    L = ids.shape[1]
    idx = int(rng.choice(range(1, L - 1)))
    orig = ids.clone()
    corrupted = ids.clone()
    r = int(rng.randint(0, tokenizer.vocab_size))
    while r == ids[0, idx].item():
        r = int(rng.randint(0, tokenizer.vocab_size))
    corrupted[0, idx] = r
    return corrupted, [idx], orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT base model dir (GPT-2 Large on ROCStories)")
    ap.add_argument("--adapter", required=True, help="GFlowNet LoRA adapter dir")
    ap.add_argument("--eps_start", type=float, default=10.5)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/3] loading base SFT and GFlowNet-merged models ...")
    tok, sft = load_tokenizer_and_model(args.base, dtype=dtype)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _, gfn = load_tokenizer_and_model_peft(args.base, args.adapter, dtype=dtype)
    print("      both models loaded OK")

    print("\n[2/3] checking the adapter actually changed the model ...")
    ll_sft = seq_logprob(sft, tok, SENTENCE, device)
    ll_gfn = seq_logprob(gfn, tok, SENTENCE, device)
    print(f"      log p(sentence)  SFT={ll_sft:.3f}   GFN={ll_gfn:.3f}   diff={abs(ll_sft - ll_gfn):.3f}")
    if abs(ll_sft - ll_gfn) < 1e-3:
        print("      FAIL: log-probs are identical. The adapter did NOT apply.")
        print("      Do not run the matrix until this is fixed (check adapter path / peft version).")
        return
    print("      PASS: GFlowNet model differs from SFT, adapter is in effect.")

    print("\n[3/3] running a short DLS trajectory through the merged GFN model ...")
    del sft
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gfn.eval()
    sampler = DiscreteLangevinSampler(
        model=gfn, tokenizer=tok, steps=args.steps, temperature=5.0, oracle=False,
        method="policy", mh_sampling=True, grad_normalization=True,
        epsilon_schedule=np.linspace(args.eps_start, args.eps_end, args.steps),
    )
    corrupted, mask_indices, orig = build_case(tok, SENTENCE, device)
    torch.manual_seed(0)
    _, metrics = sampler.optimize(corrupted.clone(), mask_indices, orig.clone())

    l2 = [m["avg_l2_distance"] for m in metrics]
    kl = [m["avg_kl_divergence"] for m in metrics]
    finite = all(np.isfinite(x) for x in l2 + kl)
    moved = abs(l2[0] - l2[-1]) > 1e-6 or abs(kl[0] - kl[-1]) > 1e-6
    gt = orig[0, mask_indices].tolist()
    rec = metrics[-1]["token_ids"]
    print(f"      steps run: {len(metrics)}   L2 {l2[0]:.3f} -> {l2[-1]:.3f}   KL {kl[0]:.3f} -> {kl[-1]:.3f}")
    print(f"      corrupted token -> recovered '{tok.decode(rec)}'  (gt '{tok.decode(gt)}')")
    print(f"      metrics finite: {finite}   trajectory moved: {moved}")

    if finite and moved:
        print("\nSMOKE TEST PASSED. Safe to launch the GFlowNet matrix on the idle GPUs.")
    else:
        print("\nSMOKE TEST INCONCLUSIVE: metrics were non-finite or flat. Inspect before scaling up.")


if __name__ == "__main__":
    main()
