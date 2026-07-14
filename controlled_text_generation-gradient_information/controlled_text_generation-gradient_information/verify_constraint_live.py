#!/usr/bin/env python3
"""
verify_constraint_live.py

Run this BEFORE any constrained sweep. It catches the exact bug that silently
invalidated our first probe run: `full` producing bitwise-identical results to
`lm_only`, because gradient normalization discarded the constraint's contribution.

It checks three things on a single real case:
  1. the constraint gradient w.r.t. s is non-zero (the autograd path is connected)
  2. grad(full) differs from grad(lm_only) by a non-trivial angle
  3. the classifier's log-prob actually responds to changes in s

If (2) shows an angle of ~0 degrees, the constraint is a no-op and the whole
ablation is meaningless. Do not run the sweep until this passes.

Usage:
  python verify_constraint_live.py --model_path <gpt2_sft> --head sentiment_head.pt
"""

import argparse
import numpy as np
import torch

from core.prep import load_tokenizer_and_model, joint_log_prob_from_inputs_embeds
from core.constraint import load_sentiment_head, constraint_log_prob

SENT = "The film was shown in theaters across the country during the summer season."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--beta_lm", type=float, default=0.8)
    ap.add_argument("--beta_c", type=float, default=0.2)
    ap.add_argument("--target_label", type=int, default=1)
    args = ap.parse_args()

    tok, model = load_tokenizer_and_model(args.model_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    device = next(model.parameters()).device
    head = load_sentiment_head(args.head, device)

    ids = tok(SENT, return_tensors="pt").input_ids.to(device)
    mask_idx = torch.tensor([3, 5, 7], device=device)
    base = model.get_input_embeddings()(ids).detach()
    s = base[0, mask_idx, :].clone().detach().requires_grad_(True)

    inputs_embeds = base.clone().detach()
    inputs_embeds[0, mask_idx, :] = s
    target_ids = ids.clone()

    lm = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids)
    c = constraint_log_prob(model, head, inputs_embeds, args.target_label)

    g_lm = torch.autograd.grad(lm, s, retain_graph=True)[0]
    g_c = torch.autograd.grad(c, s, retain_graph=True)[0]

    print("=" * 66)
    print("1. GRADIENT PATH")
    print(f"   ||grad LM||         = {g_lm.norm().item():.6e}")
    print(f"   ||grad constraint|| = {g_c.norm().item():.6e}")
    if g_c.norm().item() < 1e-10:
        print("   FAIL: constraint gradient is zero. The autograd path is broken.")
        return
    ratio = g_c.norm().item() / max(g_lm.norm().item(), 1e-12)
    print(f"   ratio ||g_c|| / ||g_lm|| = {ratio:.4f}")
    print("   PASS: constraint gradient reaches s.")

    print("\n2. DOES THE CONSTRAINT CHANGE THE PROPOSAL DIRECTION?")
    g_full = args.beta_lm * g_lm + args.beta_c * g_c
    g_lmonly = args.beta_lm * g_lm

    def unit(v):
        return v / (v.norm() + 1e-12)

    cos = torch.dot(unit(g_full).flatten(), unit(g_lmonly).flatten()).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cos)).item()
    print(f"   angle(grad_full, grad_lm_only) = {angle:.4f} degrees")

    # what the sampler ACTUALLY uses if grad_normalization is on
    print(f"   after unit-normalization, max |elementwise diff| = "
          f"{(unit(g_full) - unit(g_lmonly)).abs().max().item():.3e}")

    if angle < 0.01:
        print("   FAIL: the constraint does not measurably rotate the gradient.")
        print("   `full` will behave identically to `lm_only`. Increase beta_c or")
        print("   check that grad_normalization is OFF.")
        return
    print("   PASS: the combined gradient points somewhere different.")

    print("\n3. IS THE CLASSIFIER RESPONSIVE TO s?")
    with torch.no_grad():
        p0 = torch.softmax(
            head(model(inputs_embeds=inputs_embeds, output_hidden_states=True,
                       return_dict=True).hidden_states[-1]), dim=-1)[0, args.target_label].item()
        bumped = inputs_embeds.clone()
        bumped[0, mask_idx, :] += 0.5 * unit(g_c)
        p1 = torch.softmax(
            head(model(inputs_embeds=bumped, output_hidden_states=True,
                       return_dict=True).hidden_states[-1]), dim=-1)[0, args.target_label].item()
    print(f"   p(target) before = {p0:.4f}")
    print(f"   p(target) after a small step ALONG the constraint gradient = {p1:.4f}")
    print(f"   delta = {p1 - p0:+.4f}  {'(rises, as it should)' if p1 > p0 else '(DOES NOT RISE)'}")

    print("\n" + "=" * 66)
    if angle > 0.01 and p1 > p0:
        print("ALL CHECKS PASSED. The constraint is live. Safe to run the sweep.")
    else:
        print("CHECKS FAILED. Do not run the sweep.")
    print("=" * 66)


if __name__ == "__main__":
    main()
