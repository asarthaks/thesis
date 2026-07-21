#!/usr/bin/env python
"""
run_sedd_guided.py  -  Part G step 2: classifier-guided SEDD generation.

The concern-11 continuation task (15 MuCoLa prompts, span 20), both target labels,
guided vs unguided SEDD-medium generation. The noisy classifier ONLY guides; the
steering gain is scored by the concern-11 judge (frozen-GPT-2 sentiment head via the
same classify() as run_constrained.py). Also reports gpt2sft NLL/token of the span as
the fluency readout, so steering and fluency cost appear together.

Guidance: absorbing-SEDD analytic denoising; at each step, for currently-MASK
positions, take the top-k SEDD candidates, score the k candidate states with the noisy
classifier in ONE batched pass, multiply the categorical weight by
p_clf(target|state)^gamma, renormalize, sample. One classifier pass per step.

Harness contract: --run_name, --out_dir, atomic JSON, per-item CSV, shardable over
cases. Not best-of-N: one generation per (prompt, sample, arm, label).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.prep import load_tokenizer_and_model
from core.constraint import load_sentiment_head
from diagnostics.run_revision import bootstrap_ci
import sedd_lib
from sedd_lib import MASK_TOKEN, REAL_VOCAB
from train_noisy_classifier import NoisyClassifier, EPS

MUCOLA_PROMPTS = [
    "Once upon a time", "The book", "The chicken", "The city", "The country",
    "The horse", "The lake", "The last time", "The movie", "The painting",
    "The pizza", "The potato", "The president of the country", "The road",
    "The year is 1910.",
]


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def classify_judge(gpt2sft, head, ids):
    """concern-11 judge: frozen-GPT-2 sentiment head, argmax label. Same as
    run_constrained.classify()."""
    emb = gpt2sft.get_input_embeddings()(ids)
    out = gpt2sft(inputs_embeds=emb, output_hidden_states=True, return_dict=True)
    return int(head(out.hidden_states[-1]).argmax(-1).item())


@torch.no_grad()
def span_nll_per_tok(gpt2sft, ids, span_locs):
    """mean NLL/token of the span positions under gpt2sft (fluency readout)."""
    out = gpt2sft(ids)
    lp = torch.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
    tgt = ids[0, 1:]
    tok_nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # nll of predicting position t+1
    idx = [s - 1 for s in span_locs if 1 <= s < ids.shape[1]]
    if not idx:
        return float("nan")
    return float(tok_nll[torch.tensor(idx, device=ids.device)].mean().item())


@torch.no_grad()
def guided_generate(sedd, clf, target_label, prompt_ids, span_len, steps, gamma,
                    topk, device, seed, guided, batch=1):
    """Batched absorbing-SEDD analytic denoising with optional classifier guidance.
    Generates `batch` distinct samples of the same prompt in one pass.
    Returns (batch, L) ids and the span position list."""
    from catsample import sample_categorical
    from model.utils import get_score_fn

    graph, noise = sedd.graph, sedd.noise
    score_fn = get_score_fn(sedd.model, train=False, sampling=True)

    Lp = prompt_ids.shape[1]
    L = Lp + span_len
    span_locs = list(range(Lp, L))
    prompt_locs = list(range(Lp))
    prompt_vals = prompt_ids[0].clone()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = graph.sample_limit(batch, L).to(device)
    x[:, prompt_locs] = prompt_vals
    eps_t = 1e-5
    timesteps = torch.linspace(1, eps_t, steps + 1, device=device)
    k = min(topk, REAL_VOCAB)

    def apply_guidance(probs, x_cur, mc):
        # probs: (B,L,V). Reweight every currently-MASK cell (across the batch) by the
        # classifier's p(target | candidate state) over the top-k SEDD candidates.
        masked = (x_cur == MASK_TOKEN)                     # (B,L)
        masked[:, prompt_locs] = False
        cells = torch.nonzero(masked, as_tuple=False)      # (Nc,2): (b,pos)
        if cells.shape[0] == 0:
            return probs
        cb, cp = cells[:, 0], cells[:, 1]
        Nc = cells.shape[0]
        real = probs[cb, cp, :REAL_VOCAB]                  # (Nc, V_real)
        topv, topi = real.topk(k, dim=-1)                  # (Nc,k)
        cand = x_cur[cb].unsqueeze(1).expand(Nc, k, L).clone()
        ar = torch.arange(Nc, device=device)
        cand[ar[:, None], torch.arange(k, device=device)[None, :], cp[:, None]] = topi
        nf = torch.full((Nc * k,), float(mc), device=device)
        logits = clf(cand.view(Nc * k, L), nf)             # (Nc*k, 2)
        p_t = torch.softmax(logits.float(), dim=-1)[:, target_label].view(Nc, k)
        real = real.scatter(-1, topi, topv * (p_t.clamp_min(1e-8) ** gamma))
        probs[cb, cp, :REAL_VOCAB] = real
        return probs

    for i in range(steps):
        x[:, prompt_locs] = prompt_vals
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        curr_sigma = noise(t)[0]
        next_sigma = noise(t - (1 - eps_t) / steps)[0]
        dsigma = curr_sigma - next_sigma
        score = score_fn(x, curr_sigma)
        stag = graph.staggered_score(score, dsigma)
        probs = stag * graph.transp_transition(x, dsigma)  # (B,L,V)
        if guided:
            probs = apply_guidance(probs, x, float((1 - EPS) * t[0].item()))
        x = sample_categorical(probs)

    # final denoise step (removes any remaining MASK); guide it too
    x[:, prompt_locs] = prompt_vals
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    sigma = noise(t)[0]
    score = score_fn(x, sigma)
    stag = graph.staggered_score(score, sigma)
    probs = stag * graph.transp_transition(x, sigma)
    if guided:
        probs = apply_guidance(probs, x, float((1 - EPS) * t[0].item()))
    if graph.absorb:
        probs = probs[..., :REAL_VOCAB]
    x = sample_categorical(probs)
    x[:, prompt_locs] = prompt_vals
    return x, span_locs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--scale", default="medium")
    ap.add_argument("--gpt2sft_path", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--classifier", default="results_revision/noisy_classifier.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--samples_per_prompt", type=int, default=20)
    ap.add_argument("--span_len", type=int, default=20)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} exists")
        return

    t0 = time.time()
    tok, gpt2sft = load_tokenizer_and_model(args.gpt2sft_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gpt2sft.eval()
    for p in gpt2sft.parameters():
        p.requires_grad_(False)
    device = args.device
    head = load_sentiment_head(args.head, device)
    sedd = sedd_lib.load_sedd(args.scale, device)

    ckpt = torch.load(args.classifier, map_location=device)
    clf = NoisyClassifier(d_emb=ckpt["d_emb"], vocab=ckpt["vocab"]).to(device)
    clf.load_state_dict(ckpt["state_dict"]); clf.eval()
    print(f"[guided] loaded gpt2sft + sedd-{args.scale} + judge + noisy clf in "
          f"{time.time()-t0:.1f}s", flush=True)

    # shard over PROMPTS; each prompt generates samples_per_prompt in one batched pass.
    prompts = [(pi, MUCOLA_PROMPTS[pi]) for pi in range(len(MUCOLA_PROMPTS))
               if (pi % args.num_shards) == args.shard_idx]

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["prompt_idx", "sample_idx", "target_label", "arm", "judged_label",
                "hit_target", "clf_self_label", "span_nll", "text"])

    B = args.samples_per_prompt
    for pi, prompt in prompts:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        for target_label in (0, 1):
            for guided in (False, True):
                arm = "guided" if guided else "unguided"
                seed = (args.data_seed + 1000 * pi + (100000 if guided else 0)
                        + 500000 * target_label)
                gen, span_locs = guided_generate(
                    sedd, clf, target_label, prompt_ids, args.span_len, args.steps,
                    args.gamma, args.topk, device, seed, guided, batch=B)
                # noisy-clf self-assessment on the clean final text (DIAGNOSTIC only:
                # tells whether guidance moved the guiding classifier's own judgment,
                # separating mechanism validity from clf-vs-judge disagreement).
                nf0 = torch.zeros(gen.shape[0], device=device)
                self_lbl = clf(gen, nf0).argmax(-1).cpu().tolist()
                for k in range(B):
                    one = gen[k:k + 1]
                    judged = classify_judge(gpt2sft, head, one)
                    nll = span_nll_per_tok(gpt2sft, one, span_locs)
                    text = tok.decode(one[0], skip_special_tokens=True)[:160].replace("\n", " ")
                    w.writerow([pi, k, target_label, arm, judged,
                                int(judged == target_label), self_lbl[k], nll, text])
        print(f"[guided shard{args.shard_idx}] prompt {pi} done, "
              f"{(time.time()-t0)/60:.1f}m", flush=True)
    f.close()
    n = len(prompts)

    summary = {"experiment": "guided", "scale": args.scale, "shard_idx": args.shard_idx,
               "num_shards": args.num_shards, "n_cases": n, "gamma": args.gamma,
               "topk": args.topk, "steps": args.steps, "span_len": args.span_len,
               "shard_done": True, "run_name": args.run_name,
               "wall_time_sec": time.time() - t0}
    atomic_json(json_path, summary)
    print(f"[done] {json_path} ({summary['wall_time_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
