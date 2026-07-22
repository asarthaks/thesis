#!/usr/bin/env python
"""
run_gprime.py  -  Phase 5 Stage 1a: on-domain, trust-region guided SEDD generation.

Phase 4's Part G steered off-domain (MuCoLa continuation prompts), where the guiding
noisy classifier and the concern-11 judge agree only 56-64% on the generated text, and
gamma pushed text off the fluent manifold (NLL 7.1 -> 11.0) exactly where two sentiment
classifiers diverge. G-prime tests the same steering where the instruments are
calibrated: prompts are held-out SST-2 validation sentences (NEITHER the guide nor the
judge trained on them; both train on ds["train"] and only read ds["validation"] under
no_grad). It also adds a trust region that caps the fluency cost by construction.

Trust region (NEW): at each commitment, among the top-k SEDD candidates for a masked
cell, only those whose SEDD log-prob is within delta nats of the top candidate are
eligible; the guided categorical is restricted to that eligible set (all other mass on
the real vocab set to zero). The committed token is therefore always within delta nats
of the SEDD argmax at that cell, so guidance can only reshuffle mass among
near-equally-fluent options. The MASK column is left untouched (a cell may still stay
MASK for a later step, as in the unguided process).

Role separation (non-negotiable): the noisy classifier ONLY guides; steering is scored
by the concern-11 judge (frozen-GPT-2 sentiment head via classify_judge). Not best-of-N.

Metrics emitted per generation row: judged label + hit_target (judge), clf_self_label
(guide's own verdict on the clean text, mechanism check), gpt2sft span NLL (fluency),
and per-commitment trust-region stats (mean eligible count, fraction with <=1 eligible).
A per-prompt "realtext" row scores the full held-out sentence with both instruments
(the fully on-domain instrument-calibration point). Sharded over prompts; atomic JSON.
"""

import argparse
import csv
import json
import math
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
import sedd_lib
from sedd_lib import MASK_TOKEN, REAL_VOCAB
from train_noisy_classifier import NoisyClassifier, EPS

PROMPT_LEN = 10          # first 10 tokens of a held-out sentence (in the "8 to 12" band)
MIN_SENT_TOK = 12        # require the sentence to have real content beyond the prompt


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_heldout_prompts(tok, n_prompts):
    """First n_prompts SST-2 validation sentences (dataset order) whose GPT-2
    tokenization has >= MIN_SENT_TOK tokens. Prompt = first PROMPT_LEN tokens. Neither
    the judge head nor the noisy guide trained on this split. Returns list of
    (idx, prompt_ids_list, full_ids_list, true_label). Deterministic and reproducible."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")["validation"]
    out = []
    for ex in ds:
        ids = tok(ex["sentence"].strip(), add_special_tokens=False).input_ids
        if len(ids) < MIN_SENT_TOK:
            continue
        out.append((len(out), ids[:PROMPT_LEN], ids, int(ex["label"])))
        if len(out) >= n_prompts:
            break
    return out


@torch.no_grad()
def classify_judge(gpt2sft, head, ids):
    """concern-11 judge: frozen-GPT-2 sentiment head, argmax label (== run_constrained)."""
    emb = gpt2sft.get_input_embeddings()(ids)
    out = gpt2sft(inputs_embeds=emb, output_hidden_states=True, return_dict=True)
    return int(head(out.hidden_states[-1]).argmax(-1).item())


@torch.no_grad()
def span_nll_per_tok(gpt2sft, ids, span_locs):
    """mean NLL/token of the span positions under gpt2sft (fluency readout)."""
    out = gpt2sft(ids)
    lp = torch.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
    tgt = ids[0, 1:]
    tok_nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    idx = [s - 1 for s in span_locs if 1 <= s < ids.shape[1]]
    if not idx:
        return float("nan")
    return float(tok_nll[torch.tensor(idx, device=ids.device)].mean().item())


@torch.no_grad()
def guided_generate(sedd, clf, target_label, prompt_ids, span_len, steps, gamma,
                    topk, delta, device, seed, guided, batch=1):
    """Batched absorbing-SEDD analytic denoising with optional trust-region classifier
    guidance. Returns (x[batch,L], span_locs, elig_stats). elig_stats accumulates, over
    every guided masked-cell application, the total count, sum of eligible-candidate
    counts, and the number of applications with <=1 eligible candidate."""
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
    thresh_ratio = math.exp(-delta)   # candidate eligible if prob >= top_prob * exp(-delta)
    stats = {"n_apply": 0, "sum_elig": 0.0, "n_le1": 0}

    def apply_guidance(probs, x_cur, mc):
        masked = (x_cur == MASK_TOKEN)
        masked[:, prompt_locs] = False
        cells = torch.nonzero(masked, as_tuple=False)
        if cells.shape[0] == 0:
            return probs
        cb, cp = cells[:, 0], cells[:, 1]
        Nc = cells.shape[0]
        real = probs[cb, cp, :REAL_VOCAB]                  # (Nc, V_real), SEDD masses >= 0
        topv, topi = real.topk(k, dim=-1)                  # descending
        # trust region: eligible = within delta nats of the top candidate
        elig = topv >= (topv[:, :1] * thresh_ratio)        # (Nc,k) bool, col 0 always True
        # classifier weight over the k candidate states
        cand = x_cur[cb].unsqueeze(1).expand(Nc, k, L).clone()
        ar = torch.arange(Nc, device=device)
        cand[ar[:, None], torch.arange(k, device=device)[None, :], cp[:, None]] = topi
        nf = torch.full((Nc * k,), float(mc), device=device)
        logits = clf(cand.view(Nc * k, L), nf)
        p_t = torch.softmax(logits.float(), dim=-1)[:, target_label].view(Nc, k)
        w = topv * (p_t.clamp_min(1e-8) ** gamma)
        w = w * elig.to(w.dtype)                           # zero out non-eligible top-k
        new_real = torch.zeros_like(real)                  # zero all other real mass too
        new_real.scatter_(-1, topi, w)
        probs[cb, cp, :REAL_VOCAB] = new_real
        # record eligibility (governs the widen-to-8 decision)
        n_elig = elig.sum(-1)
        stats["n_apply"] += int(Nc)
        stats["sum_elig"] += float(n_elig.sum().item())
        stats["n_le1"] += int((n_elig <= 1).sum().item())
        return probs

    for i in range(steps):
        x[:, prompt_locs] = prompt_vals
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        curr_sigma = noise(t)[0]
        next_sigma = noise(t - (1 - eps_t) / steps)[0]
        dsigma = curr_sigma - next_sigma
        score = score_fn(x, curr_sigma)
        stag = graph.staggered_score(score, dsigma)
        probs = stag * graph.transp_transition(x, dsigma)
        if guided:
            probs = apply_guidance(probs, x, float((1 - EPS) * t[0].item()))
        x = sample_categorical(probs)

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
    return x, span_locs, stats


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

    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--span_len", type=int, default=20)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gammas", default="2,4")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--delta", type=float, default=5.0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} exists")
        return

    gammas = [float(g) for g in args.gammas.split(",")]
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

    all_prompts = load_heldout_prompts(tok, args.n_prompts)
    prompts = [p for p in all_prompts if (p[0] % args.num_shards) == args.shard_idx]
    print(f"[gprime] loaded models + {len(all_prompts)} held-out SST-2 prompts "
          f"(shard {args.shard_idx}: {len(prompts)}) in {time.time()-t0:.1f}s", flush=True)

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["prompt_idx", "target_label", "gamma", "arm", "judged_label",
                "hit_target", "clf_self_label", "span_nll", "mean_eligible",
                "frac_le1_eligible", "text"])

    agg_stats = {"n_apply": 0, "sum_elig": 0.0, "n_le1": 0}

    for pidx, prompt_toks, full_toks, true_lbl in prompts:
        prompt_ids = torch.tensor([prompt_toks], device=device)
        # realtext calibration row: score the full held-out sentence with both instruments
        full_ids = torch.tensor([full_toks], device=device)
        rt_judge = classify_judge(gpt2sft, head, full_ids)
        rt_self = int(clf(full_ids, torch.zeros(1, device=device)).argmax(-1).item())
        w.writerow([pidx, true_lbl, -1, "realtext", rt_judge, int(rt_judge == true_lbl),
                    rt_self, "", "", "", tok.decode(full_toks)[:160].replace("\n", " ")])

        seed_u = 700000 + pidx            # one seed per prompt; unguided and guided share it
        # unguided generation (gamma-independent), scored once
        u_gen, span_locs, _ = guided_generate(
            sedd, clf, 0, prompt_ids, args.span_len, args.steps, 0.0, args.topk,
            args.delta, device, seed_u, guided=False, batch=1)
        u_judge = classify_judge(gpt2sft, head, u_gen)
        u_self = int(clf(u_gen, torch.zeros(1, device=device)).argmax(-1).item())
        u_nll = span_nll_per_tok(gpt2sft, u_gen, span_locs)
        u_text = tok.decode(u_gen[0], skip_special_tokens=True)[:160].replace("\n", " ")

        for gamma in gammas:
            for target_label in (0, 1):
                # unguided row (same generation) paired under this (label, gamma)
                w.writerow([pidx, target_label, gamma, "unguided", u_judge,
                            int(u_judge == target_label), u_self, u_nll, "", "", u_text])
                # guided generation, SAME seed as unguided -> true paired comparison
                g_gen, _, st = guided_generate(
                    sedd, clf, target_label, prompt_ids, args.span_len, args.steps,
                    gamma, args.topk, args.delta, device, seed_u, guided=True, batch=1)
                g_judge = classify_judge(gpt2sft, head, g_gen)
                g_self = int(clf(g_gen, torch.zeros(1, device=device)).argmax(-1).item())
                g_nll = span_nll_per_tok(gpt2sft, g_gen, span_locs)
                g_text = tok.decode(g_gen[0], skip_special_tokens=True)[:160].replace("\n", " ")
                me = st["sum_elig"] / max(1, st["n_apply"])
                fl = st["n_le1"] / max(1, st["n_apply"])
                w.writerow([pidx, target_label, gamma, "guided", g_judge,
                            int(g_judge == target_label), g_self, g_nll,
                            round(me, 4), round(fl, 4), g_text])
                for kk in agg_stats:
                    agg_stats[kk] += st[kk]
        print(f"[gprime shard{args.shard_idx}] prompt {pidx} done "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    f.close()

    summary = {"experiment": "gprime", "scale": args.scale, "shard_idx": args.shard_idx,
               "num_shards": args.num_shards, "n_prompts": len(prompts),
               "gammas": gammas, "topk": args.topk, "delta": args.delta,
               "steps": args.steps, "span_len": args.span_len,
               "prompt_len": PROMPT_LEN, "min_sent_tok": MIN_SENT_TOK,
               "eligibility": {
                   "n_apply": agg_stats["n_apply"],
                   "mean_eligible": agg_stats["sum_elig"] / max(1, agg_stats["n_apply"]),
                   "frac_le1_eligible": agg_stats["n_le1"] / max(1, agg_stats["n_apply"])},
               "shard_done": True, "run_name": args.run_name,
               "wall_time_sec": time.time() - t0}
    atomic_json(json_path, summary)
    print(f"[done] {json_path} ({summary['wall_time_sec']/60:.1f} min); "
          f"mean_eligible={summary['eligibility']['mean_eligible']:.2f} "
          f"frac_le1={summary['eligibility']['frac_le1_eligible']:.3f}")


if __name__ == "__main__":
    main()
