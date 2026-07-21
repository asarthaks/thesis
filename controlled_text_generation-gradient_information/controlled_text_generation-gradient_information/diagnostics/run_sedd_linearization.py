#!/usr/bin/env python
"""
run_sedd_linearization.py

CONCERN 8: the positive control the thesis designed but never ran.

The central causal claim is "the AR training objective is why the gradient is
uninformative". That predicts a model trained with a DIFFERENT objective on the
same tokenizer should give a surrogate that DOES track the true change. SEDD
(Lou et al. 2024, GPT-2 tokenizer, public weights) is the cheapest such model.

This script runs ONLY the linearization diagnostic (mirrors exp_linearization in
run_diagnostic.py and Figures 5.4 / 5.5), on the SAME 200 ROCStories sequences
and masked positions. No sampler, no annealing, no MH.

  surrogate  the diffusion model's own ratio estimate for swapping the masked
             token to candidate v  (this is what SEDD gives directly: the
             concrete score is an estimate of p_t(v)/p_t(x))
  truth      the actual change in the model's per-position denoising log-prob
             when the token is swapped, measured by a forward pass per candidate

We correlate surrogate vs truth over candidates stratified by embedding distance,
and report the binned-by-distance curve, exactly as the AR figures do.

HONEST SCOPING
--------------
SEDD is not pip-installable as a clean library and its score parameterization is
not the AR gradient, so this is a PILOT, and the two hooks below (load + the two
model calls) MUST be checked against the SEDD repo you actually clone. Everything
downstream of those hooks is generic and shared with the AR experiment. Run with
--dry_run first: it wires a tiny fake model through the whole loop so you can see
the CSV/JSON come out before spending a GPU-hour, then flip to real weights.

Interpretation, both ways (both are wins, which is what makes it a control):
  corr clearly positive   -> the AR objective was the cause; thesis jumps a tier
  corr also ~0            -> the cause is deeper (discreteness / geometry), a more
                            surprising and still-publishable finding
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ==========================================================================
# ADAPT BLOCK. These three functions are the only SEDD-specific code. Verify
# each against the repo you clone (github.com/louaaron/Score-Entropy-Discrete-
# Diffusion). Signatures below match that repo's `load_model_local`, its
# `graph`/`noise` objects, and a model that returns a (B, L, V) score.
# ==========================================================================

class SEDDBundle:
    """Everything the loop needs, kept in one object so the ADAPT surface is small."""
    def __init__(self, model, graph, noise, tokenizer, emb_matrix, device):
        self.model = model
        self.graph = graph
        self.noise = noise
        self.tokenizer = tokenizer
        self.emb_matrix = emb_matrix     # V x D, for the distance stratification only
        self.device = device


def load_sedd(model_dir, device, dry_run=False):
    if dry_run:
        return _fake_bundle(device)

    # ---- ADAPT: match the repo's loader. Typical usage: ----
    #   from load_model import load_model_local
    #   model, graph, noise = load_model_local(model_dir, device)
    #   from transformers import GPT2TokenizerFast
    #   tok = GPT2TokenizerFast.from_pretrained("gpt2")
    sys.path.insert(0, model_dir)               # so the repo's modules import
    from load_model import load_model_local     # noqa: E402
    from transformers import GPT2TokenizerFast   # noqa: E402
    model, graph, noise = load_model_local(model_dir, device)
    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # SEDD-small/medium have no separate input-embedding table we can lean on for
    # the geometric stratification; reuse GPT-2's, which shares the tokenizer, so
    # the "distance" axis is comparable to the AR figures.
    from transformers import GPT2LMHeadModel     # noqa: E402
    ref = GPT2LMHeadModel.from_pretrained("gpt2")
    emb = ref.get_input_embeddings().weight.detach().to(device)
    return SEDDBundle(model, graph, noise, tok, emb, device)


@torch.no_grad()
def sedd_score_row(bundle, ids, pos, sigma):
    """
    Return a V-dim tensor: the model's estimated log-ratio for setting position
    `pos` to each candidate, at noise level sigma. This is the SURROGATE.

    ---- ADAPT: the repo computes `score = model(x, sigma)`; score[b, l, :] is the
    concrete score at (b, l). We take log of the ratio estimate at `pos`. Depending
    on the graph (absorbing vs uniform) the exact transform differs; the version
    below assumes score already estimates p_t(v)/p_t(x_l) so log score is the
    log-ratio. Check against graph.score_entropy / graph.staggered_score. ----
    """
    x = ids.clone().to(bundle.device)            # 1 x L
    sig = torch.tensor([sigma], device=bundle.device)
    score = bundle.model(x, sig)                 # 1 x L x V  (ADAPT if shape differs)
    row = score[0, pos].float()
    row = torch.clamp(row, min=1e-20)
    return torch.log(row)                        # V


@torch.no_grad()
def sedd_position_logprob(bundle, ids, pos, cand_ids, sigma):
    """
    Return a len(cand_ids) tensor: the model's per-position denoising log-prob of
    each candidate at `pos`, given the rest of the (noised) sequence. Difference
    from the current token's value is the TRUTH we correlate the surrogate against.

    ---- ADAPT: for SEDD the natural per-position readout is the reverse posterior
    q_{0|t}(x_pos = v | x_rest). The repo exposes this through the score + graph;
    a common route is graph.staggered_score(score, sigma) then a softmax over the
    vocabulary at `pos`. Replace the two lines marked below. ----
    """
    x = ids.clone().to(bundle.device)
    sig = torch.tensor([sigma], device=bundle.device)
    score = bundle.model(x, sig)                             # 1 x L x V
    # ADAPT: turn the score into a per-position categorical over the vocabulary
    post = bundle.graph.staggered_score(score, sig)          # ADAPT
    logp = torch.log_softmax(post[0, pos].float(), dim=-1)   # ADAPT
    return logp[cand_ids.to(bundle.device)]


def _fake_bundle(device):
    """A tiny stand-in so --dry_run exercises the whole loop with no SEDD."""
    V, D = 512, 32
    emb = torch.randn(V, D, device=device)

    class _M:
        def __call__(self, x, sig):
            B, L = x.shape
            return torch.rand(B, L, V, device=device) + 0.01
        def eval(self): return self

    class _G:
        def staggered_score(self, score, sig):
            return torch.log(score + 1e-6)

    class _T:
        vocab_size = V

    return SEDDBundle(_M(), _G(), None, _T(), emb, device)


# ==========================================================================
# generic loop, shared in spirit with run_diagnostic.exp_linearization
# ==========================================================================

def load_sequences(dataset, tokenizer, n, min_tok, max_tok, seed):
    import random
    texts = []
    if os.path.isfile(dataset):
        with open(dataset) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset("wza/roc_stories", split="train", trust_remote_code=True)
        key = "text" if "text" in ds.column_names else ds.column_names[0]
        texts = [ds[i][key] for i in range(min(len(ds), 20000))]
    rng = random.Random(seed)
    rng.shuffle(texts)
    out = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids[0]
        if min_tok <= len(ids) <= max_tok:
            out.append(ids)
        if len(out) >= n:
            break
    return out


def run(args):
    device = args.device
    bundle = load_sedd(args.model_dir, device, dry_run=args.dry_run)
    tok = bundle.tokenizer
    E = bundle.emb_matrix
    V, D = E.shape

    if args.dry_run:
        # fabricate short random sequences
        seqs = [torch.randint(0, V, (np.random.randint(12, 30),)) for _ in range(args.n_seqs)]
    else:
        seqs = load_sequences(args.dataset, tok, args.n_seqs,
                              args.min_tokens, args.max_tokens, args.seed)

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["seq_id", "pos", "cand_id", "stratum", "dist", "surrogate", "true_delta"])

    rng = np.random.RandomState(args.seed)
    live = (E.norm(dim=-1) > 1e-6)
    live_idx = torch.nonzero(live, as_tuple=False).squeeze(-1)
    V_live = int(live_idx.numel())
    E_live = E[live_idx]

    t0 = time.time()
    for si, ids in enumerate(seqs):
        ids = ids.unsqueeze(0).to(device)
        L = ids.shape[1]
        if L < 6:
            continue
        pos = int(rng.randint(2, L - 2))
        orig_tok = int(ids[0, pos].item())

        # surrogate: the model's ratio estimate over the whole vocabulary
        surrogate_full = sedd_score_row(bundle, ids, pos, args.sigma)  # V

        # candidate set, stratified by embedding distance (same recipe as AR exp)
        e_cur = E[orig_tok]
        dists_all = torch.cdist(e_cur.unsqueeze(0), E_live).squeeze(0)
        order = torch.argsort(dists_all)
        n_near = min(args.n_near, max(0, V_live - 1))
        mid_lo, mid_hi = int(0.1 * V_live), int(0.4 * V_live)
        n_mid = min(args.n_mid, max(0, mid_hi - mid_lo))
        n_far = max(0, min(args.n_cand - n_near - n_mid, V_live))
        near = order[1:1 + n_near]
        mid = order[torch.from_numpy(
            rng.choice(np.arange(mid_lo, mid_hi), size=n_mid, replace=False)).to(device)] \
            if n_mid else torch.empty(0, dtype=torch.long, device=device)
        far = torch.from_numpy(rng.choice(V_live, size=n_far, replace=False)).to(device)
        cands_live = torch.cat([near, mid, far])
        cands = live_idx[cands_live]
        strata = (["near"] * len(near) + ["mid"] * len(mid) + ["random"] * len(far))

        dist = (E[cands] - e_cur.unsqueeze(0)).norm(dim=-1).cpu().numpy()
        surrogate = surrogate_full[cands].cpu().numpy()

        base_lp = sedd_position_logprob(bundle, ids, pos,
                                        torch.tensor([orig_tok]), args.sigma)[0].item()
        true_lp = sedd_position_logprob(bundle, ids, pos, cands, args.sigma).cpu().numpy()
        true_delta = true_lp - base_lp

        for j in range(len(cands)):
            w.writerow([si, pos, int(cands[j].item()), strata[j],
                        float(dist[j]), float(surrogate[j]), float(true_delta[j])])
        if (si + 1) % 10 == 0:
            print(f"[sedd_lin] {si+1}/{len(seqs)}  {(time.time()-t0)/60:.1f}m", flush=True)
    f.close()

    import pandas as pd
    from scipy.stats import spearmanr, pearsonr
    df = pd.read_csv(csv_path)

    def safe(fn, a, b):
        try:
            return float(fn(a, b)[0])
        except Exception:
            return float("nan")

    summary = {
        "experiment": "sedd_linearization",
        "dry_run": bool(args.dry_run),
        "sigma": args.sigma,
        "n_sequences": len(seqs),
        "n_pairs": int(len(df)),
        "spearman_surrogate_vs_true_ALL": safe(spearmanr, df.surrogate, df.true_delta),
        "pearson_surrogate_vs_true_ALL": safe(pearsonr, df.surrogate, df.true_delta),
        "note": ("Compare spearman_surrogate_vs_true_ALL against the AR "
                 "linearization number (~ -0.01). Substantially positive here "
                 "confirms the training-objective diagnosis."),
    }
    for st in ["near", "mid", "random"]:
        sub = df[df.stratum == st]
        if len(sub) > 10:
            summary[f"spearman_{st}"] = safe(spearmanr, sub.surrogate, sub.true_delta)
            summary[f"mean_dist_{st}"] = float(sub.dist.mean())
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results_sedd")
    p.add_argument("--model_dir", default="", help="path to a cloned+downloaded SEDD checkpoint dir")
    p.add_argument("--dataset", default="roc_stories")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="use a fake model to test the loop")

    p.add_argument("--n_seqs", type=int, default=200)
    p.add_argument("--min_tokens", type=int, default=15)
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--n_cand", type=int, default=2000)
    p.add_argument("--n_near", type=int, default=500)
    p.add_argument("--n_mid", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="noise level for the score readout; small = close to clean text")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} already exists")
        return
    if not args.dry_run and not args.model_dir:
        print("ERROR: pass --model_dir (a SEDD checkpoint) or --dry_run", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    summary = run(args)
    summary["run_name"] = args.run_name
    summary["wall_time_sec"] = time.time() - t0
    summary["argv"] = vars(args)
    atomic_json(json_path, summary)
    print(f"[done] {json_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "argv"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
