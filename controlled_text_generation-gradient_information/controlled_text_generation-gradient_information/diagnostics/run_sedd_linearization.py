#!/usr/bin/env python
"""
run_sedd_linearization.py

CONCERN 8: the positive control the thesis designed but never ran, now VERIFIED
against the upstream repo (github.com/louaaron/Score-Entropy-Discrete-Diffusion)
and CORRECTED. See REVISION_LOG.md (Phase 2, SEDD section) for the full derivation
of the two design fixes below.

The central causal claim is "the AR training objective is why the input-embedding
gradient is uninformative". That predicts a model trained with a DIFFERENT objective
on the same tokenizer should give a cheap, one-pass directional signal that DOES
track the true effect of a token swap. SEDD (Lou et al. 2024, GPT-2 tokenizer,
public weights: louaaron/sedd-small) is the cheapest such model.

This mirrors the AR linearization diagnostic (run_diagnostic.exp_linearization,
Figures 5.4 / 5.5): for each sequence and one position, compare a cheap one-pass
SURROGATE against an expensive per-candidate TRUTH, correlate, and bin by embedding
distance.

TWO CORRECTIONS to the originally-designed hooks (each verified against the repo and
numerically on louaaron/sedd-small; both were necessary or the number would be
meaningless):

  1. The score is ALREADY in log space (values in [-39, 0], with negatives), so the
     original `torch.log(score)` was a double-log. We use the score directly as the
     per-position log-preference (a log_softmax over the real vocabulary).

  2. In absorbing-state SEDD the score is context-dependent ONLY at MASKED positions
     (verified: perturbing an UNMASKED token changes another unmasked position's
     readout by 0.0). So a "pseudo-log-likelihood over all positions" truth would be
     identically zero. The faithful analogue of the AR truth (which re-runs the model
     per candidate and is dominated by the change to OTHER positions) is:
        surrogate(v): mask `pos`, keep the rest clean, one pass. The model's denoising
                      log-preference for filling pos, r_pos = log_softmax(score[pos]).
                      surrogate(v) = r_pos[v] - r_pos[orig].  (SEDD's cheap proposal.)
        truth(v):     OBSERVE pos = v, MASK a fixed held-out probe set Q, keep the rest
                      clean, one pass PER CANDIDATE. Sum the model's reconstruction
                      log-prob of the TRUE probe tokens:
                        PLL_Q(v) = sum_{q in Q} log_softmax(score_modified[q])[x_q_true]
                      truth(v) = PLL_Q(v) - PLL_Q(orig).  (The real effect of
                      committing pos=v on reconstructing the rest of THIS sequence.)
     surrogate is read from a DIFFERENT forward pass (pos masked) than truth (pos
     observed = v, Q masked), so they are not trivially identical; the correlation is
     a genuine test of whether SEDD's local proposal signal points the same way as the
     true context effect.

Interpretation (both ways, which is what makes it a control):
  spearman clearly positive  -> the AR objective was the cause; thesis jumps a tier
  spearman also ~0           -> the cause is deeper (discreteness / geometry), a more
                                surprising and still-publishable finding
Compare against the AR number spearman_surrogate_vs_true_ALL ~= -0.01 to 0.057.

Run --dry_run first (fake log-space model wired through the whole loop), then real.
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


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ==========================================================================
# ADAPT BLOCK. The only SEDD-specific code. VERIFIED against the cloned repo
# (louaaron/Score-Entropy-Discrete-Diffusion) and numerically on sedd-small.
# ==========================================================================

MASK_TOKEN = 50257   # absorbing state id for the GPT-2-vocab SEDD (dim 50258 = 50257 + mask)
REAL_VOCAB = 50257


class SEDDBundle:
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

    # VERIFIED: repo's load_model tries load_model_hf (SEDD.from_pretrained) then
    # load_model_local. Passing "louaaron/sedd-small" hits the HF path. The two
    # model-internal flash-attn dependencies are patched to torch-native fallbacks
    # in the clone (model/transformer.py SDPA, model/rotary.py eager); see log.
    sys.path.insert(0, model_dir if os.path.isdir(model_dir) else
                    os.environ.get("SEDD_REPO",
                                   "/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion"))
    from load_model import load_model            # noqa: E402
    from transformers import GPT2TokenizerFast   # noqa: E402
    model, graph, noise = load_model(model_dir, device)
    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # SEDD has no separate input-embedding table for the geometric stratification;
    # reuse GPT-2's (shared tokenizer) so the distance axis matches the AR figures.
    from transformers import GPT2LMHeadModel      # noqa: E402
    ref = GPT2LMHeadModel.from_pretrained("gpt2")
    emb = ref.get_input_embeddings().weight.detach().to(device)
    return SEDDBundle(model, graph, noise, tok, emb, device)


@torch.no_grad()
def _score(bundle, ids, sigma):
    """Model score, shape (B, L, V), ALREADY in log space. B x L LongTensor in."""
    sig = torch.tensor([sigma] * ids.shape[0], device=bundle.device)
    return bundle.model(ids.to(bundle.device), sig).float()


@torch.no_grad()
def sedd_surrogate_row(bundle, ids, pos, sigma):
    """SURROGATE. Mask `pos`, rest clean, one pass. Returns the per-real-vocab
    log-preference r_pos (log_softmax over the real vocabulary) at pos. The caller
    centers it at the original token."""
    x = ids.clone()
    x[0, pos] = MASK_TOKEN
    s = _score(bundle, x, sigma)[0, pos, :REAL_VOCAB]   # V_real, log space
    return F.log_softmax(s, dim=-1)


@torch.no_grad()
def sedd_truth_pll(bundle, ids_batch, probe_pos, true_probe_ids, sigma):
    """TRUTH readout. ids_batch: B x L with pos already set to each candidate and the
    probe positions already set to MASK. probe_pos: list[int]. true_probe_ids: the
    true tokens at the probes (len = len(probe_pos)). Returns a B-length tensor:
    sum over probes of log_softmax(score[q])[true_q]."""
    s = _score(bundle, ids_batch, sigma)                 # B x L x V
    logp = F.log_softmax(s[:, :, :REAL_VOCAB], dim=-1)    # B x L x V_real
    B = ids_batch.shape[0]
    out = torch.zeros(B, device=bundle.device)
    for q, tq in zip(probe_pos, true_probe_ids):
        out += logp[:, q, int(tq)]
    return out


def _fake_bundle(device):
    """--dry_run stand-in. Returns LOG-space scores (<=0) so the no-extra-log path
    is exercised exactly as with the real model."""
    V, D = 512, 32
    emb = torch.randn(V, D, device=device)

    class _M:
        def __call__(self, x, sig):
            B, L = x.shape
            return -torch.rand(B, L, V + 1, device=device) * 5.0   # log-space, <= 0
        def eval(self): return self

    class _G:
        pass

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
    global REAL_VOCAB, MASK_TOKEN
    if args.dry_run:
        REAL_VOCAB = V
        MASK_TOKEN = V

    if args.dry_run:
        seqs = [torch.randint(0, V, (int(np.random.randint(12, 30)),)) for _ in range(args.n_seqs)]
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
    n_written = 0
    for si, ids in enumerate(seqs):
        ids = ids.unsqueeze(0).to(device)
        L = ids.shape[1]
        if L < 8:
            continue
        pos = int(rng.randint(2, L - 2))
        orig_tok = int(ids[0, pos].item())

        # fixed held-out probe set for the truth readout: up to n_probes non-pos
        # positions (positions 1..L-1, excluding pos), seeded so it is identical
        # across all candidates for this sequence.
        cand_probe = [p for p in range(1, L) if p != pos]
        k = min(args.n_probes, len(cand_probe))
        probe_pos = sorted(rng.choice(cand_probe, size=k, replace=False).tolist())
        true_probe_ids = [int(ids[0, q].item()) for q in probe_pos]

        # ---- surrogate: one pass, mask pos, rest clean ----
        r_pos = sedd_surrogate_row(bundle, ids, pos, args.sigma)   # V_real, log space
        r_orig = float(r_pos[orig_tok].item())

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
        surrogate = (r_pos[cands] - r_orig).cpu().numpy()

        # ---- truth: one pass per candidate (batched). pos=v, probes masked, rest clean.
        # baseline PLL at pos=orig
        base_in = ids.clone()
        for q in probe_pos:
            base_in[0, q] = MASK_TOKEN
        base_pll = float(sedd_truth_pll(bundle, base_in, probe_pos, true_probe_ids, args.sigma)[0].item())

        C = cands.shape[0]
        true_delta = np.zeros(C)
        for b0 in range(0, C, args.batch_size):
            b1 = min(b0 + args.batch_size, C)
            chunk = cands[b0:b1]
            B = chunk.shape[0]
            batch = base_in.repeat(B, 1)         # probes already masked
            batch[:, pos] = chunk                # set pos = candidate (observed)
            pll = sedd_truth_pll(bundle, batch, probe_pos, true_probe_ids, args.sigma)
            true_delta[b0:b1] = (pll - base_pll).cpu().numpy()

        for j in range(C):
            w.writerow([si, pos, int(cands[j].item()), strata[j],
                        float(dist[j]), float(surrogate[j]), float(true_delta[j])])
            n_written += 1
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (si + 1) * (len(seqs) - si - 1)
            print(f"[sedd_lin] {si+1}/{len(seqs)} seqs, {n_written} rows, "
                  f"{el/60:.1f}m, eta {eta/60:.1f}m", flush=True)
    f.close()

    import pandas as pd
    from scipy.stats import spearmanr, pearsonr
    df = pd.read_csv(csv_path)

    def safe(fn, a, b):
        try:
            m = np.isfinite(a) & np.isfinite(b)
            return float(fn(np.asarray(a)[m], np.asarray(b)[m])[0])
        except Exception:
            return float("nan")

    # per-sequence spearman then average, matching the AR reconcile's per_run metric,
    # plus the pooled ALL number.
    per_seq = []
    for sid, sub in df.groupby("seq_id"):
        if len(sub) > 10:
            per_seq.append(safe(spearmanr, sub.surrogate.values, sub.true_delta.values))
    per_seq = [x for x in per_seq if np.isfinite(x)]

    summary = {
        "experiment": "sedd_linearization",
        "dry_run": bool(args.dry_run),
        "sigma": args.sigma,
        "n_probes": args.n_probes,
        "n_sequences": int(df.seq_id.nunique()),
        "n_pairs": int(len(df)),
        "spearman_surrogate_vs_true_ALL": safe(spearmanr, df.surrogate, df.true_delta),
        "pearson_surrogate_vs_true_ALL": safe(pearsonr, df.surrogate, df.true_delta),
        "per_seq_spearman_mean": float(np.mean(per_seq)) if per_seq else float("nan"),
        "per_seq_spearman_median": float(np.median(per_seq)) if per_seq else float("nan"),
        "note": ("Compare spearman_surrogate_vs_true_ALL and per_seq_spearman_mean "
                 "against the AR linearization (~ -0.01 to 0.057). Substantially "
                 "positive here confirms the training-objective diagnosis."),
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
    p.add_argument("--model_dir", default="louaaron/sedd-small",
                   help="HF id (default louaaron/sedd-small) or a local checkpoint dir")
    p.add_argument("--dataset", default="roc_stories")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="use a fake log-space model")

    p.add_argument("--n_seqs", type=int, default=200)
    p.add_argument("--min_tokens", type=int, default=15)
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--n_cand", type=int, default=2000)
    p.add_argument("--n_near", type=int, default=500)
    p.add_argument("--n_mid", type=int, default=500)
    p.add_argument("--n_probes", type=int, default=8,
                   help="held-out masked probe positions for the truth PLL readout")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="noise level fed to the score model")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} already exists")
        return

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
