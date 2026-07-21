#!/usr/bin/env python
"""
run_revision.py

New GPU experiments requested by the examiner revision plan. Same contract as
run_diagnostic.py and run_experiment.py so run_queue.sh / worker.sh pick these up
unchanged:

  - a job is identified by --run_name
  - a job is DONE when <out_dir>/<run_name>.json exists
  - the JSON is written atomically at the very end
  - bulk per-item records go to <out_dir>/<run_name>.csv

Experiments
-----------
  kl_baselines    CONCERN 2 (+ CONCERN 4 step 3). Runs the non-MCMC and
                  non-gradient baselines on the IDENTICAL sequences and
                  corruption seeds the main grid used, and reports the SAME KL
                  metric that base_sampler.optimize logs, so every number is a
                  drop-in reference line for Table 5.1:
                    ground_truth       insert the original token back  (the floor)
                    untouched          the corrupted sequence, no recovery (the ceiling reference)
                    random_token       a fresh uniform token           (top of scale)
                    cond_argmax        argmax p(x_i | x_<i)            (one forward pass)
                    cond_sample        sample p(x_i | x_<i)
                    cond_topk_rescore  top-k from the conditional, re-ranked by full-seq joint
                    gibbs              Metropolized Gibbs, conditional proposal, exact-energy accept
                  The last one is the "non-gradient sampler on the same energy"
                  the scope-narrowing argument in concern 4 needs.

  model_divergence CONCERN 5 steps 1-2. Did LoRA tuning move the energy at all?
                  On a held-out set, per-sequence log-likelihood under the base
                  and under one GFlowNet variant: mean absolute difference,
                  Pearson/Spearman correlation, and the mean next-token KL
                  between the two models averaged over positions.

  continuation    CONCERN 9. The core policy-vs-random ablation on a
                  prefix-continuation task (mask a trailing span, not scattered
                  interior tokens), one model, so the null result is shown off
                  the masked-recovery task too.

Why import from run_experiment
------------------------------
kl_baselines and continuation must sit on the SAME sequences the grid used, or
they are not comparable. We therefore reuse run_experiment.load_texts and
run_experiment.build_corruption verbatim rather than re-deriving them, and we
reproduce its sample loop exactly (ti increments even on skipped sentences), so
sample_idx lines up one-to-one with the grid CSVs.
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft
# reuse the EXACT corruption and text-loading the main grid used
from run_experiment import load_texts, build_corruption, seed_all


# --------------------------------------------------------------------------
# infrastructure (kept identical to run_diagnostic.py on purpose)
# --------------------------------------------------------------------------

def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(x.mean()), float(lo), float(hi))


# --------------------------------------------------------------------------
# THE KL metric, replicated bit-for-bit from base_sampler.optimize
# --------------------------------------------------------------------------
# base_sampler computes, at every masked position m with m < seq_len-1:
#   p_ref   = softmax( logits(orig_ids)[0, m] )                  # GT-filled next-token dist
#   p_pred  = softmax( logits(orig_ids with mask<-recovered)[0, m] )
#   avg_kl  = KL(p_ref || p_pred) averaged over the valid m  (reduction='batchmean')
# We reproduce it so the baselines are on the same axis as the grid, not a new one.

@torch.no_grad()
def avg_kl_for_fill(model, orig_ids, mask_indices, fill_ids):
    """
    orig_ids:  1 x L LongTensor (the clean sequence)
    mask_indices: list[int]
    fill_ids:  1 x M LongTensor, the tokens to place at the masked positions
    Returns the scalar avg_kl exactly as base_sampler logs it, or nan if no
    masked position has a right neighbour (the same guard base_sampler uses).
    """
    seq_len = orig_ids.shape[1]
    valid = [m for m in mask_indices if m < seq_len - 1]
    if not valid:
        return float("nan")

    out_gt = model(orig_ids)
    p_ref = torch.softmax(torch.stack([out_gt.logits[0, m, :] for m in valid]), dim=-1)

    temp = orig_ids.clone()
    mt = torch.tensor(mask_indices, device=orig_ids.device)
    temp[0, mt] = fill_ids.to(orig_ids.device)
    out_pred = model(temp)
    log_p_pred = torch.log_softmax(out_pred.logits[0, valid, :], dim=-1)
    kl = F.kl_div(log_p_pred, p_ref, reduction="batchmean", log_target=False)
    return float(kl.item())


@torch.no_grad()
def joint_logprob(model, ids):
    """sum_t log p(x_t | x_<t) for a 1 x L sequence."""
    out = model(ids)
    lp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    tgt = ids[:, 1:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()


# --------------------------------------------------------------------------
# grid-faithful sample generator
# --------------------------------------------------------------------------

def iter_grid_samples(args, tokenizer, device):
    """
    Yields (sample_idx, corrupted, mask_indices, orig_ids) in the SAME order and
    with the SAME seeds as run_experiment.main, so sample_idx matches the grid.
    """
    texts = load_texts(args)
    done, ti = 0, 0
    while done < args.n_samples and ti < len(texts):
        case = build_corruption(tokenizer, texts[ti], args.num_masks,
                                args.data_seed + ti, device)
        ti += 1
        if case is None:
            continue
        corrupted, mask_indices, orig_ids = case
        yield done, corrupted, mask_indices, orig_ids
        done += 1


# --------------------------------------------------------------------------
# EXPERIMENT: kl_baselines
# --------------------------------------------------------------------------

def exp_kl_baselines(args, model, tokenizer, device):
    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "baseline", "avg_kl", "exact_match", "num_masks"])

    vocab = tokenizer.vocab_size
    t0 = time.time()
    n = 0

    for sidx, corrupted, mask_indices, orig_ids in iter_grid_samples(args, tokenizer, device):
        L = orig_ids.shape[1]
        mt = torch.tensor(mask_indices, device=device)
        gt_fill = orig_ids[0, mt].clone()
        corrupt_fill = corrupted[0, mt].clone()

        rows = {}

        # 1. ground truth floor. Under this KL metric it is 0 by construction,
        #    which is itself worth stating: the metric measures deviation from the
        #    GT-conditioned next-token distribution, so GT is exactly 0. That is the
        #    honest answer to "the model grades its own homework": the reference is
        #    the ground-truth fill, not the model's own preferred token.
        rows["ground_truth"] = (avg_kl_for_fill(model, orig_ids, mask_indices, gt_fill),
                                int((gt_fill == gt_fill).all()))

        # 2. untouched corruption ceiling reference
        rows["untouched"] = (avg_kl_for_fill(model, orig_ids, mask_indices, corrupt_fill),
                             int((corrupt_fill == gt_fill).all().item()))

        # 3. random token floor (fresh, distinct from both gt and corruption)
        rng = np.random.RandomState(10_000 + sidx)
        rand_fill = corrupt_fill.clone()
        for j in range(len(mask_indices)):
            r = int(rng.randint(0, vocab))
            while r == int(gt_fill[j].item()):
                r = int(rng.randint(0, vocab))
            rand_fill[j] = r
        rows["random_token"] = (avg_kl_for_fill(model, orig_ids, mask_indices, rand_fill),
                                int((rand_fill == gt_fill).all().item()))

        # conditional-based baselines. One forward pass on the corrupted sequence
        # gives the left-context conditional at every masked position.
        with torch.no_grad():
            out_c = model(corrupted)
            logits_c = out_c.logits[0].float()   # L x V

        cond_argmax_fill = corrupt_fill.clone()
        cond_sample_fill = corrupt_fill.clone()
        cond_topk_fill = corrupt_fill.clone()
        for j, m in enumerate(mask_indices):
            dist = torch.log_softmax(logits_c[m - 1], dim=-1)   # p(x_m | x_<m)
            cond_argmax_fill[j] = int(dist.argmax().item())
            probs = dist.exp()
            cond_sample_fill[j] = int(torch.multinomial(probs, 1).item())
            # top-k re-rescored by full-sequence joint likelihood
            k = min(args.topk, probs.numel())
            topk = torch.topk(dist, k).indices
            best_lp, best_tok = -1e30, int(topk[0].item())
            for cand in topk.tolist():
                tmp = corrupted.clone()
                tmp[0, m] = cand
                lp = joint_logprob(model, tmp)
                if lp > best_lp:
                    best_lp, best_tok = lp, cand
            cond_topk_fill[j] = best_tok

        rows["cond_argmax"] = (avg_kl_for_fill(model, orig_ids, mask_indices, cond_argmax_fill),
                               int((cond_argmax_fill == gt_fill).all().item()))
        rows["cond_sample"] = (avg_kl_for_fill(model, orig_ids, mask_indices, cond_sample_fill),
                               int((cond_sample_fill == gt_fill).all().item()))
        rows["cond_topk_rescore"] = (avg_kl_for_fill(model, orig_ids, mask_indices, cond_topk_fill),
                                     int((cond_topk_fill == gt_fill).all().item()))

        # 7. Metropolized Gibbs: conditional proposal, exact-energy accept. This is
        #    the non-gradient sampler on the SAME energy (concern 4). Start from the
        #    corrupted fill so it has the same starting point the samplers get.
        gibbs_fill = corrupt_fill.clone()
        cur = corrupted.clone()
        cur_E = -joint_logprob(model, cur)
        grng = torch.Generator(device=device).manual_seed(20_000 + sidx)
        for _sweep in range(args.gibbs_sweeps):
            for j, m in enumerate(mask_indices):
                with torch.no_grad():
                    dist = torch.log_softmax(model(cur).logits[0, m - 1].float(), dim=-1)
                probs = dist.exp()
                prop = int(torch.multinomial(probs, 1, generator=grng).item())
                cur_tok = int(cur[0, m].item())
                if prop == cur_tok:
                    continue
                tmp = cur.clone()
                tmp[0, m] = prop
                prop_E = -joint_logprob(model, tmp)
                # log accept = -(E' - E) + (log q(cur) - log q(prop))
                log_q_fwd = float(dist[prop].item())
                log_q_back = float(dist[cur_tok].item())
                log_acc = -(prop_E - cur_E) + (log_q_back - log_q_fwd)
                if float(torch.log(torch.rand((), generator=grng, device=device)).item()) < log_acc:
                    cur = tmp
                    cur_E = prop_E
        gibbs_fill = cur[0, mt].clone()
        rows["gibbs"] = (avg_kl_for_fill(model, orig_ids, mask_indices, gibbs_fill),
                         int((gibbs_fill == gt_fill).all().item()))

        for name, (kl, em) in rows.items():
            w.writerow([sidx, name, kl, em, len(mask_indices)])
        n += 1
        if n % 20 == 0:
            el = time.time() - t0
            print(f"[kl_baselines] {n}/{args.n_samples}, {el/60:.1f}m", flush=True)

    f.close()

    import pandas as pd
    df = pd.read_csv(csv_path)
    summary = {"experiment": "kl_baselines", "n_sequences": n, "by_baseline": {}}
    for name in df.baseline.unique():
        sub = df[df.baseline == name]
        mean, lo, hi = bootstrap_ci(sub.avg_kl.values, seed=args.seed)
        summary["by_baseline"][name] = {
            "n": int(len(sub)),
            "mean_kl": mean, "kl_ci95_lo": lo, "kl_ci95_hi": hi,
            "median_kl": float(sub.avg_kl.median()),
            "exact_match_pct": float(100.0 * sub.exact_match.mean()),
        }
    return summary


# --------------------------------------------------------------------------
# EXPERIMENT: model_divergence
# --------------------------------------------------------------------------

def exp_model_divergence(args, base_model, tokenizer, device):
    """
    Requires --adapter_path. Loads the tuned variant separately, then compares.
    """
    if not args.adapter_path:
        raise ValueError("model_divergence needs --adapter_path (the GFlowNet variant)")

    print(f"[model_divergence] loading tuned variant {args.adapter_path}", flush=True)
    _, tuned = load_tokenizer_and_model_peft(args.model_path, args.adapter_path,
                                             dtype=next(base_model.parameters()).dtype)
    tuned.eval()

    # held-out sequences: reuse the grid loader but ask for a large n
    texts = load_texts(args)
    seqs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        if args.min_tokens <= ids.shape[1] <= args.max_tokens:
            seqs.append(ids)
        if len(seqs) >= args.n_seqs:
            break

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["seq_id", "loglik_base", "loglik_tuned", "mean_nexttok_kl"])

    base_lls, tuned_lls, kls = [], [], []
    for i, ids in enumerate(seqs):
        lb = joint_logprob(base_model, ids)
        lt = joint_logprob(tuned, ids)
        with torch.no_grad():
            pb = torch.log_softmax(base_model(ids).logits[0, :-1, :].float(), dim=-1)
            pt = torch.log_softmax(tuned(ids).logits[0, :-1, :].float(), dim=-1)
            # KL(base || tuned) averaged over positions
            kl = F.kl_div(pt, pb.exp(), reduction="batchmean", log_target=False).item()
        base_lls.append(lb); tuned_lls.append(lt); kls.append(kl)
        w.writerow([i, lb, lt, kl])
        if (i + 1) % 100 == 0:
            print(f"[model_divergence] {i+1}/{len(seqs)}", flush=True)
    f.close()

    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    df = pd.read_csv(csv_path)
    diff = (df.loglik_tuned - df.loglik_base)
    summary = {
        "experiment": "model_divergence",
        "adapter_path": args.adapter_path,
        "n_sequences": int(len(df)),
        "mean_abs_loglik_diff": float(diff.abs().mean()),
        "mean_signed_loglik_diff": float(diff.mean()),
        "median_abs_loglik_diff": float(diff.abs().median()),
        "pearson_base_tuned_loglik": float(pearsonr(df.loglik_base, df.loglik_tuned)[0]),
        "spearman_base_tuned_loglik": float(spearmanr(df.loglik_base, df.loglik_tuned)[0]),
        "mean_nexttok_kl_base_vs_tuned": float(df.mean_nexttok_kl.mean()),
        "note": ("If mean_abs_loglik_diff is small and the correlation ~1, the LoRA "
                 "left the energy essentially unchanged, so Langevin indistinguishability "
                 "on the tuned energy is trivially expected; retitle the finding as "
                 "'light-touch amortization leaves the energy unchanged'. If the diff is "
                 "large yet the landscape diagnostics are flat, the original claim stands."),
    }
    return summary


# --------------------------------------------------------------------------
# EXPERIMENT: continuation
# --------------------------------------------------------------------------

def exp_continuation(args, model, tokenizer, device):
    """
    Core policy-vs-random ablation on a prefix-continuation task: mask a trailing
    span of `span` tokens and recover them, DLS + MH on, one schedule. Uses the
    real core samplers so the comparison is apples-to-apples with the grid.
    """
    from core.dls import DiscreteLangevinSampler

    eps = np.linspace(args.eps_start, args.eps_end, args.steps)
    methods = ["policy", "grad_norm_preserved_random_dir", "random"]

    texts = load_texts(args)
    # keep sentences long enough to host a trailing span plus a prefix
    seqs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] >= args.span + 5:
            seqs.append(ids)
        if len(seqs) >= args.n_samples:
            break

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "method", "final_kl", "final_l2", "accuracy_pct"])

    summary = {"experiment": "continuation", "span": args.span, "steps": args.steps,
               "by_method": {}}

    for method in methods:
        sampler = DiscreteLangevinSampler(
            model=model, tokenizer=tokenizer, steps=args.steps, temperature=args.temperature,
            oracle=False, method=method, mh_sampling=True, grad_normalization=True,
            noise_scale=args.noise_scale, epsilon_schedule=eps,
        )
        kls, accs = [], []
        for sidx, ids in enumerate(seqs):
            L = ids.shape[1]
            # trailing span, staying off the final token so every masked position
            # has a right neighbour for the KL metric
            hi = L - 1
            lo = max(1, hi - args.span)
            mask_indices = list(range(lo, hi))
            if not mask_indices:
                continue
            orig_ids = ids.clone()
            corrupted = ids.clone()
            rng = np.random.RandomState(args.data_seed + sidx)
            for m in mask_indices:
                r = int(rng.randint(0, tokenizer.vocab_size))
                while r == int(orig_ids[0, m].item()):
                    r = int(rng.randint(0, tokenizer.vocab_size))
                corrupted[0, m] = r
            seed_all(args.data_seed + sidx)
            _, metrics = sampler.optimize(corrupted.clone(), mask_indices, orig_ids.clone())
            final = metrics[-1]
            gt = orig_ids[0, torch.tensor(mask_indices, device=device)].tolist()
            acc = 100.0 * sum(int(a == b) for a, b in zip(final["token_ids"], gt)) / max(len(gt), 1)
            kls.append(final["avg_kl_divergence"]); accs.append(acc)
            w.writerow([sidx, method, final["avg_kl_divergence"], final["avg_l2_distance"], acc])
        mean, lo, hi = bootstrap_ci(kls, seed=args.seed)
        summary["by_method"][method] = {
            "n": len(kls), "mean_final_kl": mean, "kl_ci95_lo": lo, "kl_ci95_hi": hi,
            "mean_accuracy_pct": float(np.mean(accs)) if accs else float("nan"),
        }
        print(f"[continuation] {method}: kl={mean:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    f.close()
    return summary


EXPERIMENTS = {
    "kl_baselines": exp_kl_baselines,
    "model_divergence": exp_model_divergence,
    "continuation": exp_continuation,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results_revision")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--model_tag", default="gpt2-large")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    # sequence selection, kept identical in spirit to run_experiment
    p.add_argument("--data_file", default=None)
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=40)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--num_masks", type=int, default=1)
    p.add_argument("--data_seed", type=int, default=0)

    # kl_baselines
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--gibbs_sweeps", type=int, default=3)

    # model_divergence
    p.add_argument("--n_seqs", type=int, default=1000)
    p.add_argument("--min_tokens", type=int, default=15)
    p.add_argument("--max_tokens", type=int, default=60)

    # continuation
    p.add_argument("--span", type=int, default=20)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eps_start", type=float, default=10.5)
    p.add_argument("--eps_end", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=0.01)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} already exists")
        return

    seed_all(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    t0 = time.time()
    if args.adapter_path and args.exp != "model_divergence":
        tokenizer, model = load_tokenizer_and_model_peft(args.model_path, args.adapter_path, dtype=dtype)
    else:
        tokenizer, model = load_tokenizer_and_model(args.model_path, dtype=dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    for pparam in model.parameters():
        pparam.requires_grad_(False)
    print(f"[{args.exp}] model loaded in {time.time()-t0:.1f}s", flush=True)

    summary = EXPERIMENTS[args.exp](args, model, tokenizer, args.device)
    summary["run_name"] = args.run_name
    summary["model_path"] = args.model_path
    summary["model_tag"] = args.model_tag
    summary["seed"] = args.seed
    summary["wall_time_sec"] = time.time() - t0
    summary["argv"] = vars(args)

    atomic_json(json_path, summary)
    print(f"[done] {json_path} ({summary['wall_time_sec']/60:.1f} min)")
    print(json.dumps({k: v for k, v in summary.items() if k != "argv"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
