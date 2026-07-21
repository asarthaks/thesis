#!/usr/bin/env python
"""
run_sedd_cap.py  -  Phase 4 SEDD CAPABILITY pilots.

Experiments (--exp):
  gates      Part 0. On the first --n_gate kl_baselines sequences: variable-length
             gate (masked-position log-preference at length L vs L+tail), projection
             gate (observed positions bit-identical, fill is real vocab), and a
             native recovery sanity read. One scale per call.
  recovery   Part P1. Native SEDD absorbing conditional denoising recovers the one
             flipped token on the kl_baselines set. Metrics: exact %, top-5 %, avg KL
             under gpt2sft. Shardable.
  hybrid     Part H. Position-wise independence-MH on the SAME sequences/positions,
             exact gpt2sft energy, proposal swapped for SEDD's bidirectional
             log-preference. Reference arms on the SAME sequences: AR left-conditional
             independence MH, DLS policy, DLS random. Metrics: exact %, top-5 %,
             avg KL, acceptance. Shardable.

Harness contract: --run_name, --out_dir, atomic JSON, per-item CSV, resume by JSON
existence. Sharding over sequences via --shard_idx/--num_shards; a shard's JSON is a
marker and its CSV holds the per-sequence rows. merge_sedd_cap.py assembles the final
JSON+CSV with bootstrap CIs over the merged rows.

Corpus is EXACTLY the kl_baselines set (iter_grid_samples, WikiText-2, data_seed 0,
num_masks 1). SEDD never touches core/; it is only imported. Nothing here is best-of-N.
"""

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.prep import load_tokenizer_and_model
from run_experiment import seed_all
from diagnostics.run_revision import (
    iter_grid_samples, avg_kl_for_fill, joint_logprob, bootstrap_ci,
    _cond_logprob_at,
)
import sedd_lib
from sedd_lib import MASK_TOKEN, REAL_VOCAB


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def grid_corpus(args, tokenizer, device):
    """The exact kl_baselines sequence set, as a materialised list."""
    ga = SimpleNamespace(
        data_file=args.data_file, min_words=args.min_words, max_words=args.max_words,
        n_samples=args.n_samples, num_masks=args.num_masks, data_seed=args.data_seed,
    )
    return list(iter_grid_samples(ga, tokenizer, device))


def shard_filter(seqs, shard_idx, num_shards):
    if num_shards <= 1:
        return seqs
    return [t for t in seqs if (t[0] % num_shards) == shard_idx]


# ==========================================================================
# gates (Part 0)
# ==========================================================================

def exp_gates(args, sedd, gpt2sft, tokenizer, device):
    seqs = grid_corpus(args, tokenizer, device)[:args.n_gate]
    rows = []
    for sidx, corrupted, mask_indices, orig_ids in seqs:
        L = orig_ids.shape[1]
        pos = mask_indices[0]
        gt = int(orig_ids[0, pos].item())
        obs_locs = [i for i in range(L) if i != pos]

        # --- native recovery + projection gate ---
        rec_seq = sedd_lib.conditional_recovery(
            sedd, orig_ids, [pos], predictor=args.predictor, steps=args.steps,
            seed=args.data_seed + sidx)
        rec_tok = int(rec_seq[0, pos].item())
        obs_bit_identical = bool((rec_seq[0, obs_locs] == orig_ids[0, obs_locs]).all().item())
        fill_is_real_vocab = (rec_tok != MASK_TOKEN) and (0 <= rec_tok < REAL_VOCAB)

        # --- variable-length / tail-leakage gate ---
        r_L = sedd_lib.logpref_at(sedd, orig_ids, pos, args.sigma)          # (V,)
        tailed = torch.cat(
            [orig_ids, torch.full((1, args.tail), MASK_TOKEN, device=device,
                                  dtype=orig_ids.dtype)], dim=1)
        r_Lt = sedd_lib.logpref_at(sedd, tailed, pos, args.sigma)
        top1_L, top1_Lt = int(r_L.argmax().item()), int(r_Lt.argmax().item())
        top5_L = set(torch.topk(r_L, 5).indices.tolist())
        top5_Lt = set(torch.topk(r_Lt, 5).indices.tolist())
        kl_L_vs_Lt = float(F.kl_div(r_Lt, r_L.exp(), reduction="sum",
                                    log_target=False).item())  # KL(p_L || p_Lt)
        maxabs = float((r_L.exp() - r_Lt.exp()).abs().max().item())

        rows.append(dict(
            sample_idx=sidx, L=L, pos=pos, gt_tok=gt, rec_tok=rec_tok,
            recovered_exact=int(rec_tok == gt),
            obs_bit_identical=obs_bit_identical,
            fill_is_real_vocab=fill_is_real_vocab,
            tail=args.tail,
            top1_len_L=top1_L, top1_len_Ltail=top1_Lt,
            top1_identical=int(top1_L == top1_Lt),
            top5_overlap=len(top5_L & top5_Lt),
            gt_in_top5_L=int(gt in top5_L),
            kl_pref_L_vs_Ltail=kl_L_vs_Lt, maxabs_prob_L_vs_Ltail=maxabs,
        ))

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "experiment": "gates", "scale": args.scale, "n_gate": len(rows),
        "predictor": args.predictor, "steps": args.steps, "sigma": args.sigma,
        "projection_gate_pass": bool(all(r["obs_bit_identical"] and r["fill_is_real_vocab"] for r in rows)),
        "varlen_top1_identical_count": int(sum(r["top1_identical"] for r in rows)),
        "varlen_mean_kl_L_vs_Ltail": float(np.mean([r["kl_pref_L_vs_Ltail"] for r in rows])),
        "varlen_max_kl_L_vs_Ltail": float(np.max([r["kl_pref_L_vs_Ltail"] for r in rows])),
        "recovered_exact_count": int(sum(r["recovered_exact"] for r in rows)),
        "gt_in_top5_count": int(sum(r["gt_in_top5_L"] for r in rows)),
        "rows": rows,
    }
    return summary


# ==========================================================================
# recovery (Part P1)
# ==========================================================================

def exp_recovery(args, sedd, gpt2sft, tokenizer, device):
    all_seqs = grid_corpus(args, tokenizer, device)
    seqs = shard_filter(all_seqs, args.shard_idx, args.num_shards)

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "pos", "gt_tok", "rec_tok", "exact", "top5",
                "avg_kl_gpt2sft", "num_masks"])

    t0 = time.time()
    n = 0
    for sidx, corrupted, mask_indices, orig_ids in seqs:
        pos = mask_indices[0]
        gt = int(orig_ids[0, pos].item())
        rec_seq = sedd_lib.conditional_recovery(
            sedd, orig_ids, [pos], predictor=args.predictor, steps=args.steps,
            seed=args.data_seed + sidx)
        rec_tok = int(rec_seq[0, pos].item())
        r_pos = sedd_lib.logpref_at(sedd, orig_ids, pos, args.sigma)
        top5 = int(gt in set(torch.topk(r_pos, 5).indices.tolist()))
        fill = torch.tensor([rec_tok], device=device)
        kl = avg_kl_for_fill(gpt2sft, orig_ids, mask_indices, fill)
        w.writerow([sidx, pos, gt, rec_tok, int(rec_tok == gt), top5, kl, len(mask_indices)])
        n += 1
        if n % 25 == 0:
            print(f"[recovery {args.scale} shard{args.shard_idx}] {n} seqs, "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
    f.close()
    return {"experiment": "recovery", "scale": args.scale, "shard_idx": args.shard_idx,
            "num_shards": args.num_shards, "n_rows": n, "predictor": args.predictor,
            "steps": args.steps, "sigma": args.sigma, "shard_done": True}


# ==========================================================================
# hybrid (Part H)
# ==========================================================================

def _independence_mh(gpt2sft, corrupted, pos, log_q, start_tok, steps, gen, device):
    """Independence Metropolis with a FIXED proposal log_q (V,), exact AR energy
    S = joint_logprob. Returns (final_tok, accept_pct)."""
    probs = log_q.exp()
    cur_tok = start_tok
    cur_seq = corrupted.clone()
    cur_seq[0, pos] = cur_tok
    cur_S = joint_logprob(gpt2sft, cur_seq)
    n_acc = 0
    for _ in range(steps):
        prop = int(torch.multinomial(probs, 1, generator=gen).item())
        if prop == cur_tok:
            n_acc += 1
            continue
        tmp = cur_seq.clone()
        tmp[0, pos] = prop
        prop_S = joint_logprob(gpt2sft, tmp)
        log_a = (prop_S - cur_S) + (float(log_q[cur_tok].item()) - float(log_q[prop].item()))
        if float(torch.log(torch.rand((), generator=gen, device=device)).item()) < log_a:
            cur_tok, cur_seq, cur_S = prop, tmp, prop_S
            n_acc += 1
    return cur_tok, 100.0 * n_acc / steps


def exp_hybrid(args, sedd, gpt2sft, tokenizer, device):
    """The SEDD hybrid arm only (scale-specific): SEDD bidirectional proposal +
    exact gpt2sft energy, independence MH. The reference arms (left_conditional,
    dls_policy, dls_random) are scale-independent and produced once by hybrid_refs."""
    all_seqs = grid_corpus(args, tokenizer, device)
    seqs = shard_filter(all_seqs, args.shard_idx, args.num_shards)

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "pos", "arm", "gt_tok", "rec_tok", "exact", "top5",
                "avg_kl_gpt2sft", "accept_pct"])

    t0 = time.time()
    n = 0
    arm = f"hybrid_{args.scale}"
    for sidx, corrupted, mask_indices, orig_ids in seqs:
        pos = mask_indices[0]
        gt = int(orig_ids[0, pos].item())
        start_tok = int(corrupted[0, pos].item())
        q_sedd = sedd_lib.logpref_at(sedd, orig_ids, pos, args.sigma)      # bidirectional
        g = torch.Generator(device=device).manual_seed(args.data_seed + 60_000 + sidx)
        tok_h, acc_h = _independence_mh(gpt2sft, corrupted, pos, q_sedd, start_tok,
                                        args.mh_steps, g, device)
        top5 = int(gt in set(torch.topk(q_sedd, 5).indices.tolist()))
        fill = torch.tensor([tok_h], device=device)
        kl = avg_kl_for_fill(gpt2sft, orig_ids, mask_indices, fill)
        w.writerow([sidx, pos, arm, gt, tok_h, int(tok_h == gt), top5, kl, acc_h])
        n += 1
        if n % 20 == 0:
            print(f"[hybrid {args.scale} shard{args.shard_idx}] {n} seqs, "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
    f.close()
    return {"experiment": "hybrid", "scale": args.scale, "shard_idx": args.shard_idx,
            "num_shards": args.num_shards, "n_rows": n, "sigma": args.sigma,
            "mh_steps": args.mh_steps, "temperature_proposal": 1.0, "shard_done": True}


def exp_hybrid_refs(args, sedd, gpt2sft, tokenizer, device):
    """Scale-independent reference arms on the P1 sequence set, same task as hybrid:
    AR left-conditional independence MH (proposal p(x_m|x_<m)), DLS policy and DLS
    random (the AR-gradient and random arms). gpt2sft only; SEDD is not used."""
    from core.dls import DiscreteLangevinSampler

    all_seqs = grid_corpus(args, tokenizer, device)
    seqs = shard_filter(all_seqs, args.shard_idx, args.num_shards)
    eps = np.linspace(args.eps_start, args.eps_end, args.dls_steps)

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "pos", "arm", "gt_tok", "rec_tok", "exact", "top5",
                "avg_kl_gpt2sft", "accept_pct"])

    t0 = time.time()
    n = 0
    for sidx, corrupted, mask_indices, orig_ids in seqs:
        pos = mask_indices[0]
        gt = int(orig_ids[0, pos].item())
        start_tok = int(corrupted[0, pos].item())
        q_ar = _cond_logprob_at(gpt2sft, corrupted, pos)

        def emit(arm, rec_tok, accept):
            top5 = int(gt in set(torch.topk(q_ar, 5).indices.tolist()))
            fill = torch.tensor([rec_tok], device=device)
            kl = avg_kl_for_fill(gpt2sft, orig_ids, mask_indices, fill)
            w.writerow([sidx, pos, arm, gt, rec_tok, int(rec_tok == gt), top5, kl, accept])

        g = torch.Generator(device=device).manual_seed(args.data_seed + 61_000 + sidx)
        tok_l, acc_l = _independence_mh(gpt2sft, corrupted, pos, q_ar, start_tok,
                                        args.mh_steps, g, device)
        emit("left_conditional", tok_l, acc_l)

        for arm, method in [("dls_policy", "policy"), ("dls_random", "random")]:
            sampler = DiscreteLangevinSampler(
                model=gpt2sft, tokenizer=tokenizer, steps=args.dls_steps,
                temperature=args.temperature, oracle=False, method=method,
                mh_sampling=True, grad_normalization=True,
                noise_scale=args.noise_scale, epsilon_schedule=eps)
            seed_all(args.data_seed + sidx)
            s_hist, _ = sampler.optimize(corrupted.clone(), [pos], orig_ids.clone())
            emit(arm, int(s_hist[-1][0].item()), float("nan"))
        n += 1
        if n % 20 == 0:
            print(f"[hybrid_refs shard{args.shard_idx}] {n} seqs, "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
    f.close()
    return {"experiment": "hybrid_refs", "shard_idx": args.shard_idx,
            "num_shards": args.num_shards, "n_rows": n,
            "mh_steps": args.mh_steps, "dls_steps": args.dls_steps, "shard_done": True}


EXPERIMENTS = {"gates": exp_gates, "recovery": exp_recovery, "hybrid": exp_hybrid,
               "hybrid_refs": exp_hybrid_refs}
NO_SEDD = {"hybrid_refs"}   # experiments that use gpt2sft only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results_revision")
    p.add_argument("--scale", default="small", choices=["small", "medium"])
    p.add_argument("--gpt2sft_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")

    # corpus (kl_baselines-identical)
    p.add_argument("--data_file", default=None)
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=40)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--num_masks", type=int, default=1)
    p.add_argument("--data_seed", type=int, default=0)

    # sharding
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)

    # SEDD sampler
    p.add_argument("--predictor", default="euler")
    p.add_argument("--steps", type=int, default=128)      # recovery denoising steps
    p.add_argument("--sigma", type=float, default=0.1)    # readout noise level

    # gates
    p.add_argument("--n_gate", type=int, default=5)
    p.add_argument("--tail", type=int, default=16)

    # hybrid
    p.add_argument("--mh_steps", type=int, default=50)
    p.add_argument("--dls_steps", type=int, default=50)
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

    t0 = time.time()
    tokenizer, gpt2sft = load_tokenizer_and_model(args.gpt2sft_path, dtype=torch.float32)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gpt2sft.eval()
    for pp in gpt2sft.parameters():
        pp.requires_grad_(False)
    if args.exp in NO_SEDD:
        sedd = None
        print(f"[{args.exp}] gpt2sft loaded in {time.time()-t0:.1f}s (no SEDD)", flush=True)
    else:
        sedd = sedd_lib.load_sedd(args.scale, args.device)
        print(f"[{args.exp}] models loaded in {time.time()-t0:.1f}s "
              f"(gpt2sft + sedd-{args.scale})", flush=True)

    summary = EXPERIMENTS[args.exp](args, sedd, gpt2sft, tokenizer, args.device)
    summary["run_name"] = args.run_name
    summary["gpt2sft_path"] = args.gpt2sft_path
    summary["wall_time_sec"] = time.time() - t0
    summary["argv"] = vars(args)
    atomic_json(json_path, summary)
    print(f"[done] {json_path} ({summary['wall_time_sec']/60:.1f} min)")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("argv", "rows")},
                     indent=2)[:2000])


if __name__ == "__main__":
    main()
