#!/usr/bin/env python3
"""
run_experiment.py

One uniform entry point for every Langevin infilling experiment in the thesis,
built on the corrected unified core/ samplers. It runs a single (sampler, method,
mh, grad_norm, steps, oracle, model) configuration over a set of corrupted
sentences, aggregates the trajectories, logs everything to Weights & Biases, and
also drops a local JSON so you can replot offline.

The point is that the tmux launcher can fan out the whole thesis matrix by calling
this script with different flags, and every run lands in the same wandb project
with a consistent schema, so overlaying "With MH vs No MH" or "policy vs random"
is just a group-by in the wandb UI.

Requires the two patches applied first:
  - core/dls.py            symmetric random-walk MH fix
  - core/base_sampler.py   injectable epsilon_schedule

Example:
  python run_experiment.py \
    --sampler dls --method policy --mh --grad_norm --steps 50 \
    --model_path /path/to/gpt2_large_sft_output --model_tag gpt2-large \
    --eps_start 10.5 --eps_end 0.1 --n_samples 200 \
    --wandb_project ctg-langevin-thesis --wandb_group dls_traj_50 \
    --run_name gpt2-large.dls.policy.mh
"""

import os
import json
import argparse
import random

import numpy as np
import torch

from core.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft
from core.dls import DiscreteLangevinSampler
from core.cls import ContinuousLangevinSampler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_corruption(tokenizer, text, num_masks, seed, device):
    """Deterministic mask + random-token corruption for one sentence."""
    rng = np.random.RandomState(seed)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    if seq_len < num_masks + 3:
        return None
    valid = list(range(1, seq_len - 1))
    if num_masks > len(valid):
        return None
    mask_indices = sorted(rng.choice(valid, size=num_masks, replace=False).tolist())
    orig_ids = input_ids.clone()
    corrupted = input_ids.clone()
    vocab = tokenizer.vocab_size
    for idx in mask_indices:
        orig = input_ids[0, idx].item()
        r = int(rng.randint(0, vocab))
        while r == orig:
            r = int(rng.randint(0, vocab))
        corrupted[0, idx] = r
    return corrupted, mask_indices, orig_ids


def load_texts(args):
    """WikiText-2 validation by default; a plain one-sentence-per-line file otherwise
    (use --data_file for ROCStories or any custom set)."""
    if args.data_file:
        with open(args.data_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x["text"].strip() for x in ds]
    # word-count filter to get proper sentences, same spirit as the drivers
    texts = [t for t in texts if args.min_words < len(t.split()) < args.max_words]
    return texts


def make_sampler(args, model, tokenizer):
    eps_schedule = np.linspace(args.eps_start, args.eps_end, args.steps)
    common = dict(
        model=model,
        tokenizer=tokenizer,
        steps=args.steps,
        temperature=args.temperature,
        oracle=args.oracle,
        method=args.method,
        mh_sampling=args.mh,
        grad_normalization=args.grad_norm,
        noise_scale=args.noise_scale,
        epsilon_schedule=eps_schedule,
    )
    if args.sampler == "dls":
        return DiscreteLangevinSampler(**common)
    return ContinuousLangevinSampler(**common)


def _safe(x):
    return float(x) if x is not None else float("nan")


def main():
    p = argparse.ArgumentParser()
    # what to run
    p.add_argument("--sampler", choices=["dls", "cls"], required=True)
    p.add_argument("--method", default="policy",
                   choices=["policy", "grad_norm_preserved_random_dir", "random"])
    p.add_argument("--mh", action="store_true")
    p.add_argument("--grad_norm", action="store_true")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--steps", type=int, default=50)

    # model + data
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_tag", required=True, help="short label e.g. gpt2-large or llama3-8b")
    p.add_argument("--peft_checkpoint", default=None,
                   help="LoRA adapter dir to load on top of model_path (for GFlowNet checkpoints). "
                        "The adapter is merged into the base, so the sampler sees a plain model.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--data_file", default=None, help="plain text, one sentence per line; overrides WikiText-2")
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=40)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--num_masks", type=int, default=1)
    p.add_argument("--data_seed", type=int, default=0, help="offsets the per-sentence corruption seeds")

    # sampler knobs
    p.add_argument("--eps_start", type=float, required=True)
    p.add_argument("--eps_end", type=float, required=True)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=0.01)

    # logging
    p.add_argument("--wandb_project", default="ctg-langevin-thesis")
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--out_dir", default="results_rerun")
    p.add_argument("--overwrite", action="store_true",
                   help="redo a run even if its result JSON already exists (default: skip)")

    args = p.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    seed_all(1234)

    run_name = args.run_name or f"{args.model_tag}.{args.sampler}.{args.method}." \
                                f"{'mh' if args.mh else 'nomh'}.{'gn' if args.grad_norm else 'nogn'}." \
                                f"{'oracle' if args.oracle else 'free'}.s{args.steps}"

    # Resume: if this run already produced a complete result, skip it before doing any
    # expensive work (no model load, no wandb). The JSON is written atomically at the end,
    # so its presence means the run finished. This is what makes the queue restartable.
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, run_name + ".json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[{run_name}] already done ({out_path}); skipping. --overwrite to redo.")
        return

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"[warn] wandb disabled ({e})")
            use_wandb = False

    print(f"[{run_name}] loading {args.model_path} ({args.dtype}) ...", flush=True)
    if args.peft_checkpoint:
        print(f"[{run_name}]   + merging GFlowNet adapter {args.peft_checkpoint}", flush=True)
        tokenizer, model = load_tokenizer_and_model_peft(args.model_path, args.peft_checkpoint, dtype=dtype)
    else:
        tokenizer, model = load_tokenizer_and_model(args.model_path, dtype=dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    device = next(model.parameters()).device

    texts = load_texts(args)
    print(f"[{run_name}] {len(texts)} candidate sentences", flush=True)

    n_steps = args.steps
    sum_l2 = np.zeros(n_steps)
    sum_kl = np.zeros(n_steps)
    sum_ent = np.zeros(n_steps)
    cnt = np.zeros(n_steps)
    rejects = 0
    reject_slots = 0
    correct = 0
    total_masks = 0
    examples = []
    csv_rows = []   # per-sample rows in the legacy schema the notebook's prepare_dataset reads

    # Build the sampler ONCE. Its config is constant across sentences and it holds
    # a copy of the embedding matrix, so rebuilding per sample would re-copy that
    # (multiple GB on Llama) for nothing. optimize() takes the input per call.
    sampler = make_sampler(args, model, tokenizer)

    done = 0
    ti = 0
    while done < args.n_samples and ti < len(texts):
        case = build_corruption(tokenizer, texts[ti], args.num_masks, args.data_seed + ti, device)
        ti += 1
        if case is None:
            continue
        corrupted, mask_indices, orig_ids = case

        seed_all(args.data_seed + ti)  # reproducible sampling per sentence
        _, metrics = sampler.optimize(corrupted.clone(), mask_indices, orig_ids.clone())

        gt = orig_ids[0, mask_indices].tolist()
        final_tokens = metrics[-1]["token_ids"]
        sample_correct = sum(int(pi == ti_) for pi, ti_ in zip(final_tokens, gt))
        correct += sample_correct
        total_masks += len(gt)

        for k, m in enumerate(metrics[:n_steps]):
            cnt[k] += 1
            sum_l2[k] += _safe(m.get("avg_l2_distance"))
            sum_kl[k] += _safe(m.get("avg_kl_divergence"))
            sum_ent[k] += _safe(m.get("entropy"))
            if m.get("mh_rejected"):
                rejects += 1
            reject_slots += 1

        # per-sample record with the exact keys the notebook expects
        compact_traj = [{
            "step": m.get("step", k),
            "avg_l2_distance": _safe(m.get("avg_l2_distance")),
            "avg_kl_divergence": _safe(m.get("avg_kl_divergence")),
            "entropy": _safe(m.get("entropy")),
        } for k, m in enumerate(metrics)]
        final = metrics[-1]
        csv_rows.append({
            "sample_idx": done,
            "method": args.method,
            "mh_enabled": bool(args.mh),
            "trajectory": str(compact_traj),
            "avg_l2_dist": _safe(final.get("avg_l2_distance")),
            "avg_kl_div": _safe(final.get("avg_kl_divergence")),
            "accuracy_pct": 100.0 * sample_correct / max(len(gt), 1),
        })

        if len(examples) < 8:
            examples.append([
                tokenizer.decode(orig_ids[0], skip_special_tokens=True)[:80],
                tokenizer.decode(corrupted[0], skip_special_tokens=True)[:80],
                " | ".join(tokenizer.decode([t]) for t in final_tokens),
            ])

        done += 1
        if done % 25 == 0:
            print(f"[{run_name}] {done}/{args.n_samples}", flush=True)

    cnt = np.maximum(cnt, 1)
    mean_l2 = sum_l2 / cnt
    mean_kl = sum_kl / cnt
    mean_ent = sum_ent / cnt
    accuracy = 100.0 * correct / max(total_masks, 1)
    accept_rate = 100.0 * (1 - rejects / max(reject_slots, 1)) if args.mh else float("nan")

    print(f"[{run_name}] DONE  n={done}  acc={accuracy:.2f}%  "
          f"final_kl={mean_kl[-1]:.3f}  final_l2={mean_l2[-1]:.3f}  "
          f"accept={accept_rate:.1f}%", flush=True)

    # local dump for offline replotting. Write to a temp file and rename, so a job that
    # is killed mid-write never leaves a partial JSON that the queue would treat as done.
    out = {
        "run_name": run_name,
        "config": vars(args),
        "n": done,
        "accuracy": accuracy,
        "accept_rate": accept_rate,
        "mean_l2": mean_l2.tolist(),
        "mean_kl": mean_kl.tolist(),
        "mean_entropy": mean_ent.tolist(),
        "examples": examples,
    }
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[{run_name}] wrote {out_path}", flush=True)

    # per-sample CSV in the exact schema UnifiedResultAnalysis.ipynb reads, so the same
    # prepare_dataset/plot_graphs code can render these runs. Written atomically.
    import pandas as pd
    csv_path = os.path.join(args.out_dir, run_name + ".csv")
    tmp_csv = csv_path + ".tmp"
    pd.DataFrame(csv_rows).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)
    print(f"[{run_name}] wrote {csv_path}", flush=True)

    if use_wandb:
        import wandb
        for k in range(n_steps):
            wandb.log({"step": k, "l2": mean_l2[k], "kl": mean_kl[k], "entropy": mean_ent[k]})
        wandb.run.summary["accuracy"] = accuracy
        wandb.run.summary["final_kl"] = float(mean_kl[-1])
        wandb.run.summary["final_l2"] = float(mean_l2[-1])
        wandb.run.summary["accept_rate"] = accept_rate
        wandb.run.summary["n_samples_used"] = done
        tbl = wandb.Table(columns=["original", "corrupted", "recovered"], data=examples)
        wandb.log({"examples": tbl})
        wandb.finish()


if __name__ == "__main__":
    main()
