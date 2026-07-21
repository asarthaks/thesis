#!/usr/bin/env python
"""
run_external_judge.py

CONCERN 3 step 2: break the circularity objection. The primary KL metric is
computed under the same model whose likelihood the thesis calls a bad quality
measure. Here we re-score the recovered sequences under an INDEPENDENT judge
(Llama-3) and check whether the method ranking survives. If policy, grad-norm-
preserved and random rank the same under a model that had no part in producing
them, "your metric comes from the broken object" stops being a valid objection.

Two stages, so the small generator model and the big judge never sit in VRAM at
once:

  --stage generate   loads the GPT-2 base, runs DLS recovery for
                     {policy, grad_norm_preserved_random_dir, random} on the SAME
                     grid sequences, and writes <run_name>_gen.csv with the
                     recovered TEXT (decoded), plus the in-model KL for reference.

  --stage judge      loads the judge model (e.g. Llama-3), reads a *_gen.csv, and
                     scores each recovered text: judge perplexity and judge
                     per-token NLL. Writes <run_name>.json with the per-method
                     means and the rank correlation against the in-model KL.

Schedule the generate job on a 24 GB card and the judge job on a >=40 GB card.
The judge job depends on the generate job's CSV existing; the manifest names them
so the queue runs generate first.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft
from run_experiment import load_texts, build_corruption, seed_all


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# stage: generate
# --------------------------------------------------------------------------

def stage_generate(args):
    import numpy as np
    from core.dls import DiscreteLangevinSampler

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if args.adapter_path:
        tok, model = load_tokenizer_and_model_peft(args.model_path, args.adapter_path, dtype=dtype)
    else:
        tok, model = load_tokenizer_and_model(args.model_path, dtype=dtype)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device

    eps = np.linspace(args.eps_start, args.eps_end, args.steps)
    methods = ["policy", "grad_norm_preserved_random_dir", "random"]

    texts = load_texts(args)
    gen_path = os.path.join(args.out_dir, args.run_name + "_gen.csv")
    f = open(gen_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["sample_idx", "method", "in_model_kl", "recovered_text", "gt_text"])

    for method in methods:
        sampler = DiscreteLangevinSampler(
            model=model, tokenizer=tok, steps=args.steps, temperature=args.temperature,
            oracle=False, method=method, mh_sampling=True, grad_normalization=True,
            noise_scale=args.noise_scale, epsilon_schedule=eps,
        )
        done, ti = 0, 0
        while done < args.n_samples and ti < len(texts):
            case = build_corruption(tok, texts[ti], args.num_masks, args.data_seed + ti, device)
            ti += 1
            if case is None:
                continue
            corrupted, mask_indices, orig_ids = case
            seed_all(args.data_seed + ti)
            _, metrics = sampler.optimize(corrupted.clone(), mask_indices, orig_ids.clone())
            final = metrics[-1]
            rec = orig_ids.clone()
            rec[0, torch.tensor(mask_indices, device=device)] = torch.tensor(
                final["token_ids"], device=device)
            rec_text = tok.decode(rec[0], skip_special_tokens=True).replace("\n", " ")
            gt_text = tok.decode(orig_ids[0], skip_special_tokens=True).replace("\n", " ")
            w.writerow([done, method, final["avg_kl_divergence"], rec_text, gt_text])
            done += 1
            if done % 50 == 0:
                print(f"[judge:generate] {method} {done}/{args.n_samples}", flush=True)
    f.close()

    # a marker json so the generate stage is itself resumable
    atomic_json(os.path.join(args.out_dir, args.run_name + ".json"),
                {"stage": "generate", "gen_csv": gen_path, "run_name": args.run_name,
                 "n_samples": args.n_samples})
    print(f"[judge:generate] wrote {gen_path}")


# --------------------------------------------------------------------------
# stage: judge
# --------------------------------------------------------------------------

@torch.no_grad()
def score_texts(judge, jtok, texts, device, max_len=128):
    """Return per-text (total_nll, n_tokens). Perplexity = exp(total_nll / n)."""
    out = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            out.append((float("nan"), 0)); continue
        ids = jtok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 2:
            out.append((float("nan"), 0)); continue
        lg = torch.log_softmax(judge(ids).logits[0, :-1, :].float(), dim=-1)
        tgt = ids[0, 1:]
        nll = -lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        out.append((nll, int(tgt.numel())))
    return out


def stage_judge(args):
    import pandas as pd
    from scipy.stats import spearmanr

    # find the generate CSV: either explicit, or the *_gen.csv matching the source run
    gen_csv = args.gen_csv
    if gen_csv is None:
        cands = glob.glob(os.path.join(args.out_dir, args.source_run + "_gen.csv"))
        if not cands:
            raise FileNotFoundError(
                f"no _gen.csv for source_run={args.source_run} in {args.out_dir}. "
                f"Run the matching --stage generate job first.")
        gen_csv = cands[0]
    df = pd.read_csv(gen_csv)

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    jtok, judge = load_tokenizer_and_model(args.judge_path, dtype=dtype)
    if jtok.pad_token_id is None:
        jtok.pad_token_id = jtok.eos_token_id
    judge.eval()
    device = next(judge.parameters()).device

    t0 = time.time()
    scored = score_texts(judge, jtok, df.recovered_text.tolist(), device, args.max_len)
    df["judge_nll"] = [s[0] for s in scored]
    df["judge_ntok"] = [s[1] for s in scored]
    df["judge_ppl"] = np.exp(df.judge_nll / df.judge_ntok.replace(0, np.nan))

    out_csv = os.path.join(args.out_dir, args.run_name + ".csv")
    df.to_csv(out_csv, index=False)

    summary = {"experiment": "external_judge", "judge_path": args.judge_path,
               "gen_csv": gen_csv, "by_method": {}}
    for m in df.method.unique():
        sub = df[df.method == m]
        summary["by_method"][m] = {
            "n": int(sub.judge_ppl.notna().sum()),
            "mean_judge_ppl": float(sub.judge_ppl.mean(skipna=True)),
            "median_judge_ppl": float(sub.judge_ppl.median(skipna=True)),
            "mean_judge_nll_per_tok": float((sub.judge_nll / sub.judge_ntok).mean(skipna=True)),
            "mean_in_model_kl": float(sub.in_model_kl.mean()),
        }
    # does the judge rank methods the same way the in-model KL does?
    piv = df.groupby("method").agg(kl=("in_model_kl", "mean"),
                                   ppl=("judge_ppl", "mean")).reset_index()
    if len(piv) >= 2:
        summary["rank_spearman_kl_vs_judge_ppl"] = float(
            spearmanr(piv.kl, piv.ppl)[0])
    summary["method_means"] = piv.to_dict(orient="records")
    summary["wall_time_sec"] = time.time() - t0
    atomic_json(os.path.join(args.out_dir, args.run_name + ".json"), summary)
    print(json.dumps(summary, indent=2)[:2000])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["generate", "judge"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results_judge")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--overwrite", action="store_true")

    # generate
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--data_file", default=None)
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=40)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--num_masks", type=int, default=1)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eps_start", type=float, default=10.5)
    p.add_argument("--eps_end", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=0.01)

    # judge
    p.add_argument("--judge_path", default=None, help="path to the judge model, e.g. Llama-3")
    p.add_argument("--source_run", default=None, help="run_name of the generate job")
    p.add_argument("--gen_csv", default=None, help="explicit path to a *_gen.csv")
    p.add_argument("--max_len", type=int, default=128)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} already exists")
        return

    if args.stage == "generate":
        assert args.model_path, "--model_path required for generate"
        stage_generate(args)
    else:
        assert args.judge_path, "--judge_path required for judge"
        stage_judge(args)


if __name__ == "__main__":
    main()
