#!/usr/bin/env python
"""
collect_traces.py

Runs EXPERIMENT 2 (MH acceptance conditioned on Voronoi boundary crossing) and
EXPERIMENT 3 (embedding-space trajectories) using the patched core/dls.py and
core/cls.py.

Both experiments come out of the same runs, because the patched samplers record
mh_log and traj_log simultaneously and neither recorder touches the RNG.

  python collect_traces.py \
      --model_path ./gpt2_sft_output \
      --core_path . \
      --run_name traces_gpt2sft \
      --out_dir results_diag \
      --n_seqs 200 --n_traj_seqs 6 --steps 50

Then:
  python analyze_mh.py --csv results_diag/traces_gpt2sft_mh.csv --fig_dir figures
  python plot_diagnostics.py --res_dir results_diag --fig_dir figures

IMPORTANT: run this with M = 1 masked position (--n_masks 1). With M > 1 the MH
accept/reject is a single joint decision over all positions, so "did the proposal
cross a boundary" stops being a clean binary and the conditional acceptance rate
becomes hard to interpret. The thesis already establishes that the qualitative
behaviour is preserved for M > 1, so M = 1 is the right setting for a mechanism
measurement.

CONFIGURATIONS RUN
------------------
  cls_policy_gnoff_mh    the paralysed sampler. THIS is the one the argument needs.
  cls_policy_gnoff_nomh  the same sampler with the correction removed, for contrast
  cls_policy_gnon_mh     the "rubber band" case: steps too small to leave the cell
  dls_policy_gn_mh       the control. MH works here. Reporting both numbers is
                         what makes the CLS number mean something.
"""

import argparse
import inspect
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------
# ADAPT HERE IF ANYTHING FAILS
# --------------------------------------------------------------------------
# These two helpers do the only things I cannot verify without base_sampler.py
# and run_experiment.py. Both introspect rather than assume, so they should work
# unchanged, but if they raise, fix them here and nothing else needs to change.

def build_sampler(cls, **wanted):
    """Instantiate `cls` passing only the kwargs its __init__ actually accepts."""
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters) - {"self"}
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    if has_kwargs:
        return cls(**wanted)
    kw = {k: v for k, v in wanted.items() if k in accepted}
    missing = [k for k, p in sig.parameters.items()
               if k != "self" and p.default is inspect.Parameter.empty and k not in kw]
    if missing:
        raise TypeError(
            f"{cls.__name__}.__init__ requires {missing}, which collect_traces.py "
            f"does not know how to supply. Add them to the `wanted` dict in main(), "
            f"or edit build_sampler(). Full signature: {sig}")
    return cls(**kw)


RUN_METHOD_CANDIDATES = ["run", "sample", "optimize", "run_sampler", "__call__"]


def find_run_method(sampler):
    for name in RUN_METHOD_CANDIDATES:
        fn = getattr(sampler, name, None)
        if callable(fn):
            return name, fn
    raise AttributeError(
        f"Could not find the entry point on {type(sampler).__name__}. Tried "
        f"{RUN_METHOD_CANDIDATES}. Add its name to RUN_METHOD_CANDIDATES.")


def call_run(fn, input_ids, mask_indices, steps):
    """Call the sampler's run method, matching whatever argument names it uses."""
    sig = inspect.signature(fn)
    p = set(sig.parameters)
    kw = {}
    if "steps" in p:
        kw["steps"] = steps
    elif "n_steps" in p:
        kw["n_steps"] = steps
    elif "K" in p:
        kw["K"] = steps
    if "mask_indices" in p:
        kw["mask_indices"] = mask_indices
    elif "positions" in p:
        kw["positions"] = mask_indices
    elif "mask_idx" in p:
        kw["mask_idx"] = mask_indices
    try:
        return fn(input_ids, **kw)
    except TypeError as e:
        raise TypeError(
            f"call_run() could not match the signature of the sampler entry point.\n"
            f"  signature: {sig}\n"
            f"  attempted kwargs: {list(kw)}\n"
            f"Edit call_run() in collect_traces.py. Original error: {e}")
# --------------------------------------------------------------------------


def load_sequences(dataset, tokenizer, n, min_tok, max_tok, seed):
    import random
    texts = []
    if os.path.isfile(dataset):
        with open(dataset) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset("wza/roc_stories", split="train")
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


CONFIGS = [
    # name                    sampler  method    mh     grad_norm
    ("cls_policy_gnoff_mh",   "cls",  "policy", True,  False),
    ("cls_policy_gnoff_nomh", "cls",  "policy", False, False),
    ("cls_policy_gnon_mh",    "cls",  "policy", True,  True),
    ("dls_policy_gn_mh",      "dls",  "policy", True,  True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="results_diag")
    ap.add_argument("--core_path", default=".")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dataset", default="roc_stories")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--n_seqs", type=int, default=200,
                    help="sequences used for the MH acceptance statistics")
    ap.add_argument("--n_traj_seqs", type=int, default=6,
                    help="subset for which the full state trajectory is stored")
    ap.add_argument("--n_masks", type=int, default=1,
                    help="KEEP THIS AT 1. see the docstring.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--min_tokens", type=int, default=15)
    ap.add_argument("--max_tokens", type=int, default=60)
    ap.add_argument("--eps_start", type=float, default=10.5)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.run_name + ".json")
    if os.path.exists(json_path) and not args.overwrite:
        print(f"[skip] {json_path} exists")
        return

    sys.path.insert(0, args.core_path)
    from core.dls import DiscreteLangevinSampler
    from core.cls import ContinuousLangevinSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dt)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    emb_matrix = model.get_input_embeddings().weight.detach()
    V = emb_matrix.shape[0]

    seqs = load_sequences(args.dataset, tok, args.n_seqs,
                          args.min_tokens, args.max_tokens, args.seed)
    print(f"{len(seqs)} sequences loaded")

    eps_schedule = np.linspace(args.eps_start, args.eps_end, args.steps)
    alpha_grid = np.logspace(-2, 2, 30)

    rng = np.random.RandomState(args.seed)
    all_mh, all_traj = [], []
    summary = {"experiment": "traces", "configs": {}}

    for cfg_name, kind, method, mh, gn in CONFIGS:
        Sampler = DiscreteLangevinSampler if kind == "dls" else ContinuousLangevinSampler
        wanted = dict(
            model=model, tokenizer=tok, emb_matrix=emb_matrix,
            method=method, mh_sampling=mh, grad_normalization=gn,
            oracle=False, temperature=args.temperature,
            noise_scale=args.noise_scale, alpha_grid=alpha_grid,
            epsilon_schedule=eps_schedule, device=args.device,
            steps=args.steps,
        )
        sampler = build_sampler(Sampler, **wanted)
        run_name, run_fn = find_run_method(sampler)
        print(f"[{cfg_name}] entry point: {type(sampler).__name__}.{run_name}()")

        sampler.mh_log = [] if mh else None
        sampler.traj_log = []

        t0 = time.time()
        for i, ids in enumerate(seqs):
            ids = ids.clone()
            L = ids.shape[0]
            if L < 6:
                continue
            pos = sorted(rng.choice(np.arange(2, L - 2),
                                    size=min(args.n_masks, L - 4), replace=False).tolist())
            corrupted = ids.clone()
            for p_ in pos:
                corrupted[p_] = int(rng.randint(0, V))

            sampler._diag_seq_id = i
            # only keep the full state trajectory for the first few sequences,
            # otherwise the npz gets enormous (n_seqs x steps x D floats)
            keep_traj = i < args.n_traj_seqs
            if not keep_traj:
                sampler.traj_log = None
            elif sampler.traj_log is None:
                sampler.traj_log = []

            call_run(run_fn, corrupted.unsqueeze(0).to(args.device),
                     pos, args.steps)

            if keep_traj and sampler.traj_log:
                for r in sampler.traj_log:
                    r["config"] = cfg_name
                all_traj.extend(sampler.traj_log)
                sampler.traj_log = []

            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  [{cfg_name}] {i+1}/{len(seqs)}  "
                      f"{el/60:.1f}m elapsed, eta {el/(i+1)*(len(seqs)-i-1)/60:.1f}m",
                      flush=True)

        if sampler.mh_log:
            for r in sampler.mh_log:
                r["config"] = cfg_name
            all_mh.extend(sampler.mh_log)
            d = pd.DataFrame(sampler.mh_log)
            cr = d[d.crossed == 1]
            st = d[d.crossed == 0]
            summary["configs"][cfg_name] = {
                "n_proposals": int(len(d)),
                "accept_rate_overall": float(d.accepted.mean()),
                "accept_rate_crossed": float(cr.accepted.mean()) if len(cr) else None,
                "accept_rate_stayed": float(st.accepted.mean()) if len(st) else None,
                "n_accepted_and_crossed": int(((d.crossed == 1) & (d.accepted == 1)).sum()),
                "mean_log_target_ratio_crossed": float(cr.log_target_ratio.mean()) if len(cr) else None,
                "mean_log_proposal_ratio_crossed": float(cr.log_proposal_ratio.mean()) if len(cr) else None,
            }
            print(f"  [{cfg_name}] accept overall {d.accepted.mean():.4f}  "
                  f"| crossed {summary['configs'][cfg_name]['accept_rate_crossed']}  "
                  f"| stayed {summary['configs'][cfg_name]['accept_rate_stayed']}")

    # ---- write everything ----
    mh_csv = os.path.join(args.out_dir, args.run_name + "_mh.csv")
    pd.DataFrame(all_mh).to_csv(mh_csv, index=False)
    print("wrote", mh_csv)

    # trajectories: one npz, grouped by config
    traj_out = {}
    for cfg_name, _, _, _, _ in CONFIGS:
        recs = [r for r in all_traj if r["config"] == cfg_name]
        if not recs:
            continue
        by_seq = {}
        for r in recs:
            by_seq.setdefault(r["seq_id"], []).append(r)
        seq_ids = sorted(by_seq)
        states = np.stack([np.stack([x["state"][0] for x in sorted(by_seq[s], key=lambda z: z["step"])])
                           for s in seq_ids])                          # nseq x T x D
        dists = np.stack([np.array([x["dist_to_manifold"][0] for x in sorted(by_seq[s], key=lambda z: z["step"])])
                          for s in seq_ids])                           # nseq x T
        cells = np.stack([np.array([x["token_ids"][0] for x in sorted(by_seq[s], key=lambda z: z["step"])])
                          for s in seq_ids])
        traj_out[f"{cfg_name}__states"] = states.astype(np.float32)
        traj_out[f"{cfg_name}__dist_to_manifold"] = dists.astype(np.float32)
        traj_out[f"{cfg_name}__cell_id"] = cells.astype(np.int32)

    vsub = emb_matrix[torch.from_numpy(
        np.random.RandomState(0).choice(V, min(6000, V), replace=False)
    ).to(emb_matrix.device)].float().cpu().numpy()

    npz = os.path.join(args.out_dir, args.run_name + "_traj.npz")
    np.savez_compressed(npz, vocab_embeddings=vsub, **traj_out)
    print("wrote", npz)

    summary["run_name"] = args.run_name
    summary["n_seqs"] = len(seqs)
    summary["steps"] = args.steps
    summary["n_masks"] = args.n_masks
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, json_path)
    print("wrote", json_path)
    print(json.dumps(summary["configs"], indent=2))


if __name__ == "__main__":
    main()
