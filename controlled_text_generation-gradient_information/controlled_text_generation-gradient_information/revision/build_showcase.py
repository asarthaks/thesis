#!/usr/bin/env python
"""
build_showcase.py  -  Phase 5 Stage 1b: the qualitative recovery showcase.

Immutable selection rule (stated in the table caption): with np.random.default_rng(0),
draw 10 sample_idx from the kl_baselines sequence set via choice(200, 10, replace=False),
NO filtering, NO redrawing, failures stay in. Drawn indices (sorted): 3, 8, 14, 34, 52,
60, 98, 122, 162, 199.

For each showcase sequence: original text, corrupted text (corrupted token marked), and
the single-token recovery of each method:
  dls_policy (mh, gn), dls_random, CLS flagship (policy mh gn free s50), gibbs,
  cond_argmax, cond_topk_rescore, left_conditional independence MH,
  sedd_recovery small and medium, hybrid_medium.

Recovered tokens come from the stored per-item CSVs where present (SEDD small/medium,
hybrid_medium, left_conditional, dls_policy, dls_random); the AR conditional baselines
(cond_argmax, cond_topk_rescore, gibbs) and CLS flagship are regenerated on ONLY the 10
showcase sequences. The corruption is deterministic (run_experiment.build_corruption),
so the regeneration is exact; we VERIFY it two ways before trusting it: (1) the
regenerated corrupted token and gt token match every stored CSV row for that sequence,
and (2) the regenerated avg_kl for cond_argmax / cond_topk_rescore / gibbs matches the
stored rev_klbase CSV and the regenerated CLS avg_kl matches the grid CSV, to a tolerance.

Writes results_revision/qualitative_showcase.json and a LaTeX longtable appendix
(the filled position rendered in bold).
"""

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "diagnostics")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_experiment import load_texts, build_corruption, make_sampler, seed_all
from core.prep import load_tokenizer_and_model
from run_revision import avg_kl_for_fill, joint_logprob

GPT2SFT = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"
KL_TOL = 1e-2   # avg_kl match tolerance for the regeneration verification

SHOWCASE = sorted(np.random.default_rng(0).choice(200, size=10, replace=False).tolist())


def reconstruct_corruptions(args, tokenizer, device, want):
    """Replay iter_grid_samples tracking (sample_idx -> ti) so CLS seeding matches the
    grid's seed_all(data_seed + ti). Returns {sample_idx: (ti, corrupted, mask, orig)}."""
    texts = load_texts(args)
    out, done, ti = {}, 0, 0
    while done < args.n_samples and ti < len(texts):
        case = build_corruption(tokenizer, texts[ti], args.num_masks,
                                args.data_seed + ti, device)
        cur_ti = ti
        ti += 1
        if case is None:
            continue
        if done in want:
            corrupted, mask_indices, orig_ids = case
            out[done] = (cur_ti, corrupted, mask_indices, orig_ids)
        done += 1
        if len(out) == len(want):
            break
    return out


def regen_ar_baselines(model, tokenizer, corrupted, mask_indices, orig_ids, sidx,
                       topk, gibbs_sweeps, device):
    """cond_argmax, cond_topk_rescore, gibbs single-token fills, replicating
    run_revision.exp_kl_baselines exactly (same seeds)."""
    mt = torch.tensor(mask_indices, device=device)
    gt_fill = orig_ids[0, mt].clone()
    corrupt_fill = corrupted[0, mt].clone()
    with torch.no_grad():
        logits_c = model(corrupted).logits[0].float()
    cond_argmax_fill = corrupt_fill.clone()
    cond_topk_fill = corrupt_fill.clone()
    for j, m in enumerate(mask_indices):
        dist = torch.log_softmax(logits_c[m - 1], dim=-1)
        cond_argmax_fill[j] = int(dist.argmax().item())
        k = min(topk, dist.numel())
        topk_idx = torch.topk(dist, k).indices
        best_lp, best_tok = -1e30, int(topk_idx[0].item())
        for cand in topk_idx.tolist():
            tmp = corrupted.clone(); tmp[0, m] = cand
            lp = joint_logprob(model, tmp)
            if lp > best_lp:
                best_lp, best_tok = lp, cand
        cond_topk_fill[j] = best_tok
    # metropolized gibbs (seeded generator 20000+sidx)
    cur = corrupted.clone()
    cur_E = -joint_logprob(model, cur)
    grng = torch.Generator(device=device).manual_seed(20_000 + sidx)
    for _ in range(gibbs_sweeps):
        for j, m in enumerate(mask_indices):
            with torch.no_grad():
                dist = torch.log_softmax(model(cur).logits[0, m - 1].float(), dim=-1)
            probs = dist.exp()
            prop = int(torch.multinomial(probs, 1, generator=grng).item())
            cur_tok = int(cur[0, m].item())
            if prop == cur_tok:
                continue
            tmp = cur.clone(); tmp[0, m] = prop
            prop_E = -joint_logprob(model, tmp)
            log_acc = -(prop_E - cur_E) + (float(dist[cur_tok]) - float(dist[prop]))
            if float(torch.log(torch.rand((), generator=grng, device=device))) < log_acc:
                cur, cur_E = tmp, prop_E
    gibbs_fill = cur[0, mt].clone()
    fills = {"cond_argmax": cond_argmax_fill, "cond_topk_rescore": cond_topk_fill,
             "gibbs": gibbs_fill}
    kls = {k: avg_kl_for_fill(model, orig_ids, mask_indices, v) for k, v in fills.items()}
    return {k: int(v[0].item()) for k, v in fills.items()}, kls


def run_cls_flagship(model, tokenizer, corrupted, mask_indices, orig_ids, sample_seed, device):
    """CLS flagship = gpt2-large.cls.policy.mh.gn.free.s50 (eps 10.5->0.1, temp 5.0,
    noise 0.01). run_experiment seeds sampling with seed_all(data_seed + ti) AFTER the
    `ti += 1` that follows build_corruption, so the sampling seed is data_seed + ti + 1
    (one more than the corruption seed). sample_seed carries that exact value."""
    cargs = SimpleNamespace(sampler="cls", method="policy", mh=True, grad_norm=True,
                            oracle=False, steps=50, temperature=5.0, noise_scale=0.01,
                            eps_start=10.5, eps_end=0.1)
    sampler = make_sampler(cargs, model, tokenizer)
    seed_all(sample_seed)
    _, metrics = sampler.optimize(corrupted.clone(), mask_indices, orig_ids.clone())
    rec = int(metrics[-1]["token_ids"][0])
    kl = float(metrics[-1].get("avg_kl_divergence", float("nan")))
    return rec, kl


def marked_text(tokenizer, ids_list, pos, open_="[[", close_="]]"):
    """Decode full sequence with the token at `pos` wrapped in markers."""
    parts = []
    for i, t in enumerate(ids_list):
        piece = tokenizer.decode([t])
        if i == pos:
            piece = f"{open_}{piece.strip() or piece}{close_}"
        parts.append(piece)
    return "".join(parts).replace("\n", " ").strip()


def read_stored(csv_path, arm_col=None):
    df = pd.read_csv(csv_path)
    return df


def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--gpt2sft_path", default=GPT2SFT)
    ap.add_argument("--grid_dir", default="results_gpt2_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gibbs_sweeps", type=int, default=3)
    ap.add_argument("--tex_out", default="Doc/chapters/showcase_table.tex")
    args = ap.parse_args()

    device = args.device
    tokenizer, model = load_tokenizer_and_model(args.gpt2sft_path, dtype=torch.float32)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dargs = SimpleNamespace(data_file=None, min_words=10, max_words=40,
                            n_samples=200, num_masks=1, data_seed=0)
    corrs = reconstruct_corruptions(dargs, tokenizer, device, set(SHOWCASE))
    print(f"[showcase] reconstructed {len(corrs)} corruptions for {SHOWCASE}", flush=True)

    # stored rec_tok tables
    hyb = pd.read_csv(os.path.join(args.out_dir, "rev_sedd_hybrid.csv"))
    rec_s = pd.read_csv(os.path.join(args.out_dir, "rev_sedd_recovery_small.csv"))
    rec_m = pd.read_csv(os.path.join(args.out_dir, "rev_sedd_recovery_medium.csv"))
    klbase = pd.read_csv(os.path.join(args.out_dir, "rev_klbase_gpt2sft.csv"))

    def stored_hyb(sidx, arm):
        r = hyb[(hyb.sample_idx == sidx) & (hyb.arm == arm)]
        return None if r.empty else r.iloc[0]

    def stored_rec(df, sidx):
        r = df[df.sample_idx == sidx]
        return None if r.empty else r.iloc[0]

    def stored_klbase(sidx, name):
        r = klbase[(klbase.sample_idx == sidx) & (klbase.baseline == name)]
        return None if r.empty else float(r.iloc[0].avg_kl)

    grid_cls = pd.read_csv(os.path.join(args.grid_dir,
                           "gpt2-large.cls.policy.mh.gn.free.s50.csv"))

    verify = {"ok": [], "warn": []}
    rows = []
    for sidx in SHOWCASE:
        ti, corrupted, mask_indices, orig_ids = corrs[sidx]
        pos = mask_indices[0]
        gt_tok = int(orig_ids[0, pos].item())
        corr_tok = int(corrupted[0, pos].item())
        ids = orig_ids[0].tolist()
        corr_ids = corrupted[0].tolist()

        methods = {}

        # ---- stored-rec_tok arms; verify gt_tok + pos consistency ----
        for arm, label in [("dls_policy", "dls_policy"), ("dls_random", "dls_random"),
                           ("left_conditional", "left_conditional"),
                           ("hybrid_medium", "hybrid_medium")]:
            r = stored_hyb(sidx, arm)
            if r is None:
                methods[label] = None; continue
            if int(r.gt_tok) != gt_tok or int(r.pos) != pos:
                verify["warn"].append(f"{arm} s{sidx}: stored gt/pos "
                                      f"({int(r.gt_tok)},{int(r.pos)}) != regen ({gt_tok},{pos})")
            else:
                verify["ok"].append(f"{arm} s{sidx} gt/pos match")
            methods[label] = int(r.rec_tok)
        for df, label in [(rec_s, "sedd_small"), (rec_m, "sedd_medium")]:
            r = stored_rec(df, sidx)
            if r is None:
                methods[label] = None; continue
            if int(r.gt_tok) != gt_tok or int(r.pos) != pos:
                verify["warn"].append(f"{label} s{sidx}: stored gt/pos mismatch")
            else:
                verify["ok"].append(f"{label} s{sidx} gt/pos match")
            methods[label] = int(r.rec_tok)

        # ---- regenerated AR baselines; verify avg_kl vs stored rev_klbase ----
        ar_toks, ar_kls = regen_ar_baselines(model, tokenizer, corrupted, mask_indices,
                                             orig_ids, sidx, args.topk, args.gibbs_sweeps, device)
        for name in ("cond_argmax", "cond_topk_rescore", "gibbs"):
            methods[name] = ar_toks[name]
            stored = stored_klbase(sidx, name)
            if stored is not None and abs(stored - ar_kls[name]) > KL_TOL:
                verify["warn"].append(f"{name} s{sidx}: regen KL {ar_kls[name]:.4f} != "
                                      f"stored {stored:.4f}")
            elif stored is not None:
                verify["ok"].append(f"{name} s{sidx} KL match ({ar_kls[name]:.3f})")

        # ---- CLS flagship; verify avg_kl vs grid csv ----
        cls_tok, cls_kl = run_cls_flagship(model, tokenizer, corrupted, mask_indices,
                                           orig_ids, dargs.data_seed + ti + 1, device)
        methods["cls_flagship"] = cls_tok
        gr = grid_cls[grid_cls.sample_idx == sidx]
        if not gr.empty:
            gk = float(gr.iloc[0].avg_kl_div)
            if abs(gk - cls_kl) > KL_TOL:
                verify["warn"].append(f"cls s{sidx}: regen KL {cls_kl:.4f} != grid {gk:.4f}")
            else:
                verify["ok"].append(f"cls s{sidx} KL match ({cls_kl:.3f})")

        row = {
            "sample_idx": sidx, "pos": pos, "gt_tok": gt_tok, "corr_tok": corr_tok,
            "gt_token_str": tokenizer.decode([gt_tok]),
            "corr_token_str": tokenizer.decode([corr_tok]),
            "original_text": tokenizer.decode(ids).replace("\n", " ").strip(),
            "corrupted_text": marked_text(tokenizer, corr_ids, pos),
            "methods": {},
        }
        for name, tok in methods.items():
            if tok is None:
                row["methods"][name] = None; continue
            filled = ids.copy(); filled[pos] = tok
            row["methods"][name] = {
                "rec_tok": tok, "rec_token_str": tokenizer.decode([tok]),
                "exact": int(tok == gt_tok),
                "recovered_text": marked_text(tokenizer, filled, pos),
            }
        rows.append(row)
        print(f"[showcase] s{sidx} done (pos {pos}, gt '{row['gt_token_str']}')", flush=True)

    out = {"selection_rule": "np.random.default_rng(0).choice(200,10,replace=False), "
                             "sorted, no filtering",
           "showcase_sample_idx": SHOWCASE, "corpus": "kl_baselines (WikiText-2 val, "
           "num_masks 1, data_seed 0)", "n_verify_ok": len(verify["ok"]),
           "n_verify_warn": len(verify["warn"]), "verify_warn": verify["warn"],
           "rows": rows}
    op = os.path.join(args.out_dir, "qualitative_showcase.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[showcase] wrote {op}")
    print(f"[verify] ok={len(verify['ok'])} warn={len(verify['warn'])}")
    for wln in verify["warn"]:
        print("  WARN:", wln)


if __name__ == "__main__":
    main()
