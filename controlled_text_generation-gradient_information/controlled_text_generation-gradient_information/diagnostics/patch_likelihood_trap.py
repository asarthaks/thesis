#!/usr/bin/env python
"""
patch_likelihood_trap.py

Rewrites exp_likelihood_trap in run_diagnostic.py IN PLACE.

WHY
---
Your run showed every len_beta=0 generation pinned at length 40 with zero
variance, even under ancestral sampling. That is not a truncation nuisance. It
means these models assign near-zero probability to EOS: the length-collapse
training taught them never to stop. So free-running length is itself one of the
pathologies under study, which makes it unusable as a clean experimental axis.

WHAT THIS CHANGES
-----------------
1. Records whether EOS was emitted, and the position of the FIRST eos. The
   fraction of generations that never stop is itself a clean quantitative
   measure of the length pathology, and it goes in the thesis.

2. Scores only up to the first EOS when one exists, so a generation that DID
   stop is not contaminated by post-EOS padding.

3. Flags generations that ran to the cap (hit_cap). Repetition and distinct-n
   are computed on the pre-EOS span so a mid-word truncation does not inflate
   the repetition score.

4. Computes the length-vs-logp regression ONLY within strategies whose length
   actually varies (std > 0 and at least 20 distinct lengths' worth of points),
   and reports it per-strategy plus a pooled fit over the strategies that
   qualify. If nothing qualifies it records nan and says why, rather than
   crashing.

5. Adds eos_token_id to generate(). Harmless where EOS already works, and it
   lets any model that CAN stop actually stop.

Run:
    python patch_likelihood_trap.py            # patches ./run_diagnostic.py
    python patch_likelihood_trap.py --path /abs/path/run_diagnostic.py
Then re-run ONLY this experiment (about 20 min):
    python run_diagnostic.py --exp likelihood_trap --run_name diag_likelihood_trap_gpt2sft \\
        --model_path ./gpt2_sft_output --out_dir results_diag --overwrite --max_new_tokens 60
"""

import argparse
import os
import shutil
import sys


NEW_FUNC = '''def exp_likelihood_trap(args, model, tok, device):
    """
    Decode with several strategies and show that the LOWEST energy text
    (highest likelihood) is the WORST text. This is what makes -log p a bad
    energy function in the statistical sense, independent of any sampler.

    Length is treated as a MEASURED pathology, not a clean variable: some models
    (len_beta=0) have EOS trained out of them and never stop, so we record the
    stop behaviour explicitly and never assume free-running length is meaningful.
    """
    import csv
    from collections import Counter

    seqs = load_sequences(args.dataset, tok, args.n_seqs,
                          args.min_tokens, args.max_tokens, args.seed)

    eos_id = tok.eos_token_id

    strategies = {
        "greedy":       dict(do_sample=False, num_beams=1),
        "beam5":        dict(do_sample=False, num_beams=5),
        "beam20":       dict(do_sample=False, num_beams=20),
        "ancestral":    dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
        "topp90":       dict(do_sample=True, temperature=1.0, top_p=0.9),
        "temp07":       dict(do_sample=True, temperature=0.7, top_k=0, top_p=1.0),
    }

    def repetition_rate(ids, n=4):
        if len(ids) < n:
            return 0.0
        grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        c = Counter(grams)
        rep = sum(v - 1 for v in c.values())
        return rep / max(1, len(grams))

    def distinct_n(ids, n=2):
        if len(ids) < n:
            return 0.0
        grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        return len(set(grams)) / max(1, len(grams))

    csv_path = os.path.join(args.out_dir, args.run_name + ".csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["seq_id", "strategy", "gen_len", "scored_len",
                "emitted_eos", "hit_cap", "total_logp", "mean_logp",
                "rep4", "distinct2", "text"])

    for si, ids in enumerate(seqs):
        L = ids.shape[0]
        cut = max(4, L // 2)
        prompt = ids[:cut].unsqueeze(0).to(device)

        for name, kw in strategies.items():
            gkw = dict(kw)
            if eos_id is not None:
                gkw["eos_token_id"] = eos_id
            with torch.no_grad():
                out = model.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    pad_token_id=tok.pad_token_id,
                    **gkw,
                )
            gen = out[0, cut:]
            raw_len = int(gen.numel())
            if raw_len == 0:
                continue

            # locate the FIRST eos in the generated span
            gen_list = gen.tolist()
            emitted_eos = int(eos_id is not None and eos_id in gen_list)
            if emitted_eos:
                stop = gen_list.index(eos_id)           # score up to, excluding, eos
            else:
                stop = raw_len
            hit_cap = int((not emitted_eos) and raw_len >= args.max_new_tokens)

            scored_ids = gen_list[:stop]
            scored_len = len(scored_ids)
            if scored_len == 0:                          # emitted eos immediately
                # still record it: an empty generation is a length-collapse datum
                w.writerow([si, name, raw_len, 0, emitted_eos, hit_cap,
                            0.0, 0.0, 0.0, 0.0, ""])
                continue

            # per-token log-probs of the SCORED span only
            full = out[:, :cut + stop].to(device)
            with torch.no_grad():
                o = model(input_ids=full)
                lg = torch.log_softmax(o.logits[:, :-1, :].float(), dim=-1)
                tg = full[:, 1:]
                tok_lp = lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)[0]
            gen_lp = tok_lp[cut - 1:]                     # the generated positions
            total_lp = float(gen_lp.sum().item())
            mean_lp = total_lp / max(1, scored_len)

            w.writerow([si, name, raw_len, scored_len, emitted_eos, hit_cap,
                        total_lp, mean_lp,
                        repetition_rate(scored_ids), distinct_n(scored_ids),
                        tok.decode(scored_ids).replace("\\n", " ")[:300]])

        if (si + 1) % 20 == 0:
            print(f"[likelihood_trap] {si+1}/{len(seqs)}", flush=True)

    f.close()

    import pandas as pd
    import numpy as np
    from scipy.stats import linregress
    df = pd.read_csv(csv_path)

    summary = {"experiment": "likelihood_trap", "n_sequences": len(seqs),
               "by_strategy": {}}

    # overall stop behaviour: THIS is a headline number for the length pathology.
    summary["frac_emitted_eos"] = float(df.emitted_eos.mean())
    summary["frac_hit_cap"] = float(df.hit_cap.mean())

    for name in df.strategy.unique():
        sub = df[df.strategy == name]
        if len(sub) == 0:
            continue
        summary["by_strategy"][name] = {
            "n": int(len(sub)),
            "mean_gen_len": float(sub.gen_len.mean()),
            "std_gen_len": float(sub.gen_len.std()),
            "mean_scored_len": float(sub.scored_len.mean()),
            "frac_emitted_eos": float(sub.emitted_eos.mean()),
            "frac_hit_cap": float(sub.hit_cap.mean()),
            "mean_total_logp": float(sub.total_logp.mean()),
            "mean_per_token_logp": float(sub.mean_logp.mean()),
            "mean_rep4": float(sub.rep4.mean()),
            "mean_distinct2": float(sub.distinct2.mean()),
        }

    # regression ONLY where length varies. Use scored_len, not raw gen_len.
    def safe_fit(frame):
        frame = frame[frame.scored_len > 0]
        if len(frame) < 30 or frame.scored_len.nunique() < 5 or frame.scored_len.std() == 0:
            return None
        lr = linregress(frame.scored_len, frame.total_logp)
        return {"slope_nats_per_token": float(lr.slope),
                "intercept": float(lr.intercept),
                "r_value": float(lr.rvalue),
                "n": int(len(frame))}

    per_strategy_fit = {}
    for name in df.strategy.unique():
        fit = safe_fit(df[df.strategy == name])
        if fit:
            per_strategy_fit[name] = fit

    # pooled fit over the sampling strategies, which are the ones that vary
    pooled = safe_fit(df[df.strategy.isin(["ancestral", "topp90", "temp07"])])

    summary["length_vs_total_logp"] = {
        "per_strategy": per_strategy_fit,
        "pooled_sampling": pooled,
        "usable": bool(per_strategy_fit or pooled),
        "note": ("Slope is the change in unnormalised reward per generated token, "
                 "in nats. Negative slope is the brevity incentive; its magnitude "
                 "is the model's mean per-token entropy. Computed only where "
                 "length varies; deterministic decoders pinned at the cap are "
                 "excluded because their length carries no information."),
    }
    if not summary["length_vs_total_logp"]["usable"]:
        summary["length_vs_total_logp"]["reason_unavailable"] = (
            "No strategy produced varying scored length. For a len_beta=0 model "
            "this is itself the finding: EOS has been trained out, so the model "
            "never stops and every generation runs to the cap. See frac_emitted_eos.")

    return summary
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="run_diagnostic.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"ERROR: {args.path} not found. cd to its directory or pass --path.")

    src = open(args.path).read()

    start = src.find("def exp_likelihood_trap(")
    if start == -1:
        sys.exit("ERROR: could not find exp_likelihood_trap in the file.")

    # find the next top-level def or the EXPERIMENTS registry after it
    after = src.find("\ndef ", start + 1)
    marker = src.find("\n# ---", start + 1)
    ends = [x for x in (after, marker) if x != -1]
    if not ends:
        sys.exit("ERROR: could not find the end of exp_likelihood_trap.")
    end = min(ends)

    if "emitted_eos" in src[start:end]:
        print("Already patched (found 'emitted_eos'). Nothing to do.")
        return

    new_src = src[:start] + NEW_FUNC + "\n\n" + src[end + 1:]

    if args.dry_run:
        print(f"Would replace lines from offset {start} to {end}.")
        print(f"Old block: {src[start:end].count(chr(10))} lines")
        print(f"New block: {NEW_FUNC.count(chr(10))} lines")
        return

    bak = args.path + ".bak_ltrap"
    shutil.copy2(args.path, bak)
    with open(args.path, "w") as f:
        f.write(new_src)

    import py_compile
    try:
        py_compile.compile(args.path, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(bak, args.path)
        sys.exit(f"compile FAILED, reverted. error:\n{e}")

    print(f"backup : {bak}")
    print(f"patched: {args.path}")
    print("compile: PASS")
    print()
    print("Re-run ONLY this experiment, for all four models, with a larger cap:")
    for m, extra in [("gpt2sft", "--model_path ./gpt2_sft_output"),
                     ("gfn-lb0-500", "--model_path ./gpt2_sft_output --adapter_path ./gfn_ckpt/len_beta0_step500"),
                     ("gfn-lb0-2000", "--model_path ./gpt2_sft_output --adapter_path ./gfn_ckpt/len_beta0_step2000"),
                     ("gfn-lb1-500", "--model_path ./gpt2_sft_output --adapter_path ./gfn_ckpt/len_beta1_step500")]:
        print(f"  python run_diagnostic.py --exp likelihood_trap "
              f"--run_name diag_likelihood_trap_{m} {extra} "
              f"--out_dir results_diag --overwrite --max_new_tokens 60")


if __name__ == "__main__":
    main()
