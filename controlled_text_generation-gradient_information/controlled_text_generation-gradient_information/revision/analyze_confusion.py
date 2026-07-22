#!/usr/bin/env python
"""
analyze_confusion.py  -  Phase 6 Part 1: the per-class confusion / mechanism proof.

Purpose: the G write-up infers that residual guide-judge disagreement falls
asymmetrically by class. This demonstrates or refutes it from rev_gprime.csv, with no
rerun. Three pieces:

1. Label-alignment sanity: per-class accuracy of the guide (noisy classifier) and the
   judge (concern-11 frozen-GPT-2 head) on the 300 realtext rows (held-out SST-2
   validation sentences; target_label there IS the true label). Overall must reproduce
   guide 79.67%, judge 88.0%. A silent label-map flip would show one class near 0%.
2. The confusion table on the UNIQUE UNGUIDED on-domain generations (one per prompt, the
   neutral calibration surface): 2x2 guide-verdict vs judge-verdict, plus the two
   conditional agreement rates P(judge=pos | guide=pos) and P(judge=neg | guide=neg) with
   bootstrap CIs, and the paired-difference CI that decides asymmetry.
3. Guided-text agreement, reported as labeled context only.

Writes results_revision/rev_confusion.json and a staged LaTeX table.
"""

import argparse
import csv
import json
import os

import numpy as np

POS, NEG = 1, 0


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def boot_ci(fn, n_boot, rng, *arrays):
    """Bootstrap CI of a scalar statistic fn(*resampled_arrays). Arrays share length."""
    n = len(arrays[0])
    stats = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        stats[b] = fn(*[a[idx] for a in arrays])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_revision/rev_gprime.csv")
    ap.add_argument("--out", default="results_revision/rev_confusion.json")
    ap.add_argument("--tex", default="results_revision/rev_confusion.tex")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = read_rows(args.csv)

    # ---- 1. label-alignment sanity on realtext ----
    rt = [r for r in rows if r["arm"] == "realtext"]
    true_lbl = np.array([int(r["target_label"]) for r in rt])
    guide_rt = np.array([int(r["clf_self_label"]) for r in rt])
    judge_rt = np.array([int(r["judged_label"]) for r in rt])

    def per_class(true, pred):
        out = {}
        for c in (NEG, POS):
            m = true == c
            out[str(c)] = {
                "n": int(m.sum()),
                "acc_pct": float(100.0 * (pred[m] == c).mean()) if m.any() else float("nan"),
            }
        out["overall_acc_pct"] = float(100.0 * (pred == true).mean())
        return out

    sanity = {
        "n_realtext": len(rt),
        "guide": per_class(true_lbl, guide_rt),
        "judge": per_class(true_lbl, judge_rt),
    }

    # ---- 2. confusion table on UNIQUE unguided on-domain generations ----
    seen = set()
    u_guide, u_judge = [], []
    for r in rows:
        if r["arm"] != "unguided":
            continue
        p = int(r["prompt_idx"])
        if p in seen:
            continue
        seen.add(p)
        u_guide.append(int(r["clf_self_label"]))
        u_judge.append(int(r["judged_label"]))
    u_guide = np.array(u_guide)
    u_judge = np.array(u_judge)
    n_u = len(u_guide)

    # 2x2 counts: rows = guide verdict, cols = judge verdict
    counts = {}
    for g in (NEG, POS):
        for j in (NEG, POS):
            counts[f"guide{g}_judge{j}"] = int(((u_guide == g) & (u_judge == j)).sum())

    def cond_rate(guide_arr, judge_arr, cls):
        m = guide_arr == cls
        if m.sum() == 0:
            return float("nan")
        return float(100.0 * (judge_arr[m] == cls).mean())

    p_pos = cond_rate(u_guide, u_judge, POS)   # P(judge=pos | guide=pos)
    p_neg = cond_rate(u_guide, u_judge, NEG)   # P(judge=neg | guide=neg)
    ci_pos = boot_ci(lambda g, j: cond_rate(g, j, POS), args.n_boot, rng, u_guide, u_judge)
    ci_neg = boot_ci(lambda g, j: cond_rate(g, j, NEG), args.n_boot, rng, u_guide, u_judge)
    # paired difference (pos channel - neg channel) with a single bootstrap for the diff
    diff_ci = boot_ci(
        lambda g, j: cond_rate(g, j, POS) - cond_rate(g, j, NEG),
        args.n_boot, rng, u_guide, u_judge,
    )
    overall_agree = float(100.0 * (u_guide == u_judge).mean())
    overall_agree_ci = boot_ci(lambda g, j: 100.0 * (g == j).mean(),
                               args.n_boot, rng, u_guide, u_judge)

    # asymmetry decision: paired-difference CI excludes 0
    diff = p_pos - p_neg
    asym_by_diff = (diff_ci[0] > 0) or (diff_ci[1] < 0)
    higher = "positive" if p_pos > p_neg else "negative"
    asymmetry_confirmed = bool(asym_by_diff and higher == "positive")

    confusion = {
        "n_unguided_unique": n_u,
        "counts_guideVerdict_x_judgeVerdict": counts,
        "n_guide_pos": int((u_guide == POS).sum()),
        "n_guide_neg": int((u_guide == NEG).sum()),
        "P_judge_pos_given_guide_pos_pct": p_pos,
        "P_judge_pos_given_guide_pos_ci95": list(ci_pos),
        "P_judge_neg_given_guide_neg_pct": p_neg,
        "P_judge_neg_given_guide_neg_ci95": list(ci_neg),
        "conditional_diff_pos_minus_neg_pts": float(diff),
        "conditional_diff_ci95": list(diff_ci),
        "overall_agreement_pct": overall_agree,
        "overall_agreement_ci95": list(overall_agree_ci),
        "higher_channel": higher,
        "asymmetry_by_diff_ci": bool(asym_by_diff),
        "asymmetry_confirmed_higher_is_positive": asymmetry_confirmed,
    }

    # ---- 3. guided-text agreement, labeled context only ----
    guided_ctx = {}
    g_rows = [r for r in rows if r["arm"] == "guided"]
    gg = np.array([int(r["clf_self_label"]) for r in g_rows])
    gj = np.array([int(r["judged_label"]) for r in g_rows])
    guided_ctx["overall_agreement_pct"] = float(100.0 * (gg == gj).mean())
    guided_ctx["n"] = int(len(g_rows))
    for gval in ("2.0", "4.0"):
        sub = [r for r in g_rows if r["gamma"] == gval]
        a = np.array([int(r["clf_self_label"]) for r in sub])
        b = np.array([int(r["judged_label"]) for r in sub])
        guided_ctx[f"gamma{gval}_agreement_pct"] = float(100.0 * (a == b).mean())

    out = {
        "experiment": "confusion_mechanism_proof",
        "source_csv": args.csv,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "label_alignment_sanity": sanity,
        "confusion_unguided_ondomain": confusion,
        "guided_text_agreement_context": guided_ctx,
        "prediction": {
            "P2_asymmetric_higher_positive": True,
            "result_asymmetry_confirmed": asymmetry_confirmed,
        },
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)

    # ---- LaTeX table (placeholder substitution; avoids % / brace conflicts) ----
    # The caption text is chosen by the DATA: if the pre-registered by-class asymmetry
    # (positive channel higher, CI-separated) holds, state the mechanism; otherwise state
    # the honest null and do not invent a post-hoc class-level explanation.
    c = counts
    higher_word = "positive" if higher == "positive" else "negative"
    if asymmetry_confirmed:
        verdict_sentence = (
            r"Conditional agreement is asymmetric by class and its paired difference "
            r"excludes zero, with the positive channel higher. That higher-agreement "
            r"channel is the same direction that transferred to the judge under guidance "
            r"in the on-domain trust-region run, so the residual limit is an "
            r"instrument-alignment effect rather than a fixed directional barrier."
        )
    else:
        verdict_sentence = (
            r"The two conditional agreement rates are statistically indistinguishable "
            r"(their paired-difference $95$\% CI includes zero), and the point estimates "
            r"do not favour the positive channel. So the residual disagreement confirms "
            r"that the two independently trained instruments are only partially aligned "
            r"($@OA@$\% agreement, up from $56$--$64$\% off-domain), but this table does "
            r"not identify a by-class asymmetry that would by itself explain why positive "
            r"steering transferred to the judge and negative steering did not; that "
            r"transfer asymmetry is reported as observed and setting-contingent, and no "
            r"class-level mechanism is claimed beyond the overall alignment gap."
        )
    tex_tmpl = r"""% Auto-generated by revision/analyze_confusion.py (Phase 6 Part 1). Do not hand-edit.
\begin{table}[t]
  \centering
  \small
  \begin{tabular}{llcc}
    \toprule
     & & \multicolumn{2}{c}{Judge verdict} \\
    \cmidrule(lr){3-4}
     & & negative & positive \\
    \midrule
    \multirow{2}{*}{Guide verdict} & negative & @G0J0@ & @G0J1@ \\
     & positive & @G1J0@ & @G1J1@ \\
    \bottomrule
  \end{tabular}
  \caption{Per-class agreement of the guide (noisy classifier) and the independent
  judge (frozen GPT-2 sentiment head) on the @NU@ unguided on-domain generations (the
  neutral calibration surface; guided text is excluded to avoid circularity). The judge
  confirms the guide's positive verdict @PPOS@\% of the time ($95$\% CI
  $[@PPOSLO@, @PPOSHI@]$) and its negative verdict @PNEG@\% ($[@PNEGLO@, @PNEGHI@]$); the
  paired difference (positive minus negative) is $@DIFF@$ points ($[@DIFFLO@, @DIFFHI@]$).
  @VERDICT@ Overall agreement @OA@\% $[@OALO@, @OAHI@]$.}
  \label{tab:confusion}
\end{table}
"""
    tex_tmpl = tex_tmpl.replace("@VERDICT@", verdict_sentence)
    subs = {
        "@G0J0@": str(c["guide0_judge0"]), "@G0J1@": str(c["guide0_judge1"]),
        "@G1J0@": str(c["guide1_judge0"]), "@G1J1@": str(c["guide1_judge1"]),
        "@NU@": str(n_u),
        "@HIGHER@": higher_word,
        "@PPOS@": f"{p_pos:.1f}", "@PPOSLO@": f"{ci_pos[0]:.1f}", "@PPOSHI@": f"{ci_pos[1]:.1f}",
        "@PNEG@": f"{p_neg:.1f}", "@PNEGLO@": f"{ci_neg[0]:.1f}", "@PNEGHI@": f"{ci_neg[1]:.1f}",
        "@DIFF@": f"{diff:.1f}", "@DIFFLO@": f"{diff_ci[0]:.1f}", "@DIFFHI@": f"{diff_ci[1]:.1f}",
        "@OA@": f"{overall_agree:.1f}", "@OALO@": f"{overall_agree_ci[0]:.1f}",
        "@OAHI@": f"{overall_agree_ci[1]:.1f}",
    }
    tex = tex_tmpl
    for kk, vv in subs.items():
        tex = tex.replace(kk, vv)
    with open(args.tex, "w") as f:
        f.write(tex)

    print(json.dumps(out, indent=2))
    print(f"\n[wrote] {args.out}\n[wrote] {args.tex}")


if __name__ == "__main__":
    main()
