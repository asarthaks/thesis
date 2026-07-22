#!/usr/bin/env python
"""
make_showcase_tex.py  -  render the Stage 1b qualitative showcase as a LaTeX appendix.

Reads results_revision/qualitative_showcase.json (the infill recoveries) and
results_revision/rev_gprime.csv (the on-domain guided pairs), and writes a self-
contained appendix section with two longtables:

  1. Infill recovery showcase (10 seeded kl_baselines sequences x every method). Each
     sequence block shows the original text, the corrupted text with the corrupted
     token in bold, then one row per method with the single recovered token in bold and
     a hit marker against the ground truth.
  2. On-domain guided generation showcase (G-prime): 3 seeded pairs per target label,
     unguided vs guided (gamma 4), with the judge verdict.

Selection rules (immutable, stated in the captions):
  infill: np.random.default_rng(0).choice(200,10,replace=False), no filtering.
  gprime: np.random.default_rng(0).choice(300,6,replace=False); first 3 -> label 0,
          next 3 -> label 1.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

METHOD_ORDER = [
    ("dls_policy", "DLS policy (MH, GN)"),
    ("dls_random", "DLS random dir"),
    ("cls_flagship", "CLS flagship (policy, MH, GN, free, s50)"),
    ("gibbs", "Metropolized Gibbs"),
    ("cond_argmax", "Conditional argmax"),
    ("cond_topk_rescore", "Conditional top-k rescore"),
    ("left_conditional", "Left-conditional (independence MH)"),
    ("sedd_small", "SEDD-small recovery"),
    ("sedd_medium", "SEDD-medium recovery"),
    ("hybrid_medium", "Hybrid (SEDD proposal, exact-energy MH)"),
]


def esc(s):
    if s is None:
        return ""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def bold_marks(s):
    """Turn the [[...]] markers from build_showcase into \textbf{...} after escaping."""
    s = esc(s)
    s = s.replace("[[", r"\textbf{").replace("]]", r"}")
    return s


def tok_disp(s):
    s = s.replace("\n", " ")
    return "'" + esc(s.strip() if s.strip() else s) + "'"


def infill_table(sc):
    L = []
    L.append(r"\subsection{Qualitative infill recovery showcase}")
    L.append(r"\label{app:showcase-infill}")
    L.append(r"The ten sequences were drawn once with a fixed rule and never filtered: "
             r"\texttt{np.random.default\_rng(0).choice(200,10,replace=False)} over the "
             r"kl-baselines set (WikiText-2 validation, one random interior mask, "
             r"\texttt{data\_seed}~0), sorted. Failures are kept in. The corrupted and "
             r"recovered token at the masked position is shown in bold; a check mark means "
             r"the recovered token equals the ground truth. Numbers cross-reference "
             r"Sections~\ref{sec:results-baselines} and~\ref{sec:results-diffusion}.")
    L.append(r"\begin{center}\footnotesize")
    L.append(r"\begin{longtable}{p{0.30\textwidth} p{0.60\textwidth} c}")
    L.append(r"\caption{Infill recovery on ten seeded sequences.}"
             r"\label{tab:showcase-infill}\\")
    L.append(r"\toprule Method & Recovered token (in context) & Hit\\ \midrule")
    L.append(r"\endfirsthead \toprule Method & Recovered token (in context) & Hit\\ "
             r"\midrule \endhead \bottomrule \endfoot")
    for row in sc["rows"]:
        sidx = row["sample_idx"]
        L.append(r"\multicolumn{3}{l}{\textbf{Sequence idx " + str(sidx) +
                 r"} \quad ground truth: " + tok_disp(row["gt_token_str"]) +
                 r"\quad corrupted: " + tok_disp(row["corr_token_str"]) + r"}\\")
        L.append(r"\multicolumn{3}{p{0.92\textwidth}}{\itshape original: " +
                 esc(row["original_text"][:400]) + r"}\\")
        L.append(r"\multicolumn{3}{p{0.92\textwidth}}{\itshape corrupted: " +
                 bold_marks(row["corrupted_text"][:400]) + r"}\\")
        L.append(r"\addlinespace[2pt]")
        for key, disp in METHOD_ORDER:
            m = row["methods"].get(key)
            if m is None:
                L.append(esc(disp) + r" & \textit{n/a} & \\")
                continue
            hit = r"\checkmark" if m["exact"] else ""
            L.append(esc(disp) + " & " + bold_marks(m["recovered_text"][:300]) +
                     " & " + hit + r"\\")
        L.append(r"\midrule")
    L.append(r"\end{longtable}\end{center}")
    return "\n".join(L)


def gprime_pairs(csv_path):
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(0)
    idx = rng.choice(300, size=6, replace=False)
    assign = {0: sorted(idx[:3].tolist()), 1: sorted(idx[3:].tolist())}
    gamma = 4.0 if 4.0 in set(df.gamma.unique()) else float(sorted(df.gamma.unique())[-1])
    out = {}
    for lbl, pidxs in assign.items():
        recs = []
        for pidx in pidxs:
            u = df[(df.prompt_idx == pidx) & (df.arm == "unguided") &
                   (df.target_label == lbl) & (df.gamma == gamma)]
            g = df[(df.prompt_idx == pidx) & (df.arm == "guided") &
                   (df.target_label == lbl) & (df.gamma == gamma)]
            if u.empty or g.empty:
                continue
            recs.append({"prompt_idx": int(pidx),
                         "unguided_text": str(u.iloc[0].text), "unguided_judge": int(u.iloc[0].judged_label),
                         "guided_text": str(g.iloc[0].text), "guided_judge": int(g.iloc[0].judged_label)})
        out[lbl] = recs
    return out, gamma, assign, idx.tolist()


def gprime_table(csv_path):
    pairs, gamma, assign, raw = gprime_pairs(csv_path)
    L = []
    L.append(r"\subsection{On-domain guided generation showcase}")
    L.append(r"\label{app:showcase-gprime}")
    L.append(r"Three seeded prompts per target label (rule: "
             r"\texttt{np.random.default\_rng(0).choice(300,6,replace=False)}; first three "
             r"to the negative target, next three to the positive), each shown unguided "
             r"versus guided at $\gamma=" + f"{gamma:g}" + r"$ with the trust region "
             r"($\delta=5$ nats). ``neg''/``pos'' is the frozen concern-11 judge's verdict "
             r"on the generated span; the guide only steered. See "
             r"Section~\ref{sec:results-diffusion}.")
    L.append(r"\begin{center}\footnotesize")
    L.append(r"\begin{longtable}{c p{0.40\textwidth} p{0.40\textwidth}}")
    L.append(r"\caption{On-domain trust-region guided generation, unguided vs guided.}"
             r"\label{tab:showcase-gprime}\\")
    L.append(r"\toprule Target & Unguided (judge) & Guided (judge)\\ \midrule")
    L.append(r"\endfirsthead \toprule Target & Unguided (judge) & Guided (judge)\\ "
             r"\midrule \endhead \bottomrule \endfoot")
    names = {0: "negative", 1: "positive"}
    jn = {0: "neg", 1: "pos"}
    for lbl in (0, 1):
        for r in pairs.get(lbl, []):
            L.append(names[lbl] + " & " + esc(r["unguided_text"][:280]) +
                     r" \textit{(" + jn[r["unguided_judge"]] + r")} & " +
                     esc(r["guided_text"][:280]) + r" \textit{(" + jn[r["guided_judge"]] +
                     r")}\\ \addlinespace[3pt]")
    L.append(r"\end{longtable}\end{center}")
    return "\n".join(L), {"assign": assign, "gamma": gamma, "raw_draw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results_revision")
    ap.add_argument("--tex_out", default="Doc/chapters/showcase_appendix.tex")
    args = ap.parse_args()

    sc = json.load(open(os.path.join(args.out_dir, "qualitative_showcase.json")))
    gtex, gmeta = gprime_table(os.path.join(args.out_dir, "rev_gprime.csv"))

    body = []
    body.append(r"% Auto-generated by revision/make_showcase_tex.py from "
                r"results_revision/qualitative_showcase.json and rev_gprime.csv.")
    body.append(r"\section{Qualitative Recovery and Steering Showcase}")
    body.append(r"\label{app:showcase}")
    body.append(infill_table(sc))
    body.append(gtex)
    os.makedirs(os.path.dirname(args.tex_out), exist_ok=True)
    with open(args.tex_out, "w") as f:
        f.write("\n\n".join(body) + "\n")
    print(f"wrote {args.tex_out}")
    print("gprime showcase selection:", json.dumps(gmeta))


if __name__ == "__main__":
    main()
