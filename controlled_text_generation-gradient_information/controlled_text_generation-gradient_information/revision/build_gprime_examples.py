#!/usr/bin/env python
"""
build_gprime_examples.py  -  Phase 6 Part 2: guided-generation examples.

Makes the steering visible in text. Two blocks:

A. On-domain trust-region G-prime pairs. IMMUTABLE seeded selection: rng =
   np.random.default_rng(0); for each cell in the fixed order
   [(gamma 2, neg), (gamma 2, pos), (gamma 4, neg), (gamma 4, pos)] draw
   rng.choice(300, 3, replace=False). 12 pairs total, no filtering, no redrawing.
   Each pair shows the prompt (first 10 GPT-2 tokens of the held-out SST-2 sentence,
   reconstructed exactly), the unguided continuation, the guided continuation, and the
   guide's and judge's verdict on each.

B. Trust-region before/after. The 3 highest-NLL guided continuations from the Phase 4
   OFF-domain gamma-4 run (rev_sedd_guided_g4.csv), where guidance pushed text off the
   fluent manifold (guided NLL ~11 vs unguided ~7). Placed next to the on-domain gamma-4
   G-prime pairs, whose trust region keeps NLL bounded. Phase 4 per-item text WAS stored,
   so no regeneration is needed.

Statistics live in Part 1's confusion table and rev_gprime.json; these are illustration.
Writes results_revision/gprime_examples.md and results_revision/gprime_examples.tex.
"""

import csv
import json
import os
import sys

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPRIME_CSV = os.path.join(ROOT, "results_revision", "rev_gprime.csv")
PHASE4_CSV = os.path.join(ROOT, "results_revision", "rev_sedd_guided_g4.csv")
GPT2SFT = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"
PROMPT_LEN, MIN_SENT_TOK = 10, 12
LBL = {0: "negative", 1: "positive"}


def reconstruct_prompts(n_prompts=300):
    """prompt_idx -> first-PROMPT_LEN-token prompt string, matching run_gprime.py exactly."""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(GPT2SFT if os.path.isdir(GPT2SFT) else "gpt2-large")
    ds = load_dataset("glue", "sst2")["validation"]
    out = {}
    for ex in ds:
        ids = tok(ex["sentence"].strip(), add_special_tokens=False).input_ids
        if len(ids) < MIN_SENT_TOK:
            continue
        out[len(out)] = tok.decode(ids[:PROMPT_LEN])
        if len(out) >= n_prompts:
            break
    return out


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n].strip()


def main():
    rows = read_rows(GPRIME_CSV)
    prompts = reconstruct_prompts()

    # index unguided (one per prompt) and guided (per prompt,gamma,label)
    ung = {}   # prompt_idx -> row
    gud = {}   # (prompt_idx, gamma, label) -> row
    for r in rows:
        if r["arm"] == "unguided":
            ung.setdefault(int(r["prompt_idx"]), r)
        elif r["arm"] == "guided":
            gud[(int(r["prompt_idx"]), float(r["gamma"]), int(r["target_label"]))] = r

    # ---- IMMUTABLE selection ----
    rng = np.random.default_rng(0)
    cells = [(2.0, 0), (2.0, 1), (4.0, 0), (4.0, 1)]
    selection = {}
    for (g, lab) in cells:
        idx = sorted(int(i) for i in rng.choice(300, size=3, replace=False))
        selection[f"gamma{g}_label{lab}"] = idx

    gprime_pairs = []
    for (g, lab) in cells:
        for pidx in selection[f"gamma{g}_label{lab}"]:
            u = ung[pidx]
            gd = gud[(pidx, g, lab)]
            gprime_pairs.append({
                "cell": f"gamma{g}_{LBL[lab]}",
                "gamma": g, "target_label": lab,
                "prompt_idx": pidx,
                "prompt": prompts[pidx],
                "unguided_text": u["text"],
                "unguided_guide": int(u["clf_self_label"]),
                "unguided_judge": int(u["judged_label"]),
                "guided_text": gd["text"],
                "guided_guide": int(gd["clf_self_label"]),
                "guided_judge": int(gd["judged_label"]),
                "guided_nll": float(gd["span_nll"]),
                "unguided_nll": float(u["span_nll"]),
            })

    # ---- Phase 4 off-domain before/after: top-3 highest guided NLL ----
    p4 = read_rows(PHASE4_CSV)
    p4g = [r for r in p4 if r["arm"] == "guided"]
    p4g.sort(key=lambda r: float(r["span_nll"]), reverse=True)
    top3 = p4g[:3]
    p4u = {(int(r["prompt_idx"]), int(r["sample_idx"]), int(r["target_label"])): r
           for r in p4 if r["arm"] == "unguided"}
    phase4_pairs = []
    for gd in top3:
        key = (int(gd["prompt_idx"]), int(gd["sample_idx"]), int(gd["target_label"]))
        u = p4u[key]
        phase4_pairs.append({
            "prompt_idx": int(gd["prompt_idx"]), "sample_idx": int(gd["sample_idx"]),
            "target_label": int(gd["target_label"]),
            "prompt": lcp(u["text"], gd["text"]),
            "unguided_text": u["text"], "unguided_nll": float(u["span_nll"]),
            "unguided_guide": int(u["clf_self_label"]), "unguided_judge": int(u["judged_label"]),
            "guided_text": gd["text"], "guided_nll": float(gd["span_nll"]),
            "guided_guide": int(gd["clf_self_label"]), "guided_judge": int(gd["judged_label"]),
        })

    out = {
        "experiment": "gprime_examples",
        "selection_rule": ("np.random.default_rng(0); cells in order "
                           "[(g2,neg),(g2,pos),(g4,neg),(g4,pos)]; "
                           "sorted(rng.choice(300,3,replace=False)) per cell; unfiltered"),
        "selection": selection,
        "gprime_pairs": gprime_pairs,
        "phase4_offdomain_rule": ("3 highest guided span_nll rows in rev_sedd_guided_g4.csv "
                                  "(off-domain, gamma 4), paired to their unguided partner"),
        "phase4_pairs": phase4_pairs,
    }
    jpath = os.path.join(ROOT, "results_revision", "gprime_examples.json")
    with open(jpath + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(jpath + ".tmp", jpath)

    write_markdown(out)
    write_tex(out)
    print("selection:", json.dumps(selection))
    print("phase4 top3 idx/nll:",
          [(p["prompt_idx"], p["sample_idx"], round(p["guided_nll"], 1)) for p in phase4_pairs])
    print(f"[wrote] {jpath}")
    print(f"[wrote] {os.path.join(ROOT, 'results_revision', 'gprime_examples.md')}")
    print(f"[wrote] {os.path.join(ROOT, 'results_revision', 'gprime_examples.tex')}")


def _verdict_line(guide, judge):
    tag = "agree" if guide == judge else "DISAGREE"
    return f"guide: {LBL[guide]} | judge: {LBL[judge]} ({tag})"


def write_markdown(out):
    L = []
    L.append("# G-prime guided-generation examples\n")
    L.append("Seeded, unfiltered draws. The statistics are in the confusion table "
             "(Part 1) and rev_gprime.json; these pairs are illustration only. Verdicts "
             "are the guide's (noisy classifier) and the independent judge's (frozen "
             "GPT-2 sentiment head). Pairs where guide and judge disagree are the "
             "partial instrument alignment made visible, not failures.\n")
    L.append(f"Selection rule: {out['selection_rule']}\n")
    L.append(f"Drawn indices: {json.dumps(out['selection'])}\n")

    cur = None
    for p in out["gprime_pairs"]:
        if p["cell"] != cur:
            cur = p["cell"]
            g = p["gamma"]
            L.append(f"\n## Cell: gamma {g:g}, target {LBL[p['target_label']]}\n")
        L.append(f"### prompt_idx {p['prompt_idx']}")
        L.append(f"- **Prompt:** {p['prompt']}")
        L.append(f"- **Unguided:** {p['unguided_text']}")
        L.append(f"  - {_verdict_line(p['unguided_guide'], p['unguided_judge'])}"
                 f" | span NLL {p['unguided_nll']:.2f}")
        L.append(f"- **Guided (gamma {p['gamma']:g}, toward {LBL[p['target_label']]}):** "
                 f"{p['guided_text']}")
        L.append(f"  - {_verdict_line(p['guided_guide'], p['guided_judge'])}"
                 f" | span NLL {p['guided_nll']:.2f}")
        L.append("")

    L.append("\n## Trust-region before/after: Phase 4 off-domain, gamma 4 "
             "(the fluency climb the trust region removes)\n")
    L.append("These are the three highest-NLL guided continuations from the Phase 4 "
             "off-domain run (MuCoLa continuation prompts, no trust region). Guidance "
             "there pushed text off the fluent manifold (guided NLL ~11 vs unguided ~7). "
             "The on-domain gamma-4 G-prime pairs above keep NLL bounded by construction, "
             "which is the trust region working.\n")
    for p in out["phase4_pairs"]:
        L.append(f"### off-domain prompt_idx {p['prompt_idx']}, sample {p['sample_idx']}, "
                 f"target {LBL[p['target_label']]}")
        L.append(f"- **Prompt (shared opening):** {p['prompt']}")
        L.append(f"- **Unguided:** {p['unguided_text']}")
        L.append(f"  - {_verdict_line(p['unguided_guide'], p['unguided_judge'])}"
                 f" | span NLL {p['unguided_nll']:.2f}")
        L.append(f"- **Guided (gamma 4, no trust region):** {p['guided_text']}")
        L.append(f"  - {_verdict_line(p['guided_guide'], p['guided_judge'])}"
                 f" | span NLL {p['guided_nll']:.2f}")
        L.append("")

    path = os.path.join(ROOT, "results_revision", "gprime_examples.md")
    with open(path, "w") as f:
        f.write("\n".join(L))


_UNI_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def _to_safe_ascii(s):
    """pdfLaTeX + utf8 inputenc chokes on CJK / U+FFFD / exotic bytes that the model
    emits. Map common typographic unicode to ASCII, replace any remaining non-printable-
    ASCII run with a visible [?] marker (honest for garbled output)."""
    for a, b in _UNI_MAP.items():
        s = s.replace(a, b)
    out, prev_marker = [], False
    for ch in s:
        o = ord(ch)
        if 32 <= o <= 126:
            out.append(ch)
            prev_marker = False
        elif not prev_marker:
            out.append("[?]")
            prev_marker = True
    return "".join(out)


def _tex_escape(s):
    s = _to_safe_ascii(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def write_tex(out):
    L = []
    L.append("% Auto-generated by revision/build_gprime_examples.py (Phase 6 Part 2). "
             "Do not hand-edit.")
    L.append(r"\begin{longtable}{p{0.16\linewidth} p{0.78\linewidth}}")
    L.append(r"\caption{On-domain trust-region guided generations (G-prime). Draws are "
             r"seeded (\texttt{default\_rng(0)}, three pairs per cell) and unfiltered; "
             r"the statistics are in Table~\ref{tab:confusion} and the G-prime results "
             r"table. \emph{g} and \emph{j} are the guide's and the independent judge's "
             r"verdict. Pairs where they disagree show the partial instrument alignment "
             r"directly.}\label{tab:gprime-examples}\\")
    L.append(r"\toprule")
    L.append(r"& \\")
    L.append(r"\endfirsthead")
    L.append(r"\toprule & \\ \endhead")
    L.append(r"\bottomrule \endfoot")

    cur = None
    for p in out["gprime_pairs"]:
        if p["cell"] != cur:
            cur = p["cell"]
            L.append(r"\midrule")
            L.append(r"\multicolumn{2}{l}{\textbf{Cell: $\gamma=%g$, target %s}} \\ \midrule"
                     % (p["gamma"], LBL[p["target_label"]]))
        L.append(r"Prompt & %s \\" % _tex_escape(p["prompt"]))
        L.append(r"Unguided & %s \newline{\small [g:%s, j:%s; NLL %.2f]} \\"
                 % (_tex_escape(p["unguided_text"]), LBL[p["unguided_guide"]][:3],
                    LBL[p["unguided_judge"]][:3], p["unguided_nll"]))
        L.append(r"Guided & %s \newline{\small [g:%s, j:%s; NLL %.2f]} \\[2pt]"
                 % (_tex_escape(p["guided_text"]), LBL[p["guided_guide"]][:3],
                    LBL[p["guided_judge"]][:3], p["guided_nll"]))
    L.append(r"\end{longtable}")

    # Phase 4 before/after
    L.append("")
    L.append(r"\begin{longtable}{p{0.16\linewidth} p{0.78\linewidth}}")
    L.append(r"\caption{Trust-region before/after. The three highest-NLL guided "
             r"continuations from the Phase 4 off-domain run (no trust region), where "
             r"guidance pushed text off the fluent manifold (guided NLL near 11 versus "
             r"unguided near 7). The on-domain gamma-4 pairs in "
             r"Table~\ref{tab:gprime-examples} stay bounded, which is the trust region "
             r"working.}\label{tab:gprime-trustregion}\\")
    L.append(r"\toprule & \\ \endfirsthead \toprule & \\ \endhead \bottomrule \endfoot")
    for p in out["phase4_pairs"]:
        L.append(r"\midrule")
        L.append(r"Prompt & %s \\" % _tex_escape(p["prompt"]))
        L.append(r"Unguided & %s \newline{\small [g:%s, j:%s; NLL %.2f]} \\"
                 % (_tex_escape(p["unguided_text"]), LBL[p["unguided_guide"]][:3],
                    LBL[p["unguided_judge"]][:3], p["unguided_nll"]))
        L.append(r"Guided & %s \newline{\small [g:%s, j:%s; NLL %.2f]} \\[2pt]"
                 % (_tex_escape(p["guided_text"]), LBL[p["guided_guide"]][:3],
                    LBL[p["guided_judge"]][:3], p["guided_nll"]))
    L.append(r"\end{longtable}")

    path = os.path.join(ROOT, "results_revision", "gprime_examples.tex")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
