#!/usr/bin/env python
"""
numbers_diff_phase6.py  -  Phase 6 Part 4.8: diff the numbers written into Doc/ against
their source JSON files. Reports every check as OK or MISMATCH; never edits anything.

Sources: results_revision/{numbers.json, rev_gprime.json, rev_confusion.json},
figures/fig_traj_stats.json. Each check asserts that a value quoted in the thesis text or
tables equals (to a tolerance) the value in the producing result file.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def check(label, thesis_val, src_val, tol=0.05):
    if isinstance(thesis_val, (int, float)) and isinstance(src_val, (int, float)):
        ok = abs(thesis_val - src_val) <= tol
    else:
        ok = str(thesis_val) == str(src_val)
    print(f"[{'OK ' if ok else 'MISMATCH'}] {label}: thesis={thesis_val} source={src_val}")
    return ok


def main():
    g = load("results_revision/rev_gprime.json")
    c = load("results_revision/rev_confusion.json")
    t = load("figures/fig_traj_stats.json")
    n = load("results_revision/numbers.json")
    all_ok = True

    print("== G-prime table 5.11 (rev_gprime.json) ==")
    cells = {"gamma2_label0": (-2.3, 18.0, 6.27), "gamma2_label1": (6.7, 9.3, 5.30),
             "gamma4_label0": (-0.7, 25.7, 7.28), "gamma4_label1": (7.3, 17.7, 6.01)}
    for k, (gain, self_g, gnll) in cells.items():
        b = g["by_cell"][k]
        all_ok &= check(f"{k} gain", gain, round(b["gain_pts"], 1), 0.15)
        all_ok &= check(f"{k} self_gain", self_g, round(b["self_gain_pts"], 1), 0.15)
        all_ok &= check(f"{k} guided_nll", gnll, round(b["guided_span_nll"], 2), 0.02)
    all_ok &= check("unguided_span_nll -> 7.01", 7.01, round(g["by_cell"]["gamma2_label0"]["unguided_span_nll"], 2), 0.01)

    print("== Agreement ladder (rev_gprime.json) ==")
    all_ok &= check("on-domain unguided agree 71.7", 71.7, round(g["diagnosis_unguided_gen_agreement_pct"], 1), 0.1)
    all_ok &= check("on-domain CI lo 66.7", 66.7, round(g["diagnosis_unguided_gen_agreement_ci95"][0], 1), 0.1)
    all_ok &= check("realtext agree 79.7", 79.7, round(g["realtext_agreement_pct"], 1), 0.1)
    all_ok &= check("judge acc 88", 88.0, g["realtext_judge_acc_pct"], 0.1)
    all_ok &= check("clf acc 79.7", 79.7, round(g["realtext_clf_acc_pct"], 1), 0.1)
    all_ok &= check("offdomain band lo 56", 56, g["phase4_offdomain_agreement_band_pct"][0])
    all_ok &= check("offdomain band hi 64", 64, g["phase4_offdomain_agreement_band_pct"][1])

    print("== Confusion table 5.12 (rev_confusion.json) ==")
    cc = c["confusion_unguided_ondomain"]["counts_guideVerdict_x_judgeVerdict"]
    all_ok &= check("count neg/neg 109", 109, cc["guide0_judge0"])
    all_ok &= check("count neg/pos 35", 35, cc["guide0_judge1"])
    all_ok &= check("count pos/neg 50", 50, cc["guide1_judge0"])
    all_ok &= check("count pos/pos 106", 106, cc["guide1_judge1"])
    all_ok &= check("P(j=pos|g=pos) 67.9", 67.9, round(c["confusion_unguided_ondomain"]["P_judge_pos_given_guide_pos_pct"], 1), 0.1)
    all_ok &= check("P(j=neg|g=neg) 75.7", 75.7, round(c["confusion_unguided_ondomain"]["P_judge_neg_given_guide_neg_pct"], 1), 0.1)
    all_ok &= check("diff -7.7", -7.7, round(c["confusion_unguided_ondomain"]["conditional_diff_pos_minus_neg_pts"], 1), 0.1)
    all_ok &= check("overall agree 71.7", 71.7, round(c["confusion_unguided_ondomain"]["overall_agreement_pct"], 1), 0.1)
    gs = c["label_alignment_sanity"]
    all_ok &= check("guide acc neg 79.4", 79.4, round(gs["guide"]["0"]["acc_pct"], 1), 0.1)
    all_ok &= check("guide acc pos 79.9", 79.9, round(gs["guide"]["1"]["acc_pct"], 1), 0.1)
    all_ok &= check("judge acc neg 85.8", 85.8, round(gs["judge"]["0"]["acc_pct"], 1), 0.1)
    all_ok &= check("judge acc pos 89.9", 89.9, round(gs["judge"]["1"]["acc_pct"], 1), 0.1)

    print("== Trajectory (fig_traj_stats.json) ==")
    pc = t["per_config"]
    all_ok &= check("MH-on nn min ~118", 118, round(pc["cls_policy_gnoff_mh"]["nearest_token_dist_min"]), 1)
    all_ok &= check("MH-on nn max ~151", 151, round(pc["cls_policy_gnoff_mh"]["nearest_token_dist_max"]), 1)
    all_ok &= check("MH-off nn max ~980", 979, round(pc["cls_policy_gnoff_nomh"]["nearest_token_dist_max"]), 2)
    all_ok &= check("MH-off end ~17", 17, round(pc["cls_policy_gnoff_nomh"]["end_nearest_token_dist_mean"]), 1)
    all_ok &= check("MH-off cells ~45", 45, round(pc["cls_policy_gnoff_nomh"]["distinct_cells_mean"]), 1)
    all_ok &= check("MH-on cells ~2", 2, round(pc["cls_policy_gnoff_mh"]["distinct_cells_mean"]), 1)
    all_ok &= check("DLS-policy cells ~5", 5, round(pc["dls_policy_gn_mh"]["distinct_cells_mean"]), 1)
    all_ok &= check("DLS-policy nn dist 0", 0.0, pc["dls_policy_gn_mh"]["nearest_token_dist_max"], 0.01)
    all_ok &= check("DLS final gt ~2.2", 2.2, round(pc["dls_policy_gn_mh"]["final_dist_to_gt_mean"], 1), 0.1)
    all_ok &= check("PCA EVR 3.3%", 3.3, round(100 * t["pca_explained_variance_sum"], 1), 0.1)
    fac = t["offmanifold_factor_summary"]["factor_vs_nn_spacing_mh_on"]
    all_ok &= check("factor lo ~65", 65, round(fac[0]), 1)
    all_ok &= check("factor hi ~83", 83, round(fac[1]), 1)

    print("== Anisotropy (numbers.json), referenced in traj + results ==")
    a = n["diag_anisotropy_gpt2sft"]
    all_ok &= check("NN spacing 1.82", 1.82, round(a["mean_nearest_neighbour_l2"], 2), 0.005)
    all_ok &= check("pairwise 2.77", 2.77, round(a["mean_pairwise_l2"], 2), 0.005)

    print("\n== DELIBERATE RE-SOURCING (reported, not a silent fix) ==")
    print("  Trajectory body numbers re-sourced from canonical traces_gpt2sft_traj.npz")
    print("  (max drift 588, cells 1.0/46.2/7.0) to the regeneration run")
    print("  traces_gpt2sft_plot_traj.npz / fig_traj_stats.json (max 979, cells 2.2/44.7/5.3),")
    print("  because Part 3 mandates the FULL wte matrix and the published figure uses that run.")

    print("\nRESULT:", "ALL OK" if all_ok else "MISMATCHES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
