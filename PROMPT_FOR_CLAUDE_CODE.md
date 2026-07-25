# Prompt to paste into Claude Code

Set the model to Opus 4.8 with the highest reasoning setting before you start
(`/model`, choose Opus 4.8, and enable high/extended thinking). Then paste the
block below.

---

Read CLAUDE.md, thesis_revision_plan.md, and REVISION_README.md fully before doing
anything. Think hard and plan before you run any command. This repo holds a
diagnostic thesis and an examiner-revision effort with 19 concerns; most heavy
experiments are already run, so your first job is to find out what exists, not to
launch jobs.

Work in this order and keep a running log in REVISION_LOG.md with timestamped
entries for every audit result, decision, fix, launch, and resulting number.

1. AUDIT. Enumerate every results_* and status_* folder. For each, list the
   run_names present (JSON = done, .failed in status = failed, .lock without JSON =
   stuck). Grep across all folders for spearman_surrogate_vs_true and
   likelihood_trap and anisotropy outputs so you find diagnostics that may have run
   under ad-hoc names. Produce a table mapping the 19 concerns to: already done,
   failed, or missing. Write it to REVISION_LOG.md. Do not run anything yet.

2. FIX THE KNOWN BUG. In diagnostics/run_diagnostic.py, exp_linearization, the
   line that computes the surrogate (delta_e @ g) crashes under a bfloat16 model
   with "addmv input tensors must have the same dtype". Cast both operands to
   float32 at the matmul, and audit the whole function for the same mismatch under
   bf16 (true_delta and any other matmul/einsum). Add a tiny guard or comment.
   Verify with a short dry check on GPT-2 in bf16 if cheap. Log the fix.

3. PLAN THE GAPS. From the audit, decide the minimum set of jobs to run: only the
   missing or failed ones. If a diagnostic already exists under a different
   run_name, point the analysis at it or symlink it to the expected name instead of
   re-running. Prefer float32 where a model fits; use the dtype fix so bf16 Llama
   works on 24 GB. Present the plan and the GPU cost before launching. Wait for my
   go-ahead on anything above single-experiment cost.

4. RUN. Use a FRESH status dir (for example status_rev_v2) so stale locks from the
   old grid do not shadow new jobs. Keep each queue call on one out_dir, and if you
   need reset_incomplete.sh call it with that same out_dir. For the queue, follow
   REVISION_README.md phases. Generate manifests with gen_manifest_revision.py, but
   trim any line whose result already exists per your audit.

5. ANALYZE. Once diagnostics exist, run reconcile_numbers.py and
   analyze_likelihood_trap.py. Then run the light phase (kl_baselines,
   model_divergence, judge generate, per-method oracle) and, after you have
   verified the SEDD hooks against the upstream repo, the experiments phase. Run
   the SEDD dry-run first.

6. REPORT. Produce results_revision/numbers.json (via reconcile) and a
   concern-by-concern status section in REVISION_LOG.md: for each concern, the
   number or result, which file it came from, and whether it is settled or still
   needs a thesis-writing edit. Flag the writing-only concerns (3 step 1 and 4, 4
   scope, 7 derivation, 13, 14, 18, 19) as not-code so I know to handle them in
   LaTeX.

Constraints: do not re-run a completed job; do not launch anything destructive or
above single-experiment cost without stating the plan first; no em-dashes in any
prose you write. When you report the concern-1 stats, describe them as bounded
effects with the min-detectable-difference, not as "no effect". For concern 11,
report the paired cons_only minus cons_random contrast as the trustworthy number,
since the raw continuation/mucola gains are dominated by a fixed sentiment bias.

Whenever you are unsure what the thesis actually claims (a number, a table, a
definition, or the exact wording of a sentence), read the LaTeX source in the
`Doc/` folder rather than guessing. Those `.tex` files are the ground truth for
what was written. For the reconcile step (concerns 6, 15, 17) pull every reported
value from the `.tex`, not from memory, then diff against numbers.json. For the
writing-only concerns, quote the current wording from the `.tex` with its file and
line before proposing an edit. If the folder is named something other than `Doc/`,
find the `.tex` files first and use whatever folder holds them.
