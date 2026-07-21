# Revision Log

Timestamped log of audits, decisions, fixes, launches, and resulting numbers for
the 19-concern examiner revision. Newest entries appended at the bottom.

---

## 2026-07-21 12:40 CEST - Phase 1 AUDIT (no jobs launched)

Read CLAUDE.md, thesis_revision_plan.md, REVISION_README.md in full. Enumerated
every results_* and status_* folder. Key correction to REVISION_README's stale
"state of folders" note: **most revision analysis has already been run** and
`results_diagnosis` is NOT empty of upstream inputs (the diagnostics live in
`results_diag`).

### Folder inventory

| Folder | Contents | Meaning |
|--------|----------|---------|
| results_gpt2_v2 | 29 json + 29 csv | grid, incl. per-method `dls.*.oracle.s50` for all 3 methods |
| results_llama | 29 json + 29 csv | grid, incl. per-method `dls.*.oracle.s50` for all 3 methods |
| results_gfn | 87 json + 87 csv | GFlowNet grid |
| results_constrained | 40 json | constrained (concern 11 input) |
| results_probe | 9 json | constrained probe (concern 11 input) |
| results_diag | 11 json + 9 csv + 3 npz | diagnostics: gpt2sft + 3 gfn variants (linearization, likelihood_trap), gpt2sft+llama anisotropy, gpt2sft traces |
| results_diagnosis | 2 csv, 0 json | **llama diagnostics both incomplete** (see failures) |
| results_revision | 7 json | analysis outputs already produced (see below) |
| results_rerun | 23 json | gpt2-large cls/dls reruns (oracle + s50/s100) |
| results/ | 1 csv | single stray dls run |

### Status folders

- status_diag: 12 locks, **1 .failed** (`diag_linearization_llama3-8b`), plus one
  stuck lock (`diag_likelihood_trap_llama3-8b`: lock + log, no .done, no JSON).
- status_rev: only `rev_ltrap_within` and `rev_reconcile` locks (the two no-GPU
  analyses that depend on diagnostics). No GPU-phase jobs (klbase/divergence/judge/
  continuation/sedd) have ever been launched.
- status_constrained / status_gfn / status_gpt2_v2 / status_llama: all locks, 0
  failed (grid + constrained fully done).

### Two confirmed diagnostic failures (llama)

1. `diag_linearization_llama3-8b` -> **FAILED** at run_diagnostic.py:283,
   `surrogate = (delta_e @ g)`, "addmv input tensors must have the same dtype, but
   got Float, BFloat16, and Float". This is exactly known issue 1 (bf16 load). CSV
   in results_diagnosis has 1 line (header only). No JSON.
2. `diag_likelihood_trap_llama3-8b` -> **STUCK**: log shows progress to 240/500
   then stops; no .done, no .failed, stuck lock, empty CSV (0 lines). Killed
   mid-run, not the dtype bug. Needs a clean rerun.

### Analysis outputs already present in results_revision (verified populated)

- `rev_stats_gpt2.json` (18 comparisons, 11 groups), `rev_stats_llama.json` (18),
  `rev_stats_gfn.json` (54, 33 groups) -> **concern 1 DONE**. Each comparison has
  paired mean diff, 95% CI, Wilcoxon p, TOST (margin = 0.05 x policy mean KL),
  min_detectable_diff_80pow, and an is_len_beta1_variant flag.
- `rev_constrained.json` (8 groups, 20 contrasts) -> **concern 11 DONE**. Reports
  `constraint_direction_value` = cons_only - cons_random per group.
- `rev_ltrap_within.json` -> **concern 10 PARTIAL**: gpt2sft + 3 gfn variants
  present; **Llama missing** (its diagnostic never completed).
- `rev_reconcile.json` + `numbers.json` -> **concerns 6/15/17 PARTIAL**: has
  gpt2sft + 3 gfn linearization spearman, length slopes, MH within/boundary split,
  config count, spearman phrasing. **Missing Llama linearization spearman and Llama
  likelihood_trap** (llama diagnostics incomplete). Must re-run reconcile after the
  two llama diagnostics finish.

### Key numbers captured from existing outputs (for the report)

- Concern 1 (bounded null): example gpt2 dls/mh/gn/free/s50 policy vs gradnorm:
  mean diff = +0.171 KL, 95% CI [-0.285, +0.619], Wilcoxon p = 0.400, TOST not
  equivalent (CI wider than margin 0.327), min detectable diff @80% power = 0.652.
  Report as bounded effect with min-detectable-diff, not "no effect".
- Concern 6b length slope (gpt2sft likelihood_trap): uncensored -1.123
  nats/token (r = -0.078, n=1500); censored slope null (nearly all at cap). gfn-lb1
  uncensored -0.505, censored -2.361 (n=1148). Reconciles the -1.12 vs -0.11.
- Concern 6a/7 MH split (traces_gpt2sft_mh.csv): CLS policy no-MH within-cell accept
  0.034% (n=2906), boundary 3.67% (n=7094); CLS policy MH within-cell 0.627%,
  boundary 8.56%; DLS policy MH within-cell 100%, boundary 9.27%.
- Concern 6c inter-token distance: mean pairwise 2.773, nearest-neighbour 1.822
  (gpt2sft anisotropy); "mean_inter_token_distance" in linearization = 2.354.
- Concern 15 config count: 11 distinct diagnostic run_names present vs 240 full
  factorial; note recommends enumerating pruned cells in an appendix.
- Concern 17 spearman: max |rho| = 0.057 across the 4 present models; suggested
  sentence "negligible in magnitude (|rho| < 0.06), n = 400,000 candidate pairs".
- Concern 11 constraint-direction contrast (cons_only - cons_random), the
  trustworthy number: continuation/mucola negative target +27.33, positive target
  +36.67 (matches the +27 to +37 story); "ours"/DLS setup near zero.
- Concern 12 oracle fairness: per-method oracle runs
  `gpt2-large.dls.{policy,random,grad_norm_preserved_random_dir}.nomh.gn.oracle.s50`
  ALREADY EXIST in both results_gpt2_v2 and results_llama. The light-phase manifest
  would regenerate these same run_names -> trim them, reuse existing.

### Concern-to-status table

| # | Concern | Status | Source / gap |
|---|---------|--------|--------------|
| 1 | Stats behind null | DONE | rev_stats_{gpt2,llama,gfn}.json |
| 2 | KL baselines | MISSING (GPU) | rev_klbase_* never run (Phase C) |
| 3 | Circular eval | MISSING (GPU) + WRITING | judge gen/score never run; KL eq is writing |
| 4 | Non-grad sampler scope | MISSING (GPU) + WRITING | gibbs baseline inside concern 2; scope narrowing is writing |
| 5 | Did tuning change energy | MISSING (GPU) | rev_divergence_* never run; compare_models.py available |
| 6 | Numerical inconsistencies | PARTIAL | rev_reconcile done for gpt2/gfn; needs llama re-add |
| 7 | CLS energy / within-cell | PARTIAL + WRITING | MH split measured; derivation is writing |
| 8 | SEDD positive control | MISSING (GPU) | dry-run + real run never run; hooks need repo verify |
| 9 | Task generality | MISSING (GPU) | rev_continuation_* never run |
| 10 | Likelihood-trap confound | PARTIAL | within-strategy done for gpt2/gfn; Llama trap missing |
| 11 | Constraint CIs + anomaly | DONE | rev_constrained.json |
| 12 | Oracle fairness | DONE (reuse) | per-method oracle.s50 already in grid folders |
| 13 | "schedule worked" undefined | WRITING | one sentence in 5.1 |
| 14 | len_beta coverage | WRITING | soften / list values |
| 15 | Config count | PARTIAL | in rev_reconcile; appendix enumeration is writing |
| 16 | Seeds / variance | MISSING (GPU) | seed reruns never run |
| 17 | Spearman phrasing | DONE | in rev_reconcile; writing edit |
| 18 | Prose repetition | WRITING | editing pass |
| 19 | Cost comparison | WRITING | small table |

### Immediate blocker resolved

The only thing gating the no-GPU analyses is the two incomplete llama diagnostics.
Fixing the bf16 bug (below) unblocks the linearization one; both need a rerun.

---

## 2026-07-21 12:40 CEST - FIX known issue 1 (bf16 dtype crash)

File: `diagnostics/run_diagnostic.py`, `exp_linearization`, line 283.

Root cause: `g = grad_wrt_position(...).float()` (line 250) is float32, but the
embedding matrix `E` is bf16 under a bf16 model load, so `delta_e = E[cands] -
e_cur` is bf16 and `delta_e @ g` mixes dtypes -> addmv crash. Audited the whole
function: the ONLY matmul/einsum is line 283; `seq_logprob_terms` and
`grad_wrt_position` internally cast logits with `.float()` and the true-delta loop
casts outputs via `.float().cpu().numpy()`, and `base_tot` is a python float via
`.item()`, so no other bf16/float32 mix exists.

Fix: cast `delta_e` to float32 at construction:
`delta_e = (E[cands] - e_cur.unsqueeze(0)).float()`, and use it for both the
surrogate matmul and the `dist` norm. Added an explanatory comment.

Verification: ran a cheap bf16 dry check on GPT-2-large
(`--dtype bfloat16 --n_seqs 3 --n_cand 40`) on GPU 0 -> exit 0, JSON written, no
crash (wall 9.6s). The previously-crashing line now runs under bf16.

Hardware note: all 9 GPUs are free at ~49 GB each (not 24 GB), so Llama can run in
float32 too; bf16 still preferred for speed/footprint and the fix makes it safe.

---

## 2026-07-21 12:45 CEST - LAUNCH the two failed/stuck llama diagnostics (single-experiment each)

Decision: run only the two gaps that are failed/stuck (not the whole diagnostics
phase, which is otherwise complete for gpt2sft + 3 gfn variants + llama anisotropy).
Each is single-experiment cost. Routed both to `results_diag` (where the gpt2/gfn
diagnostics live and where reconcile globs), bf16, matching the family's per-exp
seq counts (verified from the existing CSVs): linearization n_seqs=200 (-> 400k
rows), likelihood_trap n_seqs=500 (-> 3k rows). Cleared their stale status_diag
markers first (ran directly, not via the queue).

- GPU0: `diag_linearization_llama3-8b`  (bf16, n_seqs=200) pid 2930935
- GPU1: `diag_likelihood_trap_llama3-8b` (bf16, n_seqs=500) pid 2931993

**Fix validated on the real model**: linearization passed run_diagnostic.py:283
(the previously-crashing matmul) on Llama-3-8B in bf16 and is writing rows
(10/200 seqs, eta ~34 min). Logs: `status_diag/diag_*_llama3-8b.rerun.log`.

When both JSONs land in results_diag, re-run the two dependent no-GPU analyses to
pick up Llama (they currently omit it):

```
python revision/reconcile_numbers.py --results_dirs results_gpt2_v2 results_llama results_gfn results_diag --run_name rev_reconcile --out_dir results_revision
python revision/analyze_likelihood_trap.py --results_dir results_diag --run_name rev_ltrap_within --out_dir results_revision
```

### Still-missing GPU work (NOT launched; awaiting go-ahead, above single-experiment cost)

- Phase C light: `rev_klbase_*` (concern 2 + gibbs for concern 4), `rev_divergence_*`
  (concern 5), `rev_judge_gen_*` (concern 3). Per-method oracle (concern 12) is
  ALREADY satisfied by existing grid runs -> trim from manifest.
- Phase D experiments: `rev_judge_score_*` (concern 3, needs Phase C gen output),
  `rev_sedd_*` (concern 8, run dry-run first + verify hooks vs upstream repo),
  `rev_continuation_*` (concern 9), seed reruns (concern 16).

---

## 2026-07-21 12:55 CEST - User go-ahead: full light + experiments batch, SEDD dry-run only

Decisions from user: run the full light + experiments batch; SEDD dry-run only for
now (real run deferred, needs repo clone + checkpoint + hook verification).

Env note: `run_queue.sh --env gfn` uses `conda activate`, which is a no-op for this
venv, but bare `python` already resolves to the gfn venv
(`gfn-lm-tuning/gfn/bin/python`, torch 2.12.0+cu130, all deps present), so the
manifest commands run correctly. I launched `worker.sh` directly (one per GPU)
instead of the tmux queue, for cleaner background monitoring; the lock/done/failed
+ resume semantics are identical.

Concern 12 reuse: symlinked the 3 completed per-method oracle runs
(`gpt2-large.dls.{policy,grad_norm_preserved_random_dir,random}.nomh.gn.oracle.s50`,
oracle:True, matching config) from results_gpt2_v2 into results_revision and TRIMMED
them from the light manifest (honors "do not re-run a completed job").

**LAUNCHED light phase** (8 real gaps), status_rev_v2, out_dir results_revision,
4 workers on GPUs 2 3 4 5:
- `rev_klbase_{gpt2sft,gfn-lb0-500,gfn-lb0-2000,gfn-lb1-500}` (concern 2 + gibbs/concern 4)
- `rev_divergence_{gfn-lb0-500,gfn-lb0-2000,gfn-lb1-500}` (concern 5)
- `rev_judge_gen_gpt2sft` (concern 3 generate stage)

Early check at +75s: all 4 klbase claimed and progressing (gpt2sft 80/200), no
failures. kl_baselines pipeline confirmed working. Experiments phase queued next,
after light completes (judge_score depends on judge_gen).

---

## 2026-07-21 13:05 CEST - Light-phase results (concerns 2 and 5) + thesis .tex diff (6, 15, 17)

Light phase finished 7/8 (judge_gen still running), zero failures.

### Concern 2 - KL baselines (mean final KL; ground_truth=floor, random=ceiling)

| model | ground_truth | untouched | random_token | cond_argmax | cond_sample | cond_topk_rescore | gibbs |
|-------|-----|-----|-----|-----|-----|-----|-----|
| gpt2sft | 0.00 | 9.14 | 9.39 | 8.24 | 8.62 | **4.43** | 6.69 |
| gfn-lb0-500 | 0.00 | 8.81 | 9.07 | 8.25 | 8.43 | **4.46** | 6.77 |
| gfn-lb0-2000 | 0.00 | 9.61 | 9.97 | 8.77 | 9.02 | **5.24** | 6.83 |
| gfn-lb1-500 | 0.00 | 8.20 | 8.36 | 20.17 | 21.11 | 18.27 | 6.20 |

Headline (concern 2 + 4): ground-truth anchors KL=0 (100% exact match, confirms the
metric floor). On the three well-behaved energies, **conditional top-k rescoring -
one forward pass - reaches 4.4-5.2 KL, beating every Langevin grid config (~6.5-8)**,
and the non-gradient **Gibbs sampler reaches 6.7-6.8** (concern 4's "non-gradient
sampler on the same energy"). So the exact-energy forward pass does the work and the
gradient adds nothing on top. This is the sharpened, quotable gradient-fallacy story
the plan wanted. exact_match_pct: cond_topk_rescore 33%, gibbs 18.5% on gpt2sft.
Note: gfn-lb1-500's conditional baselines are anomalously high (18-21 KL, worse than
untouched), consistent with the len_beta=1 gibberish degeneracy; Gibbs still works
there (6.2). Flag but do not over-read.
6e cross-check: untouched-corruption KL is 8.2-9.6; Table A.1 CLS final KL is 8.083
(gpt2), i.e. the CLS chain barely improves on the corrupted input, consistent with
the "MH rejects, chain does not move" explanation. Not a bitwise match (different
masking in run_revision vs the grid), so treat as corroboration, not proof.

### Concern 5 - model divergence (did LoRA move the energy?)

| variant | mean_abs_loglik_diff | pearson(base,tuned) | spearman | next-tok KL |
|---------|-----|-----|-----|-----|
| gfn-lb0-500 | 31.0 | 0.983 | 0.984 | 1.023 |
| gfn-lb0-2000 | 36.0 | 0.979 | 0.980 | 1.345 |
| gfn-lb1-500 | 57.3 | 0.770 | 0.835 | 3.053 |

Verdict: the tuning moved the energy **substantially** (31-57 nats mean abs loglik
difference, next-token KL 1.0-3.1), most for the length-normalized lb1-500. Yet the
linearization correlations on these same tuned energies are flat (0.057, 0.024,
0.046 from rev_reconcile). Large energy change + unchanged landscape diagnostics =
the plan's condition under which "amortization does not repair the energy" STANDS and
is now supported. This closes the logical gap the examiner flagged. (Do NOT retitle
to "light-touch LoRA left the energy unchanged" - the opposite is true.)

### Thesis .tex diff (ground-truth extraction done by subagent; file:line in that report)

Several examiner concerns are ALREADY RESOLVED in the current Doc/ .tex:
- 6b: no `-0.11`/`r=-0.00` anywhere; text uses -1.12 consistently. numbers.json
  confirms uncensored slope -1.123 (r=-0.078). BUT the gpt2sft censored slope is
  null (frac_hit_cap=0.997, nearly everything at the 40-token cap), so the -1.12 is
  an uncensored fit on capped data - add one sentence acknowledging the censoring
  (gfn-lb1-500: uncensored -0.505 vs censored -2.361 shows the cap matters).
- 6c: all three distances confirmed against sources (NN 1.822, pairwise 2.772,
  linearization mean 2.354). The issue is purely definitional - three different
  statistics are each loosely called "inter-token distance"; define each once.
- 6d: no `0.89` linearization-radius annotation exists; the caption already denies a
  clean threshold. RESOLVED. (The only 0.89 in text is a beam-string rep4 rate.)
- 6e: CLS MH / no-MH rows identical (8.083/7.799/8.198/13.732); caption explains it.
  No "46 cells" in text. Corroborated by the klbase untouched reference above.
- 6a: text reports 0.03% within-cell / 3.7% crossed / 2.6% overall. rev_reconcile
  mh split for ('cls','policy',mh=False) = within 0.034% / boundary 3.665% - MATCHES.
  BUT the text attributes these to "the correction ENABLED"; the matching trace is
  mh=False. The mh=True trace gives within 0.627% / boundary 8.56%. Flag this
  attribution for the author (concern 6a/7): confirm which run the 0.03/3.7 came from.
- 15: **145 = 5 energy functions x 29 configs each**, where 29 = 8 CLS (policy,random
  x 4 gn/schedule variants) + 21 DLS (policy,gradnorm,random x 7 variants). Verified:
  results_gpt2_v2 29 + results_llama 29 + results_gfn 87 (=3x29) = 145. This is the
  counting formula for the appendix (concern 15). The appendix's nominal 2x3x2x2x2x5
  =240 overcounts because CLS drops the gradnorm method and some gn/schedule cells.
- 17: text says rho=-0.011 "which is to say zero"; n=400,000 only in a source
  comment. Change to the effect-size phrasing "negligible in magnitude (|rho|<0.06
  across all models), n=400,000 candidate pairs" and give n in the caption.

### Concern 4 scope wording (writing): abstract already says the narrow "not a usable
search direction"; the CONCLUSION (07_conclusion.tex:4) still says the broad "not a
usable energy function for sampling on discrete text". Narrow the conclusion to match
the abstract, and add the Mix-and-Match / twisted-SMC related-work paragraph. The
Gibbs baseline above is the in-platform evidence that the failure is in the gradient,
not the energy.

---

## 2026-07-21 13:20 CEST - Experiments phase (independent jobs) results (concerns 8, 9, 16)

All 6 independent experiment jobs DONE, 0 failures (supervisor exit 2 was just the
trailing `ls *.failed` returning non-zero on no matches).

### Concern 8 - SEDD dry-run
`rev_sedd_dryrun.json` written in 3s via the fake bundle: the whole linearization
loop (score hook -> per-position logprob -> surrogate vs truth) runs end to end and
emits the CSV+JSON. Pipeline validated. Real run still deferred (needs repo clone,
checkpoint, and verification of the two ADAPT hooks vs
github.com/louaaron/Score-Entropy-Discrete-Diffusion).

### Concern 9 - task generality (prefix continuation, 20-token span, 100 seqs)
`rev_continuation_gpt2sft.json`, mean final KL: policy 8.818 [8.61, 9.02], gradnorm
8.850 [8.65, 9.04], random 8.850 [8.65, 9.04]. Policy vs random = -0.03, CIs fully
overlapping; gradnorm and random are numerically identical. The null result (policy
indistinguishable from random) reproduces on the free-form continuation task, not
just masked recovery. (Absolute KL ~8.8 is close to untouched, i.e. little recovery
on long spans, but the finding is the policy==random equivalence.) Removes the
"only a toy task" objection.

### Concern 16 - across-seed variance
Representative config `gpt2-large.dls.policy.mh.gn.free.s50`, data_seed 1000-1003:
final KL = 6.348, 6.589, 6.150, 6.291; mean 6.345, **sd 0.183**, range 0.438,
accept_rate ~9.6-10.0%. The across-seed sd (0.183) is BELOW the concern-1
equivalence margin (0.317 = 0.05 x mean). This justifies that margin empirically:
run-to-run noise alone is ~0.18 KL, so the policy-vs-random mean diffs in concern 1
(+0.044, +0.171) live inside seed noise, whereas the len_beta=1 gap of +0.670
exceeds it (supporting concern 1 step 6 - treat it as a real, harmful effect).
State seeds (1000-1003) and hardware (single 49 GB GPU, gfn venv, torch 2.12+cu130)
in Chapter 4 per concern 16.

---

## 2026-07-21 14:15 CEST - Llama diagnostics complete; reconcile + ltrap refreshed

Both llama diagnostics finished (linearization 36 min, likelihood_trap 90 min) and
the waiter auto-reran reconcile + analyze_likelihood_trap over
`results_gpt2_v2 results_llama results_gfn results_diag`.

### Concern 6/17 - cross-architecture linearization null now on ALL 5 energies
per_run_spearman (surrogate vs truth): gpt2sft -0.011, **llama3-8b 0.021**,
gfn-lb0-500 0.057, gfn-lb0-2000 0.024, gfn-lb1-500 0.046. max |rho| = 0.057.
Concern-17 sentence now holds across all five: "negligible in magnitude
(|rho| < 0.06) across all models, n = 400,000 candidate pairs." Llama linearization
mean_inter_token_distance = 0.659 (vs gpt2 2.354), consistent with the ~3x tighter
Llama embedding geometry (anisotropy: NN 0.585, pairwise 0.835).

### Concern 10 - likelihood trap now measured on Llama (confound addressed)
within-strategy max |r| and pooled r:
- gpt2sft: pooled 0.361, max within 0.511
- gfn-lb0-500: pooled 0.626, max within 0.830
- gfn-lb0-2000: pooled 0.785, max within 0.821
- gfn-lb1-500: pooled 0.628, max within 0.909
- **llama3-8b: pooled 0.218, max within 0.772**
The trap holds WITHIN decoding strategy (not just pooled), on both architectures.
On Llama the pooled correlation (0.218) is weaker but the within-strategy signal
(up to 0.772) is strong, so per the plan present the within-strategy + between-
strategy pattern (greedy/beam worst) as primary, not the pooled number. Llama trap:
frac_hit_cap = 1.0 (never emits EOS, always hits the 40-token cap), frac_emitted_eos
= 0.0 - a clean degeneracy signal.

### Concern 6b length slopes (only where length varies)
gpt2sft uncensored -1.123 (r=-0.078); censored null (frac_hit_cap 0.997).
gfn-lb1-500 uncensored -0.505, censored -2.361 (n=1148). Llama and gfn-lb0-* produce
NO slope because frac_hit_cap = 1.0 (all generations at the cap) - report this as
"generations saturate the length cap, so the brevity slope is only estimable where
the cap does not bind." The -1.12 headline stands but must carry the censoring caveat.

### Concern 15 (reconfirm): the reconcile config_count field now reads 158 because it
globbed grid+diag JSONs; that is NOT the grid config count. The grid count is
145 = 5 energy functions x 29 (8 CLS policy/random x 4 variants + 21 DLS 3-method x 7),
verified directly from results_gpt2_v2 (29) + results_llama (29) + results_gfn (87).
Use that formula for the appendix; ignore the 158 in config_count.note.

Remaining: judge_score (concern 3, waiting on judge_gen which is ~90% done, slow but
confirmed live at 99% CPU + active GPU).

---

## 2026-07-21 14:35 CEST - judge_score done; ALL GPU work complete, zero failures

### Concern 3 - external-judge rescoring (Llama-3 judge over 600 recovered seqs)
`rev_judge_score_gpt2sft.json`, mean judge perplexity: policy 178.4, gradnorm 181.3,
random 181.3 (gradnorm == random exactly); judge NLL/token 4.438 vs 4.450. The
differences are under 2% and no method wins - an INDEPENDENT model agrees the methods
are indistinguishable, so the "model grades its own homework" objection dissolves.
Do NOT quote rank_spearman_kl_vs_judge_ppl = -1.0 as a finding: it is computed over
only 3 method-means separated by sub-2% gaps, i.e. noise. Report it as
indistinguishability under the external judge. (KL equation itself is writing:
transcribe avg_kl_for_fill from run_revision.py into Section 4.4.)

Final state: no `.failed` markers in any status dir; all 24 expected JSONs present in
results_revision; no processes left running.

---

# FINAL REPORT - concern by concern (2026-07-21 14:35 CEST)

Legend: [SETTLED] number is in hand, no code left. [SETTLED+WRITE] number in hand,
needs a LaTeX edit. [WRITE] pure writing, no job produces it.

1.  [SETTLED] Stats behind the null. `results_revision/rev_stats_{gpt2,llama,gfn}.json`
    (18/18/54 comparisons). Report as BOUNDED effects: e.g. gpt2 dls/mh/gn/s50 policy
    vs gradnorm mean diff +0.171 KL, 95% CI [-0.285, +0.619], Wilcoxon p=0.40, TOST
    not-equivalent only because CI > margin (0.327), min detectable diff @80% power
    0.652. Treat len_beta=1 gap (+0.670) separately - it exceeds seed noise (see 16),
    so it is a real, harmful effect, not noise.

2.  [SETTLED+WRITE] KL baselines. `rev_klbase_*`. gpt2sft mean KL: ground_truth 0.00
    (100% exact match), untouched 9.14, random 9.39, cond_argmax 8.24, cond_sample
    8.62, cond_topk_rescore 4.43, gibbs 6.69. One forward pass (topk-rescore) beats
    every Langevin config. Add as reference rows/lines to Table 5.1 + trajectory figs.

3.  [SETTLED+WRITE] Circular evaluation. `rev_judge_gen_gpt2sft.json` +
    `rev_judge_score_gpt2sft.json`: independent Llama judge ppl 178-181, methods
    indistinguishable. WRITE: the exact KL equation in Section 4.4 (source =
    avg_kl_for_fill in run_revision.py). Human eval (step 4) NOT done - out of scope.

4.  [SETTLED+WRITE] Non-gradient sampler / scope. Gibbs baseline 6.69 KL (in rev_klbase)
    works where the gradient fails, on the same energy - the failure is in the gradient,
    not the energy. WRITE: narrow the conclusion (07_conclusion.tex:4 "not a usable
    energy function for sampling") to match the abstract's narrower "not a usable search
    direction"; add the Mix-and-Match + twisted-SMC related-work paragraph (Chapter 3).

5.  [SETTLED] Did tuning move the energy. `rev_divergence_*`. mean abs loglik diff
    31.0 / 36.0 / 57.3 nats; next-tok KL 1.02 / 1.35 / 3.05; base-tuned pearson
    0.98 / 0.98 / 0.77. Energy moved substantially yet linearization stays flat, so
    "amortization does not repair the landscape" STANDS (do not retitle to "unchanged").

6.  [SETTLED+WRITE] Numerical inconsistencies. `rev_reconcile.json` + numbers.json.
    Most already consistent in current .tex (no -0.11 slope, no 0.89 radius, no 46
    cells, no 6.4/7.4 rates). Remaining: (6a) confirm which run the 0.03%/3.7%
    acceptance came from - matching trace is mh=FALSE (within 0.034/boundary 3.665),
    text says "correction enabled"; mh=TRUE trace is 0.627/8.56. (6b) add censoring
    caveat to the -1.12 slope (censored slope is null for gpt2, cap binds).
    (6c) define the three "inter-token distances" once (NN 1.82, pairwise 2.77,
    linearization 2.35). All are WRITE fixes; numbers verified.

7.  [SETTLED+WRITE] CLS energy + within-cell acceptance. Measured MH split in
    rev_reconcile.mh_acceptance_by_boundary (cls policy nomh: within 0.034%, boundary
    3.665%; cls policy mh: 0.627%/8.56%; dls policy mh: 100%/9.27%). WRITE: the CLS
    energy equation in Section 2.4 (currently prose only - no eq for
    E=-log p(proj(s))) and the within-cell acceptance derivation.

8.  [SETTLED-pilot] SEDD positive control. `rev_sedd_dryrun.json` - pipeline validated
    with a fake bundle. Real run DEFERRED per user: needs cloning
    github.com/louaaron/Score-Entropy-Discrete-Diffusion, a checkpoint, and verifying
    the two ADAPT hooks in run_sedd_linearization.py (sedd_score_row,
    sedd_position_logprob). Highest-upside remaining optional item.

9.  [SETTLED] Task generality. `rev_continuation_gpt2sft.json`: policy 8.818
    [8.61,9.02] vs random 8.850 [8.65,9.04], gradnorm == random. Null reproduces on
    free-form 20-token continuation, not just masked recovery.

10. [SETTLED] Likelihood-trap confound + Llama. `rev_ltrap_within.json`. Within-strategy
    max |r|: gpt2sft 0.51, gfn 0.83/0.82/0.91, llama 0.77. Trap holds within strategy
    on both architectures. On Llama present within+between-strategy (greedy/beam worst),
    not pooled (pooled r only 0.218). Llama frac_hit_cap=1.0.

11. [SETTLED] Constraint direction + anomaly. `rev_constrained.json`. Trustworthy
    number = cons_only - cons_random: continuation/mucola +27.3 (neg target) / +36.7
    (pos target); "ours"/DLS setup ~0. Both target labels analyzed (symmetric design).
    Raw gains are bias-dominated (every arm flips sign with target) - report the paired
    contrast, per issue-2 in CLAUDE.md. cons_only anomaly = constraint-only leaves the
    fluent manifold (WRITE one paragraph).

12. [SETTLED-reuse] Oracle fairness. Per-method oracle runs
    gpt2-large.dls.{policy,gradnorm,random}.nomh.gn.oracle.s50 already existed in
    results_gpt2_v2 (oracle:True per method); symlinked into results_revision, not
    re-run. WRITE: state in 4.3 that the oracle sweep was run independently per method.

13. [WRITE] "schedule worked on Llama-3". 05_results.tex:9 "a step-size schedule that
    had already been made to work on Llama-3". Define "work" = calibrated motion, not
    successful guidance (the section's own motion-vs-guidance vocabulary).

14. [WRITE] len_beta coverage. 05_results.tex:273 "no setting between them was found to
    produce good text" - only lb0 and lb1 were run. Soften to "the two endpoints
    produce opposite degeneracies and we did not find an intermediate that resolves
    both", or list intermediate values tried.

15. [SETTLED+WRITE] Config count. 145 = 5 energy functions x 29 configs
    (8 CLS = policy,random x 4 gn/schedule variants; 21 DLS = policy,gradnorm,random
    x 7 variants). Verified: gpt2_v2 29 + llama 29 + gfn 87 = 145. WRITE the formula +
    pruned-cell enumeration in an appendix (nominal factorial 2x3x2x2x2x5=240 overcounts
    because CLS drops gradnorm and some gn/schedule cells).

16. [SETTLED+WRITE] Seeds/variance. seeds 1000-1003 of gpt2-large.dls.policy.mh.gn.
    free.s50: final KL 6.348/6.589/6.150/6.291, mean 6.345, sd 0.183, range 0.44. The
    sd (0.183) < the concern-1 margin (0.317), justifying the margin. WRITE seeds +
    hardware in Chapter 4 + a code/data availability statement.

17. [SETTLED+WRITE] Spearman phrasing. max |rho| = 0.057 across all 5 models
    (gpt2 -0.011, llama 0.021, gfn 0.024/0.046/0.057), n=400,000 pairs. WRITE: replace
    "which is to say zero" (05_results.tex:148) with the effect-size phrasing
    "negligible in magnitude (|rho| < 0.06 across all models), n = 400,000 pairs".

18. [WRITE] Prose repetition. Editing pass - compress the mini-recaps at 5.5, 5.10,
    6.2 and chapter transitions to one sentence each.

19. [WRITE] Computational-cost table. Forward+backward passes per recovered token for
    DLS, CLS, conditional resampling, Gibbs + wall-clock. (Gibbs wall from rev_klbase,
    continuation wall 2157s/100seqs, judge_score 24.8s/600 as data points.)

WRITING-ONLY concerns for the author to apply in LaTeX (no code produces them):
3 step 1 (KL equation), 3 step 4 (human eval), 4 (scope narrowing + related work),
7 (CLS energy equation + derivation), 13, 14, 18, 19. Plus the WRITE portions of
2, 6, 11, 12, 15, 16, 17 listed above.

All computed numbers live in `results_revision/`; the flat index for the LaTeX diff
is `results_revision/numbers.json` (refreshed with all 5 models).
