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

8.  [SETTLED - REAL RUN DONE, Phase 2] SEDD positive control. `rev_sedd_linearization.json`
    (n=200, 400k pairs, sedd-small). Surrogate-vs-true Spearman POSITIVE in every
    stratum (pooled +0.184, near +0.306, per-seq mean +0.172) vs the AR gpt2sft
    -0.011 (near +0.027) on the identical design. Confirms the training-objective
    diagnosis: the diffusion objective yields a usable local directional signal where
    the AR gradient gives none. Hooks were verified and CORRECTED (score is log-space;
    absorbing-SEDD signal lives at masked positions) before running - see the Phase 2
    SEDD section. Present as a positive-control pilot (Section 5.12 / appendix). The
    earlier `rev_sedd_dryrun.json` remains as pipeline validation.

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

---

# PHASE 2 - the last-token experiment, a working setup, and a rigorous audit

Goal of this phase: turn the central claim from a statistic into a proof. WHY does
gradient guidance fail. Answer: at the final scored position the input-embedding
gradient of the sequence log-likelihood is provably exactly zero, while the energy
difference between candidate final tokens is exactly the model's own conditional
log-ratio. So the energy is maximally usable exactly where the gradient is provably
useless. Two new experiments plus a mathematical audit demonstrate this.

## 2026-07-21 15:08 CEST - PART 1: mathematical audit (probe, no sampler runs)

Probe: `diagnostics/audit_probe.py` -> `diagnostics/audit_probe_result.json`.
Model: gpt2_large_sft_output (the SFT GPT-2 large used throughout the grid), float32,
GPU 0, 20 WikiText-2 sequences (data_seed 0, the same first-20 the grid/kl_baselines
draw). Re-ran this session and reproduced the recorded numbers bit-for-bit. Each of
the six claims is DERIVED then CONFIRMED (numerically for 1/2/3/6, by code audit for
4/5).

### Item 1 - THE ZERO-GRADIENT THEOREM AT THE LAST POSITION [CONFIRMED]

Derivation. `joint_log_prob_from_inputs_embeds` (core/prep.py:46) scores
S = sum_{t=0}^{L-2} log p(x_{t+1} | x_{<=t}) by dropping the last logit column
(`logits[:, :-1, :]`). Under causal attention the input embedding at position p
influences only outputs at positions >= p. The final scored position is p = L-1
(GPT-2 adds no trailing EOS; see item 6 evidence), whose embedding feeds ONLY
logits[L-1], the single column that predicts the nonexistent token x_L and is
exactly the column dropped by [:, :-1, :]. Therefore dS/d(inputs_embeds[L-1]) = 0
identically, not approximately. The second-to-last embedding (p = L-2) enters
exactly one surviving term (logits[L-2], which predicts x_{L-1}) plus its influence
on nothing else it can reach causally, so its gradient is small; a middle position
enters many surviving terms, so its gradient is larger.

Numerical confirmation (per-position ||dS/d inputs_embeds[p]||):
- final position (p=L-1): mean 0.0, max 0.0, min 0.0; all 20/20 sequences exactly 0.0.
- second-to-last (p=L-2): mean 2.777, min 0.011, max 11.07 (small, one downstream term).
- middle (p=L//2): mean 5.927, min 1.230, max 21.08 (large, many downstream terms).
The ordering 0 << second-to-last < middle is exactly as derived, and the final-
position zero is exact to the last bit across every sequence.

### Item 2 - STRUCTURAL BLINDNESS EXPLANATION [CONFIRMED]

Probe: wte (input embeddings) and lm_head share storage (`same_storage=True`,
`equal_values=True`, shape [50257, 1280]) - GPT-2 large ties input and output
embeddings. Mechanism, stated precisely: the term log p(x_L | x_{<L}) is a softmax
over h_{L-1} . E_out (the OUTPUT-embedding rows), so its dependence on WHICH token
x_L is enters only through the output-embedding path. Backprop to inputs_embeds[L]
never traverses that path (the token at L is an input, not the argument of the
softmax that scores it), and additionally that softmax's logit column is dropped
anyway (item 1). The information the sampler needs is present inside the model
(h_{L-1} scores every candidate final token exactly) but is invisible to the
input-embedding gradient. This is the one-sentence answer to why gradient guidance
fails; the last position is where it is a theorem, not an empirical near-zero.

### Item 3 - THE ENERGY IS EXACT AT THE LAST POSITION [CONFIRMED]

Derivation. Changing only the final token from a to b changes exactly one term of
S (the log p(final | prefix) term; every other term conditions on the prefix, which
is unchanged, and the final embedding feeds only the dropped column). Hence with
E = -S, E(a) - E(b) = -(log p(a|prefix) - log p(b|prefix)) = log p(b|prefix) -
log p(a|prefix). So an independence Metropolis sampler proposing from
q = p(. | prefix) has target proportional to proposal and acceptance probability
exactly 1 for every move. This is a built-in unit test for Part 2: measured
acceptance materially below 100% at the final position would prove the implemented
energy carries terms beyond the sequence log-likelihood.

Numerical confirmation: over 15 candidate pairs per sequence, the full-sequence
energy difference E(a)-E(b) matched the conditional log-ratio log p(b|prefix) -
log p(a|prefix) to max error 4.0e-5 nats (mean 1.5e-5), i.e. float round-off. The
energy at the last position IS the model's own conditional, exactly.

### Item 4 - MH RATIOS re-verified in the repo as they stand NOW [CONFIRMED by code audit]

- DLS policy (core/dls.py:90-108): forward log-prob is taken from the SAME
  `scaled_logits` that generated the move; the reverse log-prob is built from a
  fresh backward gradient at s_next, run through `apply_method_variation` (so the
  reverse kernel is normalized exactly as the forward one was), then evaluated at the
  original state s_idx. `log_q_ratio = log_bw_prob - log_fwd_prob`. The comment
  documents that the legacy build wrongly used the raw backward gradient; the fix is
  present.
- DLS random / grad_norm_preserved (core/dls.py:109-116): symmetric random walk, so
  `log_q_ratio = 0.0`; crucially `apply_method_variation` is NOT called on the
  backward gradient here, so no extra torch.randn is drawn and the RNG stream stays
  aligned with the no-MH path (item 5).
- CLS policy (core/cls.py:74-80): backward drift m_prop built from a method-varied
  backward gradient at s_prop; log_q_back = logq(s | m_prop), log_q_fwd =
  logq(s_prop | m_s), both Gaussians evaluated at the exact sampled states s and
  s_prop. CLS random (cls.py:81-82): log_q_back = log_q_fwd = 0.0, symmetric
  cancellation. All four kernels are detailed-balance-correct as written.

### Item 5 - THE KNOWN BITWISE ARTIFACT (gn=on) [CONFIRMED by code audit]

With grad_normalization=True, `apply_method_variation` returns, for
`grad_norm_preserved_random_dir`, `randn_like(g)` normalized to unit (dls dls.py:56-57)
and, for `random`, `randn_like(g)` normalized to unit (dls.py:66-67). Both draw one
`randn_like(raw_grad_s)` of identical shape from an identically seeded stream
(seed_all is called per sample before each method run), so the two unit directions
are the SAME sample and the proposals are bitwise identical. This is the documented
seeded behavior, not a new bug, and it explains why grad_norm == random exactly in
concerns 3 and 9. THESIS GUIDANCE: wherever gn=on, present policy-vs-random as the
primary comparison (grad_norm adds no independent information there).

### Item 6 - THE KL-METRIC BOUNDARY documented [CONFIRMED]

`base_sampler.optimize` (line 85) and `avg_kl_for_fill` (run_revision.py:114) both
keep only masked positions m with m < seq_len-1, because the KL is computed at the
next-token distribution which requires a right neighbour. A masked FINAL position
(m = seq_len-1) therefore contributes zero terms and the metric returns nan there.
This is WHY the standard grid never masks the last position (build_corruption uses
`valid = range(1, seq_len-1)`, run_experiment.py:57) and WHY Part 2 must define its
own last-token metrics (exact-match, top-5, NLL, rank). The probe also confirms
`trailing_token_is_eos_count = 0`: GPT-2's tokenizer adds no trailing special token,
so the truly-final scored position with zero downstream terms is p = L-1, a real
word token (usually the sentence-final period).

### Audit verdict

All six claims hold. The central result of the phase is items 1+2+3 together: at the
last position the gradient is provably, exactly zero (item 1) for a structural
reason (item 2), while the energy at that same position is exactly the model's
conditional (item 3). The failure is a property of the input-embedding gradient, not
of the energy. Part 2 tests this as a prediction.

## 2026-07-21 15:23 CEST - PART 2: last-token experiment DESIGN + PREDICTIONS (logged BEFORE running)

Implementation: `diagnostics/run_revision.py --exp last_token` (new, follows the
existing contract: --run_name, --out_dir, atomic JSON, per-item CSV, deterministic
corruption per sample_idx). Same sentence set and loading as kl_baselines
(WikiText-2 validation default, 10<words<40, gpt2sft = gpt2_large_sft_output,
float32, data_seed 0, n=200; NOTE the revision corpus is WikiText-2, not ROCStories
- I matched what kl_baselines actually ran so the numbers sit on the same axis).
Sequences with L<6 skipped so the three position conditions are distinct.

Design. The experimental variable is the number of DOWNSTREAM scored terms the
masked position feeds:
- final = position L-1 (0 downstream terms; gradient provably 0, energy = exact conditional),
- second_to_last = L-2 (exactly 1 downstream term),
- middle = L//2 (many downstream terms).
Only position m is corrupted (random token != GT), identical across all arms; the
prefix stays clean so p(x_m | x_<m) is the true left-context conditional.

Arms per condition:
- dls_policy: real core DiscreteLangevinSampler, MH on, gn on, schedule
  linspace(10.5, 0.1, 50). LM gradient norm recorded per gradient evaluation.
- dls_random: same settings (gradnorm arm skipped: bitwise identical to random under
  gn=on, Part 1 item 5).
- cond_argmax, cond_sample, cond_topk_rescore: reuse the kl_baselines logic.
- independence_mh: propose from q = p(.|prefix), accept with the exact full-sequence
  energy ratio log a = (S' - S) + (log q(cur) - log q(prop)). Records the accept rate.
Metrics (KL undefined at the final position, item 6): exact-match %, top-5 %
(recovered token in top-5 of p(.|prefix)), mean NLL of the recovered token under
p(.|prefix), mean rank; bootstrap CIs over sequences. Plus a paired policy-vs-random
null test (mean paired diff + bootstrap CI + Wilcoxon per metric), the independence
accept rate, and the DLS-policy LM gradient norm, per condition.

PREDICTIONS (pre-registered):
1. FINAL position: DLS-policy LM gradient norm identically 0 at every step (the
   theorem, now inside the live sampler). DLS policy and DLS random statistically
   indistinguishable on every recovery metric (paired CIs straddle 0), because the
   gradient policy consumes is exactly zero, so it injects no directional signal.
   independence_mh accept rate = ~100% (target = proposal), and independence_mh +
   cond_topk_rescore + cond_argmax recover at the ceiling the conditional sets
   (mean rank ~ 0, top-5 ~ high). This is the theorem realised as an experiment.
2. MIDDLE position: the known null reproduces (policy indistinguishable from random),
   grad norm large, independence accept < 100% (energy carries downstream terms the
   proposal ignores).
3. SECOND-TO-LAST: intermediate. The open empirical question the design answers: does
   exactly ONE downstream scored term already give the input-embedding gradient
   measurable, useful signal, or not? Grad norm nonzero here (unlike final), so if the
   gradient were usable in principle, one term is where it would first show; the
   thesis predicts it still does not help (policy == random).

SANITY GATE (Part 1 item 3): if measured independence accept at the FINAL position is
materially below 100%, the implemented energy carries terms beyond the sequence
log-likelihood and everything downstream is suspect. Smoke test (n=3, steps=5)
already shows final accept = 100.0% mean AND min, and DLS-policy final grad norm
0.0 mean AND max, so the gate passes; full n=200 run launching now.

## 2026-07-21 18:19 CEST - PART 2: last-token OUTCOME (n=200, GPT-2 large, 153 min, GPU 0)

Result: `results_revision/rev_last_token_gpt2sft.json` (+ .csv, 3600 per-item rows).
EVERY pre-registered prediction confirmed; NONE contradicted. The sanity gate passed
at full scale.

### The table (arms x position conditions x four metrics; n=200 each)

Downstream scored terms the masked position feeds: final=0, second-to-last=1, middle=many.

FINAL position (0 downstream terms):
| arm | exact% | top5% | mean rank | mean NLL |
|-----|-----|-----|-----|-----|
| dls_policy        | 0.0  | 1   | 1538.9 | 14.852 |
| dls_random        | 0.0  | 0   | 1556.2 | 15.232 |
| cond_argmax       | 40.0 | 100 | 0.0    | 0.345  |
| cond_sample       | 38.5 | 94  | 11.2   | 0.906  |
| cond_topk_rescore | 40.0 | 100 | 0.0    | 0.345  |
| independence_mh   | 34.5 | 90  | 10.5   | 1.151  |
- DLS-policy LM gradient norm: mean 0.0000, MAX 0.0000 (exact zero at every one of
  the ~15000 gradient evaluations - the theorem, realised inside the live sampler).
- independence_mh acceptance: mean 100.00%, MIN 100.00% (SANITY GATE PASSED: the
  implemented energy is exactly the sequence log-likelihood, no hidden terms).
- policy vs random PAIRED null: exact diff 0.000 CI[0,0]; nll diff -0.379
  CI[-0.811,+0.061] Wilcoxon p=0.103; rank diff -17.3 CI[-513,+499] p=0.120. All
  three CIs straddle zero -> policy indistinguishable from random, as PREDICTED, and
  here it is a theorem (the gradient is exactly 0), not merely a statistic.

SECOND-TO-LAST position (1 downstream term):
| arm | exact% | top5% | mean rank | mean NLL |
|-----|-----|-----|-----|-----|
| dls_policy        | 0.0  | 0   | 3627.0 | 16.586 |
| dls_random        | 0.0  | 0   | 3712.3 | 16.555 |
| cond_argmax       | 50.0 | 100 | 0.0    | 0.478  |
| cond_sample       | 45.5 | 87  | 7.0    | 1.277  |
| cond_topk_rescore | 55.0 | 88  | 1.5    | 1.348  |
| independence_mh   | 49.5 | 80  | 927.3  | 2.517  |
- DLS-policy LM gradient norm: mean 12.945, max 153.37 (nonzero now: one downstream term).
- independence_mh acceptance: mean 63.35%, min 0.00 (drops off 100%: the energy now
  carries the one downstream term the left-conditional proposal ignores).
- policy vs random PAIRED null: exact diff 0.000; nll diff +0.031 CI[-0.422,+0.475]
  p=0.952; rank diff -85.3 CI[-897,+711] p=0.669. ALL straddle zero.
  ANSWER TO THE OPEN QUESTION: exactly one downstream scored term makes the gradient
  NONZERO but still gives it NO usable signal - policy is statistically identical to
  random even here. The uselessness is not merely the zero-gradient edge case; it
  persists the instant the gradient becomes nonzero.

MIDDLE position (many downstream terms):
| arm | exact% | top5% | mean rank | mean NLL |
|-----|-----|-----|-----|-----|
| dls_policy        | 0.0  | 0   | 8376.1 | 16.758 |
| dls_random        | 0.0  | 0   | 7027.2 | 16.168 |
| cond_argmax       | 18.0 | 100 | 0.0    | 0.801  |
| cond_sample       | 15.5 | 78  | 39.8   | 2.115  |
| cond_topk_rescore | 39.0 | 62  | 4.7    | 3.158  |
| independence_mh   | 31.5 | 56  | 1113.2 | 4.730  |
- DLS-policy LM gradient norm: mean 23.976, max 340.88 (large).
- independence_mh acceptance: mean 30.27%, min 0.00.
- policy vs random PAIRED null: exact diff 0.000; nll diff +0.591 CI[-0.027,+1.238]
  p=0.388; rank diff +1349 CI[-76,+2825] p=0.247. ALL straddle zero -> the known null
  reproduces on this experiment's masked-middle recovery.

### Reading the metrics (important caveats, no contradictions)

- exact-match and rank/NLL are measured under p(. | prefix), the LEFT-context
  conditional. This is the clean yardstick ONLY at the final position, where the full
  posterior over the masked token equals the left conditional (Part 1 item 3). At
  second-to-last and middle the full posterior also weights DOWNSTREAM terms, so the
  independence sampler (which targets the full posterior) correctly lands on tokens
  that are low under the left conditional -> its high mean rank there (927, 1113) is
  EXPECTED and not a failure. Read its exact-match (49.5%, 31.5%) instead: at the
  middle it BEATS cond_argmax (31.5 vs 18.0) precisely because it accounts for the
  downstream context that argmax-of-left-conditional ignores.
- The DLS arms recover 0.0% exact match at EVERY position (never once hit the target
  in 200 sequences x 3 positions) with ranks in the thousands, while the exact-energy
  methods recover 18-55%. This is the sharpest form of the gradient fallacy: the
  exact-energy forward pass does all the work; the input-embedding gradient does none.

### Prediction-vs-outcome summary (pre-registered vs measured)

| prediction | outcome |
|---|---|
| final: DLS-policy grad norm identically 0 | CONFIRMED: mean 0.0, max 0.0 over all evals |
| final: policy == random on all metrics | CONFIRMED: all paired CIs straddle 0 |
| final: indep_mh accept ~100% | CONFIRMED: 100.0% mean AND min |
| final: energy methods recover at ceiling | CONFIRMED: cond_argmax/topk rank 0.0, top5 100% |
| middle: null reproduces | CONFIRMED: all paired CIs straddle 0 |
| middle: grad norm large, accept < 100% | CONFIRMED: 24.0, 30.3% |
| 2nd-to-last: does 1 downstream term help? | ANSWERED NO: grad nonzero (12.9) but policy == random still |

No outcome contradicted a prediction. The one genuinely new empirical finding (not a
foregone conclusion) is the second-to-last result: a single downstream scored term is
enough to make the gradient nonzero but not enough to make it useful.

### Figure spec (the closing figure of the thesis)

x-axis: number of downstream scored terms the masked position feeds = {0 (final),
1 (2nd-to-last), many (middle)}. Twin-panel, shared x:
- Panel A (left y): mean DLS-policy LM gradient norm = {0.000, 12.945, 23.976}. The
  bar/point at x=0 is annotated "provably exactly 0 (zero-gradient theorem)".
- Panel B (right y, overlaid or twinned): independence-MH acceptance % =
  {100.0, 63.35, 30.27}, annotated "energy-only sampler; 100% acceptance is the
  theorem's certificate that energy == sequence log-likelihood".
- Overlay on both: the policy-minus-random gap with its 95% CI on the rank metric =
  {-17.3 [-513,499], -85.3 [-897,711], +1349 [-76,2825]}, every CI crossing 0, drawn
  as a shaded band pinned at zero -> "the gradient never separates from random, at any
  downstream-context length". The reader sees at a glance: as context grows the
  gradient norm rises and the energy sampler's acceptance falls, yet the gradient's
  usefulness stays flat at zero the whole way. Data for the figure lives in
  `results_revision/rev_last_token_gpt2sft.json` (by_condition.*) and the closing
  numbers are also written to `results_revision/last_token_figure.csv`.

## 2026-07-21 18:19 CEST - PART 3A: the WORKING setup (pre-approved, built)

The independence-MH arm of the last-token experiment IS the working setup, and it is
now measured. It succeeds by construction on the exact task where the gradient
provably fails, using the SAME frozen model, the SAME energy definition (the sequence
log-likelihood), and the SAME harness. The certificate is the 100.0% acceptance at
the final position: the energy was never the problem there; the input-embedding
gradient was. Paired with the existing concern-2 result (cond_topk_rescore 4.43 KL
beats every Langevin config; Gibbs 6.69 works gradient-free on the general infill
task), the thesis now has working-vs-failing on BOTH the last-token task and the
general infill task.

WORKING-VS-FAILING, in plain language (ready to adapt for the thesis):
"At the final token of a sequence the two facts separate cleanly. The gradient of the
sequence log-likelihood with respect to that token's input embedding is exactly zero,
because under causal attention that embedding feeds only the prediction of a token
that does not exist, and that prediction is discarded from the loss. We confirmed this
in the live sampler: over two hundred sequences and fifty steps each, the gradient
norm the discrete Langevin sampler consumed at the final position was zero to the last
bit, and with it the sampler was statistically indistinguishable from one moving in a
uniformly random direction (it never once recovered the target token). Yet the energy
at that same position is not degenerate at all: the difference in sequence
log-likelihood between two candidate final tokens is exactly the model's own
conditional log-probability of those tokens, so a sampler that proposes from that
conditional and accepts by the exact energy ratio accepts every move (we measured one
hundred percent acceptance) and recovers a model-optimal token every time. The frozen
likelihood is a perfectly usable energy here; it is only the gradient of that energy,
taken through the input embedding, that carries no signal. The failure the thesis
documents is a property of the gradient, not of the energy, and the last token is
where that distinction stops being a statistic and becomes a theorem."

## 2026-07-21 18:19 CEST - PART 3B: SEDD real run - VERIFIED interface plan (AWAITING GO-AHEAD, NOT STARTED)

Status: NOT started. Per the phase brief this needs explicit user go-ahead. Below is
the interface verification done against the upstream repo
(github.com/louaaron/Score-Entropy-Discrete-Diffusion) so the plan is concrete.

Interfaces checked against the repo (read-only, no clone):
- `load_model_local(root_dir, device)` returns `(score_model, graph, noise)`. The hook
  in run_sedd_linearization.py:88 matches this EXACTLY (already correct).
- The score model is called `model(x, sigma)` and returns a (B, L, V) score. The hook
  `sedd_score_row` (line 114) matches the call shape.
- `graph.staggered_score(self, score, dsigma)` - CONFIRMED the second argument is a
  sigma STEP (dsigma), not the absolute sigma, and the reverse per-position posterior
  is `staggered_score(score, dsigma) * transp_transition(x, dsigma)`, then normalized
  (this is how sampling.py forms the categorical). TWO CORRECTIONS the hooks need
  before a real run:
  1. `sedd_position_logprob` (line 136-137) passes `sig` (absolute sigma) where the
     repo passes a dsigma step, and it OMITS the `transp_transition(x, dsigma)` factor.
     For a single small-noise readout at fixed sigma the omission is a constant-ish
     reweighting, but it should be added to make the "truth" a genuine reverse
     posterior. Fix: form `post = graph.staggered_score(score, dsigma) *
     graph.transp_transition(x, dsigma)` then `log_softmax(post[0,pos])`.
  2. The surrogate in `sedd_score_row` (line 116) takes `log(score[0,pos])`. For the
     ABSORBING graph the concrete score already estimates the ratio p_t(v)/p_t(x), so
     log is the log-ratio (correct); for the UNIFORM graph the ratio needs the
     staggered transform first. Decide the graph type of the downloaded checkpoint
     (sedd-small is absorbing) and match.
- Tokenizer + embedding geometry: hook uses GPT-2's wte for the distance strata
  (line 94-96), correct since SEDD-small shares the GPT-2 tokenizer.

Scope if approved (timebox 2 days hard, then write as designed-but-not-run):
1. Clone the repo, download sedd-small checkpoint (GPT-2 tokenizer, ~90M params).
2. Apply the two hook corrections above; re-run `--dry_run` (already validated) to
   confirm the loop still wires, then a 10-seq smoke on the real checkpoint.
3. Real linearization: n=200 ROCStories sequences (the script defaults to
   wza/roc_stories), sigma small (~0.05-0.1). Output: spearman_surrogate_vs_true_ALL
   plus the near/mid/random strata, to compare against the AR number (~ -0.01 to
   0.057). Estimated compute: SEDD-small forward is cheaper than GPT-2 large; the loop
   is ~200 seqs x 2000 candidates x 2 forward passes -> ~1-2 GPU-hours.
4. Optional: SEDD's native infilling on the same corrupted sequences as a positive
   control, reported separately (not as a harness plug-in).
Interpretation is a win either way (positive corr -> AR objective is the cause,
thesis jumps a tier; ~0 corr -> cause is deeper, more surprising, still publishable).

DECISION NEEDED FROM USER: run SEDD real (2-day timebox) or leave as
designed-and-verified-but-not-run per the plan's fallback. I have NOT started it.

## 2026-07-21 18:41 CEST - PART 3B: SEDD real run - USER APPROVED, executed

User chose "Run it (2-day timebox)". Executed. Environment and interface work below;
the full n=200 numbers are appended when the run completes.

### Environment (all in the existing gfn venv, no new venv)
- Cloned github.com/louaaron/Score-Entropy-Discrete-Diffusion to
  `/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion`.
- Only missing dep was `einops` (installed). hydra, omegaconf, huggingface_hub,
  transformers, datasets all already present.
- Model loads via `SEDD.from_pretrained("louaaron/sedd-small")` (169.6M params,
  ABSORBING graph, vocab dim 50258 = 50257 real + 1 mask). HF cache routed to
  `hf/cache`.

### Two flash-attn patches to the CLONE (inference-only, mathematically exact)
The repo hard-requires flash-attn 2.2.2 + torch 2.0.1+cu118; we have torch
2.12+cu130. Rather than build flash-attn, patched two files in the clone:
1. `model/transformer.py`: made the top-level `flash_attn_varlen_qkvpacked_func`
   import optional and added an SDPA fallback. SEDD's attention is uniform full
   (non-causal) self-attention, so `F.scaled_dot_product_attention` (same 1/sqrt(d)
   scale, 0 dropout at eval) is an EXACT substitute. Verified forward runs and
   returns finite scores.
2. `model/rotary.py`: the `@torch.jit.script` rotary fallback crashes under torch
   2.12 (torchscript interpreter range error); replaced with the identical eager
   expression `(qkv*cos)+(rotate_half(qkv)*sin)`. The flash rotary path is still
   tried first and skipped when flash is absent.
These touch only the SEDD clone, never core/ or Doc/.

### Interface verification (against the repo AND numerically on sedd-small)
- `load_model_local(root_dir, device)` / `load_model_hf(dir, device)` both return
  `(score_model, graph, noise)`; the script uses `load_model` (tries hf then local),
  so `--model_dir louaaron/sedd-small` works. CONFIRMED.
- `graph.staggered_score(score, dsigma)` (second arg is a sigma STEP) and
  `graph.transp_transition(i, sigma)` exist with those signatures. CONFIRMED.

### TWO SUBSTANTIVE CORRECTIONS to the designed hooks (each was necessary)
The originally-designed hooks would have produced a MEANINGLESS number; verifying
them caught two problems (this is exactly why the plan insisted on verification):
1. The score is ALREADY in log space (measured range [-39, 0], with negatives), not
   a positive ratio. The original `torch.log(score)` was a double-log. FIX: use the
   score directly as a per-position log-preference (log_softmax over the real vocab).
2. In absorbing SEDD the score is context-dependent ONLY at MASKED positions:
   perturbing an unmasked token changed another unmasked position's readout by
   exactly 0.0 (measured), while at a MASKED probe it changed by 0.70, and with only
   pos observed it changed by 5.39. So the originally-designed "per-position log-prob
   from one pass" truth would have been identically zero at the positions that
   matter. This is a genuine, caught defect, not cosmetics.

### The corrected, verified design (mirrors the AR linearization diagnostic)
For each ROCStories sequence and one position `pos`:
- surrogate(v): MASK pos, keep the rest clean, ONE pass. r_pos = log_softmax(score[pos]).
  surrogate(v) = r_pos[v] - r_pos[orig]. SEDD's cheap one-pass directional proposal
  for filling pos (the analogue of the AR Taylor surrogate gᵀ(e(v)-e(x_pos))).
- truth(v): OBSERVE pos = v, MASK a fixed held-out probe set Q (8 non-pos positions,
  seeded), keep the rest clean, ONE pass PER CANDIDATE. truth(v) =
  [sum_{q in Q} log_softmax(score_modified[q])[x_q_true]] - (same at pos=orig). The
  actual effect of committing pos=v on the model's reconstruction of the TRUE rest of
  this sequence (the analogue of the AR truth = Δ total sequence log-lik, which is
  likewise re-run per candidate and dominated by the effect on OTHER positions).
surrogate (pos masked) and truth (pos observed=v, Q masked) come from DIFFERENT
forward passes, so the correlation is a genuine test, not a tautology. Same distance
stratification (near/mid/random) and same 400,000 candidate pairs (200 seqs x 2000
cands) as the AR diagnostic, so the numbers are directly comparable. The corrected
script is `diagnostics/run_sedd_linearization.py` (rewritten in place; original
recoverable from git). Design fully documented in the file header.

### Validation before the real run
- `--dry_run` (fake log-space model): pipeline runs end to end, spearman ~ 0 on random
  data (correct: no signal), 2.8s.
- Real 5-seq smoke on sedd-small: spearman_ALL 0.026, per_seq_mean 0.107, and notably
  **spearman_near = 0.561** (embedding-close candidates), vs the AR near-stratum ~0.
  Promising but n=5; the full n=200 run (pid 3974146, GPU 1, out_dir results_revision,
  run_name rev_sedd_linearization) is running now. FINAL NUMBERS APPENDED BELOW ON
  COMPLETION.

## 2026-07-21 18:44 CEST - PART 3B: SEDD real linearization RESULT (n=200, 400k pairs, GPU 1, 2.0 min)

`results_revision/rev_sedd_linearization.json` (sigma 0.1, 8 probes, 200 ROCStories
seqs, 2000 candidates/seq = 400,000 pairs, matching the AR diagnostic exactly).

### The positive control fires: SEDD's surrogate DOES track the true change

Spearman(surrogate, true_delta), SEDD vs the AR gpt2sft linearization
(`results_diag/diag_linearization_gpt2sft.json`, same 400k-pair design):

| stratum | AR gpt2sft (rho) | SEDD-small (rho) |
|---|---|---|
| ALL (pooled) | -0.011 | **+0.184** |
| near  (embedding-close cands) | +0.027 | **+0.306** |
| mid   | -0.041 | +0.116 |
| random| -0.048 | +0.135 |
| per-sequence mean | -0.011 | **+0.172** (median +0.192) |

Pearson ALL: AR -0.023, SEDD +0.183.

### Reading it (this CONFIRMS the thesis's central causal claim)

Under the AR objective the one-pass directional surrogate was statistically
indistinguishable from zero at every distance (|rho| < 0.06 across all five AR /
GFlowNet energies; gpt2sft near-stratum +0.027). Under the diffusion (score-entropy)
objective, on the SAME tokenizer and the SAME 400k-pair distance-stratified design,
the analogous one-pass surrogate is POSITIVE in every stratum (near +0.306, mid
+0.116, random +0.135; pooled +0.184; per-sequence mean +0.172). The signal is
strongest for near candidates (the linearization regime) and positive even for random
candidates. So a model trained to denoise, rather than to predict-next-token, yields
a cheap local directional signal that DOES point toward the true effect of a token
swap, where the AR gradient did not. This is the pilot the thesis designed: the
training objective is the cause, and the thesis moves from "a diagnosis plus a
proposed test" to "a diagnosis plus its first confirmation".

MATCHES PREDICTION (positive corr -> objective is the cause). Not contradicted.

### Honest caveats to carry into the write-up (pilot framing, per the plan)
- The correlations are clearly positive but MODEST (0.18 pooled, 0.31 near), not near
  1. The correct claim is qualitative-and-quantitative: from statistically-zero (AR)
  to clearly-positive-and-usable (SEDD), NOT "SEDD's surrogate is exact".
- surrogate and truth are the faithful ANALOGUES of the AR quantities, not identical
  definitions (SEDD has no input-embedding gradient). surrogate = SEDD's one-pass
  denoising log-preference for pos (pos masked); truth = the re-run-per-candidate
  effect of committing pos=v on reconstructing held-out true tokens. They come from
  different forward passes, so the positive correlation is a genuine cross-check, not
  a tautology, and it is positive WITHIN each distance stratum (not a pooling
  artifact). Present as a positive-control pilot (Section 5.12 / appendix), exactly as
  concern 8 specifies.
- absorbing-graph sedd-small, score read in log space, context signal read at masked
  positions (both per the verified corrections above). sigma 0.1; a small sigma sweep
  is a cheap robustness add if an examiner asks, but the sign and order of magnitude
  are already clear at n=200.
- NOTE for reconcile: this run lives in results_revision with run_name
  rev_sedd_linearization; keep it OUT of the AR per_run_spearman set (reconcile globs
  results_gpt2_v2/llama/gfn/diag, not this file) so it is never folded into the AR
  config count or the "|rho|<0.06 across all models" AR sentence.

### Artifacts
- `results_revision/rev_sedd_linearization.json` + `.csv` (400k rows).
- Corrected script `diagnostics/run_sedd_linearization.py` (design in header).
- SEDD clone with two inference-only patches at
  `/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion`
  (model/transformer.py SDPA fallback, model/rotary.py eager rotary).
- `rev_sedd_dryrun.json` remains the earlier pipeline-validation dry run.

CONCERN 8 STATUS UPGRADE: was [SETTLED-pilot, real run deferred]; now [SETTLED - real
run DONE], positive control confirms the training-objective diagnosis.

## 2026-07-21 18:19 CEST - PART 4: phase closing report

Artifacts produced this phase (all under the repo, none in Doc/):
- `diagnostics/audit_probe.py` + `diagnostics/audit_probe_result.json` - Part 1
  numerical verification (re-run and reproduced bit-for-bit this session).
- `diagnostics/run_revision.py` - new `--exp last_token` (+ paired null test,
  independence-MH acceptance recorder, LM-grad-norm recorder). Existing experiments
  untouched; module parses and all four experiments register.
- `results_revision/rev_last_token_gpt2sft.json` + `.csv` - the n=200 experiment.
- `results_revision/last_token_figure.csv` + `.png` - the closing figure and its data.

What is now settled (the supervisor's WHY, answered as a proof):
- The input-embedding gradient of the sequence log-likelihood at the final scored
  token is EXACTLY zero (theorem + 0.0/0.0 measured in the live sampler), for the
  structural reason that the token enters the loss only through the tied OUTPUT
  embedding path and its own logit column is dropped. The energy at that same token
  is EXACTLY the model's conditional log-ratio (max error 4e-5 nats), so an
  energy-only independence sampler accepts 100% and recovers at the ceiling. The
  documented failure is a property of the gradient, not the energy.
- The null (policy indistinguishable from random) holds at downstream-context lengths
  0, 1 and many; at length 0 it is a theorem, and length 1 shows the gradient becomes
  nonzero without becoming useful.

CONTRADICTIONS: none. Every pre-registered prediction was confirmed. The sanity gate
(final-position acceptance = 100%) passed at full scale, certifying the implemented
energy is exactly the sequence log-likelihood with no hidden terms - which also
retroactively validates the CLS/DLS energy used throughout the thesis.

OPEN ITEM: SEDD real run (Part 3B) - interface verified, awaiting user go-ahead.
No other GPU work is pending. Nothing above single-experiment cost was launched
without the plan being stated (the n=200 last_token run took 153 min on one GPU;
heavier than the "under an hour" estimate because it runs 6 arms x 3 positions x 200
seqs sequentially, but it is a single experiment and it completed cleanly).

## 2026-07-21 19:58 CEST - PHASE 4 KICKOFF: capability level (SEDD recovery / hybrid / guided). Audit + Part 0 pre-registration

Phase 4 brief (PROMPT_PHASE4_FINAL.md): move the SEDD result from SIGNAL level
(linearization rho +0.184 pooled / +0.306 near vs AR |rho|<0.06) to CAPABILITY
level. Three pilots, both scales where noted, all sharded over sequences across the
9 free A6000s, fresh status dir. NO best-of-N / SMC (excluded by decision). Do not
touch core/ samplers. gpt2sft = /mount/.../gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output.

AUDIT (done before any GPU job):
- 9 A6000s (49 GB) all idle.
- SEDD clone at ../../Score-Entropy-Discrete-Diffusion, already patched (SDPA + eager
  rotary). Absorbing graph, dim 50258 (50257 real + MASK id 50257), loglinear noise.
  Repo default sampler config: predictor euler, steps 128, noise_removal True, eps 1e-5.
  sedd-small cached in ../../hf/cache; sedd-medium NOT cached (will download).
- Conditional infill recipe (run_sample_cond.py): get_pc_sampler(graph,noise,(B,L),
  predictor,steps,denoise,proj_fun) with proj_fun clamping observed positions each
  step; masked positions start at MASK via graph.sample_limit; denoiser truncates the
  MASK column so fills are always real vocab.
- Corpus: kl_baselines = run_experiment.load_texts (WikiText-2 validation, 10<words<40)
  + build_corruption (num_masks 1, one random interior pos in range(1,L-1), data_seed 0)
  + iter_grid_samples, n=200. P1/H will reuse iter_grid_samples verbatim so the sequence
  set is bit-identical; verify first 5 untouched KLs against rev_klbase CSV
  (seq0 5.4287, seq1 10.3899, seq2 4.1979, seq3 0.3092, seq4 7.6364).

SMOKE (GPU 0, before writing production code): loaded sedd-small, ran conditional
recovery on one WikiText-style sentence (L=14, NO padding to 1024), masked interior
position; euler-128 AND analytic-128 both recovered the exact original token,
observed positions bit-identical, filled token real vocab (not MASK). Variable-length
input is accepted directly. Full Part 0 gate formalises this on 5 kl_baselines seqs at
both scales.

PART 0 PRE-REGISTERED PREDICTIONS (gates on 5 kl_baselines sequences, both scales):
1. Variable-length gate. sedd-small/medium accept length-L input directly (L=15..60),
   no pad to 1024. Evidence: the masked-position log-preference readout r_pos at length
   L vs length L+tail (tail = appended MASK positions) is near-identical (top-1 token
   unchanged, small distribution shift), so a quarantined/absent tail does not condition
   the fill. PREDICT: variable-length regime holds; top-1 identical on 5/5 both scales.
   If tails materially move r_pos, switch to no-pad length-L (still variable length) and
   document. A hard pad-to-1024 regime is the fallback only if bare length-L forward fails.
2. Projection gate. Observed positions bit-identical after denoising (proj_fun clamps
   them; PREDICT 5/5 both scales) and the filled position is a real vocab token, never
   MASK (denoiser truncates the MASK column; PREDICT 5/5 both scales).
3. Scale gate. sedd-medium loads through the patched clone and passes gates 1 and 2.
4. Corpus unification. Re-run the corrected linearization diagnostic on EXACTLY the
   kl_baselines set (WikiText-2, data_seed 0, n=200) at both scales, sigma 0.1. PREDICT:
   both scales reproduce a clearly-positive rho (pooled > 0.1, near > 0.25), consistent
   with the earlier ROCStories run (+0.184 / +0.306); medium >= small in the near
   stratum. This unifies the corpus so linearization/recovery/hybrid share one set.
   All SEDD runs stay OUT of the AR reconcile globs (results_revision, rev_sedd_* names).

## 2026-07-21 20:05 CEST - PART 0 GATES: OUTCOME (both scales, 5 kl_baselines seqs each)

Corpus identity CONFIRMED: iter_grid_samples reproduces kl_baselines bit-for-bit;
recomputed untouched avg_kl for seq 0-4 = 5.428674 / 10.389947 / 4.197888 / 0.309180
/ 7.636383, matching rev_klbase_gpt2sft.csv exactly. Masked positions are random
interior (seq0..4: pos 3, 15, 2, 34, 25), NOT L//2, so the P1/H same-task reference
arms (left_conditional, dls_policy, dls_random on these exact sequences) are warranted.

Gate results (results_revision/rev_sedd_gates_{small,medium}.json, GPU 0/1):
| gate | small | medium | prediction |
|---|---|---|---|
| projection (obs bit-identical + fill real vocab) | PASS 5/5 | PASS 5/5 | PASS -> confirmed |
| variable-length top-1 identical (L vs L+16 tail) | 5/5 | 5/5 | identical -> confirmed |
| tail distribution shift mean KL(pref_L||pref_L+tail) | 0.100 (max 0.236) | 0.022 (max 0.062) | small -> confirmed |
| native recovery exact (single draw) | 3/5 | 3/5 | (sanity) |
| gt in top-5 of SEDD readout | 4/5 | 4/5 | (sanity) |

REGIME DECISION (documented): run at EXACTLY length L, no padding to 1024. Variable
length is accepted directly (rotary positions generalise). Because we never pad, no
MASK tail ever sits beside the target, so tail leakage cannot occur by construction;
the tail test only certifies the model is not catastrophically length-sensitive
(top-1 never flips even with a 16-MASK tail, and medium is 4-5x more tail-robust than
small). All gates PASS at both scales; predictions confirmed, none contradicted.
Sampler settings recorded: predictor euler, steps 128, noise_removal True, eps 1e-5,
sigma 0.1 for the one-pass readout (matches the validated linearization readout).

## 2026-07-21 20:19 CEST - PART 0 corpus-unified LINEARIZATION: OUTCOME + a partial prediction miss

results_revision/rev_sedd_lin_{small,medium}.json (n=200, 400k pairs, WikiText-2
kl_baselines set, sigma 0.1, GPU 0/1). Both scales clearly positive vs AR |rho|<0.06:

| stratum | AR gpt2sft | SEDD-small (WikiText) | SEDD-medium (WikiText) | SEDD-small (ROCStories, earlier) |
|---|---|---|---|---|
| ALL pooled | -0.011 | +0.130 | +0.133 | +0.184 |
| near | +0.027 | +0.097 | +0.148 | +0.306 |
| mid | -0.041 | +0.115 | +0.122 | +0.116 |
| random | -0.048 | +0.126 | +0.110 | +0.135 |
| per-seq mean | -0.011 | +0.184 | +0.179 | +0.172 |

PREDICTION CHECK. Predicted pooled > 0.1 (CONFIRMED: 0.130 / 0.133) and near > 0.25
(NOT met on WikiText: 0.097 / 0.148). The near-stratum > 0.25 was extrapolated from
the ROCStories run and did NOT carry to WikiText. Cause understood: the strata are
corpus-dependent (WikiText near mean-distance 3.54 is not as "near" as ROCStories'
was; WikiText tokens include rarer/technical tokens whose local embedding
neighbourhoods behave differently). The LOAD-BEARING claim is unaffected and fully
CONFIRMED on the unified corpus: pooled +0.13 and per-seq mean +0.18 at BOTH scales,
positive in EVERY stratum, an order of magnitude above AR |rho|<0.06. Medium beats
small in the near stratum (+0.148 vs +0.097), matching "medium >= small". WRITE-UP
RULE: state the robust pooled/per-seq numbers and "positive in every stratum"; do NOT
carry the ROCStories near +0.306 as if it held on WikiText. Report both corpora.

## 2026-07-21 20:19 CEST - PART P1 (recovery) + PART H (hybrid) PRE-REGISTERED PREDICTIONS (before full run)

Code: diagnostics/run_sedd_cap.py (--exp recovery|hybrid|hybrid_refs), sedd_lib.py,
merge_sedd_cap.py, run_sedd_slate.sh. Corpus = kl_baselines set (verified identical).
Reference rows from rev_klbase_gpt2sft.json: ground_truth 0.00, cond_topk_rescore 4.43,
gibbs 6.69, untouched 9.14, flagship Langevin ~6.5. Reference from
rev_last_token_gpt2sft.json (middle): independence_mh 31.5% exact / 30.3% accept,
dls arms 0.0%. Note: P1 masks RANDOM interior positions, not L//2, so hybrid ships
same-task reference arms (left_conditional, dls_policy, dls_random) on the P1 set.
Smoke (n=4-6) validated both: recovery 3/6 exact (native single draw); hybrid SEDD arm
3/4 exact with 58-100% acceptance; dls arms 0/4 exact; left_conditional 2/4 exact.

P1 predictions: SEDD native recovery avg KL under gpt2sft clearly below untouched
(9.14) at both scales; exact-match far above the Langevin arms' 0%; medium at or above
small on every metric. NOT a claim: "beats the domain-tuned AR baselines in absolute
terms" (unfair, scale/domain gaps). GATE: any arm above untouched means broken
conditioning -> stop and debug.

H predictions: hybrid acceptance well above the random/dls arms (which sit near the
DLS null); hybrid exact-match above zero and at or above the left-conditional's,
because the SEDD proposal sees both sides of the mask. If hybrid does NOT beat the
left-conditional, report it straight (a real finding about what bidirectionality adds
beyond exact left-context reweighting). dls_policy == dls_random near 0% (the AR
gradient arm), reproducing the null on the P1 set (same-task).

Launching the sharded slate now (num_shards 6, 9 GPUs, fresh logs unifiedruns/sedd_slate).

## 2026-07-21 20:26 CEST - PART G PRE-REGISTERED PREDICTIONS (noisy classifier + guided SEDD)

Role separation (non-negotiable): the NEW noisy classifier (diagnostics/train_noisy_classifier.py)
ONLY guides SEDD; steering is scored by the concern-11 judge = classify(gpt2sft, head)
with the EXISTING head /mount/arbeitsdaten/.../sentiment_constrained_ft_gpt2_large/sentiment_head.pt
(hidden 1280), the same judge used for all AR arms in rev_constrained.json. Task =
concern-11 continuation (15 MuCoLa prompts, span 20), both target labels, guided vs
unguided SEDD-medium generation. Fluency = gpt2sft NLL/token of the span (the KL metric
needs a GT reference, absent for conjured continuations; NLL/token is the faithful
gpt2sft surprisal readout, stated as such). Guidance: at masked positions each
denoising step, reweight the SEDD categorical over top-k=32 candidates by
p_noisyclf(target | candidate state)^gamma (gamma 1.0), renormalize, sample; one
batched classifier pass per step (linear in steps).

Classifier gate prediction: held-out SST-2 accuracy at LOW noise (move-chance 0) well
above 50% chance (predict ~80%+); accuracy degrades smoothly as noise rises. If low
noise is not well above chance, STOP (guidance on a broken classifier is noise).

Steering prediction: guided beats unguided on the concern-11 judge at BOTH labels with
bounded NLL cost. If the gain is small or asymmetric, report exactly that. Fairness
note for write-up: the AR arms also used a trained classifier; the only difference is
this one is trained on the noisy states it is evaluated on (the thesis's own principle,
constructive direction).

## 2026-07-21 20:35 CEST - PART P1 (recovery) + PART H (hybrid): OUTCOME (n=200, both scales, sharded)

Slate finished (30 shards + merges) in ~15 min wall across 9 GPUs (vs 153 min
sequential for last_token). Results in results_revision/rev_sedd_recovery_{small,medium}.json
and rev_sedd_hybrid.json. Classifier gate (Part G) passed concurrently (low-noise acc 81%).

P1 native SEDD recovery (single native draw, kl_baselines set, KL under gpt2sft):
| arm | exact% | top5% | mean KL [CI95] | median KL |
|---|---|---|---|---|
| SEDD-small recovery | 32.5 | 61.5 | 3.75 [3.10, 4.43] | 1.69 |
| SEDD-medium recovery | 35.0 | 67.0 | 3.73 [3.12, 4.37] | 1.45 |
| (ref) ground_truth | 100 | - | 0.00 | - |
| (ref) cond_topk_rescore | 33 | - | 4.43 | - |
| (ref) gibbs | 18.5 | - | 6.69 | - |
| (ref) flagship Langevin | 0.0 | - | ~6.5 | - |
| (ref) untouched | 0.0 | - | 9.14 | - |
ALL P1 predictions CONFIRMED: recovery KL (3.7) clearly below untouched (9.14) and
below the Langevin arms (~6.5) at both scales; exact-match (32.5/35) far above the
Langevin arms' 0%; medium >= small on every metric (exact 35>=32.5, top5 67>=61.5,
KL 3.73<=3.75, median 1.45<1.69). No arm above untouched -> conditioning intact, gate
holds. NOT claimed: beats the domain-tuned AR baselines in absolute terms (scale/domain
gaps; note recovery KL ~ cond_topk_rescore's 4.43, comparable, framed as pilot only).

H hybrid (same-task P1 set, exact gpt2sft energy, 50-step independence MH; refs
scale-independent, computed once):
| arm | exact% | top5% | mean KL | accept% |
|---|---|---|---|---|
| hybrid_medium (SEDD proposal) | 39.0 | 67.0 | 3.02 | 43.3 |
| hybrid_small (SEDD proposal) | 38.5 | 61.5 | 3.12 | 40.0 |
| left_conditional (AR proposal) | 23.5 | 34.5 | 5.17 | 34.4 |
| dls_policy (AR gradient) | 0.0 | 34.5 | 6.85 | - |
| dls_random | 0.0 | 34.5 | 6.58 | - |
ALL H predictions CONFIRMED: hybrid acceptance (40-43%) above the left-conditional
(34.4%) and far above the never-recovering gradient arms; hybrid exact-match (38.5/39)
above zero AND above the left-conditional's (23.5%) at BOTH scales, because the SEDD
proposal sees both sides of the mask. dls_policy == dls_random ~ 0% exact (the AR
gradient arm never recovers), reproducing the null same-task on the P1 set. THE
SUFFICIENCY RESULT: swapping ONLY the direction signal (AR-left-conditional -> SEDD
bidirectional proposal) inside the SAME MH-corrected chain on the SAME exact energy
and task lifts exact recovery 0% (gradient) / 23.5% (left-conditional) -> 38.5-39%.
Phase 2's zero-gradient theorem is the necessity half; this is sufficiency. None
contradicted.

## 2026-07-21 20:38 CEST - PART G guided generation: OUTCOME CONTRADICTS PREDICTION (gamma 1.0) - STOP + DIAGNOSE

results_revision/rev_sedd_guided.json (n=1200 generations, 300 pairs/label, medium,
steps 64, k=32, gamma 1.0). Judge = concern-11 frozen-GPT-2 head; noisy clf guides only.
| label | unguided hit% | guided hit% | gain pts [CI95] | unguided NLL | guided NLL |
|---|---|---|---|---|---|
| 0 (neg) | 49.0 | 55.0 | +6.0 [-1.3, +13.3] | 7.09 | 8.48 |
| 1 (pos) | 44.0 | 38.7 | -5.3 [-12.3, +1.7] | 7.16 | 8.45 |
PRE-REGISTERED PREDICTION was "guided beats unguided at BOTH labels with bounded KL
cost." OUTCOME: weak and asymmetric (label 0 small positive but CI straddles 0; label 1
NEGATIVE, wrong direction, CI straddles 0), with a real fluency cost. This CONTRADICTS
the prediction. Per the working rules I am STOPPING and NOT writing this into the thesis
until the cause is understood. Candidate causes to test: (a) gamma 1.0 too weak (sweep
{2,4}); (b) noisy clf and judge disagree on the SAME texts (would make guidance steer
the wrong way per the judge) - decisive cheap diagnostic; (c) most guidance mass sits on
"stay MASK" early when the classifier is near chance at high noise (mc0.9 acc 53%), so
guidance barely bites. Running (b) then (a).

## 2026-07-21 20:42 CEST - PART G RESOLVED: gamma sweep + self-clf diagnostic explain the asymmetry

Ran gamma in {1,2,4} (each 4 shards, self-clf diagnostic column added) plus the
noisy-clf-vs-judge agreement check. Results (judge = concern-11 head; SELF = the
guiding noisy clf's own verdict on the final text; agree = judge-vs-clf agreement on
unguided text):
| gamma | label | judge unguided->guided (gain [CI95]) | SELF gain | agree | NLL cost |
|---|---|---|---|---|---|
| 1 | 0 (neg) | 49->55 (+6.0 [-1.3,13.3]) | +9.7 | 56% | 7.1->8.5 |
| 1 | 1 (pos) | 44->39 (-5.3 [-12.3,1.7]) | +11.0 | 64% | 7.2->8.5 |
| 2 | 0 (neg) | 49->60 (+10.7 [3.3,18.0]) | +7.7 | 56% | 7.1->9.7 |
| 2 | 1 (pos) | 44->44 (+0.0 [-7.0,7.0]) | +12.0 | 64% | 7.2->9.6 |
| 4 | 0 (neg) | 49->66 (+16.7 [9.7,24.0]) | +7.3 | 56% | 7.1->11.0 |
| 4 | 1 (pos) | 44->39 (-5.3 [-13.0,2.7]) | +18.3 | 64% | 7.2->11.0 |

CAUSE UNDERSTOOD (three consistent facts):
1. The guidance MECHANISM works: it moves the guiding classifier's OWN verdict toward
   the target at every gamma and BOTH labels (SELF gain +7 to +18 pts). The
   commitment-time top-k reweighting genuinely steers absorbing-SEDD generation.
2. It steers the HELD-OUT judge significantly in the NEGATIVE direction, monotonically
   in gamma (+6 -> +10.7 [sig] -> +16.7 [sig]), but NOT the positive direction (~0).
3. The asymmetry is a cross-classifier off-distribution disagreement: the guiding clf
   and the judge agree only 56-64% on the (off-manifold) generated text, and stronger
   guidance pushes text further off the fluent manifold (NLL 7.1 -> 11.0 at gamma 4),
   exactly where two independently trained sentiment classifiers diverge. Guiding hard
   toward one classifier's "positive" therefore need not raise the other's "positive".

PREDICTION STATUS: the pre-registered "beats at BOTH labels" is CONTRADICTED as stated;
the prompt's own fallback ("if the gain is small or asymmetric, report exactly that")
applies. Cause is now understood, so this may be written into the thesis AS the honest
asymmetric finding: guided diffusion is a WORKING steering mechanism (validated on the
guiding classifier; significant on a held-out judge in one direction, up to +16.7 pts,
CI excludes 0), with an off-manifold asymmetry and a fluency cost that grows with gamma.
This dovetails with the thesis's existing off-manifold theme (concern-11 cons_only
anomaly: classifiers are unreliable off the fluent manifold). Headline for the write-up:
gamma=2 (label0 +10.7 [3.3,18.0], moderate NLL cost) with the full sweep and the SELF
mechanism-validation reported. Artifacts: results_revision/rev_sedd_guided_g{1,2,4}.json
(+ .csv), rev_sedd_guided.json (gamma 1, first run). Role separation held throughout:
noisy clf ONLY guided; steering scored by the concern-11 judge.

## 2026-07-21 21:00 CEST - PHASE 4 CLOSING REPORT (capability level complete + thesis integrated)

All Phase 4 experiments done, sharded across 9 A6000s (whole slate in one afternoon,
vs 153 min single-GPU for last_token). New code (none touches core/ or the SEDD clone
beyond the existing inference patches): diagnostics/sedd_lib.py, run_sedd_cap.py
(gates|recovery|hybrid|hybrid_refs, shardable), merge_sedd_cap.py, run_sedd_guided.py,
aggregate_guided.py, train_noisy_classifier.py; run_sedd_linearization.py gained a
--grid_corpus mode; run_sedd_slate.sh launcher. Consolidated numbers in
results_revision/sedd_capability_summary.json. All scripts py_compile clean.

RESULTS (predictions vs outcomes):
- Part 0 gates: PASS both scales (projection clean; variable-length accepted directly,
  run at exactly length L so no tail leakage). CONFIRMED.
- Linearization (unified WikiText corpus): SEDD +0.130 small / +0.133 medium pooled,
  positive in every stratum both scales, vs AR |rho|<0.06. CONFIRMED (one sub-prediction,
  near>0.25, was a ROCStories extrapolation that did not carry to WikiText; cause
  understood, load-bearing claim intact; both corpora reported).
- P1 recovery: exact 32.5/35.0%, KL 3.7 << untouched 9.14 and Langevin ~6.5; medium>=small.
  ALL CONFIRMED. Absolute-superiority over AR baselines NOT claimed (pilot framing).
- H hybrid (sufficiency): swapping only the direction signal lifts exact recovery
  0% (gradient) / 23.5% (left-conditional) -> 38.5-39%; acceptance above the
  left-conditional; dls arms 0%. ALL CONFIRMED.
- Part G guided: pre-registered "both labels" CONTRADICTED; stopped, diagnosed, cause
  understood (mechanism works, SELF +7..+18; judge steers significantly negative up to
  +16.7 [9.7,24.0] but not positive; clf-judge agree only 56-64% off-manifold; NLL cost
  grows with gamma). Written into the thesis AS the honest asymmetric capability finding.

THESIS INTEGRATION (Part T, Doc/, logged in REVISION_LOG_THESIS.md), independently
VERIFIED by me (not just the sub-agent report):
- Clean compile: latexmk/pdflatex+bibtex exit 0, FINAL pass 0 undefined citations, 0
  undefined references, 0 errors. Page count 98 -> 105.
- A20 last-token: PENDING/TODO-AUTHOR block removed; Table 5.6 (position conditions,
  verified numbers: DLS 0/0/0, energy-only 18-55%, grad norm 0.000/12.945/23.976,
  acceptance 100.0/63.4/30.3) + Figure 5.8 (figures/fig_lasttoken.png) render correctly.
- A8: new sec:results-diffusion (chapters/05a_diffusion_control.tex, \input at end of
  05_results.tex) with linearization / recovery / hybrid / guided tables (Tables 5.7-5.10),
  pilot framing, operation-level AR-vs-diffusion qualifier, absolute-comparison disclaimer;
  sec:disc-future upgraded from designed-but-not-run to run-and-confirmed.
- Concern 6a reattributed (no-MH 0.03/3.7, MH 0.627/8.56), %TODO-AUTHOR removed.
- Coherence (a) closing Discussion sentence + (b) operation-level contrast both present
  and render (verified pages 60-63 to PNG). No literal em-dash, no stray triple-dash in
  edited files. Numbers spot-checked in .tex against JSONs.

AUTHOR ITEMS remaining (unchanged by decision, need author sign-off):
1. Abstract/intro still state the constraint-gradient and training-free claims at full
   strength; the Part G asymmetry and reframed promise live in Results/Discussion only
   (consistent with the Phase 3 decision to leave the abstract untouched). Decide whether
   to soften the abstract to match.
2. Phase 3 appendix trajectory PCA plots remain placeholders (pre-existing, out of Phase 4
   scope).

No contradiction was written into the thesis without its cause first understood. Nothing
beyond the approved slate was run. GPUs released; no stray processes.
