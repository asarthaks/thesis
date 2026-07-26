# REVISION_RESTRUCTURING.md

Timestamped, append-only log of the move-and-rewire restructuring pass. This is a
MOVE-AND-REWIRE, not a rewrite: no experimental number, result JSON value, sampler
in `core/`, or sentence of thesis prose is changed. The only content edits are path
strings in code, shell, manifests, README, env files, and `%` LaTeX comments.

Sequencing rule obeyed: PART 1 (audit) precedes PART 3 (moves); PART 4 (sweep)
precedes PART 5 (verify); PART 6 (README) follows a green PART 5. Nothing moves
before the audit table and the move plan are logged.

No em-dashes anywhere.

---

## 2026-07-25 00:56 CEST  PART 0: pre-flight

- `git rev-parse --show-toplevel` -> `/mount/studenten-temp1/users/singhsk/thesis/thesis`.
  Confirmed this is the `thesis/` workspace root (holds the reference PDFs, Dockerfile,
  docker-compose.yaml, requirements.txt, the external clones, the HF cache, Obsidian,
  and the doubly nested code tree). This is the git root.
- **Rollback point (pre-run commit): `4bd1ee46b51f8e2277936cb841338d1b21f91d60`**
  (`4bd1ee4`, "final commit before restructuting"). Rollback command:
  `git reset --hard 4bd1ee46b51f8e2277936cb841338d1b21f91d60`.
- `git status` is clean except:
  - `M gfn-lm-tuning/infill_subj_arithmetic/wandb/debug-internal.log`
  - `M gfn-lm-tuning/infill_subj_arithmetic/wandb/debug.log`
  - `M gfn-lm-tuning/infill_subj_arithmetic/wandb/latest-run`
  - `?? controlled_text_generation-gradient_information/controlled_text_generation-gradient_information/unifiedruns/`
  - `?? gfn-lm-tuning/infill_subj_arithmetic/run-logs/`
  - `?? hf/`
  All dirty items are wandb logs (inside `gfn-lm-tuning/`, which never moves) or
  untracked logs/caches. Nothing material is uncommitted. Proceeding.
- Queue liveness: `tmux ls` shows stale `ctg_*` / `cons_*` sessions (created Jul 12-14),
  but `ps` shows NO live `run_queue.sh` / `worker.sh` / `run_experiment.py` process of
  ours. The only running GPU job is user `elattayr`'s `llama_generation.py` (a different
  user, outside our tree, not our queue). Our queue is NOT live. Safe to move `status_*`
  and logs.

---

## PART 1: audit (logged before any move)

### 1.1 / 1.2 Nesting and duplicates, resolved with evidence

- The code is **doubly nested** at
  `controlled_text_generation-gradient_information/controlled_text_generation-gradient_information/`.
  The OUTER `controlled_text_generation-gradient_information/` contains ONLY the inner
  dir and nothing else (verified: `ls` shows a single child; `git ls-files` shows zero
  tracked files under the outer that are not also under the inner). It is a pure empty
  wrapper shell. Decision: after promoting the inner contents, the emptied outer
  wrapper is moved to `archive/`.
- The INNER
  `controlled_text_generation-gradient_information/controlled_text_generation-gradient_information/`
  is the canonical code tree (core/, diagnostics/, revision/, Doc/, figures/, results_*,
  status_*, launchers). Confirmed.
- Root-vs-inner collisions (byte-compared):
  - `README.md`: **DIFFERS** (root = 22 B UTF-16 stub "# thesis"; inner = 16543 B, the
    real guide). Canonical = INNER. Keep inner's README.md at the promoted root; archive
    the root stub as `archive/README.root-stub.md`.
  - `requirements.txt`: **IDENTICAL**. Keep root's; the inner copy is a duplicate ->
    `archive/`.
  - `Dockerfile`: **IDENTICAL**. Keep root's; inner duplicate -> `archive/`.
  - `docker-compose.yaml`: **IDENTICAL**. Keep root's; inner duplicate -> `archive/`.
  - `Checklist_Masterthesis.pdf`: **IDENTICAL** (root and inner). Keep one in `refs/`;
    archive the other duplicate.
  - `Guidelines_for_academic_thesis_writing_at_the_IMS.pdf`: **IDENTICAL**. Keep one in
    `refs/`; archive the other duplicate.
- `.zip` fate: `controlled_text_generation-gradient_information.zip` (root, 35 MB) and
  `ThesisExample.zip` (inner) -> `archive/`. (`Experiments/IdealAlphaSchedule` also holds
  a README, not a zip.)

### 1.3 Import graph (the one fact that matters)

Every diagnostics/ and revision/ entry script uses the pattern:
`HERE = dirname(abspath(__file__)); ROOT = dirname(HERE); sys.path.insert(0, ROOT)`
(some also insert `HERE` or `ROOT/diagnostics`). They then import:
- `from core.prep import ...`, `from core.dls import ...` (resolved at ROOT)
- `from run_experiment import load_texts, build_corruption, seed_all` (resolved at ROOT)
- `from run_revision import ...`, `import sedd_lib`, `from train_noisy_classifier import ...`
  (resolved at `ROOT/diagnostics`, i.e. their own directory)
- `from diagnostics.run_revision import ...` (namespace package under ROOT)

`run_experiment.py` itself has NO sys.path shim and does `from core.prep import ...`;
it relies on CWD = code root (script-dir on sys.path[0]). `verify_equivalence_suite.py`
inserts its OWN dir and imports `core`. `verify_logic.py` / `verify_logic_cls.py` import
BOTH `from Methods.Scripts.dls import ...` AND `from core.dls import ...` with no shim
(rely on CWD = root). Package markers: root `__init__.py`, `core/__init__.py`,
`Methods/__init__.py`, `Methods/Scripts/__init__.py`, `Methods/Utils/__init__.py` exist;
`diagnostics/` and `revision/` have none (implicit namespace packages).

**Post-promotion convention:** with the whole code tree promoted to the git root and
CWD = git root, every import resolves, because `ROOT = dirname(dirname(diagnostics/x.py))`
= git root, and `core/`, `run_experiment.py`, `Methods/` are direct children of that root.
`import core` and `--core_path .` work with CWD at the root.

**Hard constraint this imposes (reshapes the plan):** `run_experiment.py`, `core/`,
`Methods/`, and every root-level Python entry script that imports `core`/`Methods` or is
imported by diagnostics/revision (e.g. `run_experiment.py`, `verify_equivalence_suite.py`,
`verify_logic.py`, `verify_logic_cls.py`, `evaluate.py`, `smoke_test_gfn.py`,
`test_sentiment_head.py`, `verify_constraint_live.py`, `train_sentiment_head.py`,
`compare_models.py`, `replot.py`, `notebook_plotting.py`, `summarize_constrained.py`,
`gen_*.py`) MUST remain at the promoted git root. Moving any of them into a `scripts/`
subfolder breaks the hard-coded `ROOT = dirname(dirname(__file__))` convention and/or the
`from run_experiment import ...` top-level import, and the only fix would be a Python
*logic* edit (changing import statements or sys.path math), which this pass forbids
(only path strings may change). Per the plan's own escape hatch (PART 1.3 "if any module
needs a package install instead, say so and handle it in PART 3" and the meta-rule "if
any single move cannot be rewired safely, leave it in place and record it deferred"),
this is flagged for an author decision below (see "DECISION REQUIRED").

`diagnostics/` and `revision/` themselves are promoted as direct root children (they are
already exactly one level below the code root), preserving the ROOT math. `sedd_lib.py`,
`train_noisy_classifier.py`, `run_revision.py` stay inside `diagnostics/`.

### 1.4 Path-reference index (files hit per token; drives PART 4 to zero)

Across `*.py *.sh *.md *.tex *.bib *.tsv` (excluding wandb/pycache), file counts:

```
results_gpt2_v2   14   results_diag        22   figures_gpt2   3    HF_HOME             6
results_llama     12   results_diagnosis    7   figures_gfn    0    SEDD_REPO           7
results_gfn       10   results_revision    52   figures_llama  0    Score-Entropy-...  10
results_constrained 14 results_rerun        5   figures_compare 4   gfn-lm-tuning      32
results_probe      7   results/ (bare)     11   figures/       14   gpt2_large_sft_output 30
status_             7   unifiedruns          5   rerun_logs      3   sentiment_head     17
wandb             16   hf/cache             6   doubled-path token 1
```

Notes:
- The high `results_revision` (52) and `results_diag` (22) counts include historical
  `REVISION_LOG*.md`, PROMPT_PHASE*.md, result JSONs that store internal paths, and
  Doc/*.tex `%` comments. Historical logs and result JSONs are NOT hand-edited: logs get a
  one-line pointer note (PART 4.7), result JSONs are regenerated by re-running the no-GPU
  analyses so stored paths update while values stay identical (PART 4.5).
- `reconcile_numbers.py` takes `--results_dirs` as `required=True` with NO hardcoded glob
  defaults; it globs `os.path.join(d, "*.json")` over dirs passed in. So results-dir moves
  touch only the COMMAND strings (README / docstring `%`-style usage comment), not the
  script logic. The SEDD exclusion is a run_name filter, unaffected by folder moves.
- argparse defaults that DO point at old folders and need updating:
  `gen_manifest_constrained.py:40 --out_dir=results_constrained`,
  `gen_manifest_revision.py:204/208/209/210 --out_dir/judge_dir/oracle_dir/seed_dir=results_revision`,
  `gen_manifest_revision.py:211 --diag_dir=results_diagnosis`,
  `gen_diag_manifest.py:92 --out_dir=results_diag`,
  `gen_manifest.py:134 --out_dir=results_rerun`,
  `run_queue.sh` default `OUTDIR=results_rerun`, `STATUS=queue_status`.
- Doc/*.tex build: master is `Doc/final/thesis.tex`, `\graphicspath{{../}{../figures/}{./}}`,
  all `\includegraphics` use the `figures/...` prefix which resolves to **`Doc/figures/`**
  (internal to Doc/, 20 files). The root-level `figures/` (33 files) is a SEPARATE
  code-output dir (scripts write there via `--fig_dir figures`). No `.tex` references
  `figures_gpt2/gfn/llama/compare`, `Experiments`, `Methods`, or `results_*`. Therefore the
  LaTeX build's figure link is INTERNAL to Doc/ and is preserved bit-for-bit by promoting
  Doc/ intact; the root `figures/` stays at the root as a code artifact.

### 1.5 Full file inventory (nothing-deleted ledger baseline)

Git-tracked unless marked. Sizes are du-based classes.

ROOT (git root) top-level entries:
| entry | tracked | class | role |
|---|---|---|---|
| `controlled_text_generation-gradient_information/` (outer) | tracked (via inner) | wrapper | empty nesting shell; inner is the code tree |
| `controlled_text_generation-gradient_information.zip` | tracked | 35 MB zip | code snapshot archive |
| `gfn-lm-tuning/` | tracked (+untracked venv/data, 77 GB) | clone + LIVE venv | holds `gfn` venv + gpt2 SFT ckpt + wandb; NEVER moves |
| `Score-Entropy-Discrete-Diffusion/` | tracked (256 KB, no venv) | clone | SEDD positive control code; movable to external/ |
| `hf/` | untracked (5.7 GB, no venv) | HF cache | `HF_HOME` target; movable to external/ |
| `Obsidian/` | tracked (16 MB) | vault | notes; left exactly as-is |
| `.git/`, `.claude/` | - | infra | git dir; Claude settings |
| `README.md` | tracked | 22 B stub | superseded by inner README; -> archive/ |
| `requirements.txt`, `Dockerfile`, `docker-compose.yaml` | tracked | small | canonical at root (inner copies identical) |
| `08_ims-theses_handout.pdf`, `Checklist_Masterthesis.pdf`, `Guidelines_...IMS.pdf`, `Instructions_...supervisors...pdf`, `Thesis...Grading Criteria...pdf`, `thesisioanna.pdf`, `SarthakSinghThesisCurrent.pdf`, `SarthakSinghThesisProposal.pdf` | tracked | PDFs | reference material -> refs/ |
| `5 may lukas meeting transcript.txt`, `Call with Mauch, Lukas-20260505_143722-Meeting Transcript.vtt`, `Call with Mauch, Lukas 5 may pt2.vtt` | tracked | transcripts | -> meetings/ |

INNER code tree top-level entries (the promotion source):
| entry | tracked | class | role / destination |
|---|---|---|---|
| `core/` | tracked | code | samplers+energy; PROMOTE to root (import anchor) |
| `diagnostics/` | tracked | code | PROMOTE to root child (ROOT math; holds run_revision/sedd_lib/train_noisy_classifier) |
| `revision/` | tracked | code | PROMOTE to root child |
| `Methods/` | tracked | code | legacy samplers; imported by verify_logic*; PROMOTE to root (import anchor) |
| `Experiments/` | tracked (426 MB) | code+data | old `IdealAlphaSchedule` aux experiment; self-contained (own sys.path root); PROMOTE to root, NOT referenced by main pipeline or tex |
| `Doc/` | tracked (9.5 MB) | thesis LaTeX | PROMOTE INTACT (Doc/figures/, Doc/final/, Doc/chapters/); internal figure link preserved |
| `figures/` | tracked (5.5 MB, 33 files) | code output | PROMOTE to root; scripts write here via `--fig_dir figures` |
| `figures_gpt2/`,`figures_gfn/`,`figures_llama/`,`figures_compare/` | tracked (~227 MB) | code output | per-model plot outputs (replot.py / compare_models.py); NOT tex inputs |
| `results/` (bare) | tracked (1 legacy CSV) | data | old-format single CSV `exp_dls_masks_1_...` |
| `results_gpt2_v2/`,`results_llama/`,`results_gfn/` | tracked (~205 MB) | data | grid results (per-sample CSV+JSON) |
| `results_constrained/`,`results_probe/` | tracked | data | constrained-generation aggregate JSONs |
| `results_diag/`,`results_diagnosis/` | tracked (289 MB / near-empty) | data | diagnostics outputs (distinct leaves) |
| `results_revision/` | tracked (138 MB) | data | revision analysis outputs incl numbers.json |
| `results_rerun/` | tracked (23 JSONs) | data | grid rerun JSONs |
| `status_constrained/`,`status_diag/`,`status_gfn/`,`status_gpt2_v2/`,`status_llama/`,`status_rev/`,`status_rev_v2/` | tracked | lock dirs | queue lock ledgers (keyed by run_name) |
| `unifiedruns/` | untracked | logs | run logs |
| `rerun_logs/` | tracked | logs | run logs |
| `wandb/` | ignored (182 MB) | logs | wandb run logs |
| `__pycache__/` | ignored | generated | bytecode; travels with tree or regenerated |
| `IoannaThesis/` | tracked | reference | `Ioannathesis.tex` reference thesis -> refs/ |
| `*.py` (root entry scripts) | tracked | code | run_experiment, evaluate, replot, notebook_plotting, compare_models, summarize_constrained, verify_*, test_sentiment_head, train_sentiment_head, smoke_test_gfn, gen_manifest*, gen_diag_manifest, gen_jobs; STAY at root (import anchors) |
| `*.sh` (launchers) | tracked | shell | run_queue, worker, reset_incomplete, run_sedd_slate, run_gprime_slate, launch_experiments, launch_multi, run_constraint_ablation, gpu_monitor |
| `run_constrained.py` | tracked | code | constrained runner |
| `core_*.patch` (3), `core_base_sampler_*.patch` | tracked | patch | historical sampler patches -> archive/ |
| `*.tsv` manifests (12) | tracked | generated | queue manifests (regenerable snapshots) |
| `*.md` (README, REVISION_*, PROMPT_*, THESIS_*, EXPERIMENTS_AND_PLOTS, evaluation_feedback, CLAUDE, thesis_revision_plan, THESIS_GUIDELINES_PROMPT, PROMPT_RESTRUCTURING) | tracked | docs | project docs/logs |
| `Checklist_Masterthesis.pdf`, `Guidelines_...IMS.pdf` (inner) | tracked | PDF dup | identical to root copies -> dedupe |
| `docker-compose.yaml`,`Dockerfile`,`requirements.txt` (inner) | tracked | dup | identical to root -> archive/ |
| `*.Zone.Identifier` (3) | tracked | stray | Windows metadata -> archive/ |
| `ThesisExample.zip` | tracked | zip | -> archive/ |
| `__init__.py` (root) | tracked | pkg marker | STAYS at root |

External clones/caches venv status: `gfn-lm-tuning/gfn/pyvenv.cfg` present (LIVE venv, 77 GB, NEVER moves); `Score-Entropy-Discrete-Diffusion` and `hf` carry no venv (movable to external/).

---

## DECISION REQUIRED before PART 3 (moves)

The audit surfaced one conflict between the requested target layout and the code: the
plan puts `run_experiment.py`, `verify_equivalence_suite.py`, `gen_manifest*.py`, and the
other Python entry runners under `scripts/`, but the import graph (section 1.3) hard-codes
`run_experiment.py`, `core/`, and `Methods/` at the code root and reaches them via
`from run_experiment import ...` and `ROOT = dirname(dirname(__file__))`. Moving Python
entry scripts into `scripts/` breaks these imports, and the only fixes are Python logic
edits, which this pass forbids. The safe, spec-compliant resolution is to keep all Python
at the promoted git root and reserve `scripts/` for the shell launchers (which can be
rewired via path strings). This is recorded here and put to the author before any move,
per the sequencing rule.

### AUTHOR DECISION (2026-07-25)

The author chose: **do the Python logic edits as well** (full `scripts/` layout for the
entry runners), fix the imports, and verify everything runs at the end. So this pass is
authorized to edit import statements / sys.path shims in the entry runners and the
diagnostics/revision modules that cross-import them. `core/` sampler logic is still NOT
touched. The full PART 5 verification gate proves the edited imports run.

Refactor mechanism (minimal, uniform, non-fragile):
- Package dirs stay at the promoted git root: `core/`, `Methods/`, `diagnostics/`,
  `revision/`, `Experiments/` (self-contained). Their internal imports are unchanged.
- Entry runners move to `scripts/`. Each moved runner that imports an internal package
  gets a 2-line shim inserting the repo root (parent of `scripts/`) on `sys.path`:
  `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`,
  so `from core...`, `from Methods...`, `from diagnostics...`, `from revision...` resolve
  regardless of CWD.
- The ONLY moved runner imported by other modules is `run_experiment.py` (used by
  `diagnostics/run_revision.py`, `run_external_judge.py`, `audit_probe.py`,
  `run_sedd_cap.py`, `revision/build_showcase.py`). Those importers already insert `ROOT`
  on `sys.path`; add `os.path.join(ROOT, "scripts")` so `from run_experiment import ...`
  resolves to `scripts/run_experiment.py`. No other runner is imported.
- `gen_manifest*.py` emitted command strings `python run_experiment.py ...` become
  `python scripts/run_experiment.py ...` (path strings); their `--out_dir` defaults are
  repointed to the new results paths.
- Shell launchers `cd` to the repo root (or use `$(dirname "$0")/..`) so `worker.sh`
  `eval`s `python scripts/run_experiment.py ...` with CWD = root and `import core` resolves;
  `run_queue.sh` locates `worker.sh` as `"$(dirname "$0")/worker.sh"`.

---

## PART 2: naming conventions (final)

- `results_gpt2_v2 results_llama results_gfn results_rerun` -> `results/grid/{gpt2_v2,llama,gfn,rerun}`.
  Kept `_v2` on gpt2 to disambiguate from the legacy bare `results/` CSV (audit found no
  `results_gpt2` v1 folder, but a legacy single-CSV `results/` exists, so the suffix stays
  informative). `results_rerun` is grid-family rerun data -> `results/grid/rerun`.
- `results/` (bare, 1 legacy CSV `exp_dls_masks_1_..._policy.csv`) -> `results/legacy/`
  (the CSV moves in first, then `results/` becomes the grouped parent).
- `results_constrained results_probe` -> `results/constrained/{main,probe}`.
- `results_diag results_diagnosis` -> `results/diagnostics/{diag,diagnosis}` (distinct
  leaves kept; near-duplicate noted in README caveats).
- `results_revision` -> `results/revision`.
- `status_*` -> `status/*` (leaf names unchanged: constrained, diag, gfn, gpt2_v2, llama,
  rev, rev_v2; lock ledger keyed by run_name is unaffected by the parent path).
- `unifiedruns rerun_logs wandb` -> `logs/{unifiedruns,rerun_logs,wandb}`.
- `*.tsv` queue manifests (regenerable snapshots) -> `logs/manifests/` (README workflow
  regenerates fresh manifests at the repo root each run).
- `figures/` stays at the promoted root (code output dir, `--fig_dir figures`, CWD=root).
  The thesis build uses Doc/figures/ internally (unchanged). Per-model plot outputs
  `figures_{gpt2,gfn,llama,compare}` -> `figures/{gpt2,gfn,llama,compare}` (subfolders;
  the few script/README refs are rewired).
- refs/: all reference PDFs (`08_ims-theses_handout.pdf`, `Checklist_Masterthesis.pdf`,
  `Guidelines_...IMS.pdf`, `Instructions_...supervisors...pdf`, `Thesis...Grading
  Criteria...pdf`, `thesisioanna.pdf`, `SarthakSinghThesisCurrent.pdf`,
  `SarthakSinghThesisProposal.pdf`) + `IoannaThesis/` (reference thesis .tex).
- meetings/: the `.txt` + two `.vtt` transcripts.
- external/: `Score-Entropy-Discrete-Diffusion/`, `hf/` (no venv). `gfn-lm-tuning/` stays.
- archive/: emptied nesting shells, `controlled_...zip`, `ThesisExample.zip`, root
  `README.md` stub, inner duplicate `requirements.txt`/`Dockerfile`/`docker-compose.yaml`,
  one duplicate of `Checklist`/`Guidelines` PDFs, 3 `*.Zone.Identifier` strays,
  3 `core_*.patch` files (archive/patches/).
- Historical docs stay at the root in place (PART 4.7 rule): all `PROMPT_*.md`,
  `THESIS_WRITING_GUIDE.md`, `THESIS_GUIDELINES_PROMPT.md`, `EXPERIMENTS_AND_PLOTS.md`,
  `evaluation_feedback.md`, `thesis_revision_plan.md`, plus README/CLAUDE/REVISION_* .
- `__pycache__` (gitignored, generated) left in place; regenerates.

Scripts moving to `scripts/`: shells (`run_queue.sh`, `worker.sh`, `reset_incomplete.sh`,
`run_sedd_slate.sh`, `run_gprime_slate.sh`, `launch_experiments.sh`, `launch_multi.sh`,
`run_constraint_ablation.sh`, `gpu_monitor.sh`) and Python entry runners
(`run_experiment.py`, `run_constrained.py`, `evaluate.py`, `replot.py`,
`notebook_plotting.py`, `compare_models.py`, `summarize_constrained.py`,
`verify_equivalence_suite.py`, `verify_logic.py`, `verify_logic_cls.py`,
`verify_constraint_live.py`, `test_sentiment_head.py`, `train_sentiment_head.py`,
`smoke_test_gfn.py`, `gen_manifest.py`, `gen_manifest_revision.py`,
`gen_manifest_constrained.py`, `gen_diag_manifest.py`, `gen_jobs.py`).


---

## PART 3-4 executed (2026-07-25 ~01:00-11:20 CEST)

Moves (all logged live above via the staged git status): 1491 `git mv` renames +
plain `mv` for untracked (`logs/unifiedruns`, `logs/wandb`, inner `__pycache__` ->
`archive/pycache-inner-root`) + untracked `hf` -> `external/hf`. Emptied nesting
shells -> `archive/nesting-shell-.../`. New root layout matches PART 3 target with
the AUTHOR DECISION applied: entry runners + shells in `scripts/`, packages
(`core/ diagnostics/ revision/ Methods/ Experiments/`) at root, `Doc/` + root
`figures/` promoted, data grouped under `results/ status/ logs/`, `refs/ meetings/
archive/ external/` populated. `gfn-lm-tuning/` and `Obsidian/` left in place.

Import refactor (author-authorized logic edits, path resolution only, no sampler
logic touched):
- 10 moved runners given a repo-root `sys.path` shim (9 added, verify_equivalence_suite
  changed self-dir -> parent-dir).
- 5 cross-importers (`diagnostics/run_revision.py`, `run_external_judge.py`,
  `audit_probe.py`, `run_sedd_cap.py`, `revision/build_showcase.py`) add
  `ROOT/scripts` to `sys.path` so `from run_experiment import ...` resolves.
- Emitted commands `run_experiment.py`/`run_constrained.py` -> `scripts/...`;
  argparse `--out_dir` defaults + `run_queue.sh` OUTDIR/STATUS repointed.
- Queue/slate/aux shells cd to the repo root and locate each other by `$0`-relative
  path; slate absolute `HF_HOME`/`SEDD_REPO` -> `external/...`, `OUT`/`LOGDIR`
  repointed. Lock-ledger semantics (keyed by run_name) unchanged.
- Bulk token repoint across 35 live code files; 2 docstring stragglers fixed.
- Doc/*.tex: 74 result-path tokens repointed inside `%` comments ONLY (verified 0
  non-comment occurrences beforehand); prose/numbers untouched.
- `env.sh` created (REPO_ROOT, RESULTS_DIR, STATUS_DIR, LOGS_DIR, SEDD_REPO,
  HF_HOME, GPT2SFT, SENTIMENT_HEAD, PYTHONPATH, HF_DATASETS_TRUST_REMOTE_CODE).
- `requirements.txt.lock` frozen from the gfn venv: 141 pinned lines, torch
  2.12.0+cu130 (interpreter reports the cu130 build; pip freeze emits `torch==2.12.0`),
  transformers 4.38.2, datasets 2.19.2, peft 0.9.0. Python 3.11.15.

## PART 5 VERIFICATION GATE - results

1. GIT STATUS: 1491 R (renames, history preserved) + README M/A/D triple (lossless:
   canonical inner README -> root, 22B stub -> archive/README.root-stub.md). Data
   files are pure renames; sample grid JSON blob hash IDENTICAL across the rename.
2. STATIC py_compile: 72/72 repo .py files PASS (gfn venv python).
3. IMPORTS/LIVENESS from root: `import core.*` OK; analyze_stats --selftest OK;
   --help/import OK on all entry + cross-import scripts (run_experiment,
   run_constrained, verify_equivalence_suite, evaluate, gen_manifest*,
   compare_models, run_revision, run_sedd_cap, run_external_judge, build_showcase,
   reconcile_numbers, analyze_constrained, analyze_likelihood_trap). The scripts/
   path additions resolve `from run_experiment import`.
4. NUMBERS DIFF: byte-identical renames prove no result value changed by
   construction. analyze_stats + analyze_constrained regenerated from new paths ==
   committed (numeric/structure identical). reconcile: numbers.json 158/158 common
   keys with 0 differing values; length_slopes identical. Grid config count 145 =
   gpt2_v2(29) + llama(29) + gfn(87). SEDD exclusion held (0 rev_sedd in grid). The
   pre-existing 0-byte `diag_likelihood_trap_llama3-8b.csv` (byte-identical at HEAD)
   blocks the literal documented reconcile command; unrelated to the move.
5. SAMPLER EQUIVALENCE: verify_equivalence_suite.py (dls, quick, gpt2sft, GPU0):
   identical final token set 2/2, max deviation of mean L2/KL/entropy = 0.000e+00,
   VERDICT "safe to switch, identical behaviour". core/ + Methods/ undisturbed.
6. QUEUE SMOKE: one n=5 job via scripts/run_queue.sh, fresh status/_smoke_test ->
   results/_smoke_test. Worker cd'd to root, claimed lock, ran scripts/run_experiment.py
   (import core via shim), wrote JSON+CSV, touched .done. Re-run: already_done=1,
   "nothing left to claim; exiting", JSON mtime unchanged (resume by lock + JSON).
   Throwaway run + fresh status dir removed (only removal; never tracked, 0 git entries).
7. LATEX GATE: `latexmk -pdf thesis.tex` in Doc/ (canonical report-class master):
   exit 0, converged to 0 undefined references, 0 undefined citations, 102 pages, no
   missing figures, ToC/LoF/LoT regenerate. Title page + Chapter 5 results page
   rendered to PNG and visually confirmed correct. Canonical Doc/thesis.* build
   artifacts now carry the new root path (doubled path resolved by rebuild).
8. SWEEP TO ZERO: live functional surface (scripts/ core/ diagnostics/ revision/
   env.sh + canonical Doc source/artifacts) = 0 stale references. Remaining old-token
   hits are all intentional: README.md (rewritten in PART 6), historical root docs
   (CLAUDE.md, PROMPT_*.md, THESIS_*.md, REVISION_LOG*.md; PART 4.7 leave-in-place),
   logs/manifests/*.tsv snapshots (regenerated fresh at runtime), 285 frozen result
   JSON provenance strings (never hand-edited, not re-runnable), non-canonical
   Doc/final + Doc/figures build artifacts (regenerable), and this log + the prompt +
   .claude infra.

---

## PART 6: README rewritten (2026-07-25)

README.md updated to the new grouped tree: "Repository layout" regrouped; "Environment"
now references `env.sh` and the two install paths (`requirements.txt.lock` exact /
`requirements.txt` loose) with `external/` anchors; every "Reproducing" command repointed
to `scripts/...`, `results/...`, `status/...` with `source env.sh` as the first step; the
"Artifact map" result-file paths repointed (labels/numbers unchanged); "How the queue
works" notes the `scripts/` launchers cd to root; the TODO block replaced with a
"Repository layout note" (reorg date 2026-07-25, pointer to REVISION_RESTRUCTURING.md,
rollback commit 4bd1ee4); "Known caveats" gains the near-duplicate
`results/diagnostics/{diag,diagnosis}` note. REVISION_README.md gained a one-line
"Reorg complete" note under its superseded banner. Claim map, caveats numbers, and the
artifact-map labels are byte-preserved except path strings.

---

## PART 7: CLOSING CERTIFICATION

### (1) Nothing-deleted certification (one row per original top-level inventory entry)

ROOT-level originals:
| original | disposition |
|---|---|
| `08_ims-theses_handout.pdf` | -> `refs/` |
| `5 may lukas meeting transcript.txt` | -> `meetings/` |
| `Call with Mauch, Lukas-20260505_143722-Meeting Transcript.vtt` | -> `meetings/` |
| `Call with Mauch, Lukas 5 may pt2.vtt` | -> `meetings/` |
| `Checklist_Masterthesis.pdf` (root) | -> `refs/` |
| `Guidelines_...IMS.pdf` (root) | -> `refs/` |
| `Instructions_...supervisors...pdf` | -> `refs/` |
| `Thesis...Grading Criteria...pdf` | -> `refs/` |
| `thesisioanna.pdf` | -> `refs/` |
| `SarthakSinghThesisCurrent.pdf` | -> `refs/` |
| `SarthakSinghThesisProposal.pdf` | -> `refs/` |
| `README.md` (root, 22B stub) | -> `archive/README.root-stub.md` |
| `requirements.txt` (root) | unchanged (canonical at root) |
| `Dockerfile` (root) | unchanged (canonical at root) |
| `docker-compose.yaml` (root) | unchanged (canonical at root) |
| `controlled_..._information/` (outer wrapper) | emptied -> `archive/nesting-shell-.../` |
| `controlled_..._information.zip` | -> `archive/` |
| `Score-Entropy-Discrete-Diffusion/` | -> `external/` |
| `hf/` | -> `external/hf/` (plain mv, untracked) |
| `Obsidian/` | unchanged |
| `gfn-lm-tuning/` | unchanged (live venv, never moved) |
| `.git/`, `.claude/` | unchanged (infra) |

INNER code-tree originals:
| original | disposition |
|---|---|
| `core/ diagnostics/ revision/ Methods/ Experiments/` | promoted to repo root (same names) |
| `Doc/` | promoted intact (Doc/figures, Doc/final, Doc/chapters); % comment paths only edited |
| `figures/` | -> repo-root `figures/` |
| `figures_gpt2/ figures_gfn/ figures_llama/ figures_compare/` | -> `figures/{gpt2,gfn,llama,compare}/` |
| `results/` (bare, 1 legacy CSV) | -> `results/legacy/` |
| `results_gpt2_v2/ results_llama/ results_gfn/` | -> `results/grid/{gpt2_v2,llama,gfn}/` |
| `results_rerun/` | -> `results/grid/rerun/` |
| `results_constrained/ results_probe/` | -> `results/constrained/{main,probe}/` |
| `results_diag/ results_diagnosis/` | -> `results/diagnostics/{diag,diagnosis}/` |
| `results_revision/` | -> `results/revision/` |
| `status_{gpt2_v2,llama,gfn,constrained,diag,rev,rev_v2}/` | -> `status/{...}/` |
| `unifiedruns/` | -> `logs/unifiedruns/` (plain mv, untracked) |
| `rerun_logs/` | -> `logs/rerun_logs/` |
| `wandb/` | -> `logs/wandb/` (plain mv, ignored) |
| `__pycache__/` (inner root, ignored) | -> `archive/pycache-inner-root/` (plain mv) |
| `manifest_*.tsv`, `m_an.tsv` (12) | -> `logs/manifests/` |
| 19 `*.py` entry runners | -> `scripts/` (+ import shims) |
| 9 `*.sh` launchers | -> `scripts/` (+ cd-to-root rewiring) |
| `IoannaThesis/` | -> `refs/IoannaThesis/` |
| `README.md` (inner, canonical) | -> repo-root `README.md` |
| `__init__.py` (root pkg marker), `.gitignore` | -> repo root |
| `CLAUDE.md REVISION_LOG.md REVISION_LOG_THESIS.md REVISION_README.md PROMPT_*.md THESIS_WRITING_GUIDE.md THESIS_GUIDELINES_PROMPT.md EXPERIMENTS_AND_PLOTS.md evaluation_feedback.md thesis_revision_plan.md PROMPT_FOR_CLAUDE_CODE.md PROMPT_RESTRUCTURING.md` | -> repo root, in place (content unchanged) |
| `core_constraint.py` (un-imported stray) | -> `archive/core_constraint.py.stray` |
| inner `requirements.txt Dockerfile docker-compose.yaml` (identical dups) | -> `archive/*.inner-dup` |
| inner `Checklist_Masterthesis.pdf Guidelines_...IMS.pdf` (identical dups) | -> `archive/*.inner-dup` |
| `*.Zone.Identifier` (3) | -> `archive/` |
| `core_*.patch` (3) | -> `archive/patches/` |
| `ThesisExample.zip` | -> `archive/` |

Every original path appears exactly once. Nothing deleted: the sole git deletion is the
inner `README.md` path, whose 16.5KB canonical content is now the repo-root `README.md`
and whose displaced 22B root stub is preserved at `archive/README.root-stub.md`.

### (2) Move summary

- `git mv` (tracked renames vs commit 4bd1ee4): **1491** (1422 pure R100 byte-identical +
  69 rename-with-path-string-edit R0<100%).
- Plain `mv` (untracked/ignored, never tracked): `hf`->`external/hf`,
  `unifiedruns`->`logs/unifiedruns`, `wandb`->`logs/wandb`, inner
  `__pycache__`->`archive/pycache-inner-root`, emptied outer nesting shell ->
  `archive/nesting-shell-...`. (5 operations.)
- New tracked deliverables added (A): `env.sh`, `requirements.txt.lock`,
  `REVISION_RESTRUCTURING.md`, `archive/README.root-stub.md`.
- `git diff --cached --shortstat 4bd1ee4`: 1494 files changed. Status letters vs 4bd1ee4:
  1422 R100 + 69 R0 + 4 A + 1 M (README.md) + 1 D (inner README.md).
- Staged for the author; NOT committed. The untracked HF cache (`external/hf`) and run
  logs (`logs/unifiedruns`, `logs/wandb`) are left untracked exactly as before the reorg.

### (3) PART 5 verification results

Recorded item-by-item above ("PART 5 VERIFICATION GATE - results"). All PASS: py_compile
72/72, imports/liveness from root, numbers byte-identical (1422 R100 + regenerated
analyses 0 differing values + 145=5x29 + SEDD excluded), sampler equivalence 0.000e+00,
queue smoke (lock+JSON resume), latexmk clean (0 undefined ref/cite, 102 pages, pages
rendered), sweep-to-zero on the live functional surface.

### (4) Final sweep-to-zero grep table

| surface | old-token hits | disposition |
|---|---|---|
| `scripts/ core/ diagnostics/ revision/ env.sh` | 0 | clean |
| `README.md` | 1 | intentional caveat prose ("old `results_diag` ... became results/diagnostics/diag") |
| canonical `Doc/thesis.*` source + rebuilt artifacts | 0 | doubled path resolved by rebuild |
| historical root docs (CLAUDE.md, PROMPT_*.md, THESIS_*.md, REVISION_LOG*.md) | many | PART 4.7 leave-in-place; REVISION_RESTRUCTURING.md is the authoritative map |
| `logs/manifests/*.tsv` | 11 files | regenerable manifest snapshots; README regenerates fresh at root |
| frozen result JSONs under `results/` | 285 files | provenance strings; never hand-edit a result JSON, not re-runnable |
| `Doc/final/*`, `Doc/figures/*` build artifacts | 4 files | regenerable latex build logs (not the canonical master) |
| `.claude/settings.local.json`, `PROMPT_RESTRUCTURING.md`, this log | few | infra / the prompt / the log itself |

### (5) Deferred / intentional non-edits

No restructuring work was deferred; every entry moved and every live reference was
rewired. Intentional non-edits (by rule): frozen result-JSON provenance strings (never
hand-edited), historical docs (PART 4.7), and regenerable build/manifest snapshots. One
pre-existing data condition surfaced (a byte-identical 0-byte
`results/diagnostics/diagnosis/diag_likelihood_trap_llama3-8b.csv`) that blocks the literal
documented reconcile command; it existed at HEAD and is unrelated to the move.

### (6) Rollback

`git reset --hard 4bd1ee46b51f8e2277936cb841338d1b21f91d60`

---

## ADDENDUM: final Doc tree (2026-07-26)

After the 2026-07-25 pass, the thesis document area was organized one level further
than PART 1.4 recorded. The canonical tree is now:

- `Doc/final/thesis/`   canonical thesis: `thesis.tex` master (article/template class),
  `chapters/` (01..08, 05a, abstract, showcase_appendix, gprime_examples, tab_confusion),
  `references.bib`.
- `Doc/final/proposal/` `proposal.tex`.
- `Doc/final/beamer/`   `Presentation.tex` (+ `bg.png`).
- `Doc/figures/`        thesis figure PDFs/PNGs (the LaTeX build inputs).
- `Doc/prev_version/`   the pre-revision source, kept for reference.
- `refs/ meetings/ archive/ external/` as before.

PART 1.4 recorded the master as `Doc/final/thesis.tex` with
`\graphicspath{{../}{../figures/}{./}}`, which resolved `figures/...` to `Doc/figures/`
because the master sat directly in `Doc/final/`. The master was subsequently moved one
level deeper to `Doc/final/thesis/thesis.tex`, which silently broke that resolution
(`{../figures/}` then pointed at the non-existent `Doc/final/figures/`, so every figure
fell back to a draft box). Fixed 2026-07-26 by setting
`\graphicspath{{../../}{../}{../figures/}{./}}`: `includegraphics` carries a `figures/`
prefix, so `../../` from `Doc/final/thesis/` resolves `../../figures/...` = `Doc/figures/...`.
The final master now builds with zero missing figures.

Stale `%` source-comment paths in the final chapters were also swept
(`figures_gpt2/` -> `figures/gpt2/`, `figures_compare/` -> `figures/compare/`), and the
config-count prose in the appendix now names the current folders
(`results/grid/{gpt2_v2,llama,gfn}`). Historical logs above are not hand-edited; only
live tex comments and README were updated, per the standing rule.
