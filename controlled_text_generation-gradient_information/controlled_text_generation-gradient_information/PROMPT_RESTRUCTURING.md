# Prompt for Claude Code: repository restructuring (the deferred reorg, now authorized)

One session, Opus 4.8, highest reasoning setting (`/model`, Opus 4.8, high or
extended thinking). This is the reorganization every prior phase deferred. The
author now explicitly authorizes it and has committed the current tree to git
immediately before this run, so a clean rollback point exists. Paste the block
below.

---

Read CLAUDE.md, README.md (all of it: "Repository layout", "How the queue works",
"Reproducing each experiment family", "Artifact map", "Known caveats", and the
TODO block), and REVISION_README.md before doing anything. The TODO block in
README.md is the authorizing spec: "git mv only, Doc/ never moves, nothing is
deleted, and a full reference sweep across *.py, *.sh, *.md, *.tex (including %
source comments and reconcile globs) plus the Stage 3 verification gate must
follow." Everything here obeys that, with one reframing the author has approved:
because the code is buried in a doubled directory, the tree is flattened and the
code promoted, so "Doc/ never moves" becomes "Doc/ content and its internal
structure and its link to figures/ never change; Doc/ relocates only as part of
the whole-tree promotion, together with figures/, and the LaTeX build is
re-verified after."

Log everything to a NEW file, REVISION_RESTRUCTURING.md, at the git root,
timestamped, appended. Do not write to REVISION_LOG.md or REVISION_LOG_THESIS.md.
No em-dashes anywhere. This is a MOVE-AND-REWIRE pass, not a rewrite: do not change
a single experimental number, result JSON value, sampler in core/, or sentence of
thesis prose. The only content edits permitted are path strings in code, shell,
manifests, README, env files, and path strings inside % LaTeX comments. Every
other byte is preserved.

Sequencing is a rule. PART 1 (audit) fully precedes PART 3 (moves). PART 4
(reference sweep) precedes PART 5 (verification). PART 6 (README) happens only
after PART 5 is green. Nothing moves before the audit table and the move plan are
written to the log.

## The known ground truth

- Git root is the `thesis/` workspace folder: it holds the reference PDFs, the
  meeting transcripts, Dockerfile, docker-compose.yaml, requirements.txt,
  README.md, the external clones (`gfn-lm-tuning`, `Score-Entropy-Discrete-Diffusion`),
  the HF cache (`hf`), the Obsidian vault (`Obsidian`), and the code, which is
  doubly nested at
  `controlled_text_generation-gradient_information/controlled_text_generation-gradient_information/`,
  plus a sibling `controlled_text_generation-gradient_information.zip`.
- The inner nested directory is the real code tree (core/, diagnostics/,
  revision/, Doc/, figures/, results_*, status_*, the launchers). Confirm this in
  the audit before relying on it.
- The `gfn` virtualenv lives inside `gfn-lm-tuning/`. Directories containing a live
  virtualenv are NEVER moved (venvs hardcode absolute paths); they stay and env.sh
  points at them.

## PART 0: pre-flight (backup is already done by the author)

1. Confirm the tree is committed and clean: `git rev-parse --show-toplevel` (must
   be the `thesis/` root), `git status` (clean, or list what is dirty and stop if
   anything material is uncommitted). Record the current commit hash as the
   rollback point in the log; the author's pre-run commit is the safety net, so no
   new snapshot is needed.
2. Confirm no queue is live (`tmux ls`, `ps` for `run_queue.sh`/`worker.sh`). If a
   queue is running, STOP; do not move status_* or logs under an active queue.

## PART 1: audit first (write all of this to the log before any move)

1. FILE INVENTORY. Walk the entire tree from the git root. For every top-level
   entry and every entry inside the doubled code directory, record path,
   git-tracked or untracked (`git ls-files`, `git status --ignored`), size class
   (small / large data / clone / venv-bearing), and a one-line role. This ledger
   is what the nothing-deleted certification diffs against at the end.

2. NESTING AND DUPLICATES, resolved with evidence. Confirm which of the two
   `controlled_text_generation-gradient_information` levels is the canonical code
   tree (the one git tracks, holding core/ and Doc/). Diff the two levels: is the
   outer an empty or near-empty wrapper, does either hold unique files. Do the
   same for the root-level README.md / requirements.txt / Dockerfile versus any
   copies inside the code tree: decide which is canonical (by content and git
   history), keep it, and mark the other for archive/, never deletion. Also decide
   the `.zip`'s fate: archive/. Log every decision with the evidence. No other
   move begins until this is settled.

3. IMPORT GRAPH. Grep every `.py` for `import`, `from ... import`, `--core_path`,
   `sys.path`, and relative-path opens. Map the dependencies of core/,
   diagnostics/, revision/, run_experiment.py, gen_manifest*.py. Output the one
   fact that matters: with the code promoted to the git root, does running from the
   git root make every import resolve (core/ becomes a direct child, so `import
   core` and `--core_path .` work with CWD at the root). Confirm this is the
   coherent post-promotion convention; if any module needs a package install
   instead, say so and handle it in PART 3.

4. PATH-REFERENCE INDEX (the checklist PART 4 drives to zero). Grep the whole tree,
   `*.py *.sh *.md *.tex *.bib` and the manifests, for every path token a move
   invalidates: `results_gpt2_v2`, `results_llama`, `results_gfn`,
   `results_constrained`, `results_probe`, `results_diag`, `results_diagnosis`,
   `results_revision`, `status_`, `figures/`, `unifiedruns`, `rerun_logs`,
   `wandb`, the launcher names, `hf/cache`, `HF_HOME`, `SEDD_REPO`,
   `Score-Entropy-Discrete-Diffusion`, `gfn-lm-tuning`, `gpt2_large_sft_output`,
   `sentiment_head.pt`, and the doubled `controlled_text_generation-gradient_information`
   path itself. Record file, line, and construct (glob, argparse default, env
   export, README command, % comment, \graphicspath / \includegraphics). Also grep
   result JSONs (numbers.json, rev_*.json) for stored internal paths.

Log the audit. Do not move anything yet.

## PART 2: naming conventions (decide, then apply consistently in PART 3)

The author wants proper names, not just regrouping. Rule: regroup under themed
parents, drop the redundant `results_` and `status_` prefixes, and normalize
leaves, but do NOT drop an informative suffix (a version tag, a scale) unless the
audit proves it vestigial. Proposed map, refine in the audit and log the final:

- `results_gpt2_v2 results_llama results_gfn` -> `results/grid/{gpt2,llama,gfn}`.
  Keep the `_v2` on gpt2 as `gpt2_v2` if the audit finds any `results_gpt2`
  (v1); otherwise `gpt2` is fine. Log the check.
- `results_constrained results_probe` -> `results/constrained/{main,probe}`
  (the analyzer reads both; keep them siblings under one parent).
- `results_diag results_diagnosis` -> `results/diagnostics/{diag,diagnosis}`.
  Keep both leaves distinct; do not merge two folders that may hold different
  runs. Note the confusing near-duplicate in the log and in README caveats.
- `results_revision` -> `results/revision`.
- `status_*` -> `status/*` (leaf names unchanged; the lock ledger is keyed by
  run_name, so the leaf naming must not change how a lock path is formed).
- `unifiedruns rerun_logs wandb` and stray `*.log` -> `logs/`.

## PART 3: target layout and the move plan

Write the full old-to-new move table to the log first: source, destination,
`git mv` (tracked) or plain `mv` (untracked), one-line reason each. Then execute,
`git mv` for tracked so history is preserved, plain `mv` for untracked, each move
logged as it happens. Promote the code tree to the git root and organize around it:

```
thesis/  (git root, becomes the runnable repo root)
  README.md  CLAUDE.md  REVISION_RESTRUCTURING.md
  REVISION_LOG.md  REVISION_LOG_THESIS.md  REVISION_README.md   (kept in place)
  requirements.txt  Dockerfile  docker-compose.yaml
  env.sh                          NEW, see 3.1
  Doc/                            promoted intact; content unchanged
  figures/                        promoted WITH Doc/, relative link preserved
  core/  diagnostics/  revision/  promoted; importable from the root
  scripts/                        run_queue.sh worker.sh reset_incomplete.sh
                                  run_sedd_slate.sh run_gprime_slate.sh
                                  gen_manifest*.py, run_experiment.py, verify_equivalence_suite.py,
                                  and the other top-level entry runners
  results/{grid,constrained,diagnostics,revision}/   per PART 2
  status/                         all status_* lock dirs
  logs/                           unifiedruns, rerun_logs, wandb, stray *.log
  external/                       SEDD clone and hf cache IF safely movable (3.3)
  gfn-lm-tuning/                  LEFT IN PLACE (holds the live gfn venv)
  refs/                           the IMS PDFs, checklist, guidelines, grading
                                  criteria, handout, external-supervisor instructions,
                                  thesisioanna.pdf, the two SarthakSinghThesis*.pdf
  meetings/                       the .txt and .vtt transcripts
  Obsidian/                       left exactly as is; do not restructure its insides
  archive/                        the .zip, the emptied nesting shells, duplicate
                                  README/requirements if the canonical lives at root,
                                  *.bak duplicates, dead strays
```

Collision handling during promotion: if a promoted file collides with a root-level
file of the same name (README.md, requirements.txt, Dockerfile), keep the
canonical one decided in PART 1.2 at the root and move the other to archive/ with a
suffix noting its origin. Never overwrite, never delete.

3.1 CENTRALIZE PATHS. Create `env.sh` at the root exporting `REPO_ROOT`,
   `RESULTS_DIR`, `STATUS_DIR`, `LOGS_DIR`, `SEDD_REPO`, `HF_HOME`, `GPT2SFT`,
   `SENTIMENT_HEAD`, pointing at the post-move locations. The README workflow
   becomes: cd to the repo root, `source env.sh`, run. This makes any future move
   one edit. Where an anchor is a machine-specific absolute path (the sentiment
   head under /mount/arbeitsdaten), keep it and comment it as server-specific.

3.2 FIGURES MOVE WITH Doc/. figures/ is referenced by Doc/*.tex and by the plot
   scripts. Promote it in the same relative position to Doc/ that it holds now, so
   \graphicspath / \includegraphics still resolve. Confirm the exact reference
   style in the audit; if Doc/ expects figures/ as a root-level sibling, that is
   exactly where it lands after promotion. Do not change any figure include path.

3.3 EXTERNAL CLONES AND CACHES. `gfn-lm-tuning` stays put (live venv). For
   `Score-Entropy-Discrete-Diffusion` and `hf`: if they carry no venv and moving is
   cheap, move them under `external/` and update env.sh and every reference;
   otherwise leave them at the root and point env.sh at them. Never move a
   directory with a live virtualenv. Never delete a checkpoint or a cache. Log the
   per-item decision.

3.4 The `.bak` sampler backups (`core/dls.py.bak`, `core/cls.py.bak`) travel with
   core/ and stay beside their files so verify_equivalence_suite.py restore logic
   still finds them.

3.5 FREEZE THE WORKING ENVIRONMENT. Capture the exact installed package set as a
   lockfile so the environment is reproducible, not just approximately. Run the
   freeze with the REAL working interpreter, the `gfn` venv inside
   `gfn-lm-tuning/` (README: "The active virtualenv is `gfn-lm-tuning/gfn`"), NOT
   the ambient python: `gfn-lm-tuning/gfn/bin/python -m pip freeze >
   requirements.txt.lock` (or activate it first, then `pip freeze`). Write
   `requirements.txt.lock` at the repo root beside `requirements.txt`; do not
   touch or overwrite `requirements.txt` itself, which stays as the loose,
   human-edited spec. Self-verify before moving on: the lockfile is non-empty,
   parses as one `name==version` per line, and pins the known anchors (it must
   contain `torch` at `2.12.0+cu130` and the `transformers`, `datasets`, `peft`
   lines); if `torch` is absent or unpinned you froze the wrong interpreter, so
   redo it against the venv. Log the line count and the torch pin. In env.sh add a
   comment that exact reproduction uses `pip install -r requirements.txt.lock` and
   loose setup uses `requirements.txt`.

## PART 4: the full reference sweep (drive the PART 1.4 index to zero)

Update every hit, worked through by construct type:

1. RECONCILE GLOBS. `revision/reconcile_numbers.py` globs over `results_*`. Repoint
   them at the new `results/grid`, `results/diagnostics`, `results/revision`
   layout and PRESERVE the SEDD exclusion (`rev_sedd_*` stays outside the AR
   reconcile globs). The config count must still reconcile to 145 = 5 x 29; verify
   in PART 5.
2. ARGPARSE DEFAULTS. Any script whose `--out_dir` / `--results_dir` default points
   at an old folder (for example gen_manifest_revision.py -> results_diagnosis).
   Update the default to the new path, or make README pass paths explicitly; pick
   one convention and apply it throughout. Log which.
3. SHELL LAUNCHERS. run_queue.sh, worker.sh, reset_incomplete.sh, the two slate
   scripts, now under scripts/: update any internal path assumptions and how they
   locate each other. Keep the lock-ledger semantics identical (status keyed by
   run_name); moving status_* under status/ must not change how a lock path forms.
4. LATEX % SOURCE COMMENTS. Doc/*.tex carries a % comment naming each number's
   source file. Update ONLY the path string inside each % comment to the new
   results/revision/... and results/diagnostics/... locations. Do not touch the
   number, the prose, or any non-comment text.
5. RESULT JSON INTERNAL PATHS. If numbers.json or the rev_*.json store file paths,
   regenerate them by re-running the no-GPU analyses in PART 5 so stored paths
   match the new tree, then numbers-diff the VALUES to confirm only paths moved.
   Never hand-edit a result JSON.
6. THE DOUBLED-PATH REFERENCES. Any reference to the old
   controlled_text_generation-gradient_information/controlled_text_generation-gradient_information
   prefix now resolves to the root; fix each.
7. OTHER MARKDOWN. Sweep every other .md for path references. For the historical
   PROMPT_PHASE*.md and log files, prefer a one-line note that paths changed and a
   pointer to REVISION_RESTRUCTURING.md over rewriting settled history; log the
   choice.

Produce the completed sweep table: every original hit, its new value, done or
intentionally-left with reason.

## PART 5: verification gate (every piece of code, from its new home; must pass before PART 6)

Full experiment re-runs are out of scope and forbidden (CLAUDE.md: do not re-run
completed jobs; most need GPUs and the frozen checkpoints). Verification means the
code resolves and runs its cheap paths from the new root, and the numbers are
provably untouched. Run each, log the result. On any failure, fix, or if the fix
would force an unsafe move, revert that move and log it deferred rather than ship a
broken tree.

1. `git status` shows the moves as renames (history preserved), tree otherwise
   clean.
2. STATIC: `find . -path ./external -prune -o -path ./gfn-lm-tuning -prune -o -name
   '*.py' -print | xargs -n1 python -m py_compile` passes with zero errors, across
   core/, diagnostics/, revision/, scripts/, and every utility.
3. IMPORTS AND LIVENESS from the git root: `python -c "import core.dls, core.cls,
   core.base_sampler, core.constraint, core.prep"`; then exercise every cheap hook
   that exists rather than only the ones named here, for example
   `python revision/analyze_stats.py --selftest`,
   `python diagnostics/run_sedd_linearization.py --dry_run ...` if its inputs are
   cheap, and `--help` on each entry script to confirm argparse and imports load.
   List every script checked and its result.
4. NUMBERS DIFF. Re-run the no-GPU analyses against the NEW paths (analyze_stats,
   analyze_constrained, reconcile_numbers, analyze_likelihood_trap) and diff their
   regenerated JSON against the committed copies: every value identical, only
   stored paths changed. Confirm reconcile still reports 145 = 5 x 29 and the SEDD
   exclusion held.
5. SAMPLER EQUIVALENCE. `python scripts/verify_equivalence_suite.py` (or its new
   path) passes: core/ moved without disturbing the samplers, the gn=on bitwise
   gradnorm == random behavior and RNG alignment intact.
6. QUEUE SMOKE TEST. One tiny job (one shard, n=5) through scripts/run_queue.sh with
   a FRESH status dir under status/, confirming it writes JSON to the new
   results/... out_dir and that a rerun resume-skips by lock and by JSON existence.
   Then remove only that throwaway smoke run and its fresh status dir (new, not
   existing data), logged.
7. LATEX GATE. `latexmk -pdf` clean in Doc/ from the new root, zero undefined refs
   or citations, every figure resolves, lists of tables and figures regenerate.
   Render the title page and one results page to PNG and inspect.
8. SWEEP TO ZERO. Re-grep the tree for every old path token from PART 1.4. Every
   remaining hit points at the new location or is an intentionally archived
   historical copy. Log the final grep table at zero live stale references.

## PART 6: rewrite README.md to the new structure (only after PART 5 is green)

Keep the voice and every claim, number, caveat, and the artifact map intact; change
only what the move changed. Update "Repository layout" to the new grouped tree;
update every command in "Reproducing each experiment family" to the new paths
(`scripts/run_queue.sh`, `--out_dir results/grid/...`, `--status status/...`,
results/diagnostics, results/revision) and add `cd <root> && source env.sh` as the
first step; update every result-file path in the "Artifact map" without changing a
single label or number; update "Environment" anchors to their post-move locations,
reference env.sh, and state the two install paths (exact via
`pip install -r requirements.txt.lock`, loose via `requirements.txt`); replace the
"TODO / Deferred by author decision" block with a
short "Repository layout" note stating the tree was reorganized on <date>, that the
full old-to-new move map and the nothing-deleted certification live in
REVISION_RESTRUCTURING.md, and that the pre-run commit <hash> is the rollback. Keep
Known caveats, adding the near-duplicate results/diagnostics/{diag,diagnosis} note.
Mirror a one-line "reorg complete, see REVISION_RESTRUCTURING.md" under
REVISION_README.md's existing superseded banner.

## PART 7: closing certification

Append to REVISION_RESTRUCTURING.md: (1) the nothing-deleted certification, one row
per original entry from the PART 1 inventory mapping it to its new path, "archived
to archive/", or "unchanged", every original path appearing exactly once; (2) the
move summary with git mv vs plain mv counts and the `git diff --stat <pre-run
commit>` rename summary; (3) the PART 5 results item by item, each PASS with its
evidence; (4) the final sweep-to-zero grep table; (5) any deferred item stated
plainly with its reason; (6) the rollback line (`git reset --hard <pre-run
commit>`). Do not commit; stage everything and show the author `git status` and the
rename diff so they commit.

## Constraints

Nothing is deleted; moves and archives only, the throwaway PART 5.6 smoke run the
only removal, logged. `git mv` for tracked, plain `mv` for untracked, both logged
per file. Never move a directory containing a live virtualenv. Doc/ content and its
link to figures/ are preserved bit for bit; they relocate only as part of the whole
tree and the build is re-verified. Do not change any number, any result JSON value,
any core/ sampler logic, or any thesis prose; the only content edits are path
strings in code, shell, manifests, README, env files, and % LaTeX comments. No new
experiments beyond the smoke tests. No em-dashes. If any single move cannot be
rewired safely in this pass, leave it in place and record it deferred rather than
force it and break the tree.
