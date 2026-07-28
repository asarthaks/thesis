# CHANGE_LOG_PHASE12.md

Generated from `git diff` against the snapshot commit
`ec346e1496f7a7f9c77ac035442c6004ac19b97f`, not from recollection. One row per diff hunk,
line ranges in the **new** file. Remark IDs are those of the Phase 12 brief.

---

## Doc/final/thesis/chapters/abstract.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 6-23 | file header | `%` comment block recording the rewrite, what was cut from the abstract and where each cut item still stands | A4 |
| 25 | paragraph 1 | Rewritten: "energy function" and "Langevin dynamics" glossed in place; the footnote giving a length constraint as the concrete example added here | A1, A4 |
| 27 | paragraph 2 | Rewritten: the bench glossed in place; the five energy functions named individually | A2, A4 |
| 29 | finding (1) | New numbered, bolded finding: the gradient direction is not usable | A3, A4 |
| 31 | finding (2) | New numbered, bolded finding: what fails is the coordinate the derivative is taken in | A3, A4 |
| 33-35 | finding (3) + close | New numbered, bolded finding: the remaining gap is bought by context; practical implication folded in as its last sentence | A3, A4 |

## Doc/final/thesis/chapters/abstract.tex, second round (author feedback: register)

The abstract was rewritten a second time after the author judged the first version's
phrasing to read as machine-generated. The register is now that of the reference thesis in
`refs/IoannaThesis/`: continuous prose rather than a list, long declarative sentences
carrying subordinate clauses, `i.e.` glosses, and a closing "These findings indicate
that", with no bolded headline lead-ins and no short punchy fragments.

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 25 | paragraph 1 | Reworded to flowing prose: "satisfying a property", "which cannot be verified until the sequence is complete", "i.e. a score that is low for text the model considers good", "This thesis tests the assumption on which that construction rests, namely that..." | A4, register |
| 27 | paragraph 2 | "The bench is X:" replaced by "The evaluation uses X, in which ..."; the five energy functions kept, now in one flowing list | A2, A4, register |
| 29 | findings | The three separate bolded paragraphs merged into **one** paragraph of continuous prose opening "Three results follow.", with the numbering kept as inline `(1)`, `(2)`, `(3)` markers so remark A3 is still satisfied; the practical implication folded into its last sentence as "These findings indicate that..." | A3, A4, register |

No number's value changed in this round; verified by numeral diff against the previously
verified build (removals: none in the abstract).

## Doc/final/thesis/chapters/01_introduction.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 4-11 | 1 Introduction, opening paragraph | `%` comment block; "language model" defined at first use as the document's single term; model capability re-attributed to the factorization *plus* training scale and the Transformer, with citations | I1, I2 |
| 22 | 1.1 Established Routes | "aligned assistants" to "aligned language models" | I2 |
| 26-28 | 1.1 Established Routes, closing paragraph | `%` comment; "inference-time control" defined in place at the point the capability is first named | I3 |
| 38-42 | 1.2 Energy-Based Control, after eq. (1) | `%` comment; `x`, `E`, `Z` typed; the energy-probability direction read off explicitly; a sentence on what may serve as an energy, pointing to Section 2.1 | I4, G1 |
| 49 | 1.2, after eq. (2) | `C : V^T -> R` and `lambda > 0` typed | G1 |
| 51 | 1.2, central-assumption paragraph | "input-embedding gradient" and "norm-matched random direction" glossed at first use | G3 |
| 53 | 1.2, hypothesis paragraph | "teacher forcing" glossed; "token-indicator coordinates" replaced by a plain-language description at this first mention | G3 |

## Doc/final/thesis/chapters/02_background.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 14 | 2.1, after eq. (3) | `x_<t` defined; `V` and `\|V\|` introduced; logit vector typed `R^{\|V\|}`; plain-language reading of the chain rule added | G1 |
| 33 | 2.2, after eq. (5) | State, gradient, noise and covariance all typed; one-sentence reading of the Langevin update added | G1 |
| 47 | 2.3, after eq. (6) | `s`, `s'`, `q` and `alpha` named and typed, `alpha` given its range | G1 |
| 60 | 2.4, before eq. (7) | **`v` defined** as a candidate vocabulary item, with `e(v)` and `e(x_i)` typed | M2, G1 |
| 62 | 2.4, eq. (7) | Step size renamed `alpha` to `epsilon`, removing the collision with the acceptance probability of eq. (6) and matching Sections 4.3 and 5.3.2 | G1 |
| 65 | 2.4, after eq. (7) and its footnote | `g` and the displacement typed; step size identified with eq. (5)'s; plain-language reading added; `alpha` to `epsilon` in the footnote derivation | M2, G1 |
| 69 | 2.4, before eq. (8) | `M` introduced; state typed `R^{M x D}`; `proj_V` given its signature | G1 |

## Doc/final/thesis/chapters/03_related_work.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 11 | 3.1 Controllable Text Generation | "aligned assistants" to "aligned language models" | I2 |

## Doc/final/thesis/chapters/04_methodology.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 53 | 4.3, after eq. (9) | The four parts of the continuous update named and explained one clause each (current state, interim continuous point, projection, interpolated mean), all typed; step size and covariance stated | M1, M2, G1 |
| 73 | 4.4, before eq. (10) | `p_ref` and `p_pred` typed as probability vectors in `R^{\|V\|}`; index range over `M'` stated | M2, G1 |
| 107-109 | 4.7 Diagnostic Experiments, linearization paragraph | New paragraph explaining the inner product `g` transpose `e` in words: what each factor is, what the product measures, and why it is the object under test | M3 |

## Doc/final/thesis/chapters/05_results.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 20-23 | 5.1, former 5.1.1 heading | Subsection collapsed into 5.1: heading removed, `\label{sec:results-quench}` kept at the promotion site, `%` comment recording the collapse | T2 |
| 72 | 5.1, `\caption[...]` of `fig:dls-traj-50` | Short caption: "Discrete Langevin sampler" to "Discrete Langevin Sampler", matching the defined term | T1 |
| 105-107 | 5.2, former 5.2.1 heading | Subsection collapsed into 5.2; label kept; a transition clause added so the promoted paragraph reads on from the parent; `%` comment | T2 |
| 110 | 5.2 | Sentence start adjusted for the collapse | T2 |
| 317 | 5.4 heading | Retitled "Why the Input-Embedding Surrogate Fails, **and What Recovers It**", so the parent covers the promoted subsection | T2 |
| 357-360 | 5.4, former 5.4.1 heading | Subsection collapsed into 5.4; `\label{sec:results-onehot}` kept; `%` comment | T2 |
| 365 | 5.4 | Sentence start adjusted for the collapse | T2 |
| 370 | 5.4, before eq. (11) | `E` typed `R^{\|V\| x D}`; `z_i` typed and its simplex vertex spelled out | G1 |
| 375 | 5.4, after eq. (11) | Sum index range stated; `L_future` and the mixed embedding typed; plain-language reading of the two terms | G1 |
| 657 | 5.4, after eq. (12) | Derivative's own shape stated (`R^{\|V\|}`) and contrasted with `R^D`; plain-language reading of the two terms added | G1 |
| 698-701 | 5.10 heading and former 5.10.1 | 5.10 retitled "Extension: Constrained Generation **and Classifier-Guided Steering**"; subsection collapsed into it; both labels kept; `%` comment | T2 |

## Doc/final/thesis/chapters/08_appendix.tex

| lines (new) | section / environment | change | remark |
|---|---|---|---|
| 74-75 | A.1, figure block | Forced `\clearpage` removed, replaced by a `%` comment naming the cut | ranked cut (page ledger) |
| 134-135 | A.2 to A.3 boundary | Forced `\clearpage` removed, `%` comment | ranked cut |
| 170-171 | A.3 to A.4 boundary | Forced `\clearpage` removed, `%` comment | ranked cut |
| 183 | A.4, `fig:traj-distance` | `\includegraphics` width 0.80 to 1.0 `\textwidth` | G2 |
| 186-194 | A.4, `fig:traj-distance` | `%` comment recording the partial fix, the missing checkpoint that blocks a proper regeneration, and the residual defect | G2 |
| 214 | A.4, `fig:traj-pca` | `\includegraphics` width 0.85 to 1.0 `\textwidth` | G2 |
| 216-218 | A.4, `fig:traj-pca` | `%` comment, same cause and limitation | G2 |
| 225-226 | A.4 to A.5 boundary | Forced `\clearpage` removed, `%` comment | ranked cut |
| 307-309 | A.5.2 heading | Retitled "The Guide-Judge Agreement Ladder **and Per-Class Confusion**"; `\label{app:guided-confusion}` moved here so its cross-references still resolve | T1, ranked cut |
| 344-348 | A.5.2, merged body | A.5.3's heading removed and its opening paragraph promoted; `%` comment recording the merge | ranked cut |
| 358-361, 362 | A.5.2, merged body | Two sentences that restated the paragraph above consolidated into one; every number, both tables and all `% SOURCE` comments retained | ranked cut |
| 366-367 | A.6 boundary | Forced `\clearpage` removed, `%` comment | ranked cut |
| 402-403 | A.8 boundary | Forced `\clearpage` removed, `%` comment | ranked cut |
| 407 | A.8 heading | "Use of AI-Tools" to "Use of AI Tools", matching the Erklaerung's own wording on page 1 | T1 |

## Doc/final/thesis/chapters/06_discussion.tex and 08_appendix.tex, T2 follow-through

Collapsing 5.1.1, 5.2.1, 5.4.1 and 5.10.1 into their parents made the child label and the
parent label render the **same number**. Four sentences cited both. Three of them read as
broken ("Sections 5.4 and 5.4") or as self-contradicting (insisting two answers are kept
apart while pointing at one section) and were repaired. The fourth, in the RQ1 answer,
cites Section 5.4 twice in different sentences of a long paragraph, which is ordinary
academic usage, and was left alone.

| lines (new) | section | change | remark |
|---|---|---|---|
| `06_discussion.tex:19` | 6.1, extension question E | Second pointer redirected from `sec:results-guided` (now 5.10, the same as the first pointer in the sentence) to `app:guided`, where the guided-steering results are in fact reported in full | T2 |
| `06_discussion.tex:28` | 6.2, mechanism one | "(Sections~\ref{sec:results-linradius} and~\ref{sec:results-onehot})", which rendered "Sections 5.4 and 5.4", reduced to the single "(Section~\ref{sec:results-linradius})" | T2 |
| `08_appendix.tex:175` | A.4 opener | "the acceptance statistics of Section 5.2 and the trajectory summaries of Section 5.2" rewritten as "the acceptance statistics and trajectory summaries of Section~\ref{sec:results-mh}" | T2 |

Verified after the repair: zero self-references (a section pointing the reader at itself)
and zero same-number pairs within a sentence anywhere in the thesis.

## diagnostics/plot_diagnostics.py

| lines (new) | function | change | remark |
|---|---|---|---|
| 42-76 | module rcParams | `TEXTWIDTH_IN` and a `figsize(width_frac, aspect)` helper added, so every figure is authored at the width it is printed at; explicit label, tick, legend and title sizes set | G2 |
| 84-95 | `save()` | `raster_dpi` parameter (default 350) added and passed to the PDF write, so artists marked `rasterized=True` are flattened while text and axes stay vector | F2 |
| 103-118 | `plot_linearization`, 1A | Figure drawn at printed size; scatter marked `rasterized=True`; colourbar label and ticks sized | F2, G2 |
| 145-176 | `plot_linearization`, 1B | Figure drawn at printed size; per-call font-size overrides removed in favour of the rcParams | G2 |
| 182-201 | `plot_linearization`, 1C | Figure drawn at printed size; scatter rasterized; the oversized footer string removed (it was inflating the tight bounding box and shrinking the whole figure); panel gutter widened | F2, G2 |
| 219-230 | `plot_linearization`, 1D | Figure drawn at printed size; legend size from rcParams | G2 |
| 361-392 | `plot_likelihood_trap`, 4A | Figure drawn at printed size; x-label shortened so the two panels' labels no longer collide and the right is no longer clipped; y-axis headroom added and the legend pinned upper-left in two columns so it clears the data; gutter widened | G2 |
| 399-414 | `plot_likelihood_trap`, 4B | Figure drawn at printed size; legend size from rcParams | G2 |
| 428-450 | `plot_anisotropy` | Figure drawn at printed size; x-label shortened (same collision and clipping as 4A); gutter widened | G2 |

## diagnostics/analyze_mh.py

| lines (new) | function | change | remark |
|---|---|---|---|
| 87 | imports | `FuncFormatter`, `MaxNLocator` imported | F1 |
| 92-124 | module rcParams | `TEXTWIDTH_IN`, the `figsize` helper and a `thousands` tick formatter added; explicit label, tick and legend sizes set | F1, G2 |
| 190-198 | `main`, Plot 2A | Figure drawn at printed size; bar annotation reduced from 11pt to 9.5pt | G2 |
| 203 | `main`, Plot 2B | Figure drawn at printed size | G2 |
| 205-211 | `main`, Plot 2B | Axis labels changed from a difference of logs to the equivalent log-ratio form, which is what the running text calls them and which fits the panel; the previous form was clipped | G2 |
| 219-234 | `main`, Plot 2B | **Tick locator capped and magnitudes printed in thousands**, fixing the colliding x-axis; legend pinned upper-left with added headroom so it clears the histogram; the caption-duplicating suptitle removed; gutter widened | F1, G2 |
| 240 | `main`, DLS contrast | Figure drawn at printed size | G2 |

## Binary: regenerated figures

Nine figure files were rewritten by re-running the two plotting scripts above. No image was
edited by hand.

| file | before (bytes) | after (bytes) | remark |
|---|---|---|---|
| `Doc/figures/fig_lin_scatter.pdf` | 615,644 | 317,594 | F2, G2 |
| `Doc/figures/fig_lin_decomposition.pdf` | 613,312 | 131,948 | F2, G2 |
| `Doc/figures/fig_mh_decomposition.pdf` | 28,961 | 26,356 | F1, G2 |
| `Doc/figures/fig_mh_accept.pdf` | 16,729 | 16,892 | G2 |
| `Doc/figures/fig_lin_radius.pdf` | 26,908 | 26,945 | G2 |
| `Doc/figures/fig_lin_topk.pdf` | 17,993 | 18,133 | G2 |
| `Doc/figures/fig_trap_scatter.pdf` | 81,845 | 81,961 | G2 |
| `Doc/figures/fig_trap_length.pdf` | 47,581 | 47,796 | G2 |
| `Doc/figures/fig_aniso_hist.pdf` | 24,219 | 24,195 | G2 |

The four tracked `.png` companions the scripts also emit (`fig_mh_accept`,
`fig_mh_decomposition`, `fig_trap_length`, `fig_trap_scatter`) were rewritten in the same
runs. They are not included by the thesis, which uses the `.pdf` of each; they are tracked
side-products and are updated for consistency. Five further `.png` side-products the
scripts newly emitted (`fig_lin_scatter`, `fig_lin_decomposition`, `fig_lin_radius`,
`fig_lin_topk`, `fig_aniso_hist`) were **deleted** rather than left in the tree, since they
were not in the repository before and the document does not use them.

## Binary and generated: build products

`Doc/final/thesis/thesis.pdf`, `.aux`, `.bbl`, `.fdb_latexmk`, `.fls`, `.lof`, `.log`,
`.lot`, `.out`, `.synctex.gz`, `.toc`. All are `latexmk` output, regenerated by the
rebuild. `.lof` and `.lot` were regenerated after the T2 collapses, as the gate requires.
`.bbl` is **identical modulo whitespace** to the snapshot's, verified by diffing both with
whitespace collapsed: the bibliography did not change.

---

## Pure sweeps

Not listed occurrence by occurrence, per the brief.

| sweep | convention applied | replacements | files touched |
|---|---|---|---|
| **T1** capitalization | Title Case for every chapter, section and subsection title (first and last word always; all others except articles, coordinating conjunctions and prepositions of four letters or fewer; both halves of a hyphenated compound when both are major words). Sentence case with a terminal period for every table and figure caption. | 66 headings and 37 short captions audited; **2** were non-compliant and were corrected (`08_appendix.tex:407`, `05_results.tex:72`), both listed individually above | `08_appendix.tex`, `05_results.tex` |
| **I2** terminology | Single term **language model**, defined at first use in Section 1. `LLM` verified absent from the thesis (0 occurrences). | **2** replacements of "aligned assistants" | `01_introduction.tex`, `03_related_work.tex` |

---

## VERIFICATION

### Every hunk maps to a remark ID

Every hunk in every `.tex` file and both `.py` files carries a remark ID in the tables
above. The mapping is complete for the source files.

### Every remark ID appears at least once

| ID | appears in the log |
|---|---|
| A1 | yes (`abstract.tex:25`) |
| A2 | yes (`abstract.tex:27`) |
| A3 | yes (`abstract.tex:29,31,33-35`) |
| A4 | yes (`abstract.tex`, all hunks) |
| T1 | yes (`08_appendix.tex:407`, `05_results.tex:72`, and the sweep row) |
| T2 | yes (`05_results.tex`, five hunk groups) |
| I1 | yes (`01_introduction.tex:4-11`) |
| I2 | yes (two hunks and the sweep row) |
| I3 | yes (`01_introduction.tex:26-28`) |
| I4 | yes (`01_introduction.tex:38-42`) |
| M1 | yes (`04_methodology.tex:53`) |
| M2 | yes (`02_background.tex:60,65`; `04_methodology.tex:53,73`) |
| M3 | yes (`04_methodology.tex:107-109`) |
| F1 | yes (`analyze_mh.py:87,92-124,219-234`) |
| F2 | yes (`plot_diagnostics.py:84-95,103-118,182-201`) |
| G1 | yes (all 12 equations; see the equation audit in `SUPERVISOR_FIXES.md`) |
| G2 | yes (both plotting scripts; `08_appendix.tex:183-218`) |
| G3 | yes (`abstract.tex` rewrite; `01_introduction.tex:51,53`) |

**No remark is not-actioned.** Two remarks are partially actioned and are flagged rather
than claimed complete: G2 for Figures 14 and 15 (the checkpoint needed to regenerate them
is not on disk), and A4's "noticeably under one page" (the abstract is one full page, the
space having gone to A2 and A3, which are requests in the same annotation block). Both are
put to the author in `SUPERVISOR_FIXES.md`.

### No hunk touches a numeral, statistic, confidence interval or result claim

Verified two ways.

1. **Source inspection.** No hunk edits a table body, a confidence interval, or a sentence
   that states a result. The `alpha`-to-`epsilon` rename in equation (7) changes a symbol's
   name, not the quantity, its role, or any value computed from it, and it removes a
   collision with equation (6) while aligning the background with Sections 4.3 and 5.3.2,
   which already used `epsilon` for it.
2. **Rendered-text diff.** Every decimal and every run of three or more digits was
   extracted from both PDFs and the multisets compared: 600 distinct before, 588 after.
   Every one of the 31 removals and 14 additions is accounted for in
   `SUPERVISOR_FIXES.md` under "Numbers diff, in detail" as either a deliberate abstract
   cut (with the value verified still present in the body), a regenerated figure's axis
   tick label, a section or page number, or one of the two citation years that remark I1
   required. **No number's value changed anywhere in the document.**

### UNMAPPED CHANGES

One change in the working tree does not belong to this pass and the author must see it.

| path | what | explanation |
|---|---|---|
| `refs/evaluation2.txt` deleted, `refs/evaluation2.md` added | A file rename, `.txt` to `.md` | **Not made by this pass.** The repository was clean when the pass opened and no command run here touched `refs/`. The new file is timestamped 21:04, before this session's first edit. The contents are byte-identical to the snapshot's `evaluation2.txt`, verified by diff, so nothing was lost; it is a pure rename. Reported, not reverted, since it appears to be the author's own change and reverting it would undo their work. |

No other unmapped hunk exists. Every remaining modified path is either a source file listed
above, a figure regenerated by a listed script change, or a `latexmk` build product.

---

## Snapshot disposal

No snapshot copy was made: the repository was clean, so the snapshot is the commit
`ec346e1496f7a7f9c77ac035442c6004ac19b97f` and nothing needs deleting. `Doc/final/.thesis_snapshot_phase12/`
was never created. The diff in this log was taken against that commit.
