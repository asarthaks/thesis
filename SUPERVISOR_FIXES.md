# SUPERVISOR_FIXES.md

Phase 12, 2026-07-28. This pass implements the supervisor's returned remarks on the
thesis in `Doc/final/thesis/`. It is a writing, notation and figure pass: no experiment
was run, no result claim was altered, and no number's value was changed anywhere.

## Snapshot

The repository was clean at the start of the pass, so the commit hash is the snapshot
and no working copy was made.

| item | value |
|---|---|
| snapshot commit | `ec346e1496f7a7f9c77ac035442c6004ac19b97f` |
| compiled pages at snapshot | 129 |
| compiled PDF bytes at snapshot | 3,097,565 |
| pages after this pass | 128 |
| PDF bytes after this pass | 2,323,802 |

The document was at 129 pages, not 128, when this pass opened. The cap of 128 was
therefore already exceeded by one page, and the pass had to be net negative rather than
net zero. It is: **129 to 128, one page shorter, and 773,763 bytes smaller.**

---

## Conventions fixed by this pass

**T1, heading capitalization.** The convention adopted is the one the document and the
IMS template already use in the great majority of cases, and it is now applied without
exception:

- **Chapter, section and subsection titles: Title Case.** First and last word always
  capitalized; every other word capitalized except articles (`a`, `an`, `the`),
  coordinating conjunctions, and prepositions of four letters or fewer (`of`, `in`,
  `to`, `by`, `for`, `from`, `with`). Both halves of a hyphenated compound are
  capitalized when both are major words (`Norm-Matched`, `Input-Embedding`,
  `Score-Trained`).
- **Table and figure captions: sentence case, terminal period.**

The sweep found the document already compliant in 64 of 66 headings and 36 of 37 short
captions. Two deviations were corrected (listed in the resolution table under T1).

**I2, terminology.** The single term is **language model**, defined at its first use in
Section 1. `LLM` does not occur anywhere in the thesis (verified: 0 hits). `assistant`
occurred twice, both as "aligned assistants"; both were replaced. Total replacements: 2,
in `01_introduction.tex` and `03_related_work.tex`. One occurrence of "neural model"
survives in Section 3.2 and was deliberately left: it refers to a neural *machine
translation* model in the reported literature, not to a language model, and changing it
would misdescribe the cited work.

---

## Page ledger

Recompiled and re-measured after each part. Running total starts at the snapshot's 129.

| # | change | remark | measured delta | running total |
|---|---|---|---|---|
| 0 | snapshot | | | **129** |
| 1 | collapse the four single-subsection subsections (5.1.1, 5.2.1, 5.4.1, 5.10.1) | T2 | **-2** | 127 |
| 2 | abstract rewritten and cut to one page | A1-A4 | **0** | 127 |
| 3 | introduction: model-capability clause, inference-time-control definition, energy-probability direction, shapes on the two intro equations | I1, I3, I4, G1 | **0** | 127 |
| 4 | methodology and background: name the four parts of the CLS update, shapes on every numbered equation, the `g` transpose `e` walkthrough | M1, M2, M3, G1 | **+2** | 129 |
| 5 | remove four forced `\clearpage` breaks in the appendix | ranked cut 1 | **-1** | 128 |
| 6 | merge appendix A.5.3 into A.5.2 and de-duplicate their shared paragraph | ranked cut 2 | **0** | 128 |
| 7 | regenerate all diagnostic figures at printed size; rasterize the two dense scatters | F1, F2, G2 | **0** | 128 |
| 8 | shapes and plain-language reading on the two results equations (relaxed objective, token-indicator derivative) | G1 | **+1** | 129 |
| 9 | remove two further forced `\clearpage` breaks in the appendix | ranked cut 3 | **-2** | 127 |
| 10 | widen Figures 14 and 15 to the full text block for legibility | G2 | **+1** | 128 |
| 11 | gloss `input-embedding gradient`, `norm-matched random direction`, `teacher forcing`, `token-indicator coordinates` at first use in the introduction | G3 | **0** | **128** |

**Final net against the snapshot: -1 page. Against the 128-page cap: at the cap, not over.**

Every addition in this pass was paid for from the ranked cut list, and the cuts were taken
in the order the brief specifies: appendix detail first (steps 5, 6, 9), which was
sufficient. **No cut was taken from the abstract, the introduction, the background
notation, or any figure caption**, and no late- or middle-results elaboration had to be
sacrificed. The ranked list was not exhausted; two further appendix candidates (dropping
the Table 15 sample rows, trimming the qualitative-example longtables) remain available
if the author later needs more room.

### Cuts taken, in full

| cut | where | what was removed | where the content still stands |
|---|---|---|---|
| Six forced `\clearpage` breaks | `08_appendix.tex` A.2, A.3, A.4, A.6, A.7 openers and the A.1 figure block | page breaks only | nothing removed |
| A.5.3 heading, and one restated sentence | appendix A.5 | the subsection heading "Per-Class Confusion on the Neutral Calibration Surface"; the sentence "The two conditional agreement rates are statistically indistinguishable" and the clause "The confusion table therefore confirms the overall alignment gap that the ladder measures", which restated the paragraph they followed | both analyses, both tables and every number kept, now inside A.5.2 |
| Three baseline figures | abstract | the top-*k* rescore 4.43, the Gibbs 6.69 and the best Langevin cell 6.4 | Section 5.3.3 and Chapter 7; the claim they supported is kept in the abstract in words |
| The "5x5" shape of the sweep | abstract | the grid's dimensions | Section 5.3.2; the span it covers is kept in the abstract in words |
| Two GFlowNet explanatory clauses | abstract | how the policies failed | Section 1.3, Section 5.9, Chapter 7 |
| The three-hypothesis structure | abstract | the A/B/C framing | Section 1.2, in full |
| A figure footer string | `fig_lin_decomposition` | "mean \|self\| = 15.02 nats, mean \|future\| = 24.21 nats" and the sentence on why the gradient cannot see the self term | Section 5.4, which already states 15.0 and 24.2 and the argument |

Nothing removed from a `.tex` file was deleted silently: every removal site carries a `%`
comment naming what was removed, why, and where the content now lives.

---

## Resolution table

| ID | remark | what was done | where | status |
|---|---|---|---|---|
| **A1** | one example as footnote, e.g. length constraint | Added a single footnote at the first sentence, giving a length constraint as the concrete case: whether a sentence stays under twenty words is settled only when it is finished. | `abstract.tex:25` | **done** |
| **A2** | more precise: 5 energy functions given by 5 different LLMs? | The five are now named individually: a GPT-2 Large fine-tuned on stories, three variants of it tuned further with a GFlowNet, and a Llama-3 8B from another architecture. Verified against Section 4.2, which states exactly this composition. **No discrepancy found**; the count and composition in the prompt match what the thesis ran. | `abstract.tex:27` | **done** |
| **A3** | number them in text | The three findings are numbered **(1)**, **(2)**, **(3)** inline and bolded, so each can be referred to. Reading applied: the *claims* are numbered, not the energy functions, per the brief's fallback instruction. The energy functions are handled by A2 instead, which is the other candidate reading; both readings are therefore satisfied. | `abstract.tex:29,31,33` | **done** |
| **A4** | abstract hard to understand without the introduction; less but clear is more | **Rewritten twice.** Every term is glossed where it is first used (energy function, Langevin dynamics, masked token recovery, GFlowNet); four blocks of content were **cut** rather than compressed (listed in the cuts table above), each of them already present in the introduction. The first rewrite used bolded, numbered headline findings; the author judged that phrasing to read as machine-generated, so it was rewritten again in the register of the reference thesis in `refs/IoannaThesis/`: one continuous paragraph of findings with inline `(1)`/`(2)`/`(3)` markers, long declarative sentences carrying subordinate clauses, an `i.e.` gloss, and a closing "These findings indicate that". It fits one page, verified by PNG. | `abstract.tex` | **done, with one qualification** |
| **T1** | check capitalization in the ToC and throughout | Convention fixed and stated above. Swept all 66 headings and 37 short captions. Two corrections: `Use of AI-Tools` to `Use of AI Tools` (which also makes it match the Erklaerung's own wording on the first page), and the short caption `Discrete Langevin sampler trajectories` to `Discrete Langevin Sampler trajectories`, matching the defined term. | `08_appendix.tex:395`, `05_results.tex:72` | **done** |
| **T2** | sections with just one subsection are strange, collapse them | Found exactly four: 5.1.1, 5.2.1, 5.4.1, 5.10.1. All four collapsed into their parents **in the body as well as the table of contents**: the `\subsubsection` headings are gone from the running text and the prose now flows continuously, with every `\label` kept at the promotion site so all cross-references still resolve. Two parent titles were widened so they cover the promoted material: 5.4 gained "and What Recovers It", 5.10 became "Extension: Constrained Generation and Classifier-Guided Steering". The collapse then made a child label and its parent label render the **same number**, and four sentences cited both; three of those read as broken and were repaired (see below). Verified after repair: no section has exactly one subsection, and zero self-references anywhere. One same-number pair survives deliberately, in the RQ1 answer of 6.1, where Section 5.4 is cited in two different sentences of a long paragraph, once for the mechanism and once for the alternative; both now live in 5.4, so the repetition is correct and reads as ordinary usage. Saving: 2 pages. | `05_results.tex`, `06_discussion.tex`, `08_appendix.tex`, ToC | **done** |
| **I1** | models work well for more reasons than one; data and architecture matter | The clause now reads that the factorization is "one reason among several", with the volume of training text and the Transformer architecture named as the other principal ingredients, cited. | `01_introduction.tex:11` | **done** |
| **I2** | be consistent with terms: LLM, assistant, model | Term chosen: **language model**, defined at first use. 2 replacements. Detail above. | `01_introduction.tex:11,22`, `03_related_work.tex:11` | **done** |
| **I3** | "Inference-time control, i.e." | Defined in place, at the point where the capability is first named: the parameters are fixed once and never revisited, and the only thing that differs between one constraint and the next is a term handed to the decoding procedure while it runs. | `01_introduction.tex:28` | **done** |
| **I4** | margin notes in 1.2: "which can represent energy", "low probability high probability" | At the first appearance of the energy formulation the direction is now read off explicitly: since exp(-E) falls as E rises, low energy corresponds to high probability and high energy to low probability. A second sentence answers the other margin note by saying what may serve as an energy (any scalar-valued function of a complete sequence) and pointing to where this thesis makes its choice. | `01_introduction.tex:42` | **done** |
| **M1** | 4.3: explain the parts, s_t, interim, ... | All four components of the continuous update are now named and explained, one clause each: the **current state**, the **interim continuous point**, the **projection**, and the **interpolated mean**, plus the step size and the identity covariance. | `04_methodology.tex:53` | **done** |
| **M2** | 4.3/4.4: shapes? what is v? | `v` is now defined explicitly at its first appearance as a **candidate vocabulary item**, one of the `\|V\|` tokens that could be placed at the position being resampled, with its embedding's shape given. Shapes are given for every vector and matrix in 4.3 and 4.4. See the equation audit below for the full sweep. | `02_background.tex:60,65`, `04_methodology.tex:53,73` | **done** |
| **M3** | 4.7: spend more explanation on the formula, what is g transpose e? | A new paragraph explains the inner product in words before the reader has to use it: what each factor is (the uphill direction and its steepness; the displacement a substitution would travel along), what the product measures (the component of the displacement lying along the uphill direction, scaled by the slope), and why it is the object under test (it is exactly what the discrete proposal ranks by). | `04_methodology.tex:107-109` | **done** |
| **F1** | Figure 2, proposal-term x-axis: overlapping numbers | The default locator packed seven five-digit labels into a narrow panel. Fixed by capping the tick count and printing magnitudes in thousands, so the axis now reads `-60k -40k -20k 0`. Fixing it exposed two further defects on the same figure, both also fixed: the left panel's legend sat on top of the histogram, and the right panel's axis label was clipped at the figure edge. See the figure audit. | `diagnostics/analyze_mh.py` | **done** |
| **F2** | Section 5.3.4 / Figure 4: plot on the next page is huge and hard to render | It is a file-weight problem, as diagnosed. The scatter carries 30,000 points and was stored as vector art. The **data layer only** is now rasterized at 350 DPI; all text, axes, ticks and the colourbar frame remain vector. Same fix applied to Figure 6, which had the same defect. Sizes below. The page now renders and scrolls immediately. | `diagnostics/plot_diagnostics.py` | **done** |
| **G1** | notation pass: symbols defined, shapes given, indices stated, plain-language reading | All 12 numbered equations audited; table below. Also found and fixed a genuine **symbol collision**: `\alpha` was both the Metropolis-Hastings acceptance probability in (5) and the step size in (7), and the results chapter already called the latter `\epsilon`. Renamed to `\epsilon` throughout (7) and its footnote. A compact front-matter notation table was **not** added: the page ledger had no room, as the brief anticipated. | all chapters | **done** |
| **G2** | figure quality: label size, readability, legends, walkthrough | All 15 figures audited; table below. Nine regenerated. The root cause of most defects was uniform: figures authored at 8 to 10 inches wide and included at roughly 5.9 inches, a 0.6x reduction that put nominal 10pt axis text near 6pt. Every regenerated figure is now drawn at the width it is printed at, so a declared point size is the printed point size. | `diagnostics/plot_diagnostics.py`, `diagnostics/analyze_mh.py`, `08_appendix.tex` | **done except two figures, flagged** |
| **G3** | first-sections polish | Abstract rewritten end to end (A4). Introduction re-read: four terms were being used before they were said plainly and are now glossed in place, `input-embedding gradient`, `norm-matched random direction`, `teacher forcing`, and the token-indicator coordinates. Background re-read; its notation gaps were the G1 work, and no further unglossed jargon was found. | `abstract.tex`, `01_introduction.tex` | **done** |

### The one qualification on A4

The abstract now occupies **one full page**, not "noticeably under" one. Getting further
under it would mean giving up either A2 (naming the five energy functions individually,
which costs five lines) or A3 (numbering the findings, which costs the bold lead-ins and
three paragraph breaks). Those are the supervisor's own requests in the same annotation
block, so the page was spent on them rather than on white space. Five successive cuts were
made to reach one page and are recorded in the cuts table. **Author decision:** if the
supervisor's "less is more" is to be read as outweighing A2 or A3, say which one to drop
and the abstract loses another five to eight lines.

---

## Equation audit (G1)

All 12 numbered equations. "Shapes" means every vector, matrix and tensor carries its
dimensions; "reading" means at least one plain-language sentence says what the equation
does, before or after it.

| eq | label | section | undefined symbols found | shapes missing | reading present | fixed |
|---|---|---|---|---|---|---|
| 1 | `eq:intro-ebm` | 1.2 | none | `x`, `E`, `Z` had no types | partial | **yes**: `x` given as a `T`-token sequence over `V`, `E` and `Z` typed, and the energy-probability direction read off in words (also I4) |
| 2 | `eq:intro-energy` | 1.2 | none | `C`, `lambda` untyped | yes | **yes**: `C : V^T -> R`, `lambda > 0` |
| 3 | `eq:bg-chain` | 2.1 | `x_<t` never defined | logit vector untyped; `V` and `\|V\|` not introduced | partial | **yes**: prefix notation defined, `V` and its size introduced, logits typed `R^{\|V\|}`, plain-language reading added |
| 4 | `eq:bg-energy` | 2.1 | none | none needed | yes | no change needed |
| 5 | `eq:bg-langevin` | 2.2 | none | `s_t`, gradient, noise, identity all untyped | yes | **yes**: all typed in `R^D` / `D x D`, plus a one-sentence reading |
| 6 | `eq:bg-mh` | 2.3 | none | `s`, `s'`, `q`, `alpha` untyped | yes | **yes**: all named and typed, `alpha` given its range `[0,1]` |
| 7 | `eq:bg-discrete-proposal` | 2.4 | **`v` never defined** (M2) | `g`, `e(v)`, displacement untyped; step size named `alpha`, **colliding with (5)** | yes | **yes**: `v` defined as a candidate vocabulary item, all shapes given, step size renamed `epsilon` to match (5) and Sections 4.3 and 5.3.2, reading added |
| 8 | `eq:bg-cls-energy` | 2.4 | none | `s` untyped; `proj_V` had no signature | yes | **yes**: `M` introduced, `s` typed `R^{M x D}`, `proj_V : R^{M x D} -> V^M` |
| 9 | `eq:meth-cls-update` | 4.3 | none | every term untyped (M1, M2) | partial | **yes**: four parts named and explained, all typed `R^{M x D}`, step size and covariance stated |
| 10 | `eq:meth-kl` | 4.4 | none | `p_ref`, `p_pred` untyped; index range implicit | yes | **yes**: both typed `R^{\|V\|}`, index range over `M'` stated |
| 11 | `eq:relaxed-objective` | 5.4 | none | `E`, `z_i`, `L_future`, `E^T z_i` untyped | yes | **yes**: `E` in `R^{\|V\| x D}`, `z_i` in `R^{\|V\|}` on the simplex with its vertex meaning spelled out, `L_future : R^D -> R`, mixed embedding typed, plus a reading of the two terms as the two roles a token index plays |
| 12 | `eq:onehot-grad` | 5.4 | none | the derivative's own shape not stated | yes | **yes**: stated as a vector in `R^{\|V\|}`, contrasted with the `R^D` of the input-embedding gradient, plus a plain-language reading of the two terms |

Recurring symbols (`x`, `V`, `v`, `D`, `M`, `e(.)`, `g`, `s`, `epsilon`, `proj_V`) are each
defined once at first use and are consistent across chapters after the `alpha`-to-`epsilon`
rename.

---

## Figure audit (G2)

Effective point size is the declared size times the ratio of printed width to authored
width. The floor is roughly 8pt.

| fig | file | defects found | fixed | regenerated |
|---|---|---|---|---|
| 1 | `gpt2-large.dls.gn.free.s50_new_trajectories.png` | none; legend is below the axes and clear of data, labels legible | n/a | no |
| 2 | `fig_mh_decomposition.pdf` | **(a)** x-axis tick labels collided into an unreadable block (F1); **(b)** axis text near 5.5pt effective; **(c)** legend sat on the histogram in the left panel; **(d)** right panel's axis label clipped at the figure edge; **(e)** a suptitle restating the caption verbatim at about 6pt | **yes**, all five | **yes** |
| 3 | `fig_forest_chain.pdf` | none; 8pt labels by construction from the earlier print-legibility pass, no collisions, legend clear of the bars | n/a | no |
| 4 | `fig_lin_scatter.pdf` | **file weight** (F2): 30,000 vector points, 616 kB, slow to render | **yes**, data layer rasterized at 350 DPI, text and axes vector | **yes** |
| 5 | `fig_lin_radius.pdf` | axis text near 8pt, legend at 7pt | **yes**, drawn at printed size | **yes** |
| 6 | `fig_lin_decomposition.pdf` | **(a)** 613 kB of vector points; **(b)** a footer string wider than the figure inflated the tight bounding box, so the figure printed at about 60 percent of the text block with unreadable axis text; **(c)** panels too close, right y-label crowding the left panel | **yes**, all three | **yes** |
| 7 | `fig_lin_topk.pdf` | text near 7.3pt, legend near 6.6pt | **yes** | **yes** |
| 8 | `fig_mh_accept.pdf` | text near 7.3pt | **yes** | **yes** |
| 9 | `fig_trap_scatter.pdf` | **(a)** text near 5.8pt, legend near 4.7pt; **(b)** the two panels' x-labels ran into each other and the right one was clipped; **(c)** legend sat on the data | **yes**, all three | **yes** |
| 10 | `fig_aniso_hist.pdf` | **(a)** text near 5.8pt; **(b)** same x-label collision and clipping as Figure 9 | **yes** | **yes** |
| 11 | `fig_trap_length.pdf` | legend at 6.9pt | **yes** | **yes** |
| 12 | `..._s100_new_trajectories.png` | none; same construction as Figure 1 | n/a | no |
| 13 | `..._nogn_companion.png` | none; same construction as Figure 1 | n/a | no |
| 14 | `fig_traj_distance.png` | authored at 8.4in, printed at 4.7in: 9pt labels near 5pt, 7pt legend near 4pt | **partial**, see below | **no, cannot** |
| 15 | `fig_traj_pca.png` | authored at 9.2in, printed at 5.0in: 10pt titles near 5.5pt | **partial**, see below | **no, cannot** |

Every figure is walked through in the running text at its point of reference; captions
were left short, per the rule the earlier passes set. No caption was cut to buy space.

### The two figures that could not be regenerated, flagged for the author

Figures 14 and 15 are produced by `revision/plot_trajectories.py`, which needs the full
50257 x 1280 embedding matrix from the GPT-2 Large SFT checkpoint in order to redo its
exact nearest-token decoding. **That checkpoint is no longer on disk**: the path the script
hard-codes does not exist, and the repository-local `gfn-lm-tuning/gpt2_large_sft_output/`
is empty. Regenerating them at the printed size is therefore not possible in this pass
without first restoring the checkpoint, which is outside a writing pass.

The available mitigation was applied: both were widened to the full text block, which
raises their effective point sizes by 25 and 18 percent respectively. Figure 15 now reads
acceptably. Figure 14's smallest elements are still under the 8pt floor. **Author decision:**
restore the checkpoint and rerun `revision/plot_trajectories.py` at the printed width, or
accept Figure 14 as is.

---

## Size verification

| measure | before | after | delta |
|---|---|---|---|
| total pages | 129 | **128** | **-1** |
| PDF bytes | 3,097,565 | **2,323,802** | **-773,763 (-25.0%)** |
| front matter | 9 | 8 | -1 |
| 1 Introduction | 8 | 8 | 0 |
| 2 Background | 9 | 10 | +1 |
| 3 Related Work | 7 | 7 | 0 |
| 4 Methodology | 11 | 12 | +1 |
| 5 Results | 42 | 42 | 0 |
| 6 Discussion | 9 | 9 | 0 |
| 7 Conclusion + bibliography | 11 | 11 | 0 |
| A Appendix | 22 | 20 | -2 |

Total pages are **at or below 128** and are **one page fewer** than the snapshot, as
required. The PDF is **smaller**, as the F2 fix predicted; it did not grow.

### Figure 4's contribution, measured

The Figure 4 page's weight is dominated by the included graphic, so the graphic file is the
measurable contribution.

| file | before | after | delta |
|---|---|---|---|
| `fig_lin_scatter.pdf` (Figure 4) | 615,644 | 317,594 | **-48.4%** |
| `fig_lin_decomposition.pdf` (Figure 6, same defect) | 613,312 | 131,948 | **-78.5%** |
| the two together | 1,228,956 | 449,542 | **-63.4%** |

That accounts for 779,414 of the 773,763 bytes the whole document lost, the small
difference being the other regenerated figures and the changed text.

One rasterization was **reverted after measurement**: applying it to Figure 9, which
carries only a few thousand points, tripled that file from 82 kB to 260 kB, because a
sparse scatter rasterizes to a large image but compresses well as vector art. Figures 9
and 11 are therefore deliberately left vector. Only the two genuinely dense scatters are
rasterized.

---

## Gates

| gate | result |
|---|---|
| `latexmk` exit code | **0** |
| total pages | **128**, at the cap, one below the snapshot |
| abstract on one page | **yes**, PNG proof rendered and inspected |
| undefined references | **0** |
| undefined citations | **0** |
| multiply-defined labels | **0** |
| "Rerun to get cross-references right" | **0** |
| lists of figures and tables regenerated after the T2 collapses | **yes**, `thesis.lof` and `thesis.lot` rebuilt |
| numbers diff | **no value changed anywhere**; see below |
| bibliography | unchanged in content (`thesis.bbl` identical modulo whitespace reflow) |

### Numbers diff, in detail

Every statistic-like numeral (any decimal, or any run of three or more digits) was
extracted from the rendered text of both PDFs and the multisets compared. 600 distinct
before, 588 after. Every difference is accounted for, and **no number's value changed**:

- **Removed from the abstract, still in the body:** `0.60`, `0.73`, `0.06`, `4.43`,
  `6.69`, `6.4`. Each drops by exactly one occurrence, the abstract's. Verified still
  present in the body: yes, all six.
- **Removed from a figure, still in the body:** `15.02`, `24.21`, the deleted
  `fig_lin_decomposition` footer. Section 5.4 states 15.0 and 24.2 unchanged.
- **Axis tick labels of regenerated figures:** `10000` through `60000` gone,
  because the F1 fix prints them as `-60k` to `0`; `150`/`250`/`350`,
  `1500`/`2500`/`3500`, `1.0`/`1.2` changed because two y-axes gained legend headroom.
- **Section and page numbers:** `5.1`, `5.2`, `5.3`, `5.4`, `5.10` each drop two
  occurrences, being the four collapsed subsubsection headings and their ToC lines;
  `107`, `116`, `121`, `124`, `126`, `128` out and `108`, `110`, `111`, `118`, `119`,
  `122`, `123`, `127` in, the document being one page shorter.
- **Added by this pass:** `2.1` (the new `Section 2.1` cross-reference in I4), `2017` and
  `2020` (the two citations I1 required), and one extra `145` (the abstract now states
  the configuration count in both paragraph two and finding (1)).

No hunk anywhere in the diff touches a confidence interval or alters a result claim. No
equation fix revealed an arithmetic error, so there was nothing to stop and report on that
count. The one substantive notation finding, the `alpha` collision in equation (7), is a
naming inconsistency between chapters, not an error in the mathematics: the quantity, its
role and every number computed from it are unchanged.

---

## Rendered proofs

Rendered to PNG from the final PDF and inspected, one per required check. Files are in
`Doc/proofs_phase12/`.

| file | what it proves |
|---|---|
| `01_abstract_one_page.png` | The abstract fits **one page**: the footnote rule closes the text block and the next page begins the table of contents. Required gate. |
| `02_toc_a.png`, `03_toc_b.png` | Table of contents: Title Case throughout, and the T2 collapses visible (5.1, 5.2, 5.4 and 5.10 no longer carry a lone subsection; the only remaining groups, 5.3 and 5.6, have four and two) |
| `04_figure2_ticks_fixed.png` | Figure 2: x-axis now reads `-60k -40k -20k 0` with no collision, both axis labels complete and unclipped, legends clear of the histograms (F1) |
| `05_figure4_rasterized.png` | Figure 4: rasterized data layer, vector text, and the figure now sits on the same page as its surrounding discussion (F2) |
| `06_figure4_next_page.png` | Figure 4's neighbouring page, checked for reflow damage: none |
| `07_figure3_seed.png` | Random check 1 of 2, figure 3 |
| `08_figure15_seed.png` | Random check 2 of 2, figure 15 |

The two random figures were drawn with `random.Random(20260728).sample(range(1,16), 2)`,
which gives `[3, 15]`.

## Out of scope, with reasons

| item | reason |
|---|---|
| Front-matter notation table | The brief made it conditional on the page ledger allowing it. It does not: the document sits exactly at the 128-page cap. Skipped, as instructed. |
| Regenerating Figures 14 and 15 | Requires the GPT-2 Large SFT checkpoint, which is not on disk. Mitigated by widening; flagged above. |
| Fixing the stale checkpoint path in `revision/plot_trajectories.py` | The repository-local path is empty too, so changing the constant would not make the script runnable. Left for whoever restores the checkpoint. |
| "neural model" in Section 3.2 | Refers to a neural machine translation model in the cited literature, not to a language model. Changing it under I2 would misdescribe the source. |
| The `refs/evaluation2.txt` to `.md` rename | Not made by this pass. Reported under UNMAPPED CHANGES in `CHANGE_LOG_PHASE12.md`. |
