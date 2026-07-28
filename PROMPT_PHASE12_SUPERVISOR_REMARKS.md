# Prompt for Claude Code, Phase 12: supervisor remarks, applied strictly and systematically

Opus 4.8, highest reasoning. Paste the block below.

---

Read REVISION_RESTRUCTURING.md, REVISION_WRITING.md and the thesis in
Doc/final/thesis/. This pass implements the supervisor's returned remarks.
Log to SUPERVISOR_FIXES.md (create it), ending with a resolution table:
every remark, what was done, where, done or author-decision. No em-dashes,
Ioanna register, template untouched, removed text to % comments.

## Hard limits (the author is explicit: grading outranks completeness)

- TOTAL PAGES 128 MAXIMUM. The document is at 128 now, so this pass is
  NET-ZERO OR SHORTER. Every addition must be paid for by a cut.
- ABSTRACT: one page maximum, and shorter is better here.
- If something must be explained and there is no room, CUT something less
  important rather than dropping the explanation. Ranked cut order, derived
  from the supervisor's own account of how theses are read (first sections
  closely, figures, then a few formulas at random): appendix detail first,
  then late-results elaboration, then middle-results elaboration. Do NOT cut
  from the abstract, introduction, background notation, or figure captions to
  buy space; those are the graded surfaces.
- Maintain a running PAGE LEDGER in SUPERVISOR_FIXES.md: every addition with
  its measured page cost, every cut with its saving, running net. Recompile
  and re-measure after each part, not once at the end.

## The remarks (verbatim, with the reading to apply)

ABSTRACT
 A1. "one example as footnote would make it more clear: e.g. length
     constraint" -> add ONE short footnote giving a concrete constraint
     example.
 A2. "more precise: 5 energy functions given by 5 different LLMs?" -> the
     abstract is imprecise about what the five energy functions are. State it
     exactly: the five are GPT-2 Large SFT, the three GFlowNet-tuned variants
     of it, and Llama-3 8B. Verify this against the thesis before writing,
     and if the count or composition differs, use what the thesis actually
     ran and log the discrepancy.
 A3. "Number them in text" -> number the claims/items in the abstract so they
     can be referred to. Determine from the annotation's position what is to
     be numbered (most likely the distinct claims or the energy functions); if
     genuinely ambiguous, number the claims and note the reading in the log.
 A4. "If you spend time to polish, then here. It is hard to understand
     without reading introduction. My advice: less, but clear is more", plus
     his summary note "Abstract requires some work to be understandable.
     Better leave stuff [out]. Intro can contain it."
     -> THIS IS THE HIGHEST-PRIORITY ITEM IN THE PASS. Rewrite the abstract to
     be self-contained for a reader who has not read the introduction: no
     forward references, no term used before it is glossed in place, no
     compressed chains of reasoning. Cut content rather than compress it;
     anything cut must already exist in the introduction, and if it does not,
     put it there (paying for it from the ranked cut list). Prefer plain
     sentences over dense ones. Target noticeably under one page.

TABLE OF CONTENTS AND STRUCTURE
 T1. "Make sure capitalization is fine in the table of contents and in the
     whole of the thesis" -> pick ONE heading capitalization convention,
     matching the template's own usage, and apply it to every chapter,
     section and subsection title, plus table and figure captions. Log the
     convention chosen.
 T2. "sections with just 1 subsection are strange. Can't you just collapse
     the subsection?" -> find every section containing exactly one
     subsection and collapse it into its parent, promoting the content and
     removing the redundant heading. Update all \ref targets. This also
     saves pages; record the saving in the ledger.

INTRODUCTION AND EARLY CHAPTERS
 I1. On why the models work so well: "They do. It is one of... massive data
     and architecture are other important ingredients" -> the current text
     attributes model capability too narrowly. Broaden it to name scale of
     data and architecture alongside whatever is currently credited, in one
     clause, with a citation.
 I2. "be consistent with terms: LLM, assistant, model... all the same, pick
     one" -> choose ONE term and apply it document-wide, defining it at first
     use. Grep for every alternative and replace. Log the term chosen and the
     replacement count.
 I3. "Inference-time control, i.e." -> the term is used without being
     unpacked. Define it in place at first use, briefly and concretely.
 I4. Margin notes in 1.2: "which can represent energy" and "low probability
     high probability" -> the relation between probability and energy is
     asserted before it is made intuitive. Add one short sentence making the
     direction explicit (low energy corresponds to high probability and vice
     versa) at the first point where the energy formulation appears.

METHODOLOGY, NOTATION (the systematic item; see the global pass below)
 M1. 4.3: "explain the parts: s_t, interim, ... are the" -> next to the
     continuous sampler's update equation, name and explain every component
     of the update: the current state, the interim continuous point, the
     projection, and the interpolated mean, one clause each.
 M2. 4.3/4.4: "Shapes? what is v?" -> give the shape of every vector and
     matrix at its first appearance, and define v explicitly (a candidate
     vocabulary item) where it is introduced.
 M3. 4.7: "spend more explanation on formula, what is g^T e?" -> explain the
     inner product between the gradient and the embedding displacement in
     words before or immediately after the formula: what each factor is, what
     the product measures, and why it is the object being tested.

FIGURES
 F1. Figure 2, on the proposal-term chart's x-axis: "overlapping numbers" ->
     fix the tick labels so they do not collide (fewer ticks, rotation, or
     larger figure width).
 F2. Section 5.3.4 / Figure 4: "plot on next page is huge and hard to render
     for some viewers" -> this is a file-weight problem: a dense vector plot
     with very many points. Rasterize the data layer at 300 to 400 DPI while
     keeping text and axes as vector, or reduce plotted point count by
     sensible subsampling that does not change the visual claim. Verify the
     resulting PDF page opens and scrolls quickly and report the page's size
     before and after.

## Global passes (apply to the WHOLE thesis, not only the annotated first half)

The supervisor read only the first half. The same defects exist later and he
will find them next time, so generalize every systematic remark now.

 G1. NOTATION PASS (his most repeated point: "formulas are sometimes hard to
     understand without background because you do not explain the notation,
     like shapes of vectors"). For EVERY numbered equation in the thesis:
     confirm that each symbol is defined at or before first use; that every
     vector, matrix and tensor carries its shape; that indices and their
     ranges are stated; and that at least one sentence in plain language says
     what the equation does before or after it. Produce an equation-by-
     equation audit table in the log (equation number, undefined symbols
     found, shapes missing, fixed yes/no). Where a symbol recurs across
     chapters, define it once at first use and stay consistent. Consider a
     compact notation table in the front matter ONLY if the page ledger
     allows it; if it does not, skip it and say so.
 G2. FIGURE QUALITY PASS (his checklist: "are figures explained and of good
     quality, label size, readability"). For EVERY figure: axis label and
     tick font sizes legible at print size (nothing below roughly 8pt
     effective), no overlapping or colliding labels, legends that do not
     obscure data, colour choices distinguishable when overlapping and in
     greyscale where feasible, and a sentence in the running text that walks
     the reader through what the figure shows. Produce a figure-by-figure
     audit table (figure number, defects found, fixed yes/no, regenerated
     yes/no). Regenerate from the plotting scripts; never edit images by hand.
     Captions stay short per the earlier passes; the walkthrough lives in the
     text.
 G3. FIRST-SECTIONS POLISH. He states plainly that the opening sections are
     read most closely. After the fixes above, re-read the abstract,
     introduction, and background end to end for clarity, unglossed jargon,
     and sentences that only parse if you already know the result. This is
     the one place to spend surplus effort.

## Order of work

1. T2 collapses and the ranked cuts FIRST, to bank pages before spending them.
2. Abstract rewrite (A1 to A4).
3. Introduction items (I1 to I4).
4. Notation: M1 to M3, then the G1 global pass.
5. Figures: F1, F2, then the G2 global pass.
6. T1 capitalization sweep and I2 terminology sweep last, so they catch text
   written earlier in this session too.
7. G3 final read of the opening sections.

## The change log (derived from a diff, never from memory)

Before editing anything, SNAPSHOT the current state: if the repo is under git
and clean, record the commit hash; otherwise copy Doc/final/thesis/ to
Doc/final/.thesis_snapshot_phase12/ (a working copy for diffing only, deleted
at the end after the log is written). Record the compiled PDF's page count and
file size in bytes at the same moment.

At the end of the pass, produce CHANGE_LOG_PHASE12.md GENERATED FROM THE DIFF,
not from recollection. For every changed file, in file order:

- the file path;
- one row per diff hunk: the line range IN THE NEW FILE, the section or
  environment it falls in (chapter/section number and title, or figure/table
  label), a one-line description of the change, and the remark ID it resolves
  (A1 to A4, T1, T2, I1 to I4, M1 to M3, F1, F2, G1, G2, G3);
- for pure sweeps (T1 capitalization, I2 terminology), do not list every
  occurrence: give the convention applied, the count of replacements, and the
  files touched.

Then a VERIFICATION section, and this is the point of doing it this way:
- Every hunk in the diff must map to a remark ID. Any hunk that does not is
  listed under UNMAPPED CHANGES with its content and an explanation. An
  unmapped hunk is either an accident or scope creep; either way the author
  must see it.
- Every remark ID must appear at least once in the log, or be listed as
  not-actioned with a reason.
- Confirm no hunk touches a numeral, a statistic, a confidence interval, or a
  result claim. Any such hunk is an error in this pass: revert it and report.

Keep the log tight: one row per hunk, no narrative, no restating the remark
text. It is a ledger, not an essay.

## Size verification (hard, measured, reported)

Report a small table with before and after for: total page count, per-chapter
page counts, compiled PDF file size in bytes, and the byte size of the
Figure 4 page's contribution if measurable. Requirements:
- total pages AT OR BELOW 128, and not one page more than the snapshot;
- the PDF file size should be equal or SMALLER, since the F2 fix rasterizes a
  dense vector plot; if it grew, find out why and report it;
- if the page count would exceed 128 at any point, stop adding and take the
  next cut from the ranked list before continuing, logging the trade.

## Gate and report

latexmk exit 0; total pages at or below 128, stated; abstract rendered to PNG
proving one page and attached in the log; zero undefined references or
citations; lists of tables and figures regenerated after the T2 collapses;
numbers diff unchanged (this pass must not touch any number; if an equation
fix reveals a genuine error, STOP and report rather than silently correcting).
Render to PNG and inspect: the abstract page, the table of contents, Figure 2,
Figure 4 and its neighbouring page, and two other figures chosen at random by
seed. Report: the resolution table for every remark above, the equation audit
table, the figure audit table, the page ledger with final net, the size
verification table, CHANGE_LOG_PHASE12.md complete with zero unexplained
unmapped hunks, and a short list of anything you judged out of scope with the
reason. Delete the snapshot copy only after the change log is written and
verified.

Constraints: net-zero pages or shorter; no experiments; no number changes; no
template changes; if a required explanation cannot be paid for from the ranked
cut list without losing something the earlier evaluations praised, keep the
explanation, make the cut anyway, and flag the trade in the report for the
author to confirm.
