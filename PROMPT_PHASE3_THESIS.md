# Prompt for Claude Code, Phase 3: apply the writing guide to the thesis

Run this AFTER Phase 2 (or in parallel if the last-token experiment is still
queued, leaving section 20 of the guide for last). Set the model to Opus 4.8 with
the highest reasoning setting (`/model`, Opus 4.8, high or extended thinking).
Then paste the block below.

---

Read CLAUDE.md, THESIS_WRITING_GUIDE.md, and the final report at the bottom of
REVISION_LOG.md before touching anything. Your task is to apply every item in
THESIS_WRITING_GUIDE.md to the LaTeX sources in Doc/. The guide is the
authoritative list; REVISION_LOG.md and results_revision/numbers.json are the
authoritative numbers; Doc/ is the authoritative current text; Ioanna's thesis
(IoannaThesis/Ioannathesis.tex) is the authoritative template. Never write a
number from memory: pull it from numbers.json or the results_revision JSONs and
put the source path in a % comment next to it, following the existing convention.

Working method:

1. INVENTORY FIRST. Before editing, produce a checklist in REVISION_LOG.md
   mapping every guide item (A1 to A20, all of B, C, D, E, F) to the exact file
   and line range in Doc/ where it applies, or "new section" with the insertion
   point. Quote the current wording for each edit target. Do the same for the
   template diff: list every preamble difference between Doc/ and Ioanna's
   thesis that affects chapter heading style, spacing, or the bibliography, with
   the line in her preamble you will copy.

2. EDIT IN SMALL COMMITS. Apply the guide item by item. After each item, compile
   the thesis (latexmk or the repo's build command) and confirm zero new errors
   and that the page count moved as expected. Never batch ten edits between
   compiles. Keep a running log entry per item: what changed, where, and the
   compile status.

3. TEMPLATE PASS. Apply the C items (chapter heading format, vertical spacing,
   statement of authorship, bibliography URL overflow) by copying Ioanna's
   preamble solutions, then recompile and visually check the first page of two
   chapters and bibliography page 2 (render those pages to PNG and inspect them).

4. STRUCTURE PASS. Add the list of tables and list of figures if missing, build
   the RQ-to-subsection map at the top of the Discussion, verify every RQ is
   explicitly answered with its subsection named, and verify every appendix item
   is commented on in a results subsection. Add the related-work-to-results links
   in the Discussion per guide item B.
5. METRIC AND METHOD JUSTIFICATION AUDIT. For every metric and every method in
   the Methodology, confirm a "why this was chosen" sentence with references
   exists; add the missing ones. List in the log which were missing.

6. THE WHY SPINE. After all local edits, read Results and Discussion end to end
   once and tighten them around the narrative spine in guide item F: energy fine,
   gradient defective, causally zero at the last position, diffusion the
   counterfactual. The supervisor's complaint was that the why was unclear; the
   test for this pass is that a reader of the Discussion alone can state the why
   in one sentence. If Phase 2's last-token results are in
   results_revision, write guide item A20 (the new section plus closing figure
   spec); if not, leave a clearly marked TODO block and tell the user.

7. APPENDICES. Write the AI tools appendix per guide item E, adapting Ioanna's
   structure, describing the actual uses accurately including this editing
   assistance. Write the config-count appendix (A15) and the cost table (A19).
   Add the embedding-trajectory section with plot placeholders and regeneration
   instructions per guide item D.

8. FINAL GATE. Recompile clean, then run the numbers check: extract every
   numeric claim you touched and diff it against numbers.json; report any
   mismatch instead of silently fixing it. Produce a final summary in
   REVISION_LOG.md: every guide item with done or blocked status, the page-count
   change, and a short list of decisions the author must confirm (the concern 6a
   attribution question, the SEDD section variant used, any wording where you
   chose between the guide and the existing text).

Constraints: no em-dashes anywhere. Write in the established voice of the
existing chapters and match Ioanna's register; long motivated paragraphs, varied
rhythm, no formulaic scaffolding. Do not delete the author's content, only
revise and extend; if something must go (the concern 18 compressions), keep the
removed text in a % comment block until the author confirms. Do not touch the
statement of authorship text itself beyond swapping in the updated file the
author provided. If any instruction here conflicts with what you find in Doc/ or
with the university template, stop and ask instead of guessing.


Log to REVISION_LOG_THESIS.md instead of REVISION_LOG.md, since another session is writing there. Same format. Do not edit anything outside Doc/ plus your log file.