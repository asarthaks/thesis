# PRESENTATION_LOG.md

Timestamped, append-only log of the defense-talk (beamer) build, Part 6 of
PROMPT_PHASE9_FINAL_DOCS. Design decisions, the slide-by-slide minute budget,
and the four rendered proof slides are recorded here. Thesis and proposal work
is logged in REVISION_WRITING.md.

Standing invariants: Ioanna register (short noun phrases, no bullet walls, no
AI-flavored phrasing); one number per claim; no em-dashes; numbers agree with
numbers.json and the regenerated figures.

The file ends with a "WHAT CHANGED" summary.

---

## 2026-07-26 00:28 CEST  Not yet started

Beamer build begins after Part 1 (numeric alarms) is confirmed and the thesis
prose passes are underway, per the Part 6 sequencing. Template located at
`Doc/final/beamer/Presentation.tex` (plus `bg.png`); `refs/08_ims-theses_handout.pdf`
holds the department talk expectations and will be read before slide design.

---

## 2026-07-26 ~15:30 CEST  PART 6: beamer defense talk BUILT

Reference deck evaluated first (Doc/final/beamer/Final_presentation_SU.pptx, Sergei Ukhov's
defense, same examiners Vu and Schweitzer, 21 slides). Adopted from it: the three-act
structure with section dividers (Motivation / Method / Results / Conclusions), the rhythm of
stating the claim or question before showing each result, a concrete worked example placed
early, and crisp closing takeaways. Avoided: its text-dense slides (full-sentence hypothesis
walls); the Ioanna register here is short noun phrases, one number per claim, no bullet walls.

Template: built on the located Doc/final/beamer template (Madrid/whale beamer), retitled for
the IMS with a deep-blue structure colour, the institute and both examiners on the title
page, and the UNIBG logo removed. inputenc switched latin1 -> utf8; bm added.

Story arc (follows Part 6 exactly), 14 content slides + 6 backups:
 1 title; 2 the revision problem AR cannot solve; 3 the energy-based promise and its one
 untested assumption (TikZ energy-landscape cartoon); 4 the four RQs; 5 the faithful
 implementation, DLS/CLS, MH, oracle calibration (TikZ sampler loop); 6 the null, gradient vs
 norm-matched random (paired CI + the four-energy table); 7 why 1, linearization failure with
 the jump-size picture (TikZ Voronoi cells with the valid radius vs the smallest move to
 scale, plus fig_lin_radius); 8 why 2, the MH breakdown, target +4.60 vs proposal -1325
 (fig_mh_decomposition); 9 the energy still works, rescore 4.43 and Gibbs 6.69; 10 GFlowNet,
 three failure modes compressed; 11 the last-token exact result, the zero-gradient theorem in
 one line plus the position table; 12 the diffusion positive control and the hybrid 0->39
 percent (TikZ hybrid architecture, SEDD proposes / AR energy accepts); 13 takeaways
 (evaluate, do not differentiate; training objectives determine landscapes); 14 thank you.
 Backups (mined from thesis_questions_knowledge_base.md, one per confusion hotspot):
 linearization radius and self-term blindness; the MH chain; likelihood-bad-yet-everywhere;
 MuCoLa/COLD and why they appear to work; task design and corpus; why a GFlowNet and the
 task-comparability caveat.

TikZ diagrams built (large fonts, controlled): energy-landscape cartoon (jagged 1D energy,
"gradient points where?"); Voronoi/token-spacing picture with the valid radius and the
smallest admissible move to scale; the sampler loop (state -> gradient -> MH -> project);
the hybrid architecture (SEDD score -> proposal -> MH accept by AR energy -> recovered token).
Result figures embedded as the thesis vector PDFs (fig_lin_radius, fig_mh_decomposition,
fig_lin_scatter), legible at slide scale; conceptual figures are the purpose-built TikZ.

Minute budget (target 20 min, IMS allows 25-30): title 0.5; motivation 4.5 (1.5 + 2 + 1);
method 2; null 2; why 5 (2.5 + 2.5); energy-works 1.5; amortization + last-token + positive
control 5 (1.5 + 1.5 + 2); takeaways 1. Total ~21.5 min, leaving buffer inside 25 min.

Compile: latexmk exit 0, 20 pages, 0 missing figures, no overfull above 40pt. Four proof
slides rendered to Doc/final/proofs/: beamer_energy_promise.png (slide 3),
beamer_linearization.png (slide 7), beamer_positive_control.png (slide 12),
beamer_takeaways.png (slide 13). Two minor right-margin TikZ text overflows found on first
render (slide 7 caption, slide 12 annotation) and fixed.

### WHAT CHANGED (beamer)
A complete 20-minute defense deck was created at Doc/final/beamer/Presentation.tex, replacing
the placeholder template content. It follows the diagnostic story arc from the revision
problem through the null, the two mechanisms, the gradient-free baselines, the GFlowNet and
last-token results, to the diffusion positive control and the takeaway, with four original
TikZ diagrams and six knowledge-base-derived backup slides. It compiles clean and is rendered
to four proof slides. No thesis number was changed by this part.
