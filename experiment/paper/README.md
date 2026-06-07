# Paper — "Do Context Files Help Coding Agents?"

REALM @ EMNLP 2026 (workshop), ACL format. First complete draft.

## Files
- `main.tex` — full paper (Abstract → Conclusion + mandatory statements + appendix).
- `references.bib` — bibliography.
- `data/key_numbers.md` — auto-extracted ground-truth numbers (provenance for every
  quantitative claim; regenerate from the dbs).
- `figures/` — (empty) for generated figure PDFs if added later.

## Build
Needs the ACL style files (`acl.sty`, `acl_natbib.bst`) from the ACL paper template
(https://github.com/acl-org/acl-style-files). Easiest path: upload `main.tex` +
`references.bib` to Overleaf with the "ACL" template, or locally:
```
# place acl.sty + acl_natbib.bst alongside main.tex, then:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
No local TeX here, so the PDF is not built in-repo.

## Content map (data-grounded)
- Correctness null both agents — Table 1 (Claude 53.3/55.6/55.6%, Codex 58.8/56.9/52.9%).
- TOST equivalence — Table 2 (Claude ≤10pp, Codex ≤15pp).
- Efficiency — cache signal (Claude selective) + NEW Claude×opshin process effect, Table 4
  (full-suite runs 3.67→2.44→1.67, duration −24%).
- Power/MDE — §Results (MDE >30pp; ~120 tasks for 10pp).
- Agent-specific borderline — Table 3 (3790/3769, verified vs db).
- Failure triage + NEW manipulation-validity probe — Table 5 (no flip; Claude 907 trends down).

## OPEN ITEMS before submission
1. **`paper1` / `paper2` real citations** — currently flagged PLACEHOLDERs in `references.bib`.
   These are the two source studies the paper reconciles (the "paper-1"/"paper-2" works).
   Must be replaced with real references before submission (IRON RULE: no placeholder cites).
2. **Author block / CRediT** — currently `Anonymous`; fill for camera-ready (non-anon venue).
3. **Verify the 4 verified cites' exact venues** (swebench/sweagent/memgpt/voyager/tost/
   empiricalse) against the published versions; arXiv ids noted in `.bib`.
4. **Optional figures** — Fig 1 is a self-contained tabular schematic; could upgrade to a
   proper diagram. Per-task dot-plot (overlap visualization) and power curve are nice-to-haves.
5. **Bilingual abstract / peer-review pass / format polish** — remaining academic-paper
   pipeline phases not yet run on this draft.
