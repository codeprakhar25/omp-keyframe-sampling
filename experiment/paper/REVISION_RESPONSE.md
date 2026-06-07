# Revision Response (round 1) — applied to main.tex

Mapping each roadmap item → action. Format: **[item]** Status — what changed.

## P0 — blocking
- **1. paper1/paper2 placeholder cites** — ⏳ DEFERRED (needs author input). Still flagged `[PLACEHOLDER]` in `references.bib`. The only unresolved P0.
- **2. "trends down" overclaim** — ✅ DONE. Abstract reworded ("in the single task with cross-agent range, more context did not improve correctness"). §5.2 (probe) demoted to a hedged single-task observation ("we do not claim a general downward effect — single task, n=3").
- **3. practitioner overgeneralization** — ✅ DONE. §5.3 scoped to "3 Python repos, naturalistic style-guide context, injection channel not content"; "may pay off less" softened; added the actionable process-efficiency nuance.

## P1 — major
- **4. TOST vs MDE reconciliation** — ✅ DONE. New paragraph in §4.2 explains realized-effect equivalence vs hypothesized-effect power; reports widest Claude bootstrap CI $[-4.4,+8.9]$pp numerically; flags n=15 bootstrap caution + permutation cross-check.
- **5. opshin effect exploratory + tested** — ✅ DONE. Labeled exploratory (post-hoc, excluded from Holm family); added within-task count test (Δ −1.65 runs/cell, 3/4 tasks, sign-flip p=0.25, n=4).
- **6. quantify agent-specific difficulty** — ✅ DONE. §4.5 now reports Spearman ρ=0.75 (p=0.001, n=15), 6/15 borderline-for-exactly-one, 6/15 floor/ceiling mismatch; adds the "screen per agent" methodological consequence.
- **7. between-agent confound** — ✅ DONE. §4.3 cross-agent paragraph restricted to qualitative; channel + tokenizer + cache-accounting confounds flagged; within-agent contrast noted as unaffected.

## P2 — minor (deferred to next pass)
- 8. related-work expansion (repo-context selection / RAG-codegen / SWE-bench Verified) — ⏳ not yet.
- 9. elevate process-not-outcome takeaway — ✅ partially (added to §5.3).
- 10. model-version snapshot caveat / "291 cells" softening — ⏳ not yet (Limitations already notes model versions).
- 11. per-task dot-plot + GLMM robustness — ⏳ optional, not yet.

## Integrity after revision
- All `\ref` resolve; `\cite` keys == bib keys (8/8); no stray "placeholder" in main.tex.
- All new numbers traced in `data/key_numbers.md` §5.

## Remaining before submission
1. Fill paper1/paper2 real citations (author).
2. Optional P2 polish (related work, dot-plot, GLMM).
3. Compile on Overleaf w/ ACL style; author block.
