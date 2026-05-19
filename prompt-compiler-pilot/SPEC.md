# Prompt Compiler Pilot — Design Spec

**Date:** 2026-05-19
**Type:** Directional pilot study, single-turn code generation
**Budget:** ~$200 (expected actual spend ~$20–30)

---

## 1. Motivation

Prior literature suggests middleware prompt rewriting improves LLM output quality, but the evidence base predates 2026 frontier models and does not isolate *structural* reformulation from *surface* cleanup. No published study directly measures structural reformulation of developer prompts on a frontier coding model. This pilot is a small, pre-registered first pass at that gap.

This is a *directional* pilot. Goal is to inform whether further investment (rigorous study, productized middleware) is worth pursuing. It is not powered for publication.

## 2. Hypothesis (pre-registered)

**H1 (primary):** For multi-requirement coding tasks, structurally reformulating a noisy/messy prompt before passing it to a frontier model improves pass@1 vs. passing the messy prompt directly.

**H2 (secondary):** Reformulation restores accuracy to within a small margin of the clean-prompt baseline.

**H3 (cost):** Net token cost of the reformulator + target call does not exceed the token cost of the messy-direct call by more than 30%, when measured per correct answer.

A pre-specified failure-mode check (see §8) measures cases where reformulation actively hurts.

## 3. Models

| Role | Model | Notes |
|---|---|---|
| Primary target | `claude-sonnet-4-6` | Frontier, cheap enough for full n |
| Cross-check target | GPT-5.4 (minimum version) | Run on 50% subset of tasks to test cross-family generalization |
| Reformulator | `claude-haiku-4-5-20251001` | Small, fast, same family as primary target |
| Messifier | `claude-haiku-4-5-20251001` | Separate prompt from reformulator |

Decoding: temperature=0.2, top_p=1.0, max_output_tokens=2048 for all roles. Versions pinned and recorded in `results/runs.jsonl`.

## 4. Dataset

- **BigCodeBench-Hard**, 30 tasks selected from the first 30 indices for which (a) the canonical solution exists, (b) the test suite runs cleanly in a sandbox, and (c) the original prompt is ≥80 words (to ensure there's something to mess up).
- Pinned task list committed to `data/tasks.jsonl` with content hashes before any messification runs.

## 5. Conditions

| Arm | Input pipeline |
|---|---|
| A — Clean | original prompt → target → grade |
| B — Messy raw | original → messifier → target → grade |
| C — Messy reformulated | original → messifier → reformulator → target → grade |

Same target, same decoding settings, same harness across all arms. Messifier output for a task is generated **once** and reused identically across B and C (i.e., B and C see the *same* messy text — the only difference is whether the reformulator runs). Reformulator output for a task is likewise generated **once** and reused across all 5 target seeds in arm C, so the reformulation step is fixed per task and the 5 seeds only vary target-model sampling.

## 6. Sample size and power

- 30 tasks × 3 conditions × 5 seeds (target sampling seeds) = **450 trials** on primary target
- 15 tasks × 3 conditions × 3 seeds = **135 trials** on cross-check target

At an assumed ~50% baseline pass rate, 30 tasks × 5 seeds per arm (paired by task+seed) gives ~80% power to detect a ~12pp McNemar-test effect at α=0.05. Adequate for directional, not publication-grade.

## 7. Messifier — definition and validation

Fixed prompt at `prompts/messifier.md`. The messifier must:

- Add a rambling preamble of unrelated context (60–120 words)
- Bury the original requirements inside the noise rather than presenting them cleanly
- Mix style preferences and constraints inline with the spec
- Add 2–4 mild typos
- **Preserve all original requirements semantically**

**Validity gate (must pass before main run):** the experimenter hand-checks 10 randomly sampled messified tasks. Pass criteria: ≥9/10 retain all requirements identifiably. If <9/10, the messifier prompt is revised and re-validated before proceeding. The validation samples and judgments are committed to `data/messifier_validation.md`.

This is the **single highest validity risk** in the design. Skipping it invalidates the entire pilot.

## 8. Reformulator — definition

Fixed prompt at `prompts/reformulator.md`. The reformulator must:

- Extract numbered requirements
- Separate constraints, edge cases, style preferences into labeled sections
- Drop irrelevant chatter
- **Not add information not present in the input**
- Output a fixed Markdown template

The reformulator prompt is frozen before the main run. Any change after first runs invalidates the pre-registration and requires re-running prior arms.

## 9. Metrics

**Primary:**
- pass@1 per arm (binary per trial → mean per arm)
- McNemar paired test on B vs C using (task, seed) pairs

**Secondary:**
- pass@5 across seeds per arm
- 95% bootstrap CI on (C − B) effect size

**Cost/value:**
- Total input + output tokens per arm
- Cost per correct answer ($/correct)

**Failure mode (pre-registered):**
- Count of trials where B passes and C fails on the same (task, seed) — this is reformulation-induced harm
- If this exceeds 15% of trials where B passes, the reformulator is causing meaningful damage even if average effect is positive

## 10. Analysis plan (pre-registered)

Frozen in `PREREG.md` before any target-model run:

1. Report mean pass@1 for A, B, C with 95% CIs
2. Primary test: McNemar on paired (B, C) trials; report p-value and effect size
3. Secondary: A vs C McNemar (does reformulation fully recover?)
4. Token-cost table per arm; $/correct calculation
5. Failure-mode count and rate
6. Cross-check target: same analysis on the GPT-5.4 subset

**No subgroup analyses, no post-hoc condition slicing, no dataset filtering after results are seen.** Any such exploration is labeled "exploratory" and not used for conclusions.

## 11. What this pilot will and will NOT tell you

**Will:**
- Whether structural reformulation moves pass@1 on synthetically-messified single-turn coding tasks for Sonnet 4.6 and (weakly, n=15) GPT-5.4
- Token-cost trade-off for the reformulation step
- Whether reformulation introduces meaningful failure modes

**Will not:**
- Whether the effect transfers to real Claude Code/Cursor multi-turn sessions
- Whether the effect holds on organic (non-synthetic) messy prompts
- Whether it generalizes across model families beyond two data points
- Anything publication-grade — n is too small

These are explicit v2 questions.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Messifier destroys ground truth | Mandatory hand-validation gate (§7); abort if <9/10 |
| Messifier too tame, no effect to detect | Validate "messy" feels genuinely hard on hand check |
| Reformulator hallucinates requirements | Failure-mode metric (§9) catches this; reformulator prompt forbids new info |
| Sonnet-only result doesn't generalize | Cross-check on GPT-5.4 subset |
| Stochastic noise swamps signal | 5 seeds; McNemar paired test |
| Budget overrun | Single-turn design caps cost; estimated ~$20 vs $200 ceiling |
| HARKing / post-hoc cherry-picking | Pre-registered PREREG.md frozen before runs |

## 13. Repo layout

```
claude-prompt-compiler-pilot/
  SPEC.md                          # this file
  PREREG.md                        # frozen analysis plan, written before runs
  README.md                        # how to reproduce
  prompts/
    messifier.md
    reformulator.md
  data/
    tasks.jsonl                    # 30 BigCodeBench-Hard items, content-hashed
    messy.jsonl                    # generated once, committed
    reformulated.jsonl             # generated once, committed
    messifier_validation.md        # hand-validation evidence
  src/
    messify.py                     # original → messy (one-shot, cached)
    reformulate.py                 # messy → reformulated (one-shot, cached)
    run_target.py                  # run target model per (task, condition, seed)
    grade.py                       # run BigCodeBench tests
    analyze.py                     # implements §10 analysis plan
  results/
    runs.jsonl                     # raw per-trial records
    report.md                      # generated report
```

## 14. Out of scope (explicitly)

- Agentic multi-turn workflows
- Real Claude Code / Cursor / Codex CLI as the harness (cannot isolate variables)
- Organic messy prompts (synthesizing for control)
- Model-family ablation beyond Sonnet + GPT
- Reformulator fine-tuning
- Surface-cleanup-only condition (deferred to v2 if v1 shows effect)
