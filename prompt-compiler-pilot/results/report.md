# Pilot Results

## Target: `claude-sonnet-4-6`

- n paired (B vs C): 150

| Arm | Pass rate | Correct | Cost (USD) | $/correct |
|---|---|---|---|---|
| A | 0.233 | 35 | $1.21 | $0.0345 |
| B | 0.233 | 35 | $1.53 | $0.0438 |
| C | 0.200 | 30 | $1.59 | $0.0529 |

- McNemar B vs C: p=0.0625
- McNemar A vs C: p=0.0625
- Bootstrap 95% CI on (C − B): [-0.067, -0.007]
- Failure-mode rate (B-pass, C-fail): 0.143

---

## Conclusions

### Against pre-registered hypotheses

**H1 — C > B for Sonnet 4.6, McNemar p<0.05:** NOT SUPPORTED.
C (0.200) < B (0.233). Direction is reversed. p=0.0625 is above threshold and the bootstrap 95% CI on (C − B) is [-0.067, -0.007] — entirely negative. Reformulation made performance worse, not better.

**H2 — C ≥ A − 5pp:** TECHNICALLY SUPPORTED.
C=0.200 ≥ A−0.05=0.183. Survives the threshold, but only because the damage is small, not because the intervention worked.

**H3 — $/correct for C within 30% of B:** SUPPORTED.
C=$0.0529 vs B=$0.0438, ratio=1.21 (21% higher). Within the 30% threshold, but this is a cost increase for worse results, not a trade-off.

**Failure-mode rate:** 0.143 (14.3%). Just under the 15% concern threshold. However, all 5 discordant pairs (B-pass, C-fail) come from a single task (BigCodeBench/273), meaning the failure is concentrated rather than distributed.

---

### What the data actually shows

**The mess had zero effect on performance.**
Arms A and B are identical across all 30 tasks (30/30 match). The task distribution is bimodal: 7 tasks always-pass, 23 tasks always-fail, in both arms, with no partial passes across 5 seeds. Sonnet 4.6 at temperature 0.2 on BigCodeBench-Hard is essentially deterministic and completely unaffected by 125 tokens of synthetic noise (preamble, typos, buried requirements).

This undermines the premise of the experiment. If mess does not degrade performance, there is nothing for the reformulator to fix.

**The reformulator caused one catastrophic failure.**
BigCodeBench/273: A=5/5 pass, B=5/5 pass, C=0/5 pass. The reformulator (Haiku) hallucinated structural constraints not present in the original — it invented class-based implementation requirements and named constants, promoting its own inferences to hard requirements. Sonnet implemented them faithfully and broke the test. This is consistent with the known failure mode of rewriting models: they add information when they should only restructure it.

**The reformulator made prompts longer, not shorter.**
Average input tokens: A=270, B=395, C=451. The structured Markdown template (section headers, numbered requirements, labeled subsections) added more tokens than the noise it was supposed to replace. The token-compression value proposition does not hold here.

**Seeds were wasted.**
With bimodal, deterministic outcomes, 5 seeds per task provided 5 identical copies rather than 5 independent samples. 1–2 seeds would have been sufficient. The experimental budget for seeding (~60% of runs) could have bought more tasks or a second model instead.

---

### Caveats

- n=30 tasks. The pilot is powered to detect ~12pp effects; the observed effect (-3.3pp) is below that threshold. The negative result may be noise.
- Synthetic mess. The messifier may have produced prompts that look noisy but preserve enough signal that the model never actually has to parse the noise. Real developer prompts may be harder in different ways.
- Single model. Sonnet 4.6 is unusually robust. Results may differ on weaker models or at higher temperature.
- Weak reformulator. Haiku reformulating for Sonnet is a significant capability mismatch. A same-class reformulator (Sonnet→Sonnet) would be a fairer test.
- No GPT cross-check was run. The cross-check results (Task 12) are absent from this report.

---

### Honest assessment of the research direction

The central idea — that structural reformulation helps frontier coding agents — is not dead, but this pilot found no supporting evidence and one clear risk. The most useful output of this experiment is not the negative result on H1; it is the finding that **A = B**. Before the reformulation question can be meaningfully tested, a benchmark configuration must be found where mess actually degrades performance. That prerequisite was not met here.

The middleware idea makes more sense in contexts where: (a) the baseline model is weaker, (b) the mess is organic and severe enough to cause degradation, or (c) the reformulator has equal or greater capability than the target. None of those conditions held in this pilot.

Spending more GPU/API budget repeating this experimental design would not be productive. The next experiment needs a different foundation.

---

### What would make v2 worth running

**Prerequisite check first (cheap):** Pick a benchmark where partial passes exist and establish that arm B < arm A before building the full pipeline. If A = B again, stop — the reformulation question cannot be answered on that benchmark.

Suggested changes if the prerequisite holds:

1. **Easier benchmark** — HumanEval+ or MBPP+ (60–80% baseline pass rate, partial passes at lower temperature). Richer signal, clearer degradation from mess.
2. **More severe or organic mess** — Use real developer prompts from public sources (GitHub issues, Aider conversation logs) rather than synthetic injection. Or increase synthetic severity until A > B is confirmed.
3. **Same-class reformulator** — Replace Haiku with Sonnet as the reformulator. The hallucination on BCB/273 is a capability problem; Haiku should not be reformulating tasks it cannot fully understand.
4. **Higher temperature** — 0.4–0.6 to get partial passes and real stochasticity, making seeds informative.
5. **Ablation** — Include a surface-cleanup-only arm (fix typos/grammar but do not restructure) to separate surface from structural effects. This was planned for v2 in the original spec.

**If all you want is a yes/no on the reformulation idea at low cost:** run the prerequisite check only. Find 20 tasks where Sonnet fails on B but passes on A. That dataset is the only one where the reformulation experiment is worth running.
