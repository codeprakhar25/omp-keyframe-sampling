# We Tried to Build a Prompt Compiler for AI Coding Agents. In This Pilot, the Mess Didn't Matter.

*A small experiment on whether cleaning up messy prompts helps a frontier model write better code.*

---

There's an idea that keeps coming up in AI tooling circles: what if you put a small, cheap model between the user and the big expensive model? The small one takes your messy, rambling prompt — full of typos, buried requirements, and irrelevant context — and restructures it into clean, numbered points before the frontier model ever sees it. A prompt compiler.

It sounds reasonable. Developers type fast and think messily. Frontier models are expensive. If a $0.001 Haiku call can clean up a $0.05 Sonnet call and improve the result, that's a good trade.

We decided to actually test it.

---

## The setup

We designed a controlled experiment with three arms:

- **Arm A (clean):** the original task description goes straight to the target model
- **Arm B (messy, raw):** a messifier model rewrites the task as a rambling, typo-ridden developer message, then sends it to the target model
- **Arm C (messy, reformulated):** the same messy prompt gets passed through a reformulator that extracts numbered requirements and structured sections, then sends *that* to the target model

We used 30 tasks from BigCodeBench-Hard — a benchmark of realistic, multi-requirement Python coding problems with automated tests. The target model was Claude Sonnet 4.6. The messifier and reformulator were both Claude Haiku 4.5. We made 5 repeated target calls per condition, graded each output by running the BigCodeBench test suite, and measured pass@1.

Before running the target model, we wrote an intended pre-registration document specifying our hypotheses: C > B, p<0.05. We wanted to be honest about what we were testing before we saw the results. As noted below, we did not fully freeze the preregistration correctly.

Total cost: about $4.

---

## The result we expected to find

We expected arm B to underperform arm A. That would mean the mess is hurting performance, and there's something for the reformulator to fix. Then we'd measure whether arm C recovered the loss.

That's not what happened.

---

## The result we actually found

**Arms A and B were identical in this setup. Perfectly, completely identical — across all 30 tasks.**

```
Arm A (clean):   pass@1 = 0.233   (35/150 trials)
Arm B (messy):   pass@1 = 0.233   (35/150 trials)
Arm C (reformed): pass@1 = 0.200  (30/150 trials)
```

Not one task had a different outcome between clean and messy prompts. We added preamble noise, buried requirements, mild typos, run-on sentences — and in this benchmark setup Sonnet 4.6 was unaffected. Every task that passed with a clean prompt also passed with the messy one. Every task that failed, failed identically.

The distribution was stark: 7 tasks always passed, 23 tasks always failed, regardless of whether the prompt was clean or noisy. The model either had the capability to solve the task or it didn't, and surface noise changed nothing.

---

## What the reformulator actually did

Arm C was slightly *worse* than both A and B. The reformulator didn't recover lost ground — there was no lost ground to recover. And it introduced one catastrophic failure.

**BigCodeBench/273:** clean prompt → 5/5 pass. Messy prompt → 5/5 pass. Reformulated prompt → **0/5 pass.**

When we looked at what Haiku produced for that task, the problem was clear. The original prompt already mentioned a handler class and the `SUCCESS_RESPONSE` / `ERROR_RESPONSE` constants, so this wasn't a simple case of inventing everything from scratch. The failure was subtler: the reformulator turned a compact benchmark prompt into a more prescriptive implementation brief, adding edge-case framing such as charset handling and empty/null data behavior, and weakening the exact starting-shape constraint around `task_func()`. Sonnet followed that structured brief and produced a server-style wrapper that timed out under the test harness.

That's the real risk with automatic reformulation: it doesn't just rephrase, it *interprets*. And when the interpretation is wrong, the model downstream has no way to know.

---

## Why Sonnet didn't care about the mess

Frontier models in 2026 are often robust to surface noise, and this pilot showed that clearly for one model, one benchmark, and one synthetic messifier. A messy preamble didn't confuse Sonnet here. Mild typos didn't degrade it. Buried requirements still got found. This is directionally consistent with what researchers have been finding across benchmarks, but this result should not be read as a universal claim about every model or every kind of messy prompt.

The implication is that the prompt compiler idea has a precondition that wasn't met here: **mess must actually degrade performance before there's anything to compile.** For Sonnet 4.6 on hard coding tasks with this synthetic mess injection, that precondition failed. In this setup, the middleware was solving a problem the target model did not have.

This may have been a more interesting problem two model generations ago.

---

## The honest limitations

An independent code review (Codex) caught a few things we should be upfront about:

**The pre-registration wasn't actually frozen.** The PREREG document had placeholder sha256 hashes that were never filled in before the runs. A pre-registration with TBD placeholders is just a template. The results are real, but the HARKing protection we intended isn't there.

**The messifier validation step was skipped.** The experimental design required hand-checking 10 messified samples to confirm semantic preservation before running anything. We didn't do it. If the messifier silently dropped requirements, that would inflate arm B's failure rate and confound the comparison. (Given that A = B perfectly, this probably didn't happen — but it should have been verified.)

**Anthropic's API doesn't support seeded generation.** The 5 "seeds" for Claude were 5 independent random calls, not reproducibly controlled samples. At temperature 0.2 with deterministic outputs, this didn't matter — but it's a design flaw.

We're noting these because they matter if you replicate this or extend it. The core finding (A = B, reformulation net-negative) is robust to all three issues, but the protocol wasn't as clean as we intended.

---

## What the experiment actually tested

We went in thinking we were testing: *does structural reformulation help?*

What we actually tested was: *can a Haiku reformulator improve prompts that Sonnet already handles perfectly well in this benchmark setup?*

The answer to that narrower question is clearly no. But the more useful prerequisite question — *does this kind of mess hurt this frontier model in the first place?* — got answered for free, and the answer is also no, at least for this model, benchmark, and type of synthetic noise.

---

## What would be worth building instead

The reformulation idea isn't dead. It just needs a different foundation.

The right product isn't an automatic reformulator that silently restructures your prompt. It's an **auditable intent compiler** — something that makes its inferences visible, distinguishes between what it preserved verbatim and what it inferred, and lets the developer reject inferences before the rewritten prompt is executed.

The BCB/273 failure is exactly the product risk: a compiler quietly turned "structure" into "new meaning." With an audit trail, that interpretation would have been surfaced before it reached Sonnet.

Before running another reformulation experiment, there's one prerequisite worth checking first, cheaply: find 20–30 real developer prompts where a messy version causes the model to fail when the clean version passes. That dataset — arm B genuinely underperforming arm A — is the only foundation where testing a reformulator makes sense.

If you can find that signal with organic, naturally messy prompts, the experiment is worth redoing. If A = B again on real prompts, the honest conclusion is narrower but important: frontier coding models may not need automatic prompt cleanup for this class of task.

### Where to look for better data

The ideal dataset for this experiment has five columns: raw messy user prompt → reformulated prompt → same agent output → objective pass/fail → human check that reformulation preserved intent. No public dataset has all five. But some get close enough to be useful for the discovery phase:

| Dataset | Why it's relevant | Gap |
|---|---|---|
| [SWE-chat](https://arxiv.org/abs/2604.20779) | Real coding-agent sessions with user prompts, tool calls, and commits — actual messy developer input | Observational, no A/B reformulation |
| [DevGPT](https://zenodo.org/records/16392320) | Developer ChatGPT conversations linked to GitHub issues and PRs | Older, not CLI-agent style |
| [CodeChat](https://huggingface.co/datasets/Suzhen/CodeChat) | Real developer-LLM conversations with messy prompts | Weak outcome labels |
| [Prompt Knowledge Gaps](https://arxiv.org/abs/2501.11709) | Labels prompts for missing context, unclear instructions, missing specs | Small, not an execution benchmark |
| SWE-bench / SWE-rebench | Objective grading with test suites | Issue statements, not organic messy prompts |

The right sequencing for a v2: use SWE-chat or DevGPT to *find* prompts where natural messiness correlates with agent failure, then use SWE-bench-style grading to measure whether reformulation recovers the loss. The mistake in this pilot wasn't using the wrong dataset — it was using an execution benchmark before validating that the synthetic mess actually degrades the target model. The dataset choice and the grading mechanism are fine; the pipeline order was wrong.

---

## The cheap lesson

Prompt middleware made more obvious sense when models were fragile. When older models got confused by a paragraph of rambling context, there was a real job for a cleaner layer to do. For frontier coding models, that job looks much smaller now.

The value has shifted from *cleaning prompts* to *auditing intent* — catching the cases where a developer's messy description is genuinely ambiguous, not just stylistically rough. That's a harder and more interesting problem than formatting cleanup, and it requires the compiler to show its work rather than silently transform it.

The reformulation idea isn't wrong. The assumption that this kind of mess hurts this frontier model turned out to be.

---

*Code and data: `claude-prompt-compiler-pilot/` — messify, reformulate, run, grade, analyze pipeline. Reusable with a different benchmark and model.*
