# When Does Persistent Context Help Coding Agents?

## Resolving the Context File Paradox — A Controlled Ablation Study

**Researcher:** Prakh
**Start date:** May 21, 2026
**Target completion:** ~8 weeks (mid-July 2026)
**Status:** Week 1 — Hypothesis Scoping

---

## 1. Problem Statement

Two peer-reviewed 2026 papers reach opposite conclusions about the same intervention:

| Paper | Finding |
|-------|---------|
| arXiv:2601.20404 (Jan 2026) | AGENTS.md reduces runtime **28.6%**, output tokens **16.6%**, task completion comparable |
| arXiv:2602.11988 (Feb 2026, ETH Zurich) | Context files **reduce** success 2–3%, **increase** inference cost 20%+ (up to 159%) |

Supporting evidence deepens the contradiction:
- Passive context injection → 100% pass rate vs 53% for active retrieval (Agent Context System, 2026)
- CLAUDE.md cache reads consume 99.93% of token quota in production usage (anthropic/claude-code#24147)
- Controlled benchmark: persistent memory yields 15–28% token savings but **zero quality improvement**
- Memory Transfer Learning (arXiv:2604.14004): abstract "insights" transfer well, concrete "trajectories" cause **negative** transfer
- agentmemory (file-based + search): 95.2% R@5, crushing Mem0 (68.5%) and MemGPT (83.2%) on LongMemEval-S

**Nobody has run the controlled ablation that explains *when and why* context files help vs hurt.**

---

## 2. Core Hypothesis

> The effect of persistent context on coding agent performance is **moderated by three factors**: (1) injection strategy, (2) content type, and (3) task complexity. Static always-on context helps simple tasks but hurts complex ones via over-constraining. Selective injection from accumulated history helps complex tasks at lower token cost. The two contradictory papers measured different points in this interaction space.

### Sub-hypotheses

**H1 (Injection Strategy):** Selective search-then-inject outperforms always-on paste for tasks requiring > 3 files, while always-on wins for single-file fixes.

**H2 (Content Type):** Synthesized wiki-style context (distilled knowledge) outperforms raw conversation dumps and static rules on novel tasks, but static rules win on repetitive/convention-heavy tasks.

**H3 (Token Economics):** Always-on injection token cost scales linearly with file size. Selective injection cost stays roughly constant regardless of accumulated history size, with the crossover point around 5K tokens of context.

**H4 (Quality vs Efficiency Tradeoff):** Context files reduce *exploration* (fewer tool calls, fewer files read) but do not improve *correctness*. The runtime savings in arXiv:2601.20404 came from less exploration, not better solutions.

---

## 3. Experimental Design

### 3.1 Independent Variables

| Variable | Levels | Notes |
|----------|--------|-------|
| **Context strategy** | (A) None, (B) Always-on static (AGENTS.md), (C) Always-on accumulated (MEMORY.md), (D) Selective inject (search-then-inject from wiki) | 4 levels |
| **Context size** | Small (< 2K tokens), Medium (5–8K), Large (15K+) | 3 levels, only for B/C/D |
| **Task complexity** | Simple (single-file, < 50 LOC change), Medium (multi-file, 50–200 LOC), Complex (cross-module, > 200 LOC) | 3 levels |
| **Agent** | Claude Code, Cursor (via CLI) | 2 levels, budget permitting |

Full grid: 4 strategies × 3 sizes × 3 complexities × 2 agents = 72 cells (reduced via fractional factorial if needed).

### 3.2 Dependent Variables

| Metric | How measured |
|--------|-------------|
| **Task success** | Pass/fail against repo test suite (binary) |
| **Patch quality** | Diff size, files touched, lint pass rate |
| **Token usage** | Input tokens, output tokens, cache read tokens (logged per turn) |
| **Wall-clock time** | Start to final patch |
| **Exploration depth** | Number of tool calls, unique files read, turns taken |
| **Context utilization** | % of injected context lines referenced in agent reasoning |

### 3.3 Task Set

**Primary:** Subset of SWE-Bench-Verified (Python repos), stratified by complexity. Target: 60–100 tasks.
**Secondary (if time permits):** SWE-Bench-CL chronological subset to test multi-session accumulation.

### 3.4 Context File Construction

For each repo in the task set:
- **(B) Static rules:** Use existing AGENTS.md/CLAUDE.md if available, else generate one following Anthropic's guidelines and manually review
- **(C) Accumulated history:** Simulate 5 prior sessions of agent work on earlier issues in the same repo, accumulate daily logs into MEMORY.md
- **(D) Selective wiki:** Same accumulated content as (C), but stored as searchable topic files; agent uses grep/search to pull relevant sections per-task

---

## 4. Timeline

### Week 1 (May 21 – May 27): Hypothesis Scoping & Literature Lock ← CURRENT

- [ ] Read the two contradictory papers in full (arXiv:2601.20404, arXiv:2602.11988)
- [ ] Read Memory Transfer Learning paper (arXiv:2604.14004) — specifically the abstraction-level findings
- [ ] Read SkillRet (arXiv:2605.05726) for retrieval methodology (may inform selective injection design)
- [ ] Read agentmemory design doc (github.com/jayzeng/agentmemory/blob/main/design.md) for selective injection architecture
- [ ] Read "Coding Agents are Effective Long-Context Processors" (arXiv:2603.20432) — how agents handle large context natively
- [ ] Finalize hypothesis — pick exactly which sub-hypotheses to test (don't boil the ocean)
- [ ] Write 1-paragraph "contribution statement" that distinguishes this from both prior papers
- [ ] Decision: which agent(s) to instrument (Claude Code only, or also Cursor?)
- [ ] Decision: SWE-Bench-Verified subset selection criteria

### Week 2 (May 28 – Jun 3): Harness Engineering

- [ ] Build instrumentation wrapper that logs per-turn: input tokens, output tokens, cache reads, tool calls, files read/written, wall-clock time
- [ ] Build context injection module: supports none / always-on / selective modes
- [ ] Build context file generator: creates static rules, accumulated history, and wiki-style topic files for each test repo
- [ ] Validate instrumentation on 3 pilot tasks (1 per complexity level)
- [ ] Set up results database (SQLite or CSV — keep it simple)

### Week 3 (Jun 4 – Jun 10): Context Construction & Pilot

- [ ] Generate context files for all repos in task set
- [ ] Manual quality review of generated context (sample 20%)
- [ ] Run pilot: 9 tasks (3 complexity × 3 strategy) with full instrumentation
- [ ] Analyze pilot data — check for ceiling/floor effects, calibrate complexity labels
- [ ] Adjust experimental design based on pilot (drop levels if underpowered, add if needed)

### Week 4–5 (Jun 11 – Jun 24): Main Experiment Run

- [ ] Run full ablation grid (parallelized where possible)
- [ ] Monitor for failures, re-run crashed tasks
- [ ] Daily sanity checks on data quality
- [ ] Begin exploratory analysis as data comes in

### Week 6 (Jun 25 – Jul 1): Analysis

- [ ] Statistical analysis: interaction effects (strategy × complexity × size)
- [ ] Token economics analysis: cost curves per strategy as context size grows
- [ ] Exploration pattern analysis: do context files change *how* agents search?
- [ ] Reconciliation analysis: simulate the conditions of both prior papers — do our results reproduce theirs?
- [ ] Identify the key figures (aim for 4–6 main figures)

### Week 7–8 (Jul 2 – Jul 15): Writing

- [ ] Draft: Introduction framing the contradiction
- [ ] Draft: Experimental setup (reproducible detail)
- [ ] Draft: Results with figures
- [ ] Draft: Discussion — when context helps, when it hurts, and why
- [ ] Draft: Related work positioning against both papers + Memory Transfer Learning
- [ ] Internal review, revision
- [ ] Prepare code/data release package

---

## 5. Key Papers (Reading List)

| Paper | arXiv / Source | Priority | Read? |
|-------|---------------|----------|-------|
| On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents | 2601.20404 | P0 | [ ] |
| Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents? | 2602.11988 | P0 | [ ] |
| Memory Transfer Learning in Coding Agents | 2604.14004 | P0 | [ ] |
| Coding Agents are Effective Long-Context Processors | 2603.20432 | P1 | [ ] |
| SkillRet: Large-Scale Benchmark for Skill Retrieval | 2605.05726 | P1 | [ ] |
| Agent Skills in the Wild | 2604.04323 | P2 | [ ] |
| ContextBench: Benchmark for Context Retrieval in Coding Agents | 2602.05892 | P1 | [ ] |
| ALMA: Auto-Learns Memory Architectures | Feb 2026 | P2 | [ ] |
| LoCoBench-Agent: Interactive Benchmark for Long-Context SWE | 2026 | P2 | [ ] |
| agentmemory design doc | github.com/jayzeng/agentmemory | P1 | [ ] |
| RAG vs Agent Memory vs LLM Wiki: Practical Comparison | medium.com/@visrow | P2 | [ ] |

---

## 6. Resources & Infrastructure

**Compute / API budget:**
- Estimate ~72 cells × avg 5 runs × ~$0.50–2.00/run = $180–720 for main experiment
- Pilot runs: ~$20–50
- Total estimated budget: **$200–800**

**Tools:**
- Agent under test: Claude Code (primary), Cursor CLI (secondary)
- Task set: SWE-Bench-Verified (public, Python subset)
- Instrumentation: custom wrapper scripts (Python)
- Analysis: Python (pandas, scipy, matplotlib/seaborn)
- Storage: local SQLite + CSV exports

**Key risks:**
- API cost overrun if tasks are harder than expected → mitigate with pilot + cost caps per run
- Ceiling effects if tasks are too easy → stratify by complexity carefully
- Reproducibility of agent runs (non-deterministic) → run each cell 3–5 times, report variance
- Context file quality confound → use both human-written and LLM-generated, analyze separately

---

## 7. Contribution Statement (Draft)

> We present the first controlled ablation study that systematically varies context injection strategy, content type, and task complexity for AI coding agents. Our results reconcile two contradictory 2026 findings by showing that the effect of persistent context is moderated by [TBD based on results]. We release our instrumentation harness, context file corpus, and full experimental data to enable reproduction and extension.

---

## 8. Week 1 Working Notes

*Use this section for daily notes during Week 1.*

### May 21 — Project kickoff
- Research plan drafted
- Key contradiction identified between arXiv:2601.20404 and arXiv:2602.11988
- Initial hypothesis: the two papers measured different interaction effects
- Next: deep-read both papers, look for methodological differences that explain the contradiction
