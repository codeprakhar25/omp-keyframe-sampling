# Deeper Pass — effort/outcome on CLEAN eval + thrashing decomposition

## 1. Effort -> outcome, CLEAN (eval_method='tests') vs all
- **claude_code (tests-only)**: pass% by tool-call quartile = 56% -> 65% -> 48% -> 23%  (n=102)
- **codex (tests-only)**: pass% by tool-call quartile = 77% -> 74% -> 22% -> 13%  (n=91)

## 2. Tool-mix — what dominates the long doomed runs? (tests-only)
- claude_code pass: 51 tools/run avg (n=49)
- claude_code fail: 55 tools/run avg (n=53)
- codex pass: 27 tools/run avg (n=42)
- codex fail: 37 tools/run avg (n=49)

## 3. Concentration — share of the single most-used tool (tests-only)
- claude_code pass: top tool = 63% of all calls (avg)
- claude_code fail: top tool = 56% of all calls (avg)
- codex pass: top tool = 100% of all calls (avg)
- codex fail: top tool = 100% of all calls (avg)
