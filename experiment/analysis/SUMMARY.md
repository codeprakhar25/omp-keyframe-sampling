# Context-Files Study — Exploratory Analysis

Source: `results_from_pod_codex/experiment_full.db` — 291 runs, 5609 turns.

## 1. Tool-use profile (parsed from turns.tool_calls_json)
- **claude_code**: 5452 tool calls — Bash=3238, Read=1588, Edit=626
- **codex**: 3030 tool calls — run_bash=3030

## 2. Tool calls per run, by outcome
- **claude_code**: median tool calls — pass 41 vs fail 58
- **codex**: median tool calls — pass 27 vs fail 35

## 3. Pass rate by repo × strategy
- claude_code none: 53.2% (n=47)
- claude_code always_on: 54.3% (n=46)
- claude_code selective: 55.6% (n=45)
- codex none: 58.8% (n=51)
- codex always_on: 56.9% (n=51)
- codex selective: 52.9% (n=51)

## 4. Wall-time & token composition
- **claude_code**: avg 18.7 min/run | in 470 / out 19626 / reasoning 0 / cache_read 2664324 tok
- **codex**: avg 15.2 min/run | in 853338 / out 7159 / reasoning 1900 / cache_read 778758 tok

## 5. Files read / written per run
- **claude_code**: read 8.1 / wrote 2.4 files/run avg
- **codex**: read 0.0 / wrote 3.1 files/run avg

## 6. Does more effort → success? (tool calls binned vs pass rate)
(see 06_effort_vs_pass.png — is the relationship positive, flat, or inverted?)
