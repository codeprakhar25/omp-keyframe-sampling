# Key numbers — provenance for paper claims (auto-extracted)

Source dbs: experiment_full.db (main), probe_codex.db, probe_claude.db

## 1. Correctness marginals (eval'd cells)

### claude_code
- distinct tasks: 15
- none: 24/45 = 53.3%
- always_on: 25/45 = 55.6%
- selective: 25/45 = 55.6%

### codex
- distinct tasks: 17
- none: 30/51 = 58.8%
- always_on: 29/51 = 56.9%
- selective: 27/51 = 52.9%

## 2. Efficiency: duration(s) + tools by agent×strategy

- claude_code always_on: n=46 dur_mean=1108 dur_med=952 tools_mean=50.4
- claude_code none: n=47 dur_mean=1288 dur_med=661 tools_mean=48.6
- claude_code selective: n=45 dur_mean=954 dur_med=579 tools_mean=52.8
- codex always_on: n=51 dur_mean=897 dur_med=866 tools_mean=32.0
- codex none: n=51 dur_mean=932 dur_med=856 tools_mean=31.9
- codex selective: n=51 dur_mean=907 dur_med=786 tools_mean=32.2

## 3. Claude opshin process-effect (within-task)

- 595: none=2243 always=1588 sel=1193 Δ(none-ctx)=+853
- 598: none=2377 always=1332 sel=1637 Δ(none-ctx)=+893
- 605: none=2872 always=2743 sel=3031 Δ(none-ctx)=-14
- 610: none=3108 always=2831 sel=2852 Δ(none-ctx)=+267
- 616: none=2759 always=1838 sel=1447 Δ(none-ctx)=+1117
- mean Δ = +623s ; faster-with-context on 4/5 tasks
- opshin marginal dur: {'none': 2689, 'always_on': 2066, 'selective': 2032}

## 4. Probe results (probe_*.db)

### probe_codex
- 554 always_on: 0/3
- 554 none: 0/3
- 554 selective: 0/3
- 907 always_on: 0/3
- 907 none: 0/3
- 907 selective: 0/3
### probe_claude
- 554 always_on: 0/3
- 554 none: 0/3
- 554 selective: 0/3
- 907 always_on: 1/3
- 907 none: 2/3
- 907 selective: 0/3

## 5. Revision additions (peer-review response)

### Agent-specific difficulty (quantified, shared 15 tasks)
- per-task pass-rate corr claude vs codex: Spearman rho=0.75 (p=0.001), Pearson r=0.77, n=15
- borderline (0<rate<1) for exactly one agent: 6/15
- differing floor/ceiling status across agents: 6/15

### opshin full-suite count, within-task (stream-log subset)
- per task none / mean(context): 595 2.50/0.75; 605 1.20/1.25; 610 2.50/0.67; 616 9.33/6.25
- mean Δ(none-ctx) = +1.65 full-suite runs; fewer-with-context 3/4 tasks; exact sign-flip p=0.250 (n=4)
- NOTE: exploratory (post-hoc mechanism), not in pre-registered metric family

### TOST bootstrap CIs (claude paired diffs, from power_analysis.py)
- always-none [-4.4,+8.9]pp; sel-none [+0.0,+6.7]pp; sel-always [-6.7,+6.7]pp (all within ±10pp)
