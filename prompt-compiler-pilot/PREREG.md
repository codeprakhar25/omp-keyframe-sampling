# Pre-Registration — Prompt Compiler Pilot v1

**Frozen on:** 2026-05-19
**Frozen before:** any target-model run

## Hypotheses

- **H1 (primary):** pass@1 in arm C > pass@1 in arm B for Claude Sonnet 4.6, McNemar paired p<0.05.
- **H2 (secondary):** pass@1 in arm C ≥ pass@1 in arm A − 5pp.
- **H3 (cost):** $/correct-answer for arm C is within 30% of arm B.

## Frozen artifacts

- Task list: `data/tasks.jsonl`, sha256: TBD_FILL_BEFORE_RUNNING
- Messy inputs: `data/messy.jsonl`, sha256: TBD_FILL_BEFORE_RUNNING
- Reformulated inputs: `data/reformulated.jsonl`, sha256: TBD_FILL_BEFORE_RUNNING
- Messifier prompt: `prompts/messifier.md`, sha256: TBD_FILL_BEFORE_RUNNING
- Reformulator prompt: `prompts/reformulator.md`, sha256: TBD_FILL_BEFORE_RUNNING
- Target prompt template: `src/run_target.py:TARGET_SYSTEM_PROMPT`

## Models (pinned)

- Primary target: `claude-sonnet-4-6`
- Cross-check target: `gpt-4.1` (update to GPT-5.x when available)
- Reformulator: `claude-haiku-4-5-20251001`
- Messifier: `claude-haiku-4-5-20251001`

## Decoding

- temperature=0.2, top_p=1.0, max_output_tokens=2048
- Seeds (primary): 1,2,3,4,5
- Seeds (cross): 1,2,3

## Conditions

- A: original prompt → target
- B: messy prompt → target
- C: reformulated prompt → target

## Analysis (no deviation without amending this file)

1. Mean pass@1 per arm with Wilson 95% CI.
2. McNemar paired test on (B vs C) over (task, seed) pairs. Report p-value.
3. McNemar paired test on (A vs C) (secondary).
4. Bootstrap 95% CI on (C − B), 10000 resamples, seed=42.
5. Per-arm input/output token totals; $/correct-answer using published pricing.
6. Failure-mode count: (task, seed) where B=pass and C=fail. Report rate.
7. Cross-check: same analysis on GPT-4.1 subset (15 tasks × 3 seeds).

**No subgroup analyses. No post-hoc filtering of tasks. Exploratory analyses, if any, will be clearly labeled and excluded from conclusions.**

## Stopping rule

Run completes when all 450 primary + 135 cross trials are graded. No interim peeking that changes the experiment.

## Instructions: fill sha256 before running

After running messify and reformulate (but BEFORE running run_target), replace the TBD_FILL_BEFORE_RUNNING placeholders with actual sha256 values:

```bash
for f in data/tasks.jsonl data/messy.jsonl data/reformulated.jsonl prompts/messifier.md prompts/reformulator.md; do
  echo "$f: $(sha256sum $f | cut -d' ' -f1)"
done
```

Then: `git add PREREG.md && git commit -m "chore(pilot): freeze PREREG.md sha256 hashes"`
