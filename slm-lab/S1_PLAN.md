# S1 — Real dataset plan (screen/GUI slice)

**Status:** drafting (Jun 30, 2026)
**Why this slice:** §2 of HANDOFF research — long-video-QA frame selection is a
crowded 2025–26 research area (AdaRD-Key, Focus, KFS-Bench, BOLT). The **screen /
GUI / agentic-browsing** slice is far less crowded, agent-shaped, and the part of
SPEC §6 flagged "most defensible." Incumbents (TwelveLabs/Marengo) are cloud +
general-video; nobody owns local + screen + agent-loop packaging.

## Vision this serves

**EvidenceGrep for agents** — an MCP tool: `(long visual artifact, question) ->
{few frames/regions that matter, timestamps}`. Off-the-shelf selector, local, cheap.
Contribution = honest benchmark + glued agent tool, NOT a novel selector model.

## Dataset target

~50 items. Each:

```json
{
  "id": "...",
  "media": "data/media/<id>.mp4 | <id>/*.png",
  "question": "...",
  "gold_answer": "...",
  "gold_evidence_span": [start_frame, end_frame]   // or [t0, t1] seconds
}
```

Answer must live in a SHORT span of a LONG capture (candidate frames >> k).
`gold_evidence_span` enables answerer-independent `recall@k`.

## Sources (screen-biased, public)

1. **Dev/software screencasts & tutorials** — long screen recordings; ask a Q whose
   answer is one moment (e.g. "what error code appeared in the terminal?", "which
   config value was edited?"). DOM unavailable (it's a video).
2. **Dashboard / chart walkthroughs** — analytics/BI demos; Q about a value visible
   for only a few frames ("what was peak QPS on the graph?").
3. **Agentic-browsing traces** — multi-frame screenshot sequences from an agent run;
   Q about a state only on one screen ("which button was disabled at checkout?").
4. (control) a few **long-talk video-QA** items to compare vs the crowded slice.

## Build pipeline

1. Collect recordings -> `data/media/` (gitignored, per repo convention).
2. Sample candidate frames at ~1 fps (reuse `harness/media.py`) -> the haystack.
3. Hand-write question + gold_answer per item.
4. Label `gold_evidence_span` (frame idx range that actually contains the answer).
5. **Anti-leakage (SPEC §9):** author the question from watching the video, NOT from
   the gold frame crop. Selector must NEVER see gold_evidence_span at inference.
6. Emit `data/manifest.s1.json`.

## Conditions

| Cond | Selector | Note |
|---|---|---|
| A | full_dump | cost/accuracy ceiling |
| C1 | embedding (SigLIP) | cheap, current code |
| C2 | generative VLM (SmolVLM2-500M / Moondream) | query-conditioned; matches what beats uniform in papers |

Answerer fixed across conditions (gpt-4.1 or claude-sonnet). `--judge` for accuracy.

## Metrics (per SPEC §5)

`recall@k` (selector, free w/ echo) · accuracy (LLM-judge) · input tokens · latency.
Headline = accuracy-vs-tokens, want C up-and-left of A. Report CIs (N=50 is noisy).

## Go/No-Go (SPEC §8, on real data)

```
PROCEED only if  C acc >= A acc - 2pts  AND  C tokens <= A tokens / 5
```

On REAL screen frames — not synthetic codes (SigLIP can't read rendered text well,
so the synthetic Exp-4 number is not trusted for the gate).

## Pod setup steps (fill when pod given)

```bash
# 1. clone repo / pull branch reward-hacking-context-planning -> slm-lab/
# 2. env
pip install -r requirements.txt
pip install torch transformers accelerate
pip install openai          # or anthropic
export OPENAI_API_KEY=...    # answerer + judge
# 3. plumbing smoke (synthetic) — confirm harness runs on pod
python3 -m harness.run --manifest data/manifest.synthetic_hard.json \
    --conditions A C --answerer openai --model gpt-4.1 \
    --selector embedding --k 6 --judge
# 4. once S1 manifest built -> swap --manifest data/manifest.s1.json
```

## Open TODO

- [ ] build C2 generative-VLM selector (SmolVLM2 scorer) as drop-in `Selector`
- [ ] collect first 10 screen items (S1 pilot) before scaling to 50
- [ ] decide fixed answerer: gpt-4.1 vs claude-sonnet-4-5 (cost vs ceiling)
