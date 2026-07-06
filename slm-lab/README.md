# slm-lab

Experiment: **visual evidence compression for agents.** Can a small model select &
compress visual evidence (frames/clips/regions) so a frontier agent answers
visual/video questions at a fraction of the tokens/cost, without losing accuracy?

Full design + go/no-go gate: [`SPEC.md`](./SPEC.md).

## Layout

```
SPEC.md                      # experiment spec
harness/
  media.py                   # manifest item -> candidate Frames (video/images)
  selectors.py               # FullDump (A) / Uniform / Embedding-SigLIP (C)
  answerers.py               # fixed frontier answerer: echo / anthropic / openai
  metrics.py                 # exact-match / LLM-judge accuracy + recall@k
  run.py                     # orchestrator + summary + go/no-go gate
scripts/make_synthetic.py    # tiny offline dataset for a smoke test
data/                        # manifests (media itself is gitignored)
results/                     # runs.jsonl + summary.json (gitignored)
```

## Quickstart (offline smoke test — no API key, no GPU)

```bash
pip install pillow numpy
python scripts/make_synthetic.py
python -m harness.run --manifest data/manifest.synthetic.json \
    --conditions A C --answerer echo --selector uniform --k 4
```

This validates the whole pipeline (selection -> answer -> scoring -> aggregation).
`EchoAnswerer` returns stub text, so accuracy is ~0 by design; what's meaningful
here is `mean_input_tokens`, `token_reduction_vs_A`, and `recall_at_k`.

## Pressure-test the selector + gate (harder synthetic)

Needle-in-haystack: long frame sequences where one frame is the answer and the
rest are plausible distractors. `uniform` will usually miss the needle; `embedding`
should recover it. `recall@k` is answerer-independent, so check selection quality
for free with `echo` before spending API calls.

```bash
python scripts/make_synthetic_hard.py --n-items 12 --frames 40 --seed 0

# free: does the selector even find the needle?
python -m harness.run --manifest data/manifest.synthetic_hard.json \
    --conditions A C --answerer echo --selector uniform --k 6

# then confirm accuracy tracks recall with a real answerer
python -m harness.run --manifest data/manifest.synthetic_hard.json \
    --conditions A C --answerer openai --model gpt-4.1 --selector uniform --k 6 --judge
```

Expect `uniform` recall ≈ the figure the generator prints; swapping `--selector embedding`
should raise it. Condition A sends `n-items × frames` images — keep counts modest to control cost.

## Real run

```bash
pip install -r requirements.txt          # plus torch/transformers for embedding selector
pip install anthropic                     # or: openai
export ANTHROPIC_API_KEY=...

python -m harness.run --manifest data/manifest.json \
    --conditions A C --answerer anthropic --model claude-sonnet-4-5 \
    --selector embedding --k 6 --judge
```

## MCP server (tool track)

`mcp_server.py` packages the winning selector (SigLIP-so400m-384 — hit@k 0.90 on n=20
stitched GUI needles, 13.5x token reduction at equal gpt-4.1 accuracy, FINDINGS §S2)
as an MCP tool any agent can call:

```bash
pip install "mcp[cli]"                    # plus torch/transformers/sentencepiece/protobuf
claude mcp add visual-evidence -- python /path/to/slm-lab/mcp_server.py
```

Tool `select_evidence(media_path, question, k=6)` → scores every frame of a video /
screenshot dir against the question, writes the top-k as PNGs, returns their paths +
scores. The agent then reads only those k frames instead of the full dump.

## Conditions (S0 implements A and C)

| Cond | Selector | Meaning |
|---|---|---|
| A | `full_dump` | send all candidate frames (upper bound on cost & accuracy) |
| C | `uniform` or `embedding` | small selector picks top-k frames |

Condition **B** (TwelveLabs / SmolVLM2 retrieval) and **D** (trained SLM selector)
come later — see SPEC §4. Add them by implementing a `Selector` / `Answerer` and
registering it in `run.build_selector` / `answerers.build_answerer`.

## Go / No-Go gate (SPEC §8)

After ~50 real items, proceed to training a custom selector only if:

```
C accuracy >= A accuracy - 2 pts   AND   C input tokens <= A input tokens / 5
```

`run.py` prints this verdict automatically when both conditions A and C are present.
