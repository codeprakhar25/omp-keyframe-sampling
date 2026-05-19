# Prompt Compiler Pilot

Directional pilot study. See `SPEC.md` for the experimental design.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # then fill in API keys
```

## Reproduce

```bash
python -m src.select_tasks    # → data/tasks.jsonl
python -m src.messify         # → data/messy.jsonl
# HAND-VALIDATE 10 samples (see SPEC.md §7)
python -m src.reformulate     # → data/reformulated.jsonl
# COMMIT PREREG.md before this line
python -m src.run_target --target primary
python -m src.grade
python -m src.run_target --target cross
python -m src.grade --target cross
python -m src.analyze         # → results/report.md
```
