"""S0 harness for the visual-evidence-compression experiment.

Pipeline: load media -> select evidence frames (condition) -> answer with a
frontier model -> score (accuracy, input tokens, selector recall@k).

See SPEC.md for the experiment design and go/no-go gate.
"""
