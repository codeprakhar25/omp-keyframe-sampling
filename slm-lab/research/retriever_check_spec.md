# Retriever check — is the Stage-1 recall wall SigLIP-specific or task-fundamental?

**Compiled:** Jul 9, 2026. **Question raised by:** TwelveLabs Marengo 3.0 blog hit #4 — composed
retrieval lands verbose 60–80-word compositional queries at **rank 1**, the exact regime where our
SigLIP buried gold at rank 55–323. If a purpose-built video retriever recalls the needle where SigLIP
can't, the Fork B negative is a *selector* limitation, not a *task* wall, and Stage-1 revives.

## VERDICT (already answered — mostly): TASK wall, not selector

We had **already run the check** before the doubt was raised. `scripts/marengo_ceiling.py:193` feeds
Marengo **the full verbose `it["question"]`** (not a reduced phrase) — i.e. the precise verbose-query
regime the blog claims Marengo wins on. Result (same 10 videos, hit@k localization):

| bin   | Marengo 3.0 (lenient / strict) | SigLIP so400m hit@6 |
|-------|--------------------------------|---------------------|
| 600s  | **.20 / .10**                  | **.30**             |
| 3600s | **.30 / .10**                  | **.24**             |

The commercial SOTA video retriever, on the full verbose query, **never beats cheap per-frame SigLIP**
on our needle — at 10-min it loses. **Hit #4's "SigLIP-specific weakness" hypothesis is rejected.**
Why Marengo doesn't win: it returns ~**6 s clips** (retrieval-by-design, not 1-frame localization), so
on a 1–2 s single-frame needle in homogeneous video it smears — same failure mode as X-CLIP in the
scorer-swap. A better *retriever* doesn't help because the bottleneck is discriminating one frame from
~300 near-duplicates, which no global/clip embedding does. This **hardens negative #3**: the wall is
the task (fine needle in visually-homogeneous long video), reproduced across SigLIP, X-CLIP, detection
(2 families), generative joint-read, and now the commercial flagship.

## Residual gaps (low value — the verdict already points one way)

1. **n=10 → n=30 Marengo confirm** on 600s. Same script, tightens the Wilson CI (n=10 is ±~.28).
   `python scripts/marengo_ceiling.py --manifest data/manifest.lvb.frames.100.json --videos-dir
   data/videos --bins 600s --n 30 --out results/marengo_600s_n30.json`. Cost: ~195 video-min indexing
   on TwelveLabs (`TWELVELABS_API` set, len 32) + query. **Only run if a reviewer challenges n=10.**
2. **Open frame-level retriever** (SigLIP2-so400m / InternVideo2-1B) at 1-frame granularity on pod,
   vs SigLIP baseline on cached frames — reuse `scripts/dump_scores.py`, swap the model id. But since
   Marengo (video-native, higher effective fps, 6 s clips) already lost to 1-fps SigLIP, an open frame
   retriever winning is unlikely. **Skip unless the paper needs an open-model breadth column.**

## Decision rule (for any residual run)

- retriever recall@6 ≫ SigLIP floor (.30) and → ceiling (.60) ⇒ wall is SELECTOR, Fork B revives.
- retriever recall@6 ≈ SigLIP floor ⇒ wall is TASK, negative hardened. **← where Marengo landed.**

## Bottom line

The retriever axis is closed. The lever that can still *flip* Fork B is not a better retriever but the
**end-task metric**: measure LVB MCQA accuracy under compression (`scripts/gpt_mcqa.py`), because
near-duplicate frames that fail frame-recall may still carry the evidence the answerer needs. blind
floor already measured (.48 @60s, .56 @600s); `full`/`topk`/`uniform` pending (pod for frames). That,
not another retriever, is next.
