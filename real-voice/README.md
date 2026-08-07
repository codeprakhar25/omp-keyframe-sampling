# real-voice — tag-controllable Hindi / Hinglish expressive TTS

Research sandbox for expressive, code-switched conversational speech generation in
Indian languages. The near-term milestone is narrow on purpose: **same Hindi text,
swap one tag, the ear hears the change.**

Not in scope yet: multi-speaker turn-taking, interruption modelling, full
code-switch prosody study.

## Gate

The milestone is judged by ear on held-out tag-swap pairs, never by training loss.

| Family | Pass bar |
|--------|---------:|
| emotion (angry / sad / excited / fear / surprise / disgust) | ≥70% ear-correct |
| breath / whisper / hesitation events | ≥60% ear-correct |

## Tag freeze v1

Frozen after the 55-clip smoke ear-audit (pass-2 self-agreement 96% strict,
100% treating `inhaling`↔`sigh` as one family; Rasa native-style map 100%).
Do not extend without re-running the smoke.

| Tag | Meaning by ear |
|---|---|
| `angry` | hot anger / irritation |
| `sad` | low, heavy, grief-tinged |
| `excited` | upbeat energy (Rasa `HAPPY` maps here) |
| `surprise` | sudden lift / shock |
| `fear` | tension, worry |
| `disgust` | revulsion / contempt |
| `pause` | near-silence gap, little or no vocalisation |
| `thinking` | filled hesitation — umm / uhh / hmm |
| `inhaling` | audible in-breath |
| `sigh` | audible out-breath sigh |
| `whispering` | whispered speech, not merely a quiet mic |
| `none` | no expressive tag |

IndicVoices event work adds a locked set on the catalog-only path
(`scripts/iv_tags.py`): `breathing`, `gasp`, `throat_clearing`, `cough`, `tsk`,
`ugh`, `sniffle`, `laugh`. `breathing` and `inhaling` stay separate tokens.

Caption contract: at most two tags, `[tag1] text` or `[tag1][tag2] text`.
Label confidence is `H` (clear) / `M` (plausible) / `L` (guess — dropped from train).

## Pipeline

```
1 pull / sample        scripts/pull_shard.py, pull_scale.py, ingest_hiacc_new.py
2 draft tags           scripts/draft_tags.py  (+ audio_draft.py for M/L clips)
3 human ear review     data/review-queue.html via scripts/serve_lab.py
4 build captions       scripts/build_captions.py  -> data/scale/captions.jsonl
5 inventory vs targets scripts/tag_inventory.py
6 pack + fixed holdout scripts/pack_train.py     -> train.jsonl / holdout.jsonl
7 native ceiling       stock base, no adapter, tag-swap on emotion pairs
8 LoRA                 modal_indic_mio.py / modal_iv_mio_train.py
9 tag-swap ear gate    scripts/tag_swap_eval.py + data/tag-swap*.html
```

Two collection paths feed step 6:

- **Curated path** — Rasa (native emotion styles), IndicVoices-R, HiACC Hinglish.
  Human ear label, `H`/`M` only. 247 clips.
- **IndicVoices catalog path** — labels come from the corpus's own `<>` verbatim
  event tags, never retagged from ear. Text-only gate and caption split
  (`modal_iv_hindi_split.py`), then WhisperX word alignment and slicing
  (`modal_iv_hindi_slice.py`). 450,690 rows scanned → 15,559 source clips →
  24,933 cut proposals, 24,923 aligned. Pack v7 = 23,675 train / 1,248 holdout.

On the catalog path, human accept/reject is cut-window sanity QA only. The ear
gates *evaluation*, not training labels.

## State

| Run | Result |
|-----|--------|
| MioCodec round-trip (P0.1 kill-switch) | PASS — events survive encode/decode |
| Indic-Mio LoRA, curated 227 rows, r=16, 2 epochs | tag-swap **2/10 — fail** |
| Orpheus Hindi LoRA, same pack | tag-swap **3/10 — fail**, parked |

Orpheus is parked for tag control: no native emotion tags and the generations
fail by ear. Base model is **Indic-Mio** (`SPRINGLab/Indic-Mio`, codec
`Aratako/MioCodec-25Hz-24kHz`).

Known gaps in the curated pack (`data/scale/tag_inventory.json`): every emotion
tag is far under its 40-clip target (angry 10, sad 12, excited 12, fear 9,
surprise 8, disgust 8). Event tags meet target. Retraining the same unbalanced
227 rows is explicitly not the next step.

## Datasets reviewed

Verdicts and per-dataset notes live in `data/reviews/*.json`, browsable through
`data/dataset-review.html`.

P0 keep: IndicVoices, IndicVoices-R, Rasa, HiACC, MUCS-Hinglish.
P1 later: Hinglish aggregate. Skip: Common Voice Hindi, IndicTTS-Hindi,
Rural Women ASR, HingCoS.

## Running things

```bash
.venv/bin/python scripts/serve_lab.py          # http://127.0.0.1:8765/
.venv/bin/python scripts/score_smoke.py        # smoke go/no-go
.venv/bin/python scripts/build_captions.py
.venv/bin/python scripts/pack_train.py
.venv/bin/python scripts/tag_inventory.py
.venv/bin/python scripts/tag_swap_eval.py --init
```

GPU work runs on Modal (`modal_*.py`), never locally.

## Not in this repo

Audio, model weights, and optimizer state are deliberately untracked. Wavs come
back from `scripts/pull_shard.py` / `pull_scale.py` or the Modal volumes
(`real-voice-mio`, `real-voice-iv-cuts`). Adapter configs and trainer state are
tracked so a run is reconstructable; the `.safetensors` are not.
`data/scale/iv_hindi_cut_proposals.json` (46 MB) regenerates from
`modal_iv_hindi_split.py --action run`.

Managed fine-tuning services for chat LLMs (Fireworks and similar) cannot train
these speech checkpoints — that was checked, and the credits are unrelated to
this milestone.
