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
| Indic-Mio LoRA, curated 227 rows, r=16, 2 ep | tag-swap **2/10 — fail** |
| Orpheus Hindi LoRA, same pack | tag-swap **3/10 — fail**, parked |
| IV v7, 23,675 rows, prefix captions, 2 ep | **fail** — decode runaway, bare contamination |
| IV v8_e3, 9,077 rows, inline captions, 3 ep | ear **fail** |
| IV v9, same pack, tag-contrastive objective, 1 ep | best tag conditioning so far; ear pending |

Orpheus is parked for tag control: no native emotion tags, generations fail by
ear. Base is **Indic-Mio** (`SPRINGLab/Indic-Mio`, codec
`Aratako/MioCodec-25Hz-24kHz`). Stock Indic-Mio scores **near chance on its own
native emotion tags** — unexplained, and a standing argument for a base swap.

### Teacher-forced tag ablation

`modal_tag_ablation.py` scores the *true* audio codes of held-out clips three
ways — right tag, no tag, wrong tag — and reads the model's own log-likelihood.
No generation, no listening, no judge model. `d_wrong` is the number that
defines tag control: does it matter *which* tag was given.

```
                              NLL      d_none    d_wrong    tag share of LoRA gain
BASE (control)               7.717     +0.014     -0.002          —
v8_e3  (3 ep, plain SFT)     6.649     +0.134     +0.052         4.8%
v8_e1  (1 ep, prompt-masked) 6.772     +0.095     +0.036         3.8%
v9     (1 ep, contrastive)   6.838     +0.079     +0.147        16.7%
```

Base `d_wrong` CI straddles zero (46% of clips positive = chance), which
validates the metric.

Findings:

- **Plain SFT learns domain adaptation, not tag control.** Only ~5% of what the
  v8 LoRA gained was attributable to tag identity; the rest is speaker, prosody
  and "IV cut audio sounds like this".
- **Row count does not predict per-tag learning.** `laugh` (426 rows) beats
  `tsk` (1412). Acoustic distinctiveness predicts it instead. More data is not
  the lever.
- **The effect is local.** Profiling per code token shows 3.9x concentration,
  peaking at 40–45% of the clip against a mean tag caption position of 0.435 —
  the model uses the tag at roughly the right moment, then the effect dies after
  ~70% of the clip.
- **Prompt masking did not help** (3.8% vs 4.8%) — being graded on reproducing
  the caption text may itself be forcing a sharper tag representation.
- **Tag-contrastive training works** (`modal_iv_v9_contrastive.py`): show the
  same clip with a wrong tag and require higher NLL under it, scored on the
  event window. `d_wrong` 0.052 → 0.147, and `d_wrong/d_none` flips from 0.39 to
  1.85 — identity now dominates presence. Ear verdict still pending.

Caveat on all of the above: the v8 holdout was never encoded, so these run on
training rows — the generous case.

### Open: are the labels even true?

Every IV event label came from a transcriber typing `<tsk>` in the verbatim
text, never from an ear — and `TAG_FREEZE.md` warned "event ≠ always audible"
before the pack was built. Listening suggests some tagged clips contain only a
pause. One cause would explain four findings at once (small `d_wrong`, more data
not helping, row count not predicting, and the surviving tags being exactly the
hardest ones to mislabel).

`data/label-audit.html` + `modal_label_audit.py` build a 200-clip audit
(20 random cuts per tag) to measure per-tag label precision, which then gets
correlated against per-tag `d_wrong`. Not yet judged.

### Curated-pack gaps

`data/scale/tag_inventory.json`: every emotion tag is far under its 40-clip
target (angry 10, sad 12, excited 12, fear 9, surprise 8, disgust 8). Event tags
meet target. Retraining the same unbalanced 227 rows is explicitly not a step.

## ASR for code-switched Hindi–English

`scripts/asr_hiacc_bakeoff.py` scores ASR against HiACC's own human
transcripts (30 local clips, 6.2 min).

```
arm                                WERµ    devaΔ   latin_kept
whisper-1                         0.452   +0.282      0.095
whisper-1  [no-lang-hint]         0.464   +0.012      0.196
gpt-4o-transcribe                 0.569   +0.025      0.098
gpt-4o-transcribe [no-lang-hint]  0.525   +0.041      0.134
gemini-2.5-flash                  0.331   +0.247      0.298  (n=7, quota)
```

`latin_kept` = share of the reference's English words still in Latin script.
**No configuration preserves code-switch script**: 80–90% of English words come
back transliterated into Devanagari, one model emitted Urdu script, and another
silently deleted code-switched spans. WER overstates the content error — the
model usually heard correctly; the script is what breaks. That makes downstream
code-switch labelling impossible on ASR output as it stands.

Forcing `language=hi` causes much of the drift (`latin_kept` 0.095 → 0.196
without it). `scripts/asr_backfill.py` hardcodes that flag and wrote 14 rows
into `drafts.json` — those transcripts are suspect and should be redone.

The script ranking is safe at n=30; the WER ordering between models is not.

## HiACC full corpus

Zenodo record `15551669`, CC-BY-4.0, open. `modal_hiacc.py` pulls the 531 MB
`Corpus.zip` to the `real-voice-hiacc` volume and joins audio to transcripts.

```
5,176 wavs, matching 5,176 human transcripts + code-switch labels
adult    train 2322 / val 332 / test 664
children train 1300 / val 186 / test 372
plus sentence_stats.csv and per-split transcription files
```

Official splits, so ASR evaluation and fine-tuning both get a real protocol.
This is the ASR fine-tuning set for code-switched Hindi–English and the Hinglish
audio the train pack is short of.

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
