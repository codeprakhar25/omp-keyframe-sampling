# CORRECT_FINDINGS — post-query-bug rebuild

Started 2026-07-16. Everything in the old `FINDINGS.md` that depended on a scorer is
**void** (see §1). This file only records things verified *after* the fix, plus the
protocol facts checked against the actual LDDR PDF rather than assumed.

Rule for this file: **no number enters without saying which harness produced it and
what it is comparable to.** A number with no comparator is not a finding.

---

## 1. The bug that voided a week

`scripts/build_longvideobench.py` (commit 9b34acf, 2026-07-06) built ONE field:

```python
question = f"{stem}\n\nOptions:\n{opts}\n\nAnswer with the single option letter."
```

and named it `question`. It was designed as the **answerer's** prompt. Every **scorer**
then read the same field, because it was there and it was called `question`. One field,
two consumers, opposite needs. LongVideoBench's own `lvb_val.json` keeps `question` and
`candidates` separate — we fused them.

**Fix**: `harness/text.py::question_stem()` — single source of truth, derived (not a
manifest rebuild, which would reshuffle the subset away from cached embeds keyed on id).
Wired into all 10 scorer/retriever call sites. Answerer untouched.

### Measured damage (600s, LongCLIP, n=411)

| replay query | Spearman vs archived tainted scores | top-8 overlap |
|---|---|---|
| **full** (original fused prompt) | **0.9999** (min 0.9903) | 0.9967 |
| **stem** (the fix) | 0.8377 (**min −0.2496**) | **0.5815** |

Two things at once:
- The `full` row is a **pipeline validation**: re-encoding the original prompt through the
  new cached-embeds path reproduces the old GPU scorer almost exactly. So the new path
  *is* the old scorer, and any stem-vs-full difference is the bug, not new code.
- The `stem` row is **the bug's size**: the fix moves **42% of the picked frames** at 600s.
  `min = −0.2496` means on at least one item the old ranking was *anti-correlated* with the
  stem — the options were driving frame selection outright.

### Why it was wrong (evidence, not opinion)

Official AKS (CVPR 2025) `feature_extract.py`, in one file:
- CLIP branch: `text = data['question']` — question only
- BLIP branch: `text = data['question']` — question only
- SeViLA branch: question + candidates + "Is this a good frame can answer the question?"

CLIP-family scorers get the stem alone. SigLIP and LongCLIP are CLIP-family. We deviated
for exactly our model class. (SeViLA includes candidates, but it is a QA-tuned ITM head
built to reason over them — cosine similarity is not.)

---

## 2. LDDR protocol — verified from the PDF, not assumed

`research/2605.11477v1.pdf`, extracted with PyMuPDF + coordinate clustering (the raw text
order scrambles the table; column identity was resolved by x/y position).

**Subtitles OFF.** Appendix F.1, stated twice:
> "During evaluation, no additional training, subtitles, audio inputs, or external tools are used."
> "No subtitles, audio inputs, external tools, or additional training data are used during evaluation."

(Their prompt template *says* "based on the video and the subtitles" — that is the stock
lmms-eval LVB template string, not evidence subtitles are fed. They state they use default
templates.)

**All baselines standardized to LongCLIP:**
> "we standardize the pre-encoding backbone to LongCLIP for all baselines to ensure a fair
> comparison. This standardization does not disadvantage the baselines."

⇒ **`picks_lc` is our apples-to-apples arm. `picks_sig` is off-protocol vs their table.**

**Budget is token-based, not frame-based:** `Ctotal = F × 1024`; under fixed resolution
each frame costs 1024 tokens so `|S| = K = F` frames. Decode at **1 FPS**. Official
lmms-eval pipeline, greedy, temperature 0.

### Table 1 — Qwen3-VL-8B*, #F=8, LongVideoBench (the row we must beat)

| Method | 15s | 60s | **600s** | **3600s** | Overall |
|---|---|---|---|---|---|
| Uniform | 70.9 | 66.9 | 54.6 | 45.2 | 54.53 |
| AKS | 68.8 | 72.1 | 61.2 | 51.4 | 59.54 |
| Q-frame | **75.1** | 72.7 | 59.5 | 49.8 | 59.31 |
| FOCUS | 65.6 | 75.0 | 63.1 | 50.2 | 59.53 |
| MDP3 | 73.5 | 70.9 | 60.9 | 51.1 | 59.83 |
| † LD | 70.9 | 73.3 | 66.3 | 55.9 | 63.43 |
| ‡ LDDR | 72.0 | **77.3** | **67.5** | **56.2** | **64.62** |

Per-bin SOTA at #F=8: 15s = **Q-frame 75.1** (not LDDR), 60s/600s/3600s = LDDR.
Biggest selection gain is **3600s** (uniform 45.2 → LD 55.9, +10.7).

**#F=8 is the ONLY budget LDDR publishes for Qwen3-VL-8B.** Only Qwen2.5-VL-7B has a #F=32
block. So k=16/32/64 rows compare to *nothing published* — they are an internal curve.

---

## 3. The gate — matched on the checked axes, but UNDERPOWERED (revised 2026-07-16)

| axis | ours | LDDR |
|---|---|---|
| model | Qwen3-VL-8B-Instruct | Qwen3-VL-8B* |
| budget | `max_num_frames=8` | #F=8 |
| bin | `longvideobench_val_v_15s` | LVB 15s |
| subtitles | off (`val_v`) | off (F.1) |
| harness | lmms-eval | lmms-eval |

**uniform@8 15s = 0.693 vs 70.9 → −1.6pt.** Matched on every axis above.

**But the gate cannot discriminate the decode pathway, and that is a real hole.** At 15s:

| pathway | 15s uniform@8 | vs LDDR 70.9 |
|---|---|---|
| `val_v` (native video reader) | 0.693 | **−1.6** |
| `val_i` (PIL frames off the 1-fps pool) | 0.7249 | **+1.6** |

Both pass a ±2pt gate, in **opposite directions**. So passing on `val_v` was never evidence
that `val_v` is the right comparator — a ~15s clip yields ~15 candidate frames and both
pathways pick ~8 of them, so they cannot disagree much. **A gate run only in the bin where
the arms converge cannot tell them apart.** Same shape as the query bug: an unchecked
axis, agreeing for a reason unrelated to correctness.

At 60s they diverge: `val_i` uniform@8 = **0.7267** vs LDDR **66.9** = **+5.8pt**.
Not subtitles (§3b), not subsampling (§3c), and not a token overspend (§3d — measured).

**This gap is not a defect to fix.** LDDR fixes resolution and spends `Ctotal = F × 1024`;
we feed native-resolution frames. Different setup, different budget. The gate's job was to
show the harness isn't broken (the old custom mcqa harness was ~10pt off) — it did that, and
that obligation ENDS there. Our claim rests on our own uniform control at our own budget:
same pathway, same k, same prompt, only the frames differ. See
`memory/feedback_own_methodology_over_number_matching.md`.

**Prior "second check" was interpolation, not a run** (uniform@6 = 53.64, uniform@16 = 58.01
→ ~54.6 at k=8 vs their 54.6). Both endpoints are `val_i`. The real 600s uniform@8 is in flight.

`scripts/diag_pathway_60s.sh` runs `val_v_60s` uniform@8 (gate command, task swapped) — kept
as a **pathway characterisation**, not a target to hit.

### 3d. Token budget — MEASURED (2026-07-16), and it clears us

Qwen3-VL processor: `patch_size=16, merge_size=2, temporal_patch_size=2`. Tokens =
`prod(t,h,w) / merge_size²`. On 60s docs at native 1280×720:

| pathway | grid_thw | visual tokens @ k=8 | per frame |
|---|---|---|---|
| `val_i` / picks (8 PIL images) | `[1,44,80]` ×8 | **7040** | **880** |
| `val_v` (same 8 frames as video) | `[[2,44,80]]` | 1760 | 220 |
| LDDR #F=8 | fixed res, F×1024 | 8192 | 1024 |

**`val_i` spends 880 tok/frame vs LDDR's 1024 — 86% of their budget, slightly UNDER.**
The predicted 2× overspend does **not** exist: smart_resize maps 720p onto a 44×80 patch
grid, which happens to land near their fixed-resolution spend. Our k=8 is a fair #F=8-scale
budget, and if anything we under-spend — which makes our uniform baseline *harder* to beat,
not easier. No confound.

The 4× ratio is `val_i` vs `val_v`, and it explains the pathway spread: `val_v` feeds 1760
tokens where `val_i` feeds 7040. **`val_v` is the under-budget arm** — so the 0.693 gate
matched 70.9 while spending ~⅕ the tokens. That match was luck; `val_i` (7040 ≈ 8192) is the
comparable one.

⚠️ **The `val_v` row is SOFT.** The direct `video_processor` call warned *"Asked to sample
`fps` frames per second but no video metadata was provided... Defaulting to `fps=24`"* and
returned `t=2` (≈4 frames), i.e. it resampled instead of taking the 8 given frames. 1760 is
what a naive direct call yields, not proof of what lmms-eval feeds. **The `val_i` row needs
no caveat** — `doc_to_visual` hands over PIL images and 8 × (44×80)/2² = 7040 is exact.

**Reportable fact: every arm at k=8 spends 7040 visual tokens, identically.** Token cost is
constant across uniform/topk/OMP, so it cancels in every paired test.

### 3b. Subtitles: verified OFF, in the prompts (not just the config)

`utils.py:141` — `lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False)`.
Default **False**. `val_i.yaml` sets it **True**; our `val_i_*_k8` arms override to **False**;
the picks arms omit the key and inherit False. Confirmed against the logged prompts, because
a config is not an input:

```
val_i_60s_k8  mean_prompt_len=458 | picks_lc 458 | picks_omp_lc 458
val_i vs picks_lc     : 0 of 172 prompts differ
val_i vs picks_omp_lc : 0 of 172 prompts differ
```

Prompt = stem + options + answer instruction. No interleaved transcript (that path prepends
timestamped lines — thousands of chars, not 458). **Frames are the only variable across arms.**

⚠️ **Trap:** LVB question text often *says* "After the subtitle '...' appears, what shows
up..." — that is the QUESTION referencing a subtitle, present in LDDR's runs too. Grepping
`subtitle` in a prompt and concluding subtitles are on is wrong.

⚠️ **Naming:** `val_i_*_k8` is a misnomer. The `i` is `doc_to_visual_i` (the frames pathway);
subtitles are explicitly off. It is NOT the stock subtitle-interleaved split.

### 3c. We run the FULL official LVB validation set (verified by id)

Checked against `lvb_val.json` in the HF snapshot — not against our own manifest, which
cannot validate itself:

```
OFFICIAL lvb_val.json  total=1337
   15s   official=189   ours=189   missing=0  extra=0
   60s   official=172   ours=172   missing=0  extra=0
  600s   official=412   ours=412   missing=0  extra=0
 3600s   official=564   ours=564   missing=0  extra=0
```

**1337/1337, id-for-id, every bin.** No subsample. This kills subsample instability as an
explanation for any gap, and it is a real strength of the setup — every bin is the whole
official bin.

⚠️ `manifest.lvb.full1560.json` holds **361** items (15s + 60s only), not 1560. Legacy
misnomer; `long976.json` (600s + 3600s) holds 976. Together = 1337. Rename before it misleads.

---

## 4. Clean results so far

Every arm below: lmms-eval, Qwen3-VL-8B-Instruct, greedy/temp 0, **batch_size 1**, subtitles
off, frames pathway, **7040 visual tokens at k=8** (§3d), full official LVB bin (§3c).
**The control is OUR uniform at OUR budget** — same pathway, same k, same prompt, frames are
the only variable. No external comparator is needed for these to mean something.

### 600s @ k=8 (n=412 = the FULL official 600s bin) — THE FIRST SIGNIFICANT CLEAN RESULT

| arm | acc | vs uniform | McNemar (exact, 2-sided) |
|---|---|---|---|
| uniform@8 | 0.5534 | — | — |
| **topk-lc@8** | **0.6141** | **+6.1pt** | 39v64, **p=0.018** |
| **OMP-lc@8** | **0.6311** | **+7.8pt** | 36v68, **p=0.0022** |
| topk vs OMP | — | +1.7pt (OMP) | 38v45, p=0.51 **ns** |

**Selection works at 600s.** Both methods beat the matched-budget uniform control at p<0.05
on a clean query. This is where the per-bin thesis said the game would be: 8 frames out of
~600 candidates means *which* 8 finally matters.

The bin-by-bin shape is the story:

| bin | n | uniform | topk-lc | Δ | p |
|---|---|---|---|---|---|
| 15s | 189 | 0.7249 | 0.7249 | 0.0 | 1.0 |
| 60s | 172 | 0.7267 | 0.7442 | +1.7 | 0.74 |
| 600s | 412 | 0.5534 | 0.6141 | **+6.1** | **0.018** |

Selection is worthless when the candidate pool is small and pays off as it grows.

**Negative #7 survives its 4th clean test** (topk vs OMP p=0.51). But 600s is the first bin
where OMP is *directionally* ahead by a non-trivial margin (+1.7pt; and its win over uniform
is the stronger of the two). Not evidence — worth watching at 3600s, the highest-pressure bin.

⚠️ Three arms = three contrasts; treat p=0.0022 as indicative, not exact. The
uniform-vs-selection contrasts are primary and both hold comfortably.

### 3600s @ k=8 (n=564 = the FULL official 3600s bin) — THE HIGHEST-PRESSURE BIN, AND NEGATIVE #7 BREAKS

| arm | acc | vs uniform | McNemar (exact, 2-sided) |
|---|---|---|---|
| uniform@8 | 0.4716 | — | — |
| topk-lc@8 | 0.5106 | +3.9pt | 59v81, p=0.076 ~ |
| **OMP-lc@8** | **0.5461** | **+7.45pt** | 51v93, **p=0.00059 SIG** |
| topk vs OMP | — | +3.55pt (OMP) | 43v63, p=0.064 ~ |

**First bin where OMP beats top-k** (and by more than it beats it anywhere else: +3.55pt,
near-sig p=0.064). Opposite of 600s (topk vs OMP p=0.51 ns) — as the candidate pool grows
from ~600 to ~3600 frames, flat top-k starts leaving accuracy on the table that the
orthogonal-matching-pursuit pick-math recovers. Uniform→OMP is the biggest selection gain in
the whole grid (+7.45pt), matching the paper's claim that 3600s is where selection matters most
(their uniform 45.2 → LD 55.9). **Negative #7 ("no pick-math beats flat top-k") takes its first
real hit here** — downgrade from "holds" to "breaks at the longest bin, pending replication."
Note top-k alone is only p=0.076 vs uniform (underpowered / noisier picks at this length); OMP
is the arm carrying the 3600s win.

### Bin-by-bin shape, k=8 (the paper table, clean query)

| bin | n | uniform | topk-lc | Δ topk | OMP-lc | Δ OMP | best p |
|---|---|---|---|---|---|---|---|
| 15s | 189 | 0.7249 | 0.7249 | 0.0 | 0.7249 | 0.0 | 1.0 |
| 60s | 172 | 0.7267 | 0.7442 | +1.7 | 0.7384 | +1.2 | 0.74 |
| 600s | 412 | 0.5534 | 0.6141 | +6.1 | 0.6311 | +7.8 | **0.0022** |
| 3600s | 564 | 0.4716 | 0.5106 | +3.9 | **0.5461** | **+7.45** | **0.00059** |

(OMP DID run at 15s/60s — 15s ties all three arms at .7249; 60s OMP .7384 ns. No selection
benefit in small pools; earlier "—" placeholders were wrong.)

Selection is worthless in small pools (15s), pays off strongly in the two long bins, and the
pick-math (OMP) axis only starts to matter at the very longest — exactly the monotone-in-pool-size
story the per-bin thesis predicted.

### Overall k=8 accuracy across all four bins (n=1337)

Micro-average = Σ(n_bin × acc_bin) / Σ n_bin = total_correct / total_videos (weights each VIDEO
equally). Correct-counts: 15s 189 / 60s 172 / 600s 412 / 3600s 564, total 1337.

| method | correct | **micro-avg (overall)** | Δ vs uniform | macro-avg (mean of 4 bins) |
|---|---|---|---|---|
| uniform@8 | 137+125+228+266=756 | **0.5654** | — | 0.6192 |
| topk-lc@8 | 137+128+253+288=806 | **0.6028** | +3.7 | 0.6485 |
| omp-lc@8  | 137+127+260+308=832 | **0.6223** | +5.7 (+2.0 vs topk) | 0.6601 |

Overall ranking: **omp-lc > topk-lc > uniform** (OMP +5.7pt over uniform, +2.0 over top-k).

**This IS the full LVB validation set.** Official LongVideoBench val = exactly **1337 questions across
4 duration groups** (8-15s / 15-60s / 180-600s / 900-3600s; source: HF dataset card). Our total is
1337 over those same 4 groups → we ran the COMPLETE val split, not a subsample. So the micro-average
is a legitimate LongVideoBench-val overall — the bin weights ARE LVB's own distribution.
(An earlier note here wrongly called this "subset-weighted"; corrected 2026-07-17.)

Legitimate claim, axes LABELED: **LVB val (all 1337 Q), subtitles OFF, 8 frames** →
uniform 56.5 / topk-lc 60.3 / omp-lc 62.2. Not equal to subs-ON leaderboard cells (LVB is built as
video+subtitle interleaved); ours is the pure visual-selection setting. Claim still rests on our own
uniform control at our budget, not on out-numbering a paper.

Micro (per-video, above) vs macro (mean of 4 bins: uniform .619 / topk .648 / omp .660) — macro runs
~5-6pt higher because it up-weights the easy short bins. Report micro for a per-video overall (how LVB
val overall is computed), macro only if the claim is explicitly per-bin-averaged. Label which.

Residual check **DONE 2026-07-17 — EXACT MATCH**. Official `lvb_val.json` (HF snapshot
60d1c89c) duration_group = {15:189, 60:172, 600:412, 3600:564}, total 1337 — identical to ours per
group. We ran the complete, exact official LVB val set. (Also confirms our 15/60/600/3600 labels =
LVB's duration_group values.) The overall is a fully legitimate LongVideoBench-val number.

### Budget curve — k=16 / k=32 bonus arms (internal; no published Qwen3-VL-8B baseline for k≠8)

lmms-eval, LongCLIP, clean stem query, bs=1. Both 3600s multi-k arms run split across two pods by
doc-shard (hA 0/2 + hB 1/2, 282+282) and concatenated per tag before McNemar. k16 added 2026-07-17
(k16_600 full on 1 GPU; k16_3600 sharded on 2 GPU). Full long-bin budget sweep:

| bin | k | n | uniform | topk-lc (Δ, p) | OMP-lc (Δ, p) | topk vs OMP |
|---|---|---|---|---|---|---|
| 600s | 8 | 412 | 0.5534 | 0.6141 (+6.1, **.018**) | 0.6311 (+7.8, **.0022**) | 38v45, p=0.51 ns |
| 600s | 16 | 412 | 0.5850 | 0.6505 (+6.6, **.012**) | 0.6578 (+7.3, **.0023**) | 31v34, p=0.80 ns |
| 600s | 32 | 412 | 0.6044 | 0.6481 (+4.4, .082 ~) | 0.6650 (+6.1, **.0066**) | 25v32, p=0.427 ns |
| 3600s | 8 | 564 | 0.4716 | 0.5106 (+3.9, .076 ~) | 0.5461 (+7.45, **.00059**) | 43v63, **p=0.064** |
| 3600s | 16 | 564 | 0.4770 | 0.5461 (+6.9, **.0016**) | 0.5798 (+10.3, **1.8e-6**) | 39v58, **p=0.067** |
| 3600s | 32 | 564 | 0.5142 | 0.5727 (+5.9, **.0099**) | 0.5851 (+7.1, **.00076**) | 41v48, p=0.525 ns |

Findings across the full sweep:
1. **OMP-lc beats matched-budget uniform at every long-bin budget, all significant** (p from 1.8e-6
   to .0099) — the selection win is robust to k, not a k=8 artifact. Gap over uniform peaks at
   **k=16/3600s (+10.3pt)**, i.e. mid-budget, not at scarcity or abundance.
2. **Budget curve monotone but OMP saturates by k16 in the long bins.** OMP 3600s .5461→.5798→.5851
   (8→16 = +3.4pt, 16→32 = +0.5pt); 600s .6311→.6578→.6650. Uniform rises too (3600s .4716→.4770→
   .5142, 600s .5534→.5850→.6044). Most of the frame-budget value is captured by k=16.
3. **Negative #7 (pick-math OMP>topk) is a scarcity effect** — marginal at k=8 **and k=16** for 3600s
   (p=.064, .067), gone by k=32 (p=.525). At 600s it never fires (all p≥.43, incl. k16 p=.80). So
   the OMP-over-flat-topk edge shows only at the tightest budget against the largest pool (3600s),
   and only there; it washes out with slack (more frames) or a smaller pool (shorter video).

k=16/60s replicated the earlier clean run **byte-for-byte** (uniform .7093, topk .7500, OMP .7442,
topk-vs-OMP p=1.0) — determinism check on the whole pathway passed.

⚠️ mfs gotcha (2026-07-16): a guardian that runs McNemar the instant `results.json` appears can
read the shared network volume mid-write from the *other* pod — it caught k32_600 as n=328 / one
arm before the sample jsonls propagated. Gate on final line-count (n×tags), not results.json alone;
the completed files were correct (412×3).

### 60s @ k=8 (n=172)

| arm | acc | vs uniform | McNemar |
|---|---|---|---|
| uniform@8 | 0.7267 | — | — |
| topk-lc@8 | 0.7442 | +1.7pt | 16v19, p=0.74 ns |
| OMP-lc@8 | 0.7384 | +1.2pt | 11v13, p=0.84 ns |
| topk vs OMP | — | +0.6pt | 11v10, p=1.0 ns |

### 15s @ k=8 (n=189) — selection does NOTHING

All three arms **acc=0.7249**, every contrast p=1.0. The arms genuinely differ (11v11, 9v9,
3v3 disagreements) and cancel exactly. A ~15s clip yields ~15 candidates at 1 fps and any 8
of them carry the same evidence.

### 60s @ k=8 — encoder comparison, the FIRST valid one (n=172)

| arm | acc |
|---|---|
| uniform@8 | 0.7267 |
| topk-**LongCLIP**@8 | 0.7442 |
| topk-**SigLIP**@8 | 0.7093 |

| pair | Δ | McNemar |
|---|---|---|
| LongCLIP vs SigLIP | −3.5pt | 6v12, p=0.24 ns |
| uniform vs SigLIP | −1.7pt | 13v16, p=0.71 ns |

Valid as an encoder test because every arm is the same 172 docs, same pathway, same
7040-token budget, same prompt, bs=1 — **the encoder is the only variable**. The retracted
"LongCLIP ≈ SigLIP null" (§5) compared arms fed *different amounts of option text* and was
never an encoder comparison at all.

Result: **ns.** Directionally LongCLIP > SigLIP by 3.5pt, and SigLIP top-k falls *below*
uniform (its selection hurts), but nothing reaches significance — 60s is the bin where even
LongCLIP's +1.7pt is ns. **The encoder question is only answerable at 600s**, where effects
do reach significance. SigLIP 600s embeds are already cached (412/412) ⇒ CPU rescore + one
GPU arm (~20 min), no re-decode. Not yet run (SigLIP parked by decision).

### 60s @ k=16, lmms-eval, clean stem query (n=172)

| arm | acc |
|---|---|
| uniform@16 | 0.7093 |
| topk-lc@16 | 0.7500 |
| OMP-lc@16 | 0.7442 |

| comparison | Δ | McNemar | verdict |
|---|---|---|---|
| topk-lc vs uniform | +4.1pt | 8v15, p=0.21 | **ns** |
| OMP-lc vs uniform | +3.5pt | 8v14, p=0.29 | **ns** |
| **topk vs OMP** | +0.6pt | 7v6, **p=1.0** | **ns** |

Reading: 60s is a small, weak bin (n=172) — selection is directionally up but not
significant. **Negative #7 ("no pick-math beats flat top-k") survives its FIRST clean-query
test**: OMP ≈ top-k at p=1.0. The hypothesis that OMP only lost because the query was
broken took its first real hit. One bin, one k — 600s/3600s is where it should matter.

---

## 5. Retracted / void

- **All pre-fix scores, picks, and derived analysis** → quarantined to
  `.slm-lab-archive/tainted_20260715_query/` (never deleted: they are the control arm of
  the query ablation).
- **Every `gpt_mcqa.py` run** — doubly dead: fused query AND ~10pt below lmms-eval on the
  same val set. Unpublishable even after a rescore.
- **"LongCLIP ≈ SigLIP null" (3 "confirmations")** — RETRACTED. The two arms received
  *different amounts of option text*; it was never an encoder comparison. Repetition is not
  verification.
- **"rank-1 cited only 27%"** — soft; analysis of a broken scorer.
- **Negative #7** — was downgraded settled→open; now re-tested clean at 60s/k=16 and holds
  there (§4).
- **The 600s budget matrix (n=412, 6 arms)** — its *uniform* arms are clean (uniform
  consumes no scores) and reusable; its picks arms are tainted.

---

## 6. Infrastructure facts worth not re-learning

- **Image embeds were never contaminated.** `dump_embeds.py` only calls
  `get_image_features`/`encode_image`. They are the full **1-fps** pool (`eff_fps = 1.000`,
  matching LDDR's 1-FPS decode). Coverage: 15s 189/189, 60s 172/172, 600s **412/412**,
  both towers. **This is why the whole rescore is CPU/minutes, not GPU/hours.**
- Two cache layouts: `results/embeds/<qid>.npz` (keys `siglip`/`longclip`/`dinov2`) and
  `results/embeds_lc/<qid>.npz` (key `emb`, the 600s LongCLIP fill). Resolved in ONE place:
  `harness/embeds.py::load_image_embed`.
- **SigLIP ranking equivalence**: `dump_scores.py` used `logits_per_image` =
  `logit_scale*cosine + logit_bias`. For a fixed text that is strictly monotone in cosine,
  so top-k is identical. Absolute values differ, ranks do not.
- LVB reuses each 3600s video across **exactly 3 questions** (564 items / 188 videos) →
  per-qid embedding re-decodes each video 3x. `--dedup-by-video` embeds once and hardlinks
  siblings (verified: same inode, identical content).
- `harness/media.py` decodes with **cv2 sequentially** (`cap.grab()` through every native
  frame). A 3600s video ≈ 108k grabs → ~2-3 min/video, ~8h for the 3600s bin single-process.
  This, not the GPU, is the 3600s bottleneck.
- **vLLM 0.25.0 is installed in `slmenv`** and imports under the run env (torch 2.11+cu130,
  sm_120); lmms-eval registers a `vllm` backend. Switching to it would change the answerer
  backend ⇒ **the 0.693 gate must be re-run before any vLLM number is comparable.**

---

## 6b. Decoding / batch protocol (locked 2026-07-16)

LDDR App. F.1, verbatim:
> "All experiments are conducted under a unified inference setting using the official lmms-eval
> evaluation pipeline. We use greedy decoding for all open-source MLLMs, with temperature set to 0,
> top-p disabled, beam size set to 1, and a maximum generation length of 128 tokens."

| axis | LDDR | ours | status |
|---|---|---|---|
| harness | official lmms-eval | lmms-eval | match |
| decoding | greedy, temp 0, top-p off, beam 1 | `temperature: 0, do_sample: False` | match |
| max_new_tokens | 128 | **32** | **deviation — immaterial** (observed `output_tokens: 2`; the model emits only the letter). Documented, not silently ignored. |
| batch size | **not stated (0 mentions in the paper)** | **1** | see below |

**batch_size = 1, for every arm. Not negotiable, and not a speed decision.**

`--batch_size > 1` changes eval results *even at temperature 0 with greedy decoding*:
1. **Padding** — batching pads sequences to equal length; batch size and GPU count change how
   padding is handled.
2. **Float addition is non-associative** — the GPU reduces in a different order per batch shape, so
   logits move ~1e-6; greedy flips whenever two options are near-tied. Deterministic ≠ identical
   across batch shapes.

Documented, not theoretical: lm-evaluation-harness issues #2498 ("Performance significantly drop
when increase the batch_size") and #1323; batch-invariance is an open research problem (MarginGate,
arXiv 2605.30218; arXiv 2601.06118). Community rule: **bs=1 always, never `auto`, or you cannot
reproduce your own results.**

Note the gap: papers report decoding params but not batch size, so the *field norm* (batch freely)
and the *correct norm* (bs=1) disagree. We follow the correct one — we claim 1-4pt effects and match
LDDR within 1.6pt, so a "minor variation in accuracy" would sit directly on our signal.

**Rule: never MIX batch sizes across compared arms.** A shared wobble cancels in paired tests; an
inconsistent one is drift that cannot be untangled afterwards.

---

## 6b. Token budget — ALL bins (measured 2026-07-17, real videos)

Real Qwen `smart_resize` over every video at `max_pixels=1605632` (the wrapper default
our k8/k16/k32 runs used). tokens = (h_bar/28)*(w_bar/28); factor 28 = patch14 x merge2.
Sources: long976 manifest (600s/3600s), full1560 manifest (15s/60s).

**tokens/frame ≈ 1196 (median) in EVERY bin** — it's a resolution property of the LVB video
pool, not per-bin. The 2048 cap is NEVER reached (frames sit at native ~1196).

| bin | n | tok/frame median | mean | min | max |
|-----|---|------------------|------|-----|-----|
| 15s  | 189 | 1196 | 1117 | 646 | 1196 |
| 60s  | 172 | 1196 | 1096 | 646 | 1196 |
| 600s | 412 | 1196 | 1186 | 874 | 1196 |
| 3600s| 564 | 1196 | 1191 | 676 | 1196 |

Nuance: short bins (15/60) have a fatter low-res tail (more distinct resolutions, min 646)
-> their MEAN dips to ~1100 vs ~1190 long bins, so 15/60 run slightly cheaper on average.
Typical frame is 1196 everywhere. Per-item tokens = k x (that video's tok/frame); 1196 is the
median, not a fixed constant.

vis tokens/item at the typical 1196 tok/frame (same table holds for all bins):

| k | frames | vis tokens/item (@1196) | LDDR budget (F×1024) | over |
|---|--------|-------------------------|----------------------|------|
| 8  | 8  | 9,568  | 8,192  | +16% |
| 16 | 16 | 19,136 | 16,384 | +16% |
| 32 | 32 | 38,272 | 32,768 | +16% |

- Linear in k (same frames, same res) — 8->16->32 just doubles.
- **~16% richer than LDDR's stated 1024/frame budget**, but the SAME factor at every k, so
  the budget-*curve* and all OMP-vs-uniform paired tests are unaffected (identical tokens/frame
  across arms). Absolute-vs-LDDR our points sit ~1.16x over nominal F×1024 -> footnote whenever
  a number sits beside their budget. (Rule: label the budget next to a paper's number.)

## 6c. k=64 — NOT RUNNING (decision 2026-07-17)

**Decision: we are NOT doing k=64 for any bin.** Diminishing returns not worth the cost/pain.

Why it stalled (root cause, for the record): k64 @ 3600s OOMs on the 32GB RTX PRO 4500 at
item 0, inside the Qwen3-VL vision encoder (`apply_rotary_pos_emb_vision`). 64 frames x ~1196
tokens ≈ 76k vision tokens; default `max_pixels=1605632` (up to 2048 tok/frame) blows 32GB.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did NOT fix it (still OOM item 0).
- **Sharding across GPUs does NOT fix it** — OOM is per-ITEM (one 64-frame encode), not per-dataset;
  every shard/GPU OOMs identically. Confirmed on a 2nd 2-GPU pod (same 32GB cards).
- Only real lever on 32GB = cap `max_pixels`. `802816` (=1024 tok/frame, the exact LDDR budget)
  would fit BUT downscales 1196->1024, i.e. ~14% fewer tokens/frame than the k8/16/32 curve ->
  NOT apples-to-apples with our own budget curve. To run k64 cleanly would need an 80GB card
  (keep 1196/frame) or rerun the whole curve at a fixed capped resolution. Not worth it -> dropped.

---

## 7. Open

- k=8 across 15/60/600/3600 on the clean query — the paper table. In flight.
- **OMP vs top-k at 600s/3600s** — the real test of the pick-math axis.
- 3600s embeds (0/564; the 4-shard dump is the remaining GPU cost).
- Does the clean query move the 600s budget cell that came back p=0.095 on tainted picks?
- HF_TOKEN rotation — still pending since the 2026-07-14 `ps` exposure.


## 2026-07-18 — idea-1 (α partial-deflation) + idea-B (MMR) GPU results — BOTH NEGATIVE

k=8, LongCLIP, LVB val, Qwen3-VL-8B, bs=1, doc-sharded 2 pods, full paired bins (412/564).
Slot decode: ab_ablation task lc=MMR, sig=α0.5, omp_lc=α0.75 (ARM_MAP.json). α=1 gen
byte-faithful to shipped omp_indices (412/412, 564/564).

| bin | uniform | topk(α0) | α0.5 | α0.75 | OMP(α1) | MMR |
|-----|---------|----------|------|-------|---------|-----|
| 600s  | .5534 | .6141 | .6189 | .6238 | .6311 | .5898 |
| 3600s | .4716 | .5106 | .5550 | .5496 | .5461 | .4965 |

McNemar vs OMP(α1): α arms all ns. MMR SIG WORSE — 600 p=.030 (b19/c36), 3600 p=.0066 (b36/c64).

- MMR diversity-fill DEAD: below OMP AND below topk both bins. Redundant late picks not waste;
  de-dup trades relevance for diversity, answerer loses. Kill.
- α: RESULTS LOGGED, VERDICT HELD. All α arms ns vs OMP (600 p=.58/.71 monotone topk<α.5<α.75<OMP;
  3600 α.5 .5550/α.75 .5496 vs OMP .5461, p=.66/.90). Do NOT call α closed/optimal yet — revisit.
- Candidate next hypothesis = temporal (before/after); redundancy provably not the lever. Later decision.


## 2026-07-18 — SELECTION AXIS CLOSED (saturated at OMP α=1) + k16 α + visual analysis

Anti-drift ablation (rfloor@0.15/0.33, iteration-α) driven by 2 independent Sonnet-5 visual
analyses (93 cases, sonnet_analysis.md). Plus k16/3600 α check. All k=8 LongCLIP LVB val,
Qwen3-VL-8B, bs=1, sharded, McNemar vs OMP(α1). Baselines already banked.

**Full k=8 selection ladder (nothing beats OMP, all new arms ns):**
- 600s (n=412):  uniform .5534 < MMR .5898 < iteralpha .6092 < topk .6141 < rf15 .6189 < rf33 .6214 < OMP .6311
- 3600s (n=564): uniform .4716 < MMR .4965 < topk .5106 < iteralpha .5355 < rf15 .5443 < rf33 .5461 = OMP .5461  (α0.5 .5550 ns)

- rfloor ≈ OMP EXACTLY: 3600 rf33 = .5461 = OMP, b=19 c=19 (perfectly balanced swaps). Constraining
  to the relevant pool changes WHICH frames on drift cases but nets ZERO — fixes one, breaks one.
- iteration-α slightly worse both bins (ns).
- Visual finding (both agents, both bins): OMP #1 failure = drift to IRRELEVANT-but-distinctive
  frames (~half–2/3 of cases; up to 7/8 frames off-topic on hour-long heterogeneous videos). REAL
  in the frames but ACCURACY-NULL (rfloor proves it). Consistent with 53-60% same-letter fails
  ("frames don't change the answer") + 41% subtitle-cued (unanswerable, subs OFF).

**CONCLUSION: selection saturated at OMP(α1).** Entire axis tested at k8 — uniform<topk<OMP, and
α-sweep / MMR / relevance-floor(×2) / iteration-α ALL ≤ OMP. Frame-selection is a dead lever. Gains
live in scorer (LongCLIP retrieval), subtitles (41% of fails, off-protocol), or answerer.

**k16/3600 α (n=564):** topk .5461 < α0.25 .5567 < α0.5 .5638 < OMP .5798 — MONOTONE, α=1 best.
α-optimum SHIFTS UP with budget (k8/3600 peaked ~α0.5 ns; k16 monotone to α1). More frames ->
aggressive deflation pays.

**α VERDICT (was HELD 2026-07-18) -> NOW CLOSED: α=1 is the robust default.** No budget where softer
α significantly wins; the k8/3600 α0.5 +0.9pt was ns and REVERSES at k16.

Untested selection idea left = temporal-neighbor expansion (targets ~9-16% before/after, heavily
overlapping unanswerable/same-letter) — low prior given everything else nulled. Pods stopped.

## 2026-07-18 (session 3) — DPP + BUDGET COMPLETE (full 412/564, NSHARD=1, lmms-eval, LongCLIP, Qwen3-VL-8B, bs=1)
Runs survived a pod deletion (volume persisted). Doc-sharding ABANDONED — lmms-eval caches
processed docs by task-name (shard-blind) on shared /workspace/hf, both shards ran same half
(206/412, verified twice). Use NSHARD=1 full only. Numbers in results/RESULTS_SUMMARY.txt (pod).

DPP (exact conditional k-DPP, greedy log-det MAP, our wiring, fixed tokens):
  600:  OMP .6311 | DPP b1 .6456 b2 .6505(+1.9 p.37) b4 .6408  -> DPP ties OMP, best +1.9 ns
  3600: OMP .5461 | DPP b1 .5674(+2.1 p.22) b2 .5372 b4 .5426  -> DPP ties OMP, best +2.1 ns
  => SELECTION SATURATED confirmed on the field's OWN objective (LDDR/MDP3 use DPP). DPP==OMP.

ADAPTIVE-k (stop when marginal gain<tau, floor3 cap8) vs FIXED-k at matched avg budget:
  600:  adapt t.01 .5995 (meanK4.61) ~= fixed k5 .5947  (+0.5, tie)
  3600: adapt t.01 .5248 (meanK5.15) ~= fixed k5 .5230  (+0.2, tie)
  => ADAPTIVE == FIXED-k. Per-question k allocation adds nothing (gain curve near-uniform across
     questions, pre-check predicted this). Use fixed small-k; simpler + equal.

BUDGET FRONTIER (fixed-k OMP, tok_save vs k8):
  600:  k8 63.1 | k6 60.9(-2.2 p.12) | k5 59.5(-3.6 p.024*) | k4 59.0(-4.1 p.019*)
  3600: k8 54.6 | k6 53.0(-1.6 ns)   | k5 52.3(-2.3 ns)     | k4 52.0(-2.7 ns)
  => DURATION-DEPENDENT COMPRESSIBILITY: 3600 tolerates 25-50% token cut nearly free (all ns);
     600 degrades significantly by k5. Long videos = more redundant = more compressible.

VS LDDR (paper, Qwen3-VL-8B, LVB; their selection relevance = MinMax-cosine, SAME as ours; GD is
dynamic-res token allocation only): ours behind their LD (linear-DPP fixed-res) by ~1-1.3pt at
matched setup (600 DPP 65.1 vs LD 66.3; 3600 OMP 54.6 vs LD 55.9). Gap = pipeline (fps pool /
tokens-per-frame), NOT method (we matched DPP). Uniform matches theirs (55.3 vs 54.6) = harness OK.
Improve-OMP directions in lever1_spec.md (r-normalization MinMax, 2fps pool, non-CLIP scorer).

## 2026-07-19 — Lever-1 Arm A: MinMax r-norm DPP (NEGATIVE)
Tests whether LDDR's exact quality normalization (r_i = MinMax(cos), no temperature) — vs our
banked z-DPP r_i = exp(beta*z(cos)) — closes the k=8 gap to LDDR-LD. Selector-only swap, same
wiring (LongCLIP, 1fps pool, fixed tokens, greedy log-det MAP, k8). Full coverage n=412/564.
Script scripts/gen_dpp_minmax.py + run_dppmm_bin.sh; finalize_dppmm.py.

| bin | MinMax-DPP | banked OMP | best z-DPP | LDDR-LD |
|-----|-----------|-----------|-----------|---------|
| 600  | 0.6286 | 0.6311 (-0.24, p=1.00) | 0.6505 (b2) | 0.663 |
| 3600 | 0.5674 | 0.5461 (+2.13, p=0.18) | 0.5674 (b1) | 0.559 |

VERDICT — NEGATIVE. MinMax lands in the same saturated cluster (0.629-0.651 @600, 0.567 @3600).
At 600s it is WORSE than z-DPP best and still -3.4pt vs LDDR-LD; at 3600s it exactly equals z-DPP
b1 (+0.00). The quality-weighting SHAPE (MinMax vs z/exp) is NOT the source of the LDDR gap. This
is the THIRD selection-side negative: (1) OMP=z-DPP objective, (2) adaptive-k=fixed-k allocation,
(3) MinMax=z/exp relevance shape. The residual -3.2pt @600 is upstream of the selector -> pool
density (Arm B, 2fps) or input-detail confound (their fixed-res tokens/frame), NOT selection math.

## 2026-07-19 — SHARDING CORRUPTION AUDIT (critical)
Coverage audit (unique doc_ids vs official per-bin counts) of EVERY banked lmms-eval run.
Script scripts/audit_coverage.py + scripts/cov_gate.py. Sharding bug = shards ran the SAME
subset (all N shards identical), merged to full LINE count but HALF unique -> silent corruption.

### CORRUPT / VOID (sharded, incomplete unique coverage)
| run | method | bins·k | unique/exp |
|-----|--------|--------|-----------|
| ab_ablation   | MMR hybrid + alpha0.5 + alpha0.75 | 600·k8, 3600·k8 | 206/412, 282/564 |
| afix_ablation | rfloor0.15 + rfloor0.33 + iteralpha | 600·k8, 3600·k8 | 206/412, 282/564 |
| k16_3600 (ak16_ablation, matrix merged) | uniform/topk/OMP @k16 | 3600·k16 | 141 & 282/564 |
| k32_3600 (hA/hB/merged) | uniform/topk/OMP @k32 | 3600·k32 | 282/564 |

VOIDED CLAIMS: OMP@k16/3600 .5798 + k32/3600 .5851 (both invalid); "OMP peaks +10.3 over
uniform at k16/3600" and "saturates by k16" (rested on corrupt k16/k32); MMR/alpha/rfloor/
iteralpha accuracy numbers; sonnet_analysis_3600_k16.md (ran on corrupt 282-subset).

### VALID (NSHARD=1 full, exact coverage) — all STAND
All k8 (15/60/600/3600 incl. overall micro-avg + per-bin gate); k16_600, k16_60; k32_600;
dpp_ablation, dppmm_ablation, adapt_ablation, ksweep_ablation (both bins). **adaptive-k is
VALID** (adapt_ablation 412/564 full) — NOT corrupt.

### RE-RUN (NSHARD=1, coverage-gated, 2026-07-19)
- VALID k16/3600 pair (n=564 both): **OMP .5816** / LDDR-MinMax .5745 (diff -0.71, p=.75 = tie).
  Corrupt .5798 happened to sit near true .5816 (282-subset was representative) -> conclusion
  held but is now VERIFIED. OMP ties LDDR-selection at k16 just as at k8.
- IN FLIGHT: uniform+topk @k16/3600 (rebuild void budget row); lean valid re-run of the
  CONTENDER variants only — alpha0.5, alpha0.75, rf15(rfloor0.15), rfloor(0.33). SKIPPED
  mmr + iteralpha (rough corrupt-subset acc clearly <= OMP both bins; no precise number needed
  for a "does not beat OMP" negative). k32/3600 skipped per user.
- Rough corrupt-subset acc (triage only): mmr 600/.563 3600/.504 (worst); iteralpha .612/.539;
  alpha0.5 .597/.571; alpha0.75 .617/.560; rf15 .626/.557; rfloor0.33 .641/.560 (nominally beats
  OMP -> MUST verify). OMP valid ref .6311/.5461.

FIX: cov_gate.py hard-asserts unique doc_ids == full pool per arm; run_valid_arms.sh /
run_k16rebuild.sh force NSHARD=1. No .done marker unless coverage complete -> bug cannot ship.

## 2026-07-19 (session 4, PM) — ALL re-runs LANDED, complete valid matrix

Every voided variant re-run finished NSHARD=1 + cov-gated (9/9 .done, exact per-bin unique
counts). Pods stopped after. Numbers below are DEFINITIVE valid, supersede all corrupt-subset
triage in the RE-RUN block above. (mmr + iteralpha WERE re-run too, despite earlier "skip".)

### Variant table @k8 (all coverage-gated: 600s n=412, 3600s n=564)
| method       | 600s  | 3600s | vs OMP 600 | vs OMP 3600 |
|--------------|-------|-------|-----------|------------|
| OMP-lc       | .6311 | .5461 | —         | —          |
| alpha0.5     | .6214 | .5585 | -0.97     | **+1.24**  |
| alpha0.75    | .6262 | .5496 | -0.49     | +0.35      |
| rfloor0.33   | .6238 | .5496 | -0.73     | +0.35      |
| rf15 (0.15)  | .6189 | .5426 | -1.22     | -0.35      |
| iteralpha    | .6068 | .5284 | -2.43     | -1.77      |
| mmr          | .5947 | .4965 | -3.64     | -4.96      |
| uniform ref  | .5534 | .4716 | -7.8      | -7.45      |

READS: (1) NO variant beats OMP at BOTH bins. alpha0.5 is the ONLY nominal cross (+1.24 @3600
but -0.97 @600) -> mixed sign = wash; McNemar pending but ~7 videos, near-certain ns. (2) MMR
(diversity-first) worst real method — 3600 .4965 barely over uniform .4716 (+2.5) — collapses
long. (3) Whole variant band sits at/under OMP -> selection-side FULLY saturated (4th+ negative:
DPP=OMP, adaptive=fixed, MinMax=OMP, now alpha/rfloor/mmr/iteralpha<=OMP).

### k16/3600 budget row REBUILT (n=564, valid, replaces void)
| method       | 3600s@k16 |
|--------------|-----------|
| OMP-lc       | **.5816** |
| LDDR-MinMax  | .5745     |
| topk-lc      | .5532     |
| uniform      | .4770     |
OMP tops the budget row; ties LDDR-MinMax (-0.71, p.75). topk .5532 (prior corrupt .5461).

### LDDR appendix (A-F) confirmations folded in
- App B limitation (verbatim): "LDDR relies on text-visual features from external encoders, so
  its performance can be affected by the quality of that external module." = OUR Lever-2 thesis,
  authors' own words. Encoder (LongCLIP) IS the named ceiling.
- App D proves kernel-free greedy DPP == greedy MAP on L~=diag(r)S diag(r) == Gram-Schmidt
  orthogonalization of query-weighted features phi=r*fhat. Their "Global Linear DPP" IS OMP
  (r-weighted). Explains our DPP==OMP empirical tie at THEORY level; kills "novel selector" framing.
- App C Table 5: their default density-prior tau=1 gives LVB-3600 55.85, but tau=0 (prior OFF)
  = 57.09 -> their own sophistication LOSES 1.2pt at 3600. Selection extras hurt long.
- App F.1/F.2: official lmms-eval, greedy temp0 topp-off beam1 max_gen128, 1fps, no subs; budget
  F*1024, fixed-res 1024/frame; single A6000; LongCLIP for their method + ALL baselines; Qwen3-VL-8B
  in backbone list. F.2: candidate pool = 1fps SHARED across methods, "differences come from
  selection not pool construction." -> Arm B (denser 2fps pool) is DEAD by their design; strike it.
  Only unverified protocol axis vs our gate = max_gen (they use 128).

### Lever-1 spec status
- Arm A (MinMax r-norm): NEGATIVE (done, ties OMP both bins).
- Arm B (2fps pool): KILLED by App F.2 (both 1fps, pool held constant) — remove from spec.
- Item 3 (token/res confound): still measure-only; pull their fixed-res token count from
  github.com/JingfengSteven/LDDR.
- Only live lever left = Lever 2 (non-CLIP / VLM-as-scorer), now backed by App B.

Deliverable: results_mega_table.html (tabbed by k, all methods x bins, blanks where not run).

## 2026-07-19 — router/oracle CPU analysis (k=8, zero GPU) — NEGATIVE #8 + oracle map

Script `scripts/router_oracle.py`, run on CPU pod against `lmmseval_matrix_clean` k8 outputs.
Verification gates passed: exact official n per bin (189/172/412/564), id sets identical across
arms, recomputed accuracies == mega table to 4dp. Output: `results/router_oracle_k8.json`.

**NEGATIVE #8 — pre-registered temporal-keyword router FAILS.** Rule (fixed before scoring:
stem-only, before/after/first/immediately/when+says -> topk, else OMP) scores BELOW plain OMP
everywhere: 600s .6238 vs .6311, 3600s .5284 vs .5461, overall .6103 vs .6223 (all McNemar ns,
best p=.12). Temporal keywords do not identify where topk wins.

**Oracle union (topk∪omp) large but per-item, not per-category:**
- 600s .7233 (+9.2 over OMP) · 3600s .6223 (+7.6) · overall .6933 (+7.1); 3-arm oracle .7457.
- Gold-category routing ceiling (LEAKAGE diagnostic, in-sample): 600s .6553 (+2.4), 3600s .5674
  (+2.1). So even PERFECT category knowledge captures only ~30% of union headroom -> no text/
  category router can reach the oracle; the discordance is item-level.
- Category winners FLIP between bins (E3E, T2O, S2E topk-favored at 3600 but omp-favored or tie
  at 600; SSS omp +9 at 3600 but topk +1 at 600) -> category signal unstable, routing fragile.
- Rescue structure symmetric: 600s topk-rescues-omp 38 / omp-rescues-topk 45; 3600s 43/63.

**Honest read:** big oracle headroom is partly answerer knife-edge noise (visual analysis found
near-identical frame sets flipping answers: d5JlCEDlHGE_1 5/8 shared timestamps, gtX_oRpLClY_0
6/8 shared), so +7-9pt oracle is an inflated upper bound; the stable routable part is the ~+2pt
category ceiling, which leakage-free routing can't fully reach. Selection-side conclusion
unchanged: single-arm pick-math closed; routing closed (this entry); remaining frame-set-level
untested arm = hybrid 4+4 (dense topk anchors + OMP spread in ONE frame set), which attacks the
per-item discordance directly but needs GPU and carries the knife-edge caveat.

**2026-07-19 DECISION: hybrid 4+4 CLOSED WITHOUT RUN** (user + analysis agree). Rationale:
(1) MMR — the continuous relevance/diversity blend of the same idea — was the WORST k8 variant
(.5947/.4965, below plain topk); (2) rescues are symmetric, and 4+4 halves both mechanisms
(4 dense anchors < 8, 4 spread picks < 8) so expected landing is between arms, not above;
(3) oracle headroom is mostly answerer knife-edge noise (~+2pt stable ceiling, leakage-only).
=> **SELECTION AXIS AT k=8 FULLY CLOSED**: pick-math (10 variants, all in band or below),
text routing (NEGATIVE #8), hybrid frame-set mixing (closed on prior). Remaining levers are
NOT selection: subs-ON arm, Lever 2 (VLM-as-scorer), token-budget/compression framing.

## 2026-07-19 — cross-budget McNemar (CPU, local jsonls) — TOKEN-REDUCTION CLAIM

Script `scripts/cross_budget_mcnemar.py`, output `results/cross_budget_mcnemar_k8.json`.
Pairing exact (same official val items per bin across budgets). Gates: exact n + unique ids
+ accuracy==mega-table 4dp per file. Gate CAUGHT stale data: `k16_3600_merged` topk arm
scores .5461 != mega .5532 (void sharded remnant; valid topk re-run not local) -> excluded;
uniform + OMP merged arms verified clean.

**HEADLINE: "half the visual tokens, still significantly better than uniform" at BOTH long bins:**
- 3600s: OMP@8f .5461 vs uniform@16f .4770 = **+6.9pt at 1/2 tokens, p=.0011** (88/49 discordant)
- 600s: OMP@16f .6578 vs uniform@32f .6044 = **+5.3pt at 1/2 tokens, p=.018** (51/29)
Supporting (parity-or-better, ns):
- 600s: OMP@8f .6311 vs uniform@16f .5850 (+4.6, p=.067) and vs uniform@32f .6044 (+2.7, p=.29)
  -> at 1/4 tokens OMP@8 is never worse than uniform@32, direction positive
- 60s: OMP@8f .7384 vs uniform@16f .7093 (+2.9, p=.42)

Framing rule (per methodology memory): token-reduction claim vs OUR uniform control, NOT vs
LDDR — LDDR published only F=8 for Qwen3-VL-8B (no lower-budget row to undercut), and at
3600s their F=8 (55.85) is above our OMP@8 (.5461), so "beat LDDR at fewer tokens" is NOT
supported; at 600s we're above their published F=8 (63.11 vs 60.68/61.65) at matched frame
budget (+16% vis-token footnote applies).

## 2026-07-19 — OMP residual-query trace (CPU, full pools) + NEGATIVE #9 (residual-plateau stop)

### Residual trace, k=8->16 (scripts/omp_residual_trace.py, both bins full: 3600 n=564, 600 n=412)
Instrumented textbook OMP per video, tracked query residual each pick. Both bins agree to 3dp.
Numbers (3600s; 600s identical):
| pick | cos_resid (rel to RESIDUAL, what OMP maxes) | cos_orig (rel to ORIGINAL q) | resid_frac ||q_t||/||q_0|| |
|------|------|------|------|
| 1  | 0.233 | 0.233 | 0.972 |
| 8  | 0.004 | 0.203 | 0.967 |
| 16 | ~0.000 | 0.199 | 0.967 |

Reads:
1. **Query-reachable subspace is TINY** — OMP ever explains only ~3.3% of query norm; residual
   flat 0.972->0.967 and STOPS. Picks 9-16 drain **0.0pt** more. This is the **CLIP modality gap**:
   LongCLIP text query sits ~97% orthogonal to the entire image-embed subspace (no frame combo
   reaches it). cos_orig maxes ~0.23, never climbs.
2. **Residual drained (in relevance direction) by pick ~5** — cos_resid collapses 0.233->0.004
   by pick 8, ~0 by pick 16. After ~pick 5 every remaining frame is orthogonal to what the
   residual still wants -> objective is FLAT -> this is WHY every selector (OMP/DPP/MMR/…) ties
   (mechanism behind the saturation already observed empirically).
3. **BUT late picks stay individually query-relevant** (cos_orig ~0.20 flat) — they are
   relevant-but-REDUNDANT (new frame, same query direction already covered). Diversity, not new
   query signal. Explains why k8->k16 gives a small real bump (redundant coverage = more chances
   answerer sees needle) and 16->32 saturates (residual fully drained by 16).
-> strongest mechanistic case for **Lever 2**: the ceiling is the ~3.3% reachable query mass
   (modality gap), NOT pick-math. A non-CLIP VLM-scorer (cross-attention, no dual-encoder gap)
   has real residual to drain across more picks and can exceed cos 0.23. Caveat: cos~0.20 still
   RANKS fine (top-k works); the point is the residual OMP orthogonalizes against is exhausted by
   ~pick 5. Saved: results/omp_resid_{600,3600}.json.

### NEGATIVE #9 — residual-plateau early-stop (adaptive-k by residual saturation) LOSES
Idea: stop OMP when ||q_res|| stops moving (delta<1e-3 for `patience` consecutive steps); k =
whatever it lands on. Variant2 (floor at min-8) reproduces k8 exactly (552/564 identical picks,
plateau always fires <8) -> skipped. Variant1 (plain plateau) ran on 3600s, NSHARD=1, cov-gated
564, Qwen3-VL-8B bs=1 (scripts/gen_plateau_picks.py + run_plateau_3600.sh, k-agnostic task
longvideobench_val_picks_omp_lc_3600s reading $LVB_PICKS_OMP_LC):
| arm | mean k | vis-budget vs k8 | acc | vs k8 .5461 | McNemar |
|-----|--------|------------------|-----|-------------|---------|
| plateau p2 | 4.47 | 56% | .5124 | **-3.37pt** | b29/c48, **p=.040 SIG** |
| plateau p3 | 5.48 | 68% | .5160 | **-3.01pt** | b22/c39, **p=.040 SIG** |
Both SIG worse; consistent with existing adaptive-k (.5248, mean 5.15). **Insight: the selector's
own residual-saturation is NOT a valid stopping rule for the ANSWERER.** Picks 6-8 that plateau
drops are relevant-but-redundant to CLIP (cos_resid~0) yet the VLM still uses them (k8 wins 48
videos p2 would've gotten). Query-coverage in CLIP space != evidence the VLM needs. You cannot
cut budget by watching OMP's residual — the drained tail still pays off downstream. **k=8 flat
beats residual-adaptive stopping.** Picks banked: results/picks_lmmseval/picks_omp_lc_3600_plateau_p{2,3}.json;
samples: results/plateau_3600/p{2,3}/.

## 2026-07-20 — PATH 3 (multi-facet query decomposition) — NEGATIVE #10, killed on CPU, zero GPU

Spec `path3_spec.md`. Hypothesis: the residual trace says one stem vector reaches the image
subspace only to cos .233; LVB stems are compound (scene + subtitle anchor + interrogative,
median **44 words**), so decomposing into facets (median **8 words**) should open several
distinct reachable subspaces. Attacks the query side, not the pick geometry — the first
direction that changes q rather than the pick rule.

Pipeline: `gen_facets.py` (GPT-4o-mini, 976/976, 0 fails, mean 3.39 facets) ->
`dump_facet_embeds.py` (LongCLIP-L, stem re-encode asserted vs cached text_lc, worst
cos=1.00000) -> `facet_diagnostic.py` -> `gen_facet_picks.py`.

### Stage C geometry — gates PASSED
| | 600s (n=412) | 3600s (n=564) |
|---|---|---|
| facet-facet cos | .650 | .655 | (not a cosmetic split) |
| stem max cos | .2312 | .2327 |
| best facet max cos | .2581 | .2641 |
| reach gain | +.0269 | +.0314 | (+12-13% relative) |
| top-1 frame moved | 95.4% | 96.6% |
| explained mass k=8 | 3.18 -> 3.38% | 3.28 -> 3.60% |

Facets DO reach further and DO move picks (73-80% of frames moved, vs the 42% reference
from the query-bug fix). Explained mass barely moves: **the modality gap is structural to
CLIP, not an artifact of compound queries.**

### But every facet is WORSE than the intact stem (top-1 in gold window)
| | stem | action | scene | subject | temporal | interrogative |
|---|---|---|---|---|---|---|
| 600s | **14.3%** | 11.9 | 9.7 | 8.4 | 7.5 | 6.5 |
| 3600s | **9.6%** | 4.3 | 5.7 | 3.2 | 2.7 | 2.1 |

The *interrogative* facet — predicted to carry the answer signal — is the **worst**. So the
recall loss is NOT a budget-allocation flaw, and **type-weighted allocation is ruled out
before building it** (no type to up-weight beats the stem).

### Conjunctive recombination also fails (pre-registered kill)
| op | 600 top1 | 600 hit@8 | 3600 top1 | 3600 hit@8 |
|---|---|---|---|---|
| stem | **14.3** | 31.8 | **9.6** | **20.4** |
| min | 9.5 | 24.8 | 4.6 | 15.8 |
| mean | 11.4 | 30.8 | 7.1 | 18.6 |
| mean+stem | 12.9 | 32.5 | 7.3 | 20.2 |
| min+stem | 12.9 | **33.7** | 7.1 | 19.1 |

min+stem's +1.9pt hit@8 at 600s is contradicted at 3600s (-0.2pt) => noise. Pre-registered
rule (no operator beats stem top-1 => dead, no GPU) fired. **No GPU spent.**

### The finding (inverts the hypothesis)
**The long LVB stem is a CONJUNCTIVE FINGERPRINT, not a diluted average.** "green lake AND
small building AND light yellow AND two windows" jointly identifies one frame; each fragment
alone is generic and matches hundreds. This explains the stage-C paradox directly: facets
achieve *higher* max cosine (+.03) on *worse* frames — higher similarity, wrong frame.
Decomposition destroys the conjunction that makes retrieval work.

Corollary: the problem is not the query representation. Two axes are now closed with a
measured mechanism (selection geometry, query decomposition), both pointing at the same
place — the CLIP dual-encoder itself. **Lever 2 (non-CLIP VLM scorer) is the only remaining
lever**, and this result strengthens its motivation.

Caveat on the metric: gold-evidence recall is a HOSTILE proxy (near-duplicate frame outside
the window counts as a miss) and is used here only as a *relative* signal between arms
measured identically. It is not a paper number. The kill rests on it being negative on both
bins, for every facet type, and for every recombination operator.

Hallucination guard worth keeping: the decomposer invented scenes for stems with no visual
content ("Which of the following sequences of scenes is correct?" -> "a bustling city
street"). Ungrounded facets DO move picks and would pass a naive pick-overlap kill-switch
while pointing at random frames. `gen_facets.py::grounding` scores lexical overlap with the
stem; facets < 0.5 are dropped (61 of 3313). Any future query-rewriting work needs this.

### Ops note — eval env was touched
CPU pod had no torch; CPU torch+torchvision were installed into **/workspace/lmmsenv**
(the shared eval env) to run the LongCLIP text tower, then uninstalled — lmmsenv is back to
no-torch and `import lmms_eval` verified OK. GPU runs were never at risk (they get torch
from slmenv via PYTHONPATH, which precedes site-packages). **But collateral upgrades were
NOT reverted and prior versions were not recorded**: pillow 12.2.0, setuptools 78.1.0,
Jinja2 3.1.6, MarkupSafe 3.0.3, fsspec 2026.4.0, networkx 3.6.1, sympy 1.14.0, mpmath 1.3.0.
Lesson: install to a scratch dir on local disk, never into lmmsenv.

## 2026-07-20 — ST-OMP (temporal-trajectory S-OMP) — NEGATIVE #11, killed on CPU, zero GPU

Proposal (external): replace query-reconstruction OMP with Simultaneous OMP over a
query-gated velocity matrix. v_t = f_{t+1}-f_t; w_t = max(0, v_t.q); Y_t = w_t*v_t;
S_i = sum_t (R_t.f_i)^2, Gram-Schmidt out of ALL rows. Motivation given was "residual
collapse" in vector OMP — **already measured false** (residual keeps 96.7% of norm at
k=16). Script `scripts/stomp_diagnostic.py`.

### Results (full pools, both bins)
| | 600s n=412 | 3600s n=564 |
|---|---|---|
| gate ratio true/mismatched query | 1.301 | 1.233 |
| gate size vs per-frame \|f.q\| | 4.1% | 3.4% |
| query-swap overlap, ST-OMP | 20.6% | 17.1% |
| query-swap overlap, cosine top-k (control) | 3.7% | 2.5% |
| gold recall@8, stem-OMP | **32.8%** | **16.5%** |
| gold recall@8, ST-OMP | 24.5% | 8.0% |
| delta | **-8.3pt** | **-8.5pt** |

### Reads
1. **The query barely enters.** w_t = v_t.q = (f_{t+1}.q) - (f_t.q) is a difference of two
   small similar image-text cosines (those max at .233). Result is 3.4-4.1% the magnitude
   of the raw frame-query signal, and only 1.23-1.30x above its mismatched-query null.
2. **ST-OMP is substantially query-BLIND.** Fed another video's question it keeps 17-21% of
   its picks; cosine top-k keeps 2.5-3.7%. So 5.5-6.8x more query-independent than the
   simplest possible baseline. It is largely a motion/shot-boundary detector in OMP notation.
3. **And those motion frames are WORSE.** At 3600s it recovers less than half the gold
   evidence stem-OMP does (8.0% vs 16.5%). Query-blind AND worse.

### Process note — a 40-video smoke lied
Smoke n=40 (600s) gave ST-OMP 35.0% vs stem 27.5% = **+7.5pt**, i.e. the opposite sign.
That was 14 videos vs 11. Full n=412 gave **-8.3pt**. Flagged as noise-level when reported;
acting on it would have bought a GPU arm for a method that loses by 8pt. **Never act on a
40-video smoke** — this is the same class of error as the subsample-instability worry in the
thesis-direction notes.

### Novelty note (independent of the numbers)
This is **S-OMP (Tropp et al. 2006, "Algorithms for simultaneous sparse approximation")**
applied to ReLU-gated velocity vectors; motion/temporal-difference keyframe selection is
long-standing in video summarization. The proposal's claimed O(KTd) runtime is also wrong —
`R @ E.T` is O(T^2 d). Identical scores come from S_i = f_i^T (R^T R) f_i at O(T d^2), which
is what was implemented.

### What it DOES buy
Shuffle-invariance was the one legitimate critique in the external teardowns (our selector is
permutation-invariant; shuffle the video, identical picks). We now have a measured answer:
a temporal-trajectory selector was built and it LOSES by ~8pt on both bins. That closes the
objection with evidence instead of a hand-wave, and is a real section for the analysis paper.

**Selection axis remains closed. Query axis closed (NEG#10). Temporal axis now closed
(NEG#11). All three point at the CLIP dual-encoder. Lever 2 is the only remaining lever.**
