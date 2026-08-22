# Handoff — paper session 2026-08-21

Two files now. **`paper/arxiv.tex` is the source of truth** (16pp, post to arXiv as-is).
`paper/acl.tex` is derived from it for ARR and differs in **length only**.

Pre-session snapshots: `paper/archive/arxiv.tex.prerewrite` (before the day's rewrite),
`paper/archive/arxiv.tex.pre-review-2026-08-21` (before the review pass).

Every edit site in `arxiv.tex` is greppable: `grep REV-2026-08-21 paper/arxiv.tex` (14 sites).
Every ACL cut site: `grep ACL-TRIM paper/acl.tex`. Full revision log is the comment block
at the top of `arxiv.tex`.

---

## 1. Experiments run today (all on Modal profile `p-khatri`, ~$19 total)

App: `slm-lab/modal_siglip.py`. Volume `slm-lab-sig`. Stages are separately invokable:
`gate` -> `picks` -> `stage` -> `smoke` -> `run_all`/`run_missing` -> `audit_results` -> `analyze`.

### Item 1 — SigLIP scorer ablation. DONE. Ranking survives.

600s (n=412), all five arms re-run together in ONE environment, paired on identical ids:

| arm | acc | vs uniform | p |
|--|--:|--:|--:|
| uniform | .5534 | — | — |
| top-k LongCLIP | .6068 | +5.34 | .039 |
| OMP LongCLIP | .6311 | +7.77 | .0024 |
| top-k SigLIP | .6068 | +5.34 | .033 |
| OMP SigLIP | .6408 | +8.74 | .00026 |

`uniform < top-k < OMP` under BOTH scorers, despite OMP sharing only 1.32/8 frames
across scorers (84% of answerer input differs, 7.1x above chance). 60s bin: all null.

**GUARD:** do NOT claim "OMP prefers SigLIP". OMP-vs-topk is sig under SigLIP (+3.40,
p=.049) and ns under LongCLIP (+2.43, p=.33), but the interaction is p=.68. See
[[R2]] below — the interaction p was WRONG in the first draft.

Coverage gate result (why 600s+60s only): siglip coverage is 15s 189/189, 60s 172/172,
**600s 412/412**, **3600s only 33/564**. 3600s needs 531 GPU re-encodes -> skipped.

### Item 2 — published vs matched. DONE, then CORRECTED. See §3.

### Item 3 — GPU split. CLOSED, zero GPU. Was a reporting bug, not a missing run.
All four 2x2 arms already existed in one environment on Azure. Clean interaction
**+1.43 pt, p=.2614** (was +1.02, p=.4435 when computed across environments).
Recompute: `slm-lab/scripts/interaction_same_env.py`.

### Item 5 — prompt boundary. DONE. Direction is OPPOSITE to what the paper claimed.

Fused (question+options) vs stem-only, LVB-600s n=412, paired against stem arms in
the same environment:

| scorer | selector | stem | fused | delta | p |
|--|--|--:|--:|--:|--:|
| LongCLIP | top-k | .6068 | .6408 | **+3.40** | .087 |
| LongCLIP | OMP | .6311 | **.6699** | **+3.88** | **.033** |
| SigLIP | top-k | .6068 | .6117 | +0.49 | .87 |
| SigLIP | OMP | .6408 | .6529 | +1.21 | .50 |

Options in the scorer **HELP**. Pick divergence: LongCLIP top-k 41.9% (this is the
paper's old "~40%"), LongCLIP OMP **53.3%**, SigLIP 30.2%/35.0%.
"Could reverse the ordering between rules" was FALSE — OMP beats top-k under every
query and scorer — and is deleted.

**SigLIP fused numbers are deliberately OUT of the paper**: SigLIP caps at 64 text
tokens and saw only 63% of the fused string (93% of items truncated). Its near-null
is a weaker intervention, not scorer robustness.

---

## 2. Five-seat review panel (all returned MAJOR REVISION)

Configured seats: Journal-Fit (area chair), R1 Methodology, R2 Domain, R3 Perspective
(measurement/evaluation), Devil's Advocate. Each committed without seeing peer output.

**Venue read (Journal-Fit):** TMLR = Accept/Minor, **best fit** — its rubric is
claims-supported-by-evidence with novelty explicitly out of scope. Findings =
Accept/Minor. ACL main = Major Revision; soundness clears the bar, excitement as
framed does not.

**Strongest reject risk identified:** a *circularity* charge, not novelty —
"you standardize everything to a LongCLIP stem scorer at k=8, then conclude subset
rules matter little, having stripped each method of the component carrying its gain."
The scorer swap partially answers it; the **budget axis does not** (k=8 is the
scarcity regime most favorable to the headline).

**Two real errors the panel caught — both mine, both introduced this session:**
1. The LDDR published-vs-matched row used the wrong cell (see §3).
2. A p-value came from a two-arm test where the sentence needed a four-arm one (R2).

**Consensus items across seats:** duplicate sentence (4/5); cost ledger excludes
selection compute (2 independently); missing discordant counts; scorer-invariance
overclaimed; Table 7 does what the paper condemns (4/5).

**Strengths the panel said to protect:** the refusal to over-read the sig-vs-ns
subgroup pair (§5.2 + App B); TOST including the margin that FAILS (±2, p=.054);
the token audit showing the winning arm is cheaper; the quantified two-environment
drift disclosure; FOCUS* starred with the boundary stated three times.

---

## 3. The LDDR correction (most important single fix)

**The old table row was wrong on three axes.** It cited +3.92 = Qwen2.5-VL-7B / 32F /
**full** LDDR (stage1+2). The matched cell is Qwen3-VL-8B / **8F** / **stage-1 †LD** /
LongCLIP — verified from `video-understanding/2605.11477v1.pdf`, Table 1, page 6.

| benchmark | LDDR published (†LD, 8F) | ours | delta |
|--|--:|--:|--:|
| Video-MME | 57.19 -> 63.15 = **+5.96** | +5.56 | **-0.40** |
| LongVideoBench | 54.53 -> 63.43 = **+8.90** | +6.66 | **-2.24** |
| LVBench | 28.08 -> 43.06 = **+14.98** | +12.39 | **-2.59** |

We are **BELOW** on all three, not above. "Exceeds its published gain ... at a quarter
of the frames" was false; the comparable cell is 8 frames vs 8 frames.

**Reframed (Prakhar's call) as a cross-harness disagreement measurement**, which is
stronger and true: two carefully controlled harnesses running the same published rules
at the same budget with the same encoder differ by **0.4 to 3.8 points** — a range
covering most of the gaps published selector comparisons are built on. Uniform
baselines also differ (LVBench 28.08 vs our 34.54), so headroom differs too.

**Do NOT claim a "native scorer reproduces better" pattern** — it does not hold
(AKS is closer than LD on LongVideoBench despite a substituted scorer).

Table build script: `slm-lab/scripts/published_vs_matched.py` (carries the
three-axis-confound limitation in its output; do not remove it).

**Also from that same table, still open:** their matched AKS is +4.29 on Video-MME
against our +1.48; matched FOCUS +1.70 against our -0.59; and **MDP3 and Q-Frame are
in their comparison and absent from ours**.

---

## 4. Applied this pass (R1-R14, all DONE)

R1 LDDR table rebuilt · R2 p corrected .73 -> .68 (four-arm test) · R3 duplicate
sentence deleted · R4 cost claims scoped to "answerer visual tokens" with the
uncounted selection cost stated · R5 scorer-invariance softened everywhere incl.
the section heading, CI [-3.6,+5.6] given · R6 discordant counts added · R7 AKS
marked scorer-substituted, LDDR-select's native-encoder advantage disclosed ·
R8 modality gap cited (liang2022modality, added to bib) and the false geometric
claim fixed · R10 Claim B repositioned — **LDDR Table 2b already reports the
resolution-for-frames tradeoff** (1024 tok/frame 59.76 -> 512 tok/frame 61.03 ->
256 tok/frame 59.31, verified p7); ours is now an audited/paired/selector-agnostic
replication · R14 the "room between rules is smaller than uniform->query-aware"
claim FAILED on Video-MME and is now bounded.

Also: AKS bib author fixed (Tian, Yunhong -> **Yunjie**), arXiv:2502.21271 added.

---

## 5. OPEN — next session

| id | item | cost |
|--|--|--|
| R9 | MDP3 + Q-Frame absent from our table, present in LDDR's | ~$3, cached embeds |
| R11 | blind / text-only floor per benchmark (no bound on headroom) | ~$2 |
| R12 | fused-query ordering check for AKS/FOCUS*/LDDR-select — Table 1's ordering rests on one point of an axis worth ~3.9pt | ~$4 |
| R13 | k=8 only; below AKS's and FOCUS's published 32-64F regime | needs k=32 run |
| — | `acl.tex` body is ~8.6pp, ARR limit 8. Compress the Method section (926 words, untouched) | writing only |
| — | Item 4 (harness artifact: selector plugin API + publish the 1.85GB embed cache) — only if NeurIPS | 1-2 days, no compute |

**R12 is the one that matters most** — it is a live threat to the main claim and cheap
because the pipeline exists.

AWS: both G-quota requests **DENIED** (new-account ramp-up). Cases 178732194600183 /
178732194700551. Reply drafts (Deeksha at AWS Activate + case appeal at 4 vCPU not 16)
are in the session transcript. SageMaker GPU quotas are all 0 too — not a bypass.
Modal needs no quota and is the working path.
