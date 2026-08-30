# Venue targets — selector paper

*(Filename spelled `VENUE_TARGETS_POSSIBLE.md`. Deadlines below were fetched from
official pages on 2026-08-28 unless marked otherwise. Re-verify before acting:
CFP dates move.)*

Last updated 2026-08-30.

## DECISION — LOCKED 2026-08-30

**Primary: ARR October cycle, deadline Oct 12 2026**, committing to NAACL/COLING
2027 or ACL 2027. Chosen on fit evidence from the venues' own calls, not timing:

- ARR's CFP **names this paper's type**: "reproduction study", "negative results",
  "model analysis papers", and negative results covering "non-reproducibility or
  non-generalizability of previously published results".
- ICLR's call lists "datasets and benchmarks" as a topic but **does not mention
  negative results or reproducibility studies at all**, and asks for "your most
  complete and most exciting work" — excitement being the exact axis the
  2026-08-21 panel said this paper does not clear.
- ARR has a Findings fallback inside the same submission; ICLR has none.
- ARR exempts Limitations, Ethics and References from the 8-page limit.

An earlier steer in this session toward ICLR-first was **wrong** and is retracted:
it rested on an unverified belief that ICLR's rubric is broader for empirical
work. The CFPs say the opposite.

## Can TMLR and a main conference both happen? (verified 2026-08-30)

**Not in parallel.** TMLR forbids text, figures or results shared with any paper
"submitted in parallel at another archival, peer-reviewed venue", and conferences
forbid submitting already-published work. One at a time.

**But there is a sequential route.** TMLR joined the NeurIPS/ICML/ICLR
**Journal-to-Conference (J2C) track**. A TMLR paper carrying a J2C, Featured or
Outstanding Certification can be presented at one of those three conferences
*with no further peer review*. Important qualifications:

- The J2C track itself is **not selective** — "all requests meeting eligibility
  criteria are accepted until capacity limits are reached" (150 slots each).
- The **certification is** the selective gate, and it is reportedly rare; a
  long-serving TMLR action editor has said very few papers get forwarded.
- **It is a presentation, not a proceedings paper**: "It shall not be considered
  as being published in the proceedings of the chosen conference."
- Eligibility window is at most 2 years since TMLR publication.

J2C request deadlines: ICML 2026 May 3 2026 (passed) · NeurIPS 2026 Sep 26 2026 ·
**ICLR 2027 Dec 18 2026**. Firm, no extensions.

TMLR aims for a decision about **9 weeks** after submission (bodies over 12 pages
take longer; ours is 8).

**Why this supports ARR-first rather than TMLR-first.** ARR-first preserves TMLR:
if the ARR route fails, TMLR is still open afterwards and a later J2C slot is
still reachable. TMLR-first blocks ARR for at least the ~9-week review, and its
conference component is a presentation rather than the proceedings line that was
the stated goal.

## The full set

| Target | Deadline | Page limit | Status |
|---|---|---|---|
| **ICLR 2027** | abstract **Sep 18 2026**, paper **Sep 25 2026** (11:59 PM AOE) | ~9 pp | Verified 2026-08-28. arXiv allowed. **Nearest deadline.** |
| **ARR October cycle** | **Oct 12 2026** | 8 pp body | Verified. Not a venue itself — see below. |
| **TMLR** | rolling, no deadline | none | Verified. arXiv preprints allowed anytime. |
| **NeurIPS Evaluations & Datasets** | 2026 cycle **PASSED** | — | Confirmed by user 2026-08-30. Track reportedly renamed from "Datasets & Benchmarks"; next cycle only. |
| CVPR 2027 | ~Nov 13 2026 | — | Third-party trackers only, **no official CFP**. The 2026-07-28 handoff recommends *avoiding* CVPR/ICCV for this paper type. |
| EMNLP / Findings | — | — | From the 2026-07-28 handoff. Superseded by the ARR route. |

## Three clarifications that matter

**1. ARR is not a venue.** It is the ACL Rolling Review pool. You submit to ARR by
Oct 12, get reviews, then *commit* the paper to a conference: NAACL or COLING 2027
(commit ~Dec 20) or ACL 2027 (commit ~Jan 2027). The main-conference line comes
from the commitment step, not from ARR itself. Findings is a fallback offered
inside that same process rather than a separate resubmission.

**2. TMLR = Transactions on Machine Learning Research.** A journal from the JMLR
family, run by the same community that runs NeurIPS/ICML/ICLR — not a workshop and
not a lesser tier, but a *journal*, so it carries no conference line and no talk.
Its distinguishing rule is the acceptance criterion: claims supported by accurate,
convincing evidence, plus interest to some subset of the audience. **Novelty and
significance are explicitly excluded as reasons to reject.** Rolling submission,
no page limit, no deadline. Reviews are public. It is the reason both the 5-seat
panel and Codex named it this paper's best fit: the paper's weakness at conference
venues is precisely the thing TMLR's rubric refuses to penalise.

**3. ICLR and ARR cannot both be targeted in sequence this year.** ICLR 2027
decisions land ~Jan 2027, well after the Oct 12 ARR deadline, and concurrent
submission of the same paper to both violates dual-submission policy at both. The
same applies to TMLR alongside either. **Exactly one of these three is the primary
this cycle.** The others become fallbacks only after a decision returns.

## Where the two independent reviews landed

| Reviewer | ACL main | TMLR | Note |
|---|---|---|---|
| 5-seat panel (2026-08-21) | Major Revision | **Accept/Minor, best fit** | "soundness clears the bar, excitement as a main-conference paper does not" |
| Codex (2026-08-30) | "difficult fit; Findings more realistic" | **"cleanest current fit"** | flagged claim-scope, not soundness, as the blocker |

Both say the same thing: the work is sound, and the gap at conference venues is
that it deliberately claims no new method. That is a rubric mismatch, not a
quality verdict.

## Standing constraint from the user

Main-conference line is wanted (2026-08-30). That is a real bet against both
reviews, taken knowingly. The open question the record has flagged since
2026-08-28 and only the user can answer is whether a *conference* line is required
specifically (thesis committee, visa, funding) — if yes, the bet is correct
regardless of the odds.

## Stale pointers — do not act on

- `PAPER_DRAFT.md:20` "Venue (verified 2026-08-15)" lists arXiv → ARR Oct 12 →
  NAACL/COLING as Active. **Pre-panel and stale.**
- `HANDOFF_PAPER_2026-07-28.md:93` recommends EMNLP/NeurIPS D&B. Superseded.

## Current readiness

- Both `paper/arxiv.tex` and `paper/acl.tex` compile clean, no undefined refs.
- arXiv body ~14 pp (no limit). ACL body **~11 pp against an 8 pp limit** — a
  ~3 pp cut, not a trim. ICLR's ~9 pp would need ~2 pp.
- Cut strategy deliberately deferred until the venue is fixed, so the work is
  done once.

---

## Preprint / arXiv policy per venue — verified 2026-08-30

Every venue below was read directly on 2026-08-30. Quotes are verbatim.
Bottom line: **no venue blocks an arXiv preprint.** The only cost is at ARR.

| Venue | Policy URL | Verdict |
|---|---|---|
| ARR / ACL | https://aclrollingreview.org/cfp | Allowed, but forfeits award eligibility + borderline priority |
| NeurIPS 2026 main | https://neurips.cc/Conferences/2026/MainTrackHandbook | Allowed |
| NeurIPS 2026 E&D | https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets | Allowed |
| ICLR 2027 | https://iclr.cc/Conferences/2027/CallForPapers | Allowed |
| CVPR 2026 | https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines | Allowed |
| TMLR | https://jmlr.org/tmlr/editorial-policies.html | Allowed |
| AAAI | https://aaai.org/conference/aaai/aaai-26/submission-guidelines/ | Allowed (page served stale AAAI-23 text; re-check for the target year) |

### ARR — the one that costs something

> "Beginning February 15, 2024, there is no anonymity period or limitation on
> posting or discussing non-anonymous preprints while the work is under peer
> review. However, the new policy does incentivize anonymous submissions through
> special paper awards and priority in acceptance decisions for borderline
> papers. You will be asked to select the preprint status of the submission on
> our submission form. If you choose the binding 'no non-anonymous preprint'
> option, you commit to not preprinting until the metareviews are released,
> under the penalty of desk rejection."

Our 5-seat panel (2026-08-21) returned MAJOR REVISION from every seat, which is
the band "priority in acceptance decisions for borderline papers" targets.
Recommendation: take the binding no-preprint option for the Oct 12 cycle and
post to arXiv when metareviews release (~Dec 2026).

ARR multiple-submission ban covers "journals and refereed and archival
conferences and workshops" — arXiv is not one of these.

### NeurIPS

> "The existence of non-anonymous preprints (on arXiv or other online
> repositories, personal websites, social media) will not result in rejection.
> If you choose to use the NeurIPS style for the preprint version, you must use
> the 'preprint' option rather than the 'final' option. The public versions of
> the submission should not say 'Under review at NeurIPS' or similar."

> "Note: While having a nonanonymized preprint alone is not a violation of the
> double-blind reviewing policy, aggressive advertising of papers under
> submission may be deemed a violation."

Dual submission covers archival venues only: "dual submissions to nonarchival
workshops are permitted."

Track rename CONFIRMED: the track is "Evaluations and Datasets" and is now
double-blind by default. NeurIPS 2026 deadline was May 6 2026 (passed;
notification Sept 24 2026), so the next window is NeurIPS 2027.

### ICLR 2027

> "Having papers on arxiv is allowed per the dual submission policy outlined in
> the author guidelines."

### CVPR 2026

> "Under the above definition, arXiv preprints and university technical reports
> are not considered as publications."

> "Q. Can I post my submission on arXiv? A. Yes."

Caveat that matters for later workshop plans: "peer-reviewed workshop papers are
considered as publications if their length is more than four pages (excluding
references), even if they do not appear in a proceedings." An archival workshop
paper would block CVPR; an arXiv preprint would not.

Media rule: "you should not list CVPR submissions on public websites or on
media."

### TMLR

> "It is acceptable for a submission to overlap with the author's previous work
> if it was shared at venues or tracks that are publicly declared, in writing,
> to be non-archival, such as workshops, or on preprint servers such as arXiv
> and bioRxiv."

### AAAI

> "workshops and preprint servers such as arXiv are acceptable"

### The rule that holds everywhere

Post the preprint; never publicly state it is under review at a named venue.

---

## 2026-08-30 — ARR lock reopened, ICLR 2027 found open

The Oct 12 ARR lock was made before three facts were known. All three are now
verified and they change the ranking. **No decision taken yet — user's call.**

**1. The Oct 12 cycle does not lead to ACL.** Per the ARR dates page, that cycle's
participating venues are **NAACL 2027 and COLING 2027**, commitment date
Dec 20 2026. ACL 2027 takes ARR submissions in **January 2027**.

**2. ARR takes ~10 weeks to a meta-review, and a meta-review is not a decision.**
Measured across three completed cycles: Mar 16 to May 21 (9.6 wks), May 25 to
Jul 30 (9.4 wks), Aug 3 to Oct 8 (9.4 wks). The Oct 2026 cycle ends Dec 20.

**3. ICLR 2027 is open, direct-submission, and resolves sooner.**
https://iclr.cc/Conferences/2027/CallForPapers

| | ARR Oct 12 | ICLR 2027 |
|---|---|---|
| Abstract | — | Sep 18 2026 |
| Paper | Oct 12 2026 | Sep 25 2026 |
| Reviews | ~Dec 17 | Nov 5 2026 |
| Author-reviewer discussion | in-cycle | Nov 5-18 2026 |
| Decision | not in this cycle | **Dec 16 2026** |

ICLR decides four days before the ARR cycle ends, and an ICLR rejection on
Dec 16 still leaves the January ARR cycle open, which is the one feeding
ACL 2027.

**The cost of the ICLR route:** ICLR keeps every submission public with author
names attached, including rejected and withdrawn ones. Our five-seat panel
returned MAJOR REVISION from all five seats, so a public rejection is a live
risk. ARR and TMLR rejections are private.

**Also settled: direct submission to the ACL family no longer exists.**
"Starting from 2024, all main *ACL conferences used ARR exclusively."
NeurIPS, ICLR, CVPR, AAAI and TMLR all remain direct-submission. No venue
requires institutional affiliation; the only affiliation-sensitive gate in the
pipeline is arXiv endorsement.

### Standing recommendation

1. **ICLR 2027** if the paper can be made worth a public record by Sep 25
   (reformat to ICLR 9-page style + close the panel items; no new runs needed).
2. **TMLR** if 26 days is too tight. Its criterion is evidence quality and
   reader interest "even if the contribution or significance of the work is
   modest" -- written for a paper strong on evidence and modest on novelty.
   Rolling, private, ~9 weeks, no preprint penalty.
3. **ARR Oct 12** third: slowest to a decision, lands at NAACL/COLING rather
   than ACL, and is the only option that charges for preprinting.

All three forbid parallel submission. Picking one determines the arXiv call.
