# Venue targets — selector paper

*(Filename spelled `VENUE_TARGETS_POSSIBLE.md`. Deadlines below were fetched from
official pages on 2026-08-28 unless marked otherwise. Re-verify before acting:
CFP dates move.)*

Last updated 2026-08-30.

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
