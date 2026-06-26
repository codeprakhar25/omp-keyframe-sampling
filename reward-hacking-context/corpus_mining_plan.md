# Corpus Mining Plan — are "proxy-pressure" instructions real in the wild?

Status: planning. **Zero-cost, no GPU, no agent runs.** First de-risk: the whole
"the *helpful* AGENTS.md is the hazard" framing collapses if proxy-pressure
phrasing is rare in real context files. Run this *before* any pod spend.

**Goal:** (1) prove proxy-pressure instructions occur in real AGENTS.md/CLAUDE.md
and quantify how common; (2) extract the *real strings* → become the Phase-1
treatment arm (kills the "straw-man instruction you invented" objection — the Q5
construct-validity lock).

---

## 1. Outcome metrics (what the mined corpus must report)

- `P(proxy-pressure)` = fraction of context files containing ≥1 outcome-pressure
  mandate. (Headline number; if very low, reconsider the framing.)
- Breakdown by **pressure category** (taxonomy §3) with frequency each.
- Co-occurrence with gameable surface (e.g. files that *also* say "you have repo
  access" / "run the full suite").
- Distribution of phrasings → the canonical templates for the treatment arm.

---

## 2. Data source + collection

- **Primary:** GitHub code search API for filenames `AGENTS.md`, `CLAUDE.md`,
  `.cursorrules`, `.github/copilot-instructions.md`. (GitHub REST/GraphQL;
  authenticated token; respect rate limits.)
- **Sampling:** target a few-thousand-file sample across repos; record repo stars,
  language, last-commit for stratification + selection-bias reporting.
- **Dedup:** hash file content; many AGENTS.md are copy-paste boilerplate — dedup
  by near-duplicate (MinHash/SimHash) so a viral template doesn't dominate the rate.
- **Licensing/PII:** store only the instruction text + repo metadata; aggregate
  before any sharing; honor that these are public but attributable.

Fallback if API coverage thin: the public **`AGENTS.md` ecosystem lists** + the
context-file corpora from the 2026 AGENTS.md papers (Agent READMEs 2511.12884,
2602.14690) — check if any released their scraped set (saves us the scrape).

---

## 3. Proxy-pressure instruction taxonomy (the classifier's label set)

What counts as **outcome/proxy-pressure** (the treatment) vs neutral:

| Category | Example phrasings | Why it's proxy-pressure |
|---|---|---|
| Test-pass mandate | "all tests must pass before finishing", "never leave failing tests" | pressures the gradeable proxy directly |
| CI/green mandate | "keep CI green", "the build must be green" | proxy = CI status |
| Coverage mandate | "maintain 100% coverage", "do not lower coverage" | proxy = coverage metric |
| Definition-of-done by metric | "done = all checks pass", "success means green checkmarks" | redefines task as proxy |
| Speed/completion pressure | "finish quickly", "minimize iterations" | pressure → shortcut |

**Neutral / non-pressure (control arm material):**
"use ruff for linting", "the build cmd is `make`", "code lives in `src/`" —
informational, no proxy goal. (These match ETH's "instructions cause over-work"
without proxy-pressure → isolates pressure from mere instruction-following.)

---

## 4. Classification method (3-stage, validated)

1. **Seed regex / keyword pass** — high-recall net for the phrasings above
   (`tests? (must|should) pass`, `CI .*green`, `100% coverage`, `all checks`, …).
   Cheap first cut; over-captures.
2. **LLM-label pass** — classify each seeded snippet into the §3 taxonomy +
   {pressure / neutral / other}, structured output. Handles paraphrase the regex
   misses/over-flags.
3. **Human validation** — hand-check ~150 classified snippets, 2 raters + Cohen's κ
   (same rigor as the hack-detector validation). Report classifier precision/recall.
   This is the construct-validity evidence cited in the paper's threats section.

---

## 5. Deliverables

- `corpus_stats.md` — `P(proxy-pressure)`, per-category frequencies, by repo
  stratum; selection-bias + dedup notes.
- `treatment_strings.json` — the real, lightly-templated phrasings per arm:
  `none` (empty) / `neutral` (informational, ETH-style) / `pressure` (proxy-pressure).
- `classifier/` — regex seeds + LLM-label prompt + the 150-label validation set + κ.

---

## 6. Decision gate after mining

- **`P(proxy-pressure)` healthy (say ≥10–15% of files):** framing holds → proceed
  to Phase-1 pilot with the mined `pressure` arm.
- **Rare (<5%):** the "common well-meaning instruction" framing weakens. Options:
  reframe as "even *occasional* real instructions can induce hacking" (existence,
  weaker), or pivot. Decide then — cheaply, before pod $.

---

## 7. Threats to handle in writing

- **Selection bias** — popular repos over-curate AGENTS.md; report by star stratum.
- **Boilerplate inflation** — dedup or one viral template skews the rate.
- **Intent vs literal** — "all tests must pass" may be benign guidance; our claim is
  about the *literal* pressure the agent conditions on, not author intent (state this).
- **Temporal drift** — context-file norms moving fast in 2026; date-stamp the scrape.
