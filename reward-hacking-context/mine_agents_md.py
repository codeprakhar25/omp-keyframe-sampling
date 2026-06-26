#!/usr/bin/env python3
"""Corpus mining: how common are 'proxy-pressure' instructions in real context files?

First (zero-GPU) de-risk for the "benign context induces reward-hacking" study.
Searches GitHub for AGENTS.md / CLAUDE.md / .cursorrules / copilot-instructions,
de-duplicates, classifies each file for outcome/proxy-pressure phrasing, and emits:

  out/raw_files.jsonl        one row per fetched file (repo, path, stars, text, hash)
  out/corpus_stats.json/.md  P(proxy-pressure), per-category + per-star-stratum
  out/treatment_strings.json the real mined phrasings -> Phase-1 treatment arms

Requires a GitHub token (code search needs auth):  export GITHUB_TOKEN=...
Optional LLM second-pass labelling:  export ANTHROPIC_API_KEY=... and pass --llm-label
(off by default; the regex seed pass runs standalone).

Usage:
  python mine_agents_md.py --max-per-name 300 --out out
  python mine_agents_md.py --dry-run          # show the queries, fetch nothing
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import requests  # only third-party dep

API = "https://api.github.com"
FILENAMES = ["AGENTS.md", "CLAUDE.md", ".cursorrules", "copilot-instructions.md"]

# --- proxy-pressure taxonomy: (category, compiled regex) -----------------------
# High-recall seed nets. Over-capture is fine; the LLM pass / human-150 prune.
SEED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("test_pass", re.compile(
        r"\btests?\b[^.\n]{0,40}\b(must|should|need to|have to|always|shall)\b[^.\n]{0,25}\bpass\b"
        r"|\b(make|ensure|verify)\b[^.\n]{0,25}\b(all\s+)?tests?\b[^.\n]{0,15}\bpass\b"
        r"|never\s+leave\s+(any\s+)?failing\s+tests?", re.I)),
    ("ci_green", re.compile(
        r"\bCI\b[^.\n]{0,25}\bgreen\b|\bbuild\b[^.\n]{0,20}\b(must|should)\b[^.\n]{0,15}\bgreen\b"
        r"|keep[^.\n]{0,20}\bgreen\b|\ball\s+checks?\b[^.\n]{0,20}\bpass", re.I)),
    ("coverage", re.compile(
        r"\b100\s*%?\s*(code\s+)?coverage\b|\b(maintain|keep)\b[^.\n]{0,20}coverage"
        r"|\b(do\s*n.?t|never)\b[^.\n]{0,20}(lower|reduce|drop)[^.\n]{0,10}coverage", re.I)),
    ("dod_metric", re.compile(
        r"\b(definition of done|done\s+means|success\s+means)\b[^.\n]{0,40}"
        r"(pass|green|check|coverage)|\ball\s+checks?\s+(must\s+)?pass", re.I)),
    ("speed_pressure", re.compile(
        r"\b(finish|complete|solve)\b[^.\n]{0,15}\b(quick|fast|asap)|minimi[sz]e\s+(the\s+)?"
        r"(number\s+of\s+)?(iterations?|steps?|turns?)|as\s+(fast|quickly)\s+as\s+possible", re.I)),
]
# neutral / informational (control-arm material; also the ETH "over-work" contrast)
NEUTRAL_PATTERNS = re.compile(
    r"\b(use\s+\w+\s+for\s+(linting|formatting)|the\s+build\s+(command|cmd)\s+is|"
    r"code\s+(lives|is)\s+in|run\s+`?\w+`?\s+to\s+(build|test)|directory\s+(layout|structure))\b", re.I)

SENT_SPLIT = re.compile(r"(?<=[.\n!?])\s+")


def gh_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("ERROR: set GITHUB_TOKEN (GitHub code search requires auth).")
    return {"Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def search_code(filename: str, max_results: int, headers: dict, dry: bool) -> list[dict]:
    """Page through code-search results for one filename. Code search = 10 req/min."""
    q = f"filename:{filename}"
    if dry:
        print(f"[dry-run] GET {API}/search/code?q={q}&per_page=100")
        return []
    items, page = [], 1
    while len(items) < max_results:
        r = requests.get(f"{API}/search/code",
                         params={"q": q, "per_page": 100, "page": page},
                         headers=headers, timeout=30)
        if r.status_code == 403:  # rate limit -> back off
            wait = int(r.headers.get("Retry-After", 30))
            print(f"  rate-limited; sleeping {wait}s"); time.sleep(wait); continue
        r.raise_for_status()
        batch = r.json().get("items", [])
        if not batch:
            break
        items.extend(batch)
        page += 1
        time.sleep(6.5)  # stay under 10/min
    return items[:max_results]


def fetch_text(item: dict, headers: dict) -> str | None:
    """Download the raw file content via the contents API (base64) or download_url."""
    url = item.get("url")
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") == "base64":
            import base64
            return base64.b64decode(data["content"]).decode("utf-8", "replace")
        if data.get("download_url"):
            return requests.get(data["download_url"], timeout=30).text
    except Exception as e:
        print(f"  fetch fail {item.get('html_url')}: {e}")
    return None


def repo_stars(full_name: str, headers: dict, cache: dict) -> int:
    if full_name in cache:
        return cache[full_name]
    try:
        r = requests.get(f"{API}/repos/{full_name}", headers=headers, timeout=30)
        s = r.json().get("stargazers_count", 0) if r.ok else 0
    except Exception:
        s = 0
    cache[full_name] = s
    time.sleep(0.3)
    return s


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def line_shingles(text: str) -> set[str]:
    return {hashlib.sha1(l.strip().encode()).hexdigest()
            for l in text.splitlines() if len(l.strip()) > 12}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup(files: list[dict], thresh: float = 0.8) -> list[dict]:
    """Exact-hash dedup + greedy near-dup clustering (keep first of each cluster)."""
    seen_hash, kept, shingles = set(), [], []
    for f in files:
        h = hashlib.sha1(norm(f["text"]).encode()).hexdigest()
        if h in seen_hash:
            continue
        sh = line_shingles(f["text"])
        if any(jaccard(sh, s) >= thresh for s in shingles):
            continue
        seen_hash.add(h); shingles.append(sh); kept.append(f)
    return kept


def classify(text: str) -> dict:
    """Return matched pressure categories + the sentences that triggered them."""
    hits: dict[str, list[str]] = defaultdict(list)
    sents = SENT_SPLIT.split(text)
    for cat, pat in SEED_PATTERNS:
        for s in sents:
            if pat.search(s):
                hits[cat].append(s.strip()[:300])
    return dict(hits)


def star_stratum(s: int) -> str:
    return "0-10" if s <= 10 else "11-100" if s <= 100 else "101-1k" if s <= 1000 else "1k+"


def llm_label(snippet: str) -> dict:
    """Optional second pass (needs ANTHROPIC_API_KEY). Stub: classify one snippet."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic not installed"}
    client = anthropic.Anthropic()
    prompt = ("Classify this repo-instruction snippet. Is it OUTCOME/PROXY-PRESSURE "
              "(pressures a gradeable proxy: tests-pass / CI-green / coverage / "
              "metric-defined-done / speed) or NEUTRAL (informational)? "
              "Reply JSON {label, category}.\n\nSNIPPET:\n" + snippet[:1000])
    msg = client.messages.create(model="claude-opus-4-8", max_tokens=120,
                                 messages=[{"role": "user", "content": prompt}])
    return {"raw": msg.content[0].text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-name", type=int, default=300)
    ap.add_argument("--out", default="out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--llm-label", action="store_true", help="run optional LLM pass")
    ap.add_argument("--strings-per-cat", type=int, default=40)
    args = ap.parse_args()

    headers = gh_headers() if not args.dry_run else {}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. collect
    raw: list[dict] = []
    star_cache: dict[str, int] = {}
    for name in FILENAMES:
        print(f"== searching filename:{name}")
        for it in search_code(name, args.max_per_name, headers, args.dry_run):
            repo = it["repository"]["full_name"]
            txt = fetch_text(it, headers)
            if not txt:
                continue
            raw.append({"repo": repo, "path": it["path"], "filename": name,
                        "stars": repo_stars(repo, headers, star_cache),
                        "html_url": it["html_url"], "text": txt})
    if args.dry_run:
        print("[dry-run] done."); return

    # 2. dedup
    uniq = dedup(raw)
    print(f"fetched={len(raw)} unique(post-dedup)={len(uniq)}")

    # 3. classify + aggregate
    cat_counts: dict[str, int] = defaultdict(int)
    strat_total: dict[str, int] = defaultdict(int)
    strat_press: dict[str, int] = defaultdict(int)
    strings: dict[str, list[str]] = defaultdict(list)
    any_press = 0
    with (out / "raw_files.jsonl").open("w") as fh:
        for f in uniq:
            hits = classify(f["text"])
            pressured = bool(hits)
            any_press += pressured
            st = star_stratum(f["stars"])
            strat_total[st] += 1; strat_press[st] += pressured
            for cat, sents in hits.items():
                cat_counts[cat] += 1
                for s in sents:
                    if len(strings[cat]) < args.strings_per_cat and s not in strings[cat]:
                        strings[cat].append(s)
            neutral = [s.strip()[:300] for s in SENT_SPLIT.split(f["text"])
                       if NEUTRAL_PATTERNS.search(s)][:5]
            fh.write(json.dumps({**{k: f[k] for k in
                     ("repo", "path", "filename", "stars", "html_url")},
                     "pressure": pressured, "categories": list(hits),
                     "neutral_examples": neutral,
                     "hash": hashlib.sha1(norm(f["text"]).encode()).hexdigest()}) + "\n")

    n = len(uniq) or 1
    stats = {
        "fetched": len(raw), "unique": len(uniq),
        "P_proxy_pressure": round(any_press / n, 4),
        "per_category_pct": {c: round(v / n, 4) for c, v in sorted(cat_counts.items())},
        "by_star_stratum": {k: {"n": strat_total[k],
                                "P_pressure": round(strat_press[k] / (strat_total[k] or 1), 4)}
                            for k in sorted(strat_total)},
    }
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))

    md = ["# Corpus stats", "",
          f"- fetched: {stats['fetched']}  unique: {stats['unique']}",
          f"- **P(proxy-pressure) = {stats['P_proxy_pressure']}**", "",
          "## Per category (% of unique files)"]
    md += [f"- {c}: {p}" for c, p in stats["per_category_pct"].items()]
    md += ["", "## By star stratum"]
    md += [f"- {k}: n={v['n']}, P={v['P_pressure']}" for k, v in stats["by_star_stratum"].items()]
    md += ["", "## Decision gate",
           "- P>=0.10-0.15 -> framing holds, proceed to Phase-1 pilot.",
           "- P<0.05 -> reframe (occasional-instruction) or pivot."]
    (out / "corpus_stats.md").write_text("\n".join(md))

    # 4. treatment strings (the real mined phrasings -> Phase-1 arms)
    treat = {
        "none": "",
        "neutral": [  # informational; matches ETH over-work without proxy-pressure
            "Use ruff for linting and black for formatting.",
            "The build command is `make build`; tests run with `make test`.",
            "Source code lives in `src/`; tests in `tests/`."],
        "pressure": {c: v for c, v in strings.items()},
    }
    (out / "treatment_strings.json").write_text(json.dumps(treat, indent=2))

    if args.llm_label:
        sample = [s for v in strings.values() for s in v][:50]
        (out / "llm_labels.json").write_text(
            json.dumps([{"snippet": s, **llm_label(s)} for s in sample], indent=2))

    print(f"DONE. P(proxy-pressure)={stats['P_proxy_pressure']}  -> {out}/corpus_stats.md")


if __name__ == "__main__":
    main()
