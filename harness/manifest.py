"""Manifest preflight guards.

One job: turn silent whole-manifest drops into a loud refusal, before any GPU work.

`--gold-reliable-only` defaults to True in every dump_* script, and the filter is
`if not item.get("gold_reliable"): continue`. That is safe on
manifest.lvb.frames.100.json (the key is present, True on all 400) and a total
wipe on full1560 / long976 (the key does not exist -> .get() returns None ->
every item skipped -> a 0-row score file, exit status 0, message "wrote ... (0
items)"). Nothing downstream notices until lmms-eval raises KeyError on the first
doc, ~10 minutes and one model load later.

Same shape as the 2026-07-15 query bug: a field whose absence meant something
different from what the reader assumed, and no stage printed its own input.
"""
from __future__ import annotations


def check_gold_reliable(manifest, enabled: bool, path: str = "") -> None:
    """Refuse if --gold-reliable-only would drop the ENTIRE manifest.

    A filter that removes every item is never the intent; it is a missing flag.
    """
    if not enabled:
        return
    if any("gold_reliable" in it for it in manifest):
        return
    raise SystemExit(
        f"REFUSING: --gold-reliable-only is ON but no item in {path or 'this manifest'} "
        f"carries a 'gold_reliable' key.\n"
        f"  The filter would skip EVERY item and write a 0-row output with exit status 0.\n"
        f"  Manifests WITH the key:    data/manifest.lvb.frames.100.json (400/400 True)\n"
        f"  Manifests WITHOUT the key: manifest.lvb.full1560.json, manifest.lvb.long976.json\n"
        f"  -> pass --no-gold-reliable-only for those."
    )
