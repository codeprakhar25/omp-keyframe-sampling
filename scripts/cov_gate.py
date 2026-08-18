#!/usr/bin/env python3
"""HARD coverage gate. Exits nonzero unless EVERY arm output has exactly EXP unique
doc_ids AND EXP lines (no dupes, no missing). This is the guard that makes the
sharding bug impossible to ship silently: a run that didn't cover the full pool
FAILS here and never gets a .done marker."""
import json, glob, sys, os

d, exp = sys.argv[1], int(sys.argv[2])
fs = glob.glob(os.path.join(d, "**", "*samples_longvideobench*.jsonl"), recursive=True)
assert fs, f"NO output files in {d}"
ok = True
for f in sorted(fs):
    ids = [json.loads(l)["doc_id"] for l in open(f) if l.strip()]
    u = len(set(ids))
    good = (u == exp and len(ids) == exp)
    ok = ok and good
    print(f"  {os.path.basename(f)[-46:]:46s} lines={len(ids):4d} uniq={u:4d} exp={exp} {'OK' if good else 'FAIL <<<'}")
sys.exit(0 if ok else 1)
