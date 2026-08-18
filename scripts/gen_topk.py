#!/usr/bin/env python3
"""Top-k-by-LongCLIP-cosine picks (the 'topk-lc' arm). Same data path as gen_fixedk /
gen_dpp_minmax so it is consistent with every other pick file. Emits {qid: sorted([secs])}.
Usage: gen_topk.py BIN K"""
import json, sys
import numpy as np
from harness.embeds import l2, load_image_embed

BIN, K = sys.argv[1], int(sys.argv[2])
tx = np.load(f"results/embeds_text/text_lc_{BIN}.npz")
qmap = dict(zip(tx["ids"].tolist(), tx["text"]))
out, miss = {}, 0
for q in qmap:
    t, emb = load_image_embed(q, "lc")
    if t is None:
        miss += 1
        continue
    E = l2(emb)
    q0 = l2(qmap[q].astype(np.float32))
    sim = E @ q0
    idx = np.argsort(sim)[::-1][:K]
    out[q] = sorted(round(float(t[i]), 1) for i in idx)
assert miss == 0, f"{miss} missing image embeds"
p = f"results/picks_lmmseval/picks_topk_lc_{BIN}_k{K}.json"
json.dump(out, open(p, "w"))
print(f"wrote {p}: {len(out)} items k={K}")
