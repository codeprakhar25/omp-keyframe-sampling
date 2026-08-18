#!/usr/bin/env python3
"""OMP-fail taxonomy + per-category alpha-effect, from existing clean-matrix + ab_ablation
sample files. Zero GPU. Two lenses on 'why does OMP fail', kept SEPARATE on purpose:

  TYPE  (from LVB question_category): Subtitle / Temporal(before-after) / Sequence / Tracking / Perception
  MECH  (from rescue + same-letter): Wrong-retrieval(=rescue) / Model-or-reasoning(=same-letter both-fail) / Ambiguous

Plus per-category accuracy for topk(a0)/a0.5/a0.75/OMP(a1) — does any category prefer a softer alpha?
"""
import glob, json

AB="results/ab_ablation"; BL="results/lmmseval_matrix_clean"

# LVB category -> primary TYPE bucket. Priority: text-cue(Subtitle) wins first (we run subs OFF),
# then relation kind. Documented, single-assignment to avoid double counting.
SUBTITLE={"T2E","T2O","T2A","T3E","T3O","TOS","TAA"}   # cue = subtitle/text
TEMPORAL={"E3E","O3O"}                                  # non-text before/after
SEQUENCE={"SSS"}                                        # order of many scenes
TRACKING={"SOS"}                                        # non-text object tracking
# everything else -> Perception (single-moment S2E/S2O/S2A/E2O/O2E/SAA)
def typ(c):
    if c in SUBTITLE: return "Subtitle"
    if c in TEMPORAL: return "Temporal_b/a"
    if c in SEQUENCE: return "Sequence"
    if c in TRACKING: return "Tracking"
    return "Perception"

def load(p):
    d={}
    for f in glob.glob(p):
        for l in open(f):
            if l.strip():
                la=json.loads(l)["lvb_acc"]
                d[la["id"]]=(la["answer"]==la["parsed_pred"], la["parsed_pred"], la["question_category"])
    return d

for BIN in ("600","3600"):
    a0 =load(f"{BL}/k8_{BIN}/**/*picks_lc_{BIN}s_k8.jsonl")
    a05=load(f"{AB}/*_sh*/**/*picks_sig_{BIN}s_k8.jsonl")
    a75=load(f"{AB}/*_sh*/**/*picks_omp_lc_{BIN}s_k8.jsonl")
    a1 =load(f"{BL}/k8_{BIN}/**/*picks_omp_lc_{BIN}s_k8.jsonl")
    ids=sorted(set(a0)&set(a05)&set(a75)&set(a1)); N=len(ids)
    fails=[i for i in ids if not a1[i][0]]              # OMP failures
    nf=len(fails)
    print(f"\n########## {BIN}s  n={N}  OMP fails={nf} ({nf/N:.0%}) ##########")

    # --- TYPE lens ---
    from collections import Counter
    tc=Counter(typ(a1[i][2]) for i in fails)
    print("  TYPE (% of OMP failures):")
    for k in ("Subtitle","Temporal_b/a","Sequence","Tracking","Perception"):
        print(f"    {k:14s} {tc[k]:3d}  {tc[k]/nf:5.1%}")

    # --- MECH lens ---
    rescue=sum(1 for i in fails if a0[i][0])                      # OMP wrong, topk right
    same  =sum(1 for i in fails if (not a0[i][0]) and a0[i][1]==a1[i][1])  # both wrong, same letter
    ambig =nf-rescue-same
    print("  MECH (% of OMP failures):")
    print(f"    Wrong-retrieval(rescue by topk) {rescue:3d}  {rescue/nf:5.1%}")
    print(f"    Model/reasoning(same-letter)    {same:3d}  {same/nf:5.1%}")
    print(f"    Ambiguous(both-fail diff letter){ambig:3d}  {ambig/nf:5.1%}")

    # --- per-category alpha effect (acc of each arm) ---
    cats=Counter(a1[i][2] for i in ids)
    print("  PER-CATEGORY acc  topk / a0.5 / a0.75 / OMP   (cats n>=15; * = best arm != OMP by >=2 items):")
    for c,cn in cats.most_common():
        if cn<15: continue
        cid=[i for i in ids if a1[i][2]==c]
        accs={n_:sum(d[i][0] for i in cid)/cn for n_,d in (("topk",a0),("a0.5",a05),("a0.75",a75),("OMP",a1))}
        bestarm=max(accs,key=accs.get)
        flag=" *" if bestarm!="OMP" and (accs[bestarm]-accs["OMP"])*cn>=2 else ""
        print(f"    {c:4s} n={cn:3d}  {accs['topk']:.3f} {accs['a0.5']:.3f} {accs['a0.75']:.3f} {accs['OMP']:.3f}  best={bestarm}{flag}")
