"""Plan for your first `plan_rounds` (6) picks, per draft slot -> results/plan6.json
Beam search instead of enumerating every set:
  for each possible round-1 anchor, extend one pick at a time. Candidates at each pick = players plausibly still on the board
  (window rule, `reach` rule, top-`pool`, never do_not_draft / first_four_exclude), cut to the best `K_CAND` by Fit.
  Each partial roster is scored by simulating the rest of the draft (forced prefix, then the usual Fit rule) with shared
  random draws; keep the best `BEAM` per anchor. Final plans are re-scored on fresh draws (no winner's-curse bias).
Run: python src/build_plan6.py            (all slots)
     python src/build_plan6.py 4 7        (just those slots)
     python src/build_plan6.py --quick    (small, smoke test)"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
from draftlib import *

L6 = CFG.get("plan_rounds", 6)
QUICK = "--quick" in sys.argv
BEAM, K_CAND, N_SEARCH, FINAL, N_FINAL = (3, 8, 8, 10, 40) if QUICK else (6, 22, 24, 30, 400)
BLOCK = DND.copy()
for n in CFG.get("first_four_exclude", []): BLOCK[IDX[n]] = True
PN = lambda s: pick_numbers(s, L6)

def noise_set(s, n, base):
    return [np.random.default_rng(base + 10000 * s + i).normal(size=(T, N)) * (SIG0 if i % 2 == 0 else SIG2) for i in range(n)]

def score(s, seq, noises):
    return float(np.mean([play(s, nz, forced=tuple(seq))[0].sum() for nz in noises]))

def candidates(s, seq):
    j = len(seq); P = PN(s)[j]; mine = set(seq)
    cand = [ORDER[r] for r in range(window_start(P, j, mine), POOL)
            if ORDER[r] not in mine and not BLOCK[ORDER[r]] and ADP[ORDER[r]] <= P + REACH]
    if not cand: return []
    w = balanced_weights(Z[list(seq)].sum(0), j) if j else np.ones(9)
    cand.sort(key=lambda p: -(Z[p] @ w))
    return cand[:K_CAND]

def run_slot(s):
    t0 = time.time(); search = noise_set(s, N_SEARCH, 0)
    finals = {}
    for anchor in candidates(s, ()) if not QUICK else candidates(s, ())[:3]:
        beam = [((anchor,), None)]
        for d in range(1, L6):
            kids = {}
            for seq, _ in beam:
                for p in candidates(s, seq):
                    k = frozenset(seq + (p,))
                    if k not in kids: kids[k] = seq + (p,)      # same set reached in another order: keep first
            if not kids: break
            scored = sorted(((score(s, q, search), q) for q in kids.values()), reverse=True)
            beam = [(q, v) for v, q in scored[:BEAM]]
        finals.update({frozenset(q): q for q, _ in beam if len(q) == L6})
    fresh = noise_set(s, N_FINAL, 777000)
    # pre-rank on a fresh small set, then full re-score the best FINAL
    pre = sorted(((score(s, q, fresh[:N_FINAL // 8]), q) for q in finals.values()), reverse=True)[:FINAL]
    res = []
    for _, q in pre:
        v = np.array([play(s, nz, forced=q)[0] for nz in fresh])
        tot = v.sum(1); res.append((tot.mean(), tot.std() / np.sqrt(len(tot)), v.mean(0), q))
    res.sort(key=lambda x: -x[0])
    plans = []
    for m, se, win, q in res[:15]:
        q = [int(x) for x in q]
        plans.append(dict(seq=q, names=[NAMES[x] for x in q], adp=[round(float(ADP[x]), 1) for x in q], picks=PN(s),
                          cats=round(float(m), 2), se=round(float(se), 2), win=[round(float(x), 2) for x in win],
                          gap=[round(float(x), 2) for x in Z[q].mean(0) - PAR[L6]]))
    best = {}                                                       # best plan per round-1 anchor
    for p in plans:
        best.setdefault(p["names"][0], p)
    return s, dict(top=plans, by_anchor=list(best.values()), candidates=len(finals), secs=round(time.time() - t0))

if __name__ == "__main__":
    slots = [int(x) for x in sys.argv[1:] if x.isdigit()] or list(range(1, T + 1))
    with Pool(4) as p: out = p.map(run_slot, slots, chunksize=1)
    path = os.path.join(ROOT, "results", "plan6_quick.json" if QUICK else "plan6.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(s): d for s, d in out})
    json.dump(old, open(path, "w"))
    for s, d in out:
        t = d["top"][0]; print(s, d["secs"], "s |", " + ".join(t["names"]), t["cats"], "+-", t["se"])
