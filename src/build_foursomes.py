"""Best first-four-pick combinations for each draft slot -> results/foursomes.json
1. Enumerate every ordered set of 4 players you could get at your 4 picks if opponents follow ADP
   (window rule: at your j-th pick, the players others already took are gone; reach <= config.reach picks past ADP;
   top-`pool` players only; players in config.first_four_exclude and the sheet's Do-Not-Draft column are never used).
2. Shortlist by a cheap heuristic (team z vs par, capped) plus raw z sum.
3. Simulate the other 9 rounds of each shortlisted foursome (30x), then re-run the top 40 (300x).
4. Save the top 12 per slot. Differences under ~0.1 expected categories are effectively ties.
Run: python src/build_foursomes.py            (all slots, ~1 hr)
     python src/build_foursomes.py 8 9        (just those slots)"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
from draftlib import *

NS2, NS3, SHORT_H, SHORT_R, FINAL = 30, 300, 1200, 400, 40
EXCL = {IDX[n] for n in CFG.get("first_four_exclude", [])}
bad = [n for n in CFG.get("first_four_exclude", []) if n not in IDX]
if bad: raise SystemExit(f"first_four_exclude names not found in players.json: {bad}")
BLOCK = DND.copy()
for i in EXCL: BLOCK[i] = True
PN = lambda s: pick_numbers(s, 4)

def enumerate_sets(s):
    P = PN(s); best = {}
    def rec(j, mine, pen):
        if j == 4:
            key = frozenset(mine)
            if key not in best or pen < best[key][1]: best[key] = (tuple(mine), pen)
            return
        L = window_start(P[j], j, set(mine))
        for r in range(L, POOL):
            p = ORDER[r]
            if p in mine or BLOCK[p] or ADP[p] > P[j] + REACH: continue
            rec(j + 1, mine + [p], pen + max(0.0, ADP[p] - P[j]))
    rec(0, [], 0.0)
    return best

def evaluate(s, seq, noises):
    return np.array([play(s, nz, forced=seq)[0] for nz in noises]).sum(1)

def run_slot(s):
    t0 = time.time()
    best = enumerate_sets(s); keys = list(best); M = len(keys)
    seqs = np.array([best[k][0] for k in keys])
    avg = Z[seqs].mean(1); gap = avg - PAR[4]
    h = np.clip(gap, -1.5, 1.0).sum(1); raw = avg.sum(1)
    idx = sorted(set(np.argsort(-h)[:SHORT_H]) | set(np.argsort(-raw)[:SHORT_R]))
    n2 = make_noise(s, NS2)
    sc2 = sorted(((evaluate(s, tuple(seqs[i]), n2).mean(), i) for i in idx), reverse=True)
    fin = [i for _, i in sc2[:FINAL]]
    n3 = make_noise(s, NS3); res = []
    for i in fin:
        v = evaluate(s, tuple(seqs[i]), n3); res.append((v.mean(), v.std() / np.sqrt(len(v)), i))
    res.sort(reverse=True); top = []
    for m, se, i in res[:12]:
        seq = [int(x) for x in seqs[i]]
        top.append(dict(seq=seq, names=[NAMES[x] for x in seq], adp=[round(float(ADP[x]), 1) for x in seq],
                        picks=PN(s), cats=round(float(m), 2), se=round(float(se), 2),
                        gap=[round(float(x), 2) for x in (Z[seq].mean(0) - PAR[4])],
                        avgz=[round(float(x), 2) for x in Z[seq].mean(0)]))
    return s, dict(total=M, short=len(idx), top=top, secs=round(time.time() - t0))

if __name__ == "__main__":
    slots = [int(x) for x in sys.argv[1:]] or list(range(1, T + 1))
    with Pool(2) as p: out = p.map(run_slot, slots, chunksize=1)
    path = os.path.join(ROOT, "results", "foursomes.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(s): d for s, d in out})
    json.dump(old, open(path, "w"))
    for s, d in out: print(s, d["total"], d["short"], d["secs"], "s", "|", " + ".join(d["top"][0]["names"]), d["top"][0]["cats"])
