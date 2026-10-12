"""Top 6-pick combos per slot -> results/top.json
Takes the best leaves of results/tree.json for each slot, de-duplicates by player set, and re-scores the best
`CAND` on fresh random draws (not the ones used to build the tree, so the winner's-curse bias is gone).
Saves the top `KEEP` with expected categories won, std. error and per-category win % (CATS order).
Run after build_tree.py:  python src/build_top.py"""
import json, os
import numpy as np
from multiprocessing import Pool
from draftlib import *

CAND, KEEP, NS = 60, 10, 300
tree = json.load(open(os.path.join(ROOT, "results", "tree.json")))

def leaves(n, path=()):
    path = path + (n["n"],)
    return [(n["c"], path)] if not n["k"] else [l for k in n["k"] for l in leaves(k, path)]

def run_slot(s):
    seen, cands = set(), []
    for c, path in sorted(leaves(tree[str(s)]["tree"]), key=lambda x: -x[0]):
        key = frozenset(path)
        if key in seen: continue
        seen.add(key); cands.append(tuple(IDX[n] for n in path))
        if len(cands) == CAND: break
    noises = [np.random.default_rng(777000 + 10000 * s + i).normal(size=(T, N)) * (SIG0 if i % 2 == 0 else SIG2) for i in range(NS)]
    res = []
    for q in cands:
        v = np.array([play(s, nz, forced=q)[0] for nz in noises]); tot = v.sum(1)
        res.append((tot.mean(), tot.std() / np.sqrt(NS), v.mean(0), q))
    res.sort(key=lambda x: -x[0])
    return s, dict(pick=tree[str(s)]["pick"], top=[dict(
        names=[NAMES[i] for i in q], adp=[round(float(ADP[i]), 1) for i in q], c=round(float(m), 2), se=round(float(se), 2),
        w=[int(round(100 * x)) for x in w]) for m, se, w, q in res[:KEEP]])

if __name__ == "__main__":
    with Pool(4) as p: out = dict(p.map(run_slot, range(1, T + 1), chunksize=1))
    json.dump({str(s): d for s, d in out.items()}, open(os.path.join(ROOT, "results", "top.json"), "w"), separators=(",", ":"))
    for s in sorted(out): t = out[s]["top"][0]; print(s, " + ".join(t["names"]), t["c"], "+-", t["se"])
