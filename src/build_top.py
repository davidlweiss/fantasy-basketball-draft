"""Top 6-pick combos per slot -> results/top_<slug>.json
Takes the best leaves of results/tree.json for each slot, de-duplicates by player set, and re-scores the best
`CAND` on fresh random draws (not the ones used to build the tree, so the winner's-curse bias is gone).
RANKING uses only the 6 picks (play_prefix: your 6 players vs each opponent's 6 players); no later picks assumed.
Saves the top `KEEP` with: c/se/w = that picks-only score, std. error and per-category win %;
  g = 10 x (avg z - par) per category (negative = behind an average team after 6 rounds -> emphasize later);
  fill = the most common picks in rounds 7-10 if you then follow the Fit rule (guidance only, not used for ranking);
  fw = projected final win % per category with that fill (guidance only).
Run after build_tree.py (same DRAFT_CONFIG):  python src/build_top.py"""
import json, os
import numpy as np
from collections import Counter
from multiprocessing import Pool
from draftlib import *

CAND, KEEP, NS, NF = 60, 10, 300, 60
tree = json.load(open(os.path.join(ROOT, "results", f"tree_{SLUG}.json")))

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
        v = np.array([play_prefix(s, nz, q) for nz in noises]); tot = v.sum(1)
        res.append((tot.mean(), tot.std() / np.sqrt(NS), v.mean(0), q))
    res.sort(key=lambda x: -x[0])
    pn = pick_numbers(s, R); out = []
    for m, se, w, q in res[:KEEP]:
        runs = [play(s, nz, forced=q) for nz in noises[:NF]]
        fw = np.mean([r[0] for r in runs], axis=0); picks = np.array([r[1] for r in runs])
        fill = [dict(r=r + 1, pick=pn[r], t=[[NAMES[i], round(100 * n / NF)] for i, n in Counter(picks[:, r]).most_common(3)])
                for r in range(len(q), len(q) + 4)]
        out.append(dict(names=[NAMES[i] for i in q], adp=[round(float(ADP[i]), 1) for i in q], c=round(float(m), 2),
                        se=round(float(se), 2), w=[int(round(100 * x)) for x in w],
                        g=[int(round(10 * x)) for x in Z[list(q)].mean(0) - PAR[len(q)]],
                        fill=fill, fw=[int(round(100 * x)) for x in fw]))
    return s, dict(pick=tree[str(s)]["pick"][:6], top=out)

if __name__ == "__main__":
    with Pool(4) as p: out = dict(p.map(run_slot, range(1, T + 1), chunksize=1))
    json.dump({str(s): d for s, d in out.items()}, open(os.path.join(ROOT, "results", f"top_{SLUG}.json"), "w"), separators=(",", ":"))
    for s in sorted(out): t = out[s]["top"][0]; print(s, " + ".join(t["names"]), t["c"], "+-", t["se"])
