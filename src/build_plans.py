"""Per-slot strategy comparison + round-by-round target plans -> results/plans.json
Compares Balanced (fit) with punting options at each draft slot (avg of tight and loose opponents),
then simulates the recommended strategy 400x to get the most common pick in each round.
Run: python src/build_plans.py            (about 20-40 min; use --quick for a fast smoke test)"""
import sys, json, os
from collections import Counter
import numpy as np
from draftlib import *

QUICK = "--quick" in sys.argv
NS, NP = (20, 20) if QUICK else (300, 400)
CANDS = [("Balanced (fit)", ()), ("Punt AST", (5,)), ("Punt TO", (8,)), ("Punt AST+TO", (5, 8)), ("Punt BLK", (7,))]
cmp, plans = {}, {}
for slot in range(1, T + 1):
    cmp[slot] = {}
    for name, pu in CANDS:
        v = []
        for sig in (SIG0, SIG2):
            v.append(np.mean([play(slot, noise_from_seed(20000 + s, sig), punts=pu)[0].sum() for s in range(NS)]))
        cmp[slot][name] = float(np.mean(v))
    base = cmp[slot]["Balanced (fit)"]; best = max(cmp[slot], key=cmp[slot].get)
    rec = best if cmp[slot][best] - base >= 0.06 else "Balanced (fit)"     # punt only if clearly better
    pu = dict(CANDS)[rec]
    out = [play(slot, noise_from_seed(31000 + s, SIG0), punts=pu) for s in range(NP)]
    wins = np.mean([w for w, _ in out], axis=0); picks = np.array([p for _, p in out])
    rounds = []
    for r in range(R):
        pn = pick_numbers(slot, R)[r]
        c = Counter(picks[:, r]).most_common(5)
        tg = [dict(n=NAMES[i], pct=round(100 * n / len(picks)), adp=round(float(ADP[i]), 1),
                   z=[round(float(x), 1) for x in Z[i]]) for i, n in c if n / len(picks) >= 0.04]
        rounds.append(dict(r=r + 1, pick=pn, t=tg))
    plans[str(slot)] = dict(strategy=rec, cats=round(float(wins.sum()), 2), win=[round(float(x), 2) for x in wins],
                            compare={n: round(x, 2) for n, x in cmp[slot].items()}, rounds=rounds)
    print(slot, rec, plans[str(slot)]["cats"], flush=True)
out = os.path.join(ROOT, "results", "plans_quick.json" if QUICK else "plans.json")
json.dump(plans, open(out, "w"))
print("saved", out)
