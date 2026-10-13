"""Data for the Draft plan tab -> results/live_<slug>.json
The page ranks players in the browser with a closed-form version of the simulation: your players' projected season totals
(per-game x games) vs the typical team after the same number of picks, turned into a win chance per category with the
fitted weekly noise. Validated against the simulation (src/check_formula.py).
Exports: per-player season totals, ADP, games, sheet targets, and the opponent baseline (mean and spread by pick count).
Run: python src/build_live.py   (DRAFT_CONFIG=config_12team.json for the 12 team league)"""
import json, os
import numpy as np
from draftlib import *

def opp_stats(nsim=240):
    """Mean and spread (across teams) of each category after n rounds when every team drafts by ADP + noise."""
    mu = np.zeros((R + 1, 9)); var = np.zeros((R + 1, 9))
    for s in range(nsim):
        noise = np.random.default_rng(31000 + s).normal(size=(T, N)) * (SIG0 if s % 2 == 0 else SIG2)
        avail = np.ones(N, bool); tot = np.zeros((T, 11))
        for r in range(R):
            for t in (range(T) if r % 2 == 0 else range(T - 1, -1, -1)):
                k = ADP + noise[t]; k[~avail] = 1e9; p = int(np.argmin(k)); avail[p] = False; tot[t] += SS[p]
            C = season_cats(tot, np.ones((T, 7)), 1.0) * SIGN
            mu[r + 1] += C.mean(0); var[r + 1] += C.var(0)
    return mu / nsim, np.sqrt(var / nsim)

if __name__ == "__main__":
    raw = json.load(open(os.path.join(ROOT, "data", "players.json")))
    targets = raw["t"]; players = []
    for i, r in enumerate(raw["p"]):
        players.append(dict(n=NAMES[i], adp=r[3], dnd=int(DND[i]), gp=r[16], t=targets.get(NAMES[i]),
                            ss=[round(float(x), 2) for x in SS[i]]))
    mu, sd = opp_stats()
    live = dict(name=CFG.get("name", "10 team"), teams=T, rounds=R, reach=REACH,
                mu=[[round(float(x), 5) for x in row] for row in mu], sd=[[round(float(x), 5) for x in row] for row in sd],
                k=[float(x) for x in SEASON["k"]], cv=[float(x) for x in SEASON["cv"]], players=players)
    path = os.path.join(ROOT, "results", f"live_{SLUG}.json")
    json.dump(live, open(path, "w"), separators=(",", ":"))
    print("saved", path, len(players), "players")
