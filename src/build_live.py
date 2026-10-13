"""Data for the live draft tab -> results/live_<slug>.json
Everything the page needs to rank players in the browser: season-total z-scores (games-played aware), par per roster size,
ADP, projected games, per-game stats and the sheet's starting-target windows.
Run: python src/build_live.py   (DRAFT_CONFIG=config_12team.json for the 12 team league)"""
import json, os
import numpy as np
from draftlib import *

raw = json.load(open(os.path.join(ROOT, "data", "players.json")))
targets = raw["t"]; players = []
for i, r in enumerate(raw["p"]):
    players.append(dict(n=NAMES[i], adp=r[3], rk=r[1], dnd=int(DND[i]), gp=r[16], mpg=r[17], t=targets.get(NAMES[i]),
                        s=[round(r[5] / r[6], 3), round(r[7] / r[8], 3), r[9], r[10], r[11], r[12], r[13], r[14], r[15]],
                        z=[round(float(x), 3) for x in ZS[i]]))
live = dict(name=CFG.get("name", "10 team"), teams=T, rounds=R, reach=REACH, cats=CATS,
            par=[None] + [[round(float(x), 4) for x in PARS[n]] for n in range(1, R + 1)], players=players)
path = os.path.join(ROOT, "results", f"live_{SLUG}.json")
json.dump(live, open(path, "w"), separators=(",", ":"))
print("saved", path, len(players), "players")
