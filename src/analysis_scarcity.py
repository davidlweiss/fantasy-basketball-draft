"""Category scarcity by round: how many players with z >= 1 (and >= 1.5) in each category sit in each ADP band,
plus the average z by round for an average team (par). Prints tables and writes results/scarcity.json.
Run: python src/analysis_scarcity.py"""
import json, os
import numpy as np
from draftlib import *
bands = [(i * T, (i + 1) * T) for i in range(R)]
out = {"cats": CATS, "elite_z1": [], "elite_z15": [], "round_avg_z": []}
print("round   " + "  ".join(f"{c:>5}" for c in CATS) + "   (count of players with z>=1 taken in that round by ADP)")
for r, (a, b) in enumerate(bands):
    ids = ORDER[a:b]
    e1 = [int((Z[ids, k] >= 1).sum()) for k in range(9)]
    e15 = [int((Z[ids, k] >= 1.5).sum()) for k in range(9)]
    out["elite_z1"].append(e1); out["elite_z15"].append(e15)
    out["round_avg_z"].append([round(float(x), 2) for x in Z[ids].mean(0)])
    print(f"R{r+1:<3}    " + "  ".join(f"{x:5d}" for x in e1))
json.dump(out, open(os.path.join(ROOT, "results", "scarcity.json"), "w"))
