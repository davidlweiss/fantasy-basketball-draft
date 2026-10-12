"""Live helper: rank what to take next by Fit, given your roster and who is gone.
  python src/next_pick.py --roster "Nikola Jokic,Stephen Curry" --pick 24 [--gone "A,B"] [--top 15]
Fit = sum over categories of (player z x weight), weight = clip(1 - 0.5*(your avg z - par(n)), 0.4, 1.8).
Players with ADP more than `reach` picks after the current pick are skipped (reach rule)."""
import argparse
import numpy as np
from draftlib import *
ap = argparse.ArgumentParser()
ap.add_argument("--roster", default=""); ap.add_argument("--gone", default="")
ap.add_argument("--pick", type=int, required=True); ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
names = lambda s: [x.strip() for x in s.split(",") if x.strip()]
mine = [IDX[n] for n in names(a.roster)]; gone = {IDX[n] for n in names(a.gone)} | set(mine)
cnt = len(mine)
w = balanced_weights(Z[mine].sum(0), cnt) if cnt else np.ones(9)
print("weights:", "  ".join(f"{c} {x:.2f}" for c, x in zip(CATS, w)))
sc = Z @ w
rows = [i for i in np.argsort(-sc) if i not in gone and not DND[i] and ADP[i] <= a.pick + REACH]
print(f"{'player':24s} {'ADP':>5s} {'fit':>6s}  " + " ".join(f"{c:>5s}" for c in CATS))
for i in rows[:a.top]:
    print(f"{NAMES[i]:24s} {ADP[i]:5.1f} {sc[i]:6.2f}  " + " ".join(f"{x:5.1f}" for x in Z[i]))
