"""Fit the season model (draftlib.SEASON) to the league's real last-season standings and print a comparison.
Real data (10 teams): category totals, and H2H records -> `k` (weekly noise) was fitted separately from the records.
This script: simulate many leagues where every team drafts by ADP + noise, take season totals (per-game x games), then
  1. set the volume spread sig per category so the spread across teams (std/mean) matches the real league,
  2. report how well category correlations with PTS and the spread of team win totals match.
Run: python src/calibrate_season.py"""
import numpy as np
import draftlib as D
from draftlib import *

CATS = ["FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
REAL_CV = np.array([1.4, 2.0, 14.1, 14.8, 18.1, 15.8, 14.3, 24.0, 17.7]) / 100
REAL_CORR = np.array([0.35, -0.44, 0.70, 1.0, 0.92, 0.95, 0.84, 0.47, 0.98])           # with PTS
REAL_WINS_SD, REAL_TOP3, REAL_BOT2 = 1.4, 5.96, 2.4                                      # sd of team win totals; avg of top 3; avg of bottom 2
NL = 60

def leagues(n):
    out = []
    for s in range(n):
        noise = np.random.default_rng(555000 + s).normal(size=(T, N)) * (SIG0 if s % 2 == 0 else SIG2)
        avail = np.ones(N, bool); tot = np.zeros((T, 11))
        for r in range(R):
            for t in (range(T) if r % 2 == 0 else range(T - 1, -1, -1)):
                k = ADP + noise[t]; k[~avail] = 1e9; p = int(np.argmin(k)); avail[p] = False; tot[t] += SS[p]
        out.append(tot)
    return out

def stats(tots, sig, lam, gamma):
    D.SEASON["sig"], D.SEASON["lam"], D.SEASON["gamma"] = sig, lam, gamma
    cv, co, wsd, top3, bot2 = [], [], [], [], []
    for i, tot in enumerate(tots):
        w = D.make_season_world(900 + i, SIG0)
        C = D.season_cats(tot, w["vol"])
        cv.append(C.std(0) / C.mean(0)); co.append([np.corrcoef(C[:, c], C[:, 3])[0, 1] for c in range(9)])
        W = np.array([D.season_win(C * SIGN, m, D.SEASON["k"]).sum() for m in range(T)]); Ws = np.sort(W)[::-1]
        wsd.append(W.std()); top3.append(Ws[:3].mean()); bot2.append(Ws[-2:].mean())
    return np.mean(cv, 0), np.nanmean(co, 0), np.mean(wsd), np.mean(top3), np.mean(bot2)

def fit(tots, gamma):
    cat7 = [2, 3, 4, 5, 6, 7, 8]                                   # 3PM PTS REB AST STL BLK TO in CATS order
    cv0 = stats(tots, np.zeros(7), np.ones(7), gamma)[0]
    sig = np.sqrt(np.maximum(REAL_CV[cat7] ** 2 - cv0[cat7] ** 2, 0.03 ** 2)); lam = np.clip(REAL_CORR[cat7], 0, 1)
    for it in range(4):                                            # re-fit once the volume factor is in (draft + volume spreads add)
        cv, co, *_ = stats(tots, sig, lam, gamma)
        sig = sig * np.clip(REAL_CV[cat7] / cv[cat7], 0.7, 1.4) ** 1.5
        lam = np.clip(lam + 0.5 * (REAL_CORR[cat7] - co[cat7]), 0.0, 1.0); lam[1] = 1.0
    return sig, lam, stats(tots, sig, lam, gamma)

def loss(res):
    cv, co, wsd, top3, bot2 = res; c = [2, 4, 5, 6, 8]
    return float(np.mean((co[c] - REAL_CORR[c]) ** 2) + 0.1 * (wsd - REAL_WINS_SD) ** 2 + 0.05 * (top3 - REAL_TOP3) ** 2)

if __name__ == "__main__":
    tots = leagues(NL); best = None
    for gamma in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        sig, lam, res = fit(tots, gamma); L = loss(res)
        print(f"gamma {gamma:.2f}: loss {L:.4f} | corr REB {res[1][4]:.2f} AST {res[1][5]:.2f} 3PM {res[1][2]:.2f} TO {res[1][8]:.2f} | win sd {res[2]:.2f} top3 {res[3]:.2f} bot2 {res[4]:.2f}")
        if best is None or L < best[0]: best = (L, gamma, sig, lam, res)
    _, gamma, sig, lam, (cv, co, wsd, top3, bot2) = best
    print(f"best: gamma = {gamma} | sig = {np.round(sig, 3).tolist()} | lam = {np.round(lam, 2).tolist()}")
    print("             " + " ".join(f"{c:>6s}" for c in CATS))
    print("spread  sim  " + " ".join(f"{100*x:5.1f}%" for x in cv)); print("spread real  " + " ".join(f"{100*x:5.1f}%" for x in REAL_CV))
    print("corrPTS sim  " + " ".join(f"{x:6.2f}" for x in co)); print("corrPTS real " + " ".join(f"{x:6.2f}" for x in REAL_CORR))
    print(f"team win totals: sd sim {wsd:.2f} vs real {REAL_WINS_SD} | top-3 avg sim {top3:.2f} vs real {REAL_TOP3} | bottom-2 avg sim {bot2:.2f} vs real {REAL_BOT2}")
