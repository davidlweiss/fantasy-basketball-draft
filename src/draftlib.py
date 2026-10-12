"""Shared model: player values (z-scores), par, snake order, and the draft simulator."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CFG = json.load(open(os.path.join(ROOT, "config.json")))
T, R, POOL, K, REACH = CFG["teams"], CFG["rounds"], CFG["pool"], CFG["first_rounds"], CFG["reach"]
_d = json.load(open(os.path.join(ROOT, "data", "players.json")))["p"]

NAMES = [r[0] for r in _d]
RK = np.array([r[1] for r in _d], float)
ADP = np.array([r[3] if r[3] is not None else 250 for r in _d], float)
DND = np.array([bool(r[4]) for r in _d])               # 'Do not draft' column in the sheet: never picked by you
FGM = np.array([r[5] for r in _d]); FGA = np.array([r[6] for r in _d])
FTM = np.array([r[7] for r in _d]); FTA = np.array([r[8] for r in _d])
C = np.array([r[9:16] for r in _d], float)             # 3PM PTS REB AST STL BLK TO
N = len(_d)
CATS = ["FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
IDX = {n: i for i, n in enumerate(NAMES)}

def zscores(pool=POOL):
    """z = (value - mean) / sd over the top `pool` players by projected rank. TO is flipped. FG%/FT% use
    makes above what an average shooter makes on the same attempts."""
    top = np.argsort(RK)[:pool]
    fgR = FGM[top].sum() / FGA[top].sum(); ftR = FTM[top].sum() / FTA[top].sum()
    fgi, fti = FGM - fgR * FGA, FTM - ftR * FTA
    Z = np.zeros((N, 9))
    Z[:, 0] = fgi / fgi[top].std(); Z[:, 1] = fti / fti[top].std()
    for k in range(7):
        z = (C[:, k] - C[top, k].mean()) / C[top, k].std()
        Z[:, 2 + k] = -z if k == 6 else z
    return Z
Z = zscores()

ORDER = [int(i) for i in np.argsort(ADP, kind="stable")]          # players by ADP
_oa = np.argsort(ADP)
# par[n]: average z of the players an average team holds after n picks (top T*n by ADP)
PAR = [None] + [Z[_oa[:min(N, T * n)]].mean(0) for n in range(1, R + 1)]
S = np.column_stack([FGM, FGA, FTM, FTA, C])                       # per-player stat rows for team totals
SIGN = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1])
SIG0 = 2 + 0.12 * np.minimum(ADP, 200)                             # 'tight' opponents: ADP + modest noise
SIG2 = 4 + 0.25 * np.minimum(ADP, 200)                             # 'loose' opponents: roughly double the noise

def pick_numbers(slot, rounds=K):
    return [r * T + slot if r % 2 == 0 else (r + 1) * T - slot + 1 for r in range(rounds)]

def team_cats(tot):
    return np.array([tot[0] / tot[1], tot[2] / tot[3], tot[4], tot[5], tot[6], tot[7], tot[8], tot[9], tot[10]])

def noise_from_seed(seed, sig):
    return np.random.default_rng(seed).normal(size=(T, N)) * sig

def make_noise(slot, k):
    """k noise matrices for a slot, alternating tight/loose opponents (same draws reused across foursomes)."""
    return [np.random.default_rng(10000 * slot + i).normal(size=(T, N)) * (SIG0 if i % 2 == 0 else SIG2) for i in range(k)]

def balanced_weights(zsum, cnt):
    """weight = 1 - 0.5 * (your average z - par), kept in [0.4, 1.8]. This is the board's 'Fit' rule."""
    return np.clip(1 - 0.5 * (zsum / cnt - PAR[cnt]), 0.4, 1.8)

def play(slot, noise, forced=(), punts=(), reach=REACH):
    """Simulate one full snake draft. Opponents take the lowest (ADP + noise) player. You take `forced`
    players first (held back from opponents until your turn), then the best-scoring player by the balanced rule.
    Returns (share of the 9 opponents beaten per category, your picks in order)."""
    me = slot - 1
    avail = np.ones(N, bool); reserved = np.zeros(N, bool); reserved[list(forced)] = True
    tot = np.zeros((T, 11)); zsum = np.zeros(9); cnt = 0; pickno = 0; picks = []
    wbase = np.ones(9)
    for j in punts: wbase[j] = 0
    for r in range(R):
        for t in (range(T) if r % 2 == 0 else range(T - 1, -1, -1)):
            pickno += 1
            if t == me:
                if cnt < len(forced):
                    p = forced[cnt]; reserved[p] = False
                else:
                    w = (balanced_weights(zsum, cnt) if cnt > 0 else 1.0) * wbase
                    sc = Z @ w; sc[~avail | DND | reserved | (ADP > pickno + reach)] = -1e9
                    if sc.max() <= -1e8:                      # nobody within reach: take best available
                        sc = Z @ w; sc[~avail | DND | reserved] = -1e9
                    p = int(np.argmax(sc))
                zsum += Z[p]; cnt += 1; picks.append(p)
            else:
                k = ADP + noise[t]; k[~avail | reserved] = 1e9; p = int(np.argmin(k))
            avail[p] = False; tot[t] += S[p]
    cats = np.array([team_cats(tot[t]) for t in range(T)]) * SIGN
    mine = cats[me]; oth = np.delete(cats, me, 0)
    return ((mine > oth) + 0.5 * (mine == oth)).mean(0), picks
