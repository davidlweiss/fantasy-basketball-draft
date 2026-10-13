"""Shared model: player values (z-scores), par, snake order, and the draft simulator."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
# League profile: config.json (10 team) by default; DRAFT_CONFIG=config_12team.json python src/build_tree.py for 12 team
CFG = json.load(open(os.path.join(ROOT, os.environ.get("DRAFT_CONFIG", "config.json"))))
SLUG = CFG.get("slug", "10team")
T, R, POOL, K, REACH = CFG["teams"], CFG["rounds"], CFG["pool"], CFG["first_rounds"], CFG["reach"]
_d = json.load(open(os.path.join(ROOT, "data", "players.json")))["p"]

NAMES = [r[0] for r in _d]
RK = np.array([r[1] for r in _d], float)
ADP = np.array([r[3] if r[3] is not None else 250 for r in _d], float)
DND = np.array([bool(r[4]) for r in _d])               # 'Do not draft' column in the sheet + config do_not_draft: never picked by you
_bad = [n for n in CFG.get("do_not_draft", []) if n not in NAMES]
if _bad: raise SystemExit(f"do_not_draft names not found in players.json: {_bad}")
for _n in CFG.get("do_not_draft", []): DND[NAMES.index(_n)] = True
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

def zscores_season(pool=POOL):
    """Like zscores() but on season totals (per-game x projected games), so durable players are worth more."""
    top = np.argsort(RK)[:pool]
    fgR = SS_[top, 0].sum() / SS_[top, 1].sum(); ftR = SS_[top, 2].sum() / SS_[top, 3].sum()
    fgi, fti = SS_[:, 0] - fgR * SS_[:, 1], SS_[:, 2] - ftR * SS_[:, 3]
    Zs = np.zeros((N, 9)); Zs[:, 0] = fgi / fgi[top].std(); Zs[:, 1] = fti / fti[top].std()
    for k in range(7):
        c = SS_[:, 4 + k]; z = (c - c[top].mean()) / c[top].std(); Zs[:, 2 + k] = -z if k == 6 else z
    return Zs

ORDER = [int(i) for i in np.argsort(ADP, kind="stable")]          # players by ADP
_oa = np.argsort(ADP)
# par[n]: average z of the players an average team holds after n picks (top T*n by ADP)
PAR = [None] + [Z[_oa[:min(N, T * n)]].mean(0) for n in range(1, R + 1)]
S = np.column_stack([FGM, FGA, FTM, FTA, C])                       # per-player stat rows for team totals
GP = np.array([r[16] if len(r) > 16 and r[16] else 70.0 for r in _d], float)    # projected games played (data/games_played.csv)
SS = S * GP[:, None]                                               # per-player season totals = per-game stats x projected games
SIGN = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1])
SS_ = SS
ZS = zscores_season()                                              # season-total z-scores (games-played aware)
PARS = [None] + [ZS[np.argsort(ADP)[:min(N, T * n)]].mean(0) for n in range(1, R + 1)]
SIG0 = 2 + 0.12 * np.minimum(ADP, 200)                             # 'tight' opponents: ADP + modest noise
SIG2 = 4 + 0.25 * np.minimum(ADP, 200)                             # 'loose' opponents: roughly double the noise

def window_start(pickno, j, mine):
    """Index in ORDER of the best-ADP player still plausibly on the board at your j-th pick (0-based):
    the (pickno - 1 - j) picks made by others are assumed to be the best-ADP players not in `mine`."""
    m = pickno - 1 - j; cnt = 0; r = 0
    while cnt < m:
        if ORDER[r] not in mine: cnt += 1
        r += 1
    return r

def pick_numbers(slot, rounds=K):
    return [r * T + slot if r % 2 == 0 else (r + 1) * T - slot + 1 for r in range(rounds)]

def team_cats(tot):
    return np.array([tot[0] / tot[1], tot[2] / tot[3], tot[4], tot[5], tot[6], tot[7], tot[8], tot[9], tot[10]])

def noise_from_seed(seed, sig):
    return np.random.default_rng(seed).normal(size=(T, N)) * sig

def make_noise(slot, k):
    """k noise matrices for a slot, alternating tight/loose opponents (same draws reused across foursomes)."""
    return [np.random.default_rng(10000 * slot + i).normal(size=(T, N)) * (SIG0 if i % 2 == 0 else SIG2) for i in range(k)]

def balanced_weights(zsum, cnt, par=None):
    """weight = 1 - 0.5 * (your average z - par), kept in [0.4, 1.8]. This is the board's 'Fit' rule."""
    return np.clip(1 - 0.5 * (zsum / cnt - (PAR if par is None else par)[cnt]), 0.4, 1.8)

# ---- season model (opt-in via play(..., season=world)): calibrated to the league's last-season standings ----------------
# Teams get a random 'volume' multiplier (games played / active roster / streaming) that moves their counting stats
# together, and a category is won with probability Phi(edge / (k x spread of teams)) instead of by strict rank, because
# weekly results are noisy (FG% and FT% most of all). sig/lam/k are fitted by src/calibrate_season.py.
SEASON = dict(
    sig=np.array([0.131, 0.153, 0.183, 0.157, 0.146, 0.257, 0.183]),      # sd of log volume multiplier: 3PM PTS REB AST STL BLK TO
    lam=np.array([0.76, 1.00, 1.00, 1.00, 0.90, 0.54, 1.00]),      # share of it that is common to all categories (corr with PTS)
    k=np.array([6.0, 3.4, 1.45, 1.0, 1.05, 1.05, 1.75, 1.9, 1.25]),  # weekly noise / team spread: FG% FT% 3PM PTS REB AST STL BLK TO
    gamma=0.5,                                                     # share of draft-day roster differences that survives the season (waivers, trades)
)
VOL_COL = np.array([1, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6])              # which volume column scales each of the 11 stat columns

def make_season_world(seed, sig_adp):
    """Opponent ADP noise (as before) plus per-team volume multipliers; returns a dict you pass as play(season=...)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(T, N)) * sig_adp
    g = rng.normal(size=(T, 1)); e = rng.normal(size=(T, 7)); sg, lam = SEASON["sig"], SEASON["lam"]
    vol = np.exp(sg * (lam * g + np.sqrt(1 - lam ** 2) * e) - 0.5 * sg ** 2)
    return dict(noise=noise, vol=vol, k=SEASON["k"])

def season_cats(tot, vol):
    m = tot.mean(0); t = (m + SEASON["gamma"] * (tot - m)) * vol[:, VOL_COL]
    return np.column_stack([t[:, 0] / t[:, 1], t[:, 2] / t[:, 3], t[:, 4], t[:, 5], t[:, 6], t[:, 7], t[:, 8], t[:, 9], t[:, 10]])

def season_win(cats, me, k):
    """Chance of beating each opponent per category, averaged over the T-1 opponents. cats are signed so higher is better."""
    d = (cats[me] - np.delete(cats, me, 0)) / (k * cats.std(0))
    return (0.5 * (1 + np.tanh(0.7978845608 * (d + 0.044715 * d ** 3)))).mean(0)

def play(slot, noise, forced=(), punts=(), reach=REACH, season=None, catw=None, zmat=None):
    """Simulate one full snake draft. Opponents take the lowest (ADP + noise) player. You take `forced`
    players first (held back from opponents until your turn), then the best-scoring player by the balanced rule.
    Returns (share of the 9 opponents beaten per category, your picks in order). With season=<make_season_world(...)> the teams'
    totals are season totals (per-game x games) with volume multipliers, and the result is a win probability per category."""
    me = slot - 1; M = S if season is None else SS; ZM = Z if zmat is None else zmat
    avail = np.ones(N, bool); reserved = np.zeros(N, bool); reserved[list(forced)] = True
    tot = np.zeros((T, 11)); zsum = np.zeros(9); cnt = 0; pickno = 0; picks = []
    wbase = np.ones(9) if catw is None else np.array(catw, float)       # catw: fixed per-category emphasis (e.g. de-emphasise FG%/FT%, ignore TO)
    for j in punts: wbase[j] = 0
    for r in range(R):
        for t in (range(T) if r % 2 == 0 else range(T - 1, -1, -1)):
            pickno += 1
            if t == me:
                if cnt < len(forced):
                    p = forced[cnt]; reserved[p] = False
                else:
                    w = (balanced_weights(zsum, cnt, None if zmat is None else PARS) if cnt > 0 else 1.0) * wbase
                    sc = ZM @ w; sc[~avail | DND | reserved | (ADP > pickno + reach)] = -1e9
                    if sc.max() <= -1e8:                      # nobody within reach: take best available
                        sc = ZM @ w; sc[~avail | DND | reserved] = -1e9
                    p = int(np.argmax(sc))
                zsum += ZM[p]; cnt += 1; picks.append(p)
            else:
                k = ADP + noise[t]; k[~avail | reserved] = 1e9; p = int(np.argmin(k))
            avail[p] = False; tot[t] += M[p]
    if season is not None:
        return season_win(season_cats(tot, season["vol"]) * SIGN, me, season["k"]), picks
    cats = np.array([team_cats(tot[t]) for t in range(T)]) * SIGN
    mine = cats[me]; oth = np.delete(cats, me, 0)
    return ((mine > oth) + 0.5 * (mine == oth)).mean(0), picks

def play_prefix(slot, noise, forced):
    """Draft only as far as your forced picks go (len(forced) rounds). Opponents take the lowest (ADP + noise) player.
    Returns the share of the 9 opponents your forced-pick roster beats in each category, comparing equal-size rosters
    (your k players vs each opponent's k players). Nothing after your last forced pick is assumed."""
    me = slot - 1; L = len(forced)
    avail = np.ones(N, bool); reserved = np.zeros(N, bool); reserved[list(forced)] = True
    tot = np.zeros((T, 11)); cnt = 0
    for r in range(L):
        for t in (range(T) if r % 2 == 0 else range(T - 1, -1, -1)):
            if t == me:
                p = forced[cnt]; reserved[p] = False; cnt += 1
            else:
                k = ADP + noise[t]; k[~avail | reserved] = 1e9; p = int(np.argmin(k))
            avail[p] = False; tot[t] += S[p]
    cats = np.array([team_cats(tot[t]) for t in range(T)]) * SIGN
    mine = cats[me]; oth = np.delete(cats, me, 0)
    return ((mine > oth) + 0.5 * (mine == oth)).mean(0)
