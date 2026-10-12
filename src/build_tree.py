"""Draft decision tree for your first `plan_rounds` (6) picks -> results/tree_<slug>.json (10team default; DRAFT_CONFIG=config_12team.json for 12 team)
Anchors = the top `teams` players by ADP (never do_not_draft); anchor k is what you take at slot k in round 1.
Every node gets up to BRANCH children for the next round, ranked 1..BRANCH:
  feasible = still plausibly on the board at that pick (window rule from draftlib, `reach` rule, top-`pool` players,
  never do_not_draft / first_four_exclude). The best K_CAND by Fit are scored by simulating the draft only up to that round
  (opponents draft by ADP + noise; same random draws for every candidate) -- the rest of your draft is NOT simulated; the top BRANCH become children.
  Node 'c' = categories won (of 9) by the picks so far vs opponents' rosters of the same size (no later picks assumed);
  'w' = % of opponents beaten in each category (CATS order); 'g' = 10 x (avg z of your picks - par for that many picks). Gaps under ~0.1 are ties.
  BRANCH=5 over rounds 2-6 -> 3,125 leaf paths per anchor, 31,250 in all.
Run: python src/build_tree.py            (all anchors, ~20-30 min)
     python src/build_tree.py 1 6        (just those slots)
     python src/build_tree.py --quick    (small, smoke test)"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
from draftlib import *

LV = CFG.get("plan_rounds", 6)
QUICK = "--quick" in sys.argv
BRANCH, K_CAND, NS = (2, 5, 6) if QUICK else (5, 20, 40)
BLOCK = DND.copy()
for n in CFG.get("first_four_exclude", []): BLOCK[IDX[n]] = True
ANCHORS = [p for p in ORDER if not DND[p]][:T]                      # slot k (1-based) -> ANCHORS[k-1]

def candidates(s, seq):
    j = len(seq); P = pick_numbers(s, LV)[j]; mine = set(seq)
    cand = [ORDER[r] for r in range(window_start(P, j, mine), POOL)
            if ORDER[r] not in mine and not BLOCK[ORDER[r]] and ADP[ORDER[r]] <= P + REACH]
    w = balanced_weights(Z[list(seq)].sum(0), j)
    cand.sort(key=lambda p: -(Z[p] @ w))
    return cand[:K_CAND]

def evaluate(s, seq, noises):
    """Per-category share of the 9 opponents your picks-so-far roster beats, vs their rosters of the same size
    (play_prefix). Independent of whatever you would draft later."""
    return np.mean([play_prefix(s, nz, seq) for nz in noises], axis=0)

def build(s, seq, noises, win):
    node = dict(n=NAMES[seq[-1]], adp=round(float(ADP[seq[-1]]), 1), c=round(float(win.sum()), 2),
                w=[int(round(100 * x)) for x in win],
                g=[int(round(10 * x)) for x in Z[list(seq)].mean(0) - PAR[len(seq)]], k=[])
    if len(seq) == LV: return node
    scored = sorted(((evaluate(s, seq + (p,), noises), p) for p in candidates(s, seq)),
                    key=lambda t: -t[0].sum())[:BRANCH]
    node["k"] = [build(s, seq + (p,), noises, v) for v, p in scored]
    return node

def run_slot(s):
    t0 = time.time(); a = ANCHORS[s - 1]
    noises = [np.random.default_rng(10000 * s + i).normal(size=(T, N)) * (SIG0 if i % 2 == 0 else SIG2) for i in range(NS)]
    c0 = evaluate(s, (a,), noises)
    tree = build(s, (a,), noises, c0)
    return s, dict(pick=pick_numbers(s, LV), tree=tree, secs=round(time.time() - t0))

def leaves(n):
    return [n] if not n["k"] else [l for k in n["k"] for l in leaves(k)]

if __name__ == "__main__":
    slots = [int(x) for x in sys.argv[1:] if x.isdigit()] or list(range(1, T + 1))
    with Pool(4) as p: out = p.map(run_slot, slots, chunksize=1)
    path = os.path.join(ROOT, "results", "tree_quick.json" if QUICK else f"tree_{SLUG}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(s): d for s, d in out})
    json.dump(old, open(path, "w"), separators=(",", ":"))
    for s, d in out: print(s, d["tree"]["n"], len(leaves(d["tree"])), "leaves", d["secs"], "s")
