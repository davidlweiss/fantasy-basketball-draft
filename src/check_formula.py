"""Check that the closed-form formula used by the Draft plan tab matches the simulation (play_prefix, fixed-roster season model).
 (b) rounds 7-13: formula vs simulation for rosters of n players, and how well the formula ranks candidate next picks;
 (c) the 'chain' estimate (greedy completion through round 6) vs the best chain stored in results/tree_10team.json.
Last run: corr 0.998-0.999 in rounds 7-13, next-pick ranking Spearman 0.98-1.00 (simulation #1 always in the formula's top 3),
chain within 0.03 categories of the tree's best chain (corr 0.998).  Run: python src/check_formula.py   (takes several minutes)"""
import sys, os, json, math, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from draftlib import *
random.seed(2)
NMAX=R
def opp_stats(nsim=240):
    mu=np.zeros((NMAX+1,9)); var=np.zeros((NMAX+1,9))
    for s in range(nsim):
        noise=np.random.default_rng(31000+s).normal(size=(T,N))*(SIG0 if s%2==0 else SIG2)
        avail=np.ones(N,bool); tot=np.zeros((T,11))
        for r in range(NMAX):
            for t in (range(T) if r%2==0 else range(T-1,-1,-1)):
                k=ADP+noise[t]; k[~avail]=1e9; p=int(np.argmin(k)); avail[p]=False; tot[t]+=SS[p]
            C=season_cats(tot,np.ones((T,7)),1.0)*SIGN
            mu[r+1]+=C.mean(0); var[r+1]+=C.var(0)
    return mu/nsim, np.sqrt(var/nsim)
MU,SD=opp_stats(); K=SEASON["k"]; CV=SEASON["cv"]
Phi=lambda x:0.5*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def heur(picks):
    n=len(picks); t=SS[list(picks)].sum(0)
    c=np.array([t[0]/t[1],t[2]/t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10]])*SIGN
    return Phi((c-MU[n])/np.sqrt((K*CV*np.abs(MU[n]))**2+SD[n]**2))
ORD=[int(i) for i in np.argsort(ADP,kind="stable")]
def pool(s,seq):
    """feasible next picks: window rule + reach, like the tree used (generalised to all rounds)"""
    j=len(seq); P=pick_numbers(s,R)[j]; mine=set(seq)
    return [ORD[r] for r in range(window_start(P,j,mine),POOL) if ORD[r] not in mine and not DND[ORD[r]] and ADP[ORD[r]]<=P+REACH]
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def spear(a,b): 
    ra,rb=rank(a),rank(b); return float(np.corrcoef(ra,rb)[0,1]) if ra.std()>0 and rb.std()>0 else float('nan')
# ---- (b) late rounds: formula vs full simulation for rosters of n players, and ranking of candidate next picks ----
NW=50
res={n:[] for n in range(7,14)}; rk={n:[] for n in range(7,13)}; top={n:[] for n in range(7,13)}
for s in (1,3,5,7,9,10):
    for rep in range(3):
        wl=[make_season_world(55000+1000*s+10*rep+i, SIG0 if i%2==0 else SIG2, fixed=True) for i in range(NW)]
        picks=play(s,wl[0]["noise"],season=wl[0],zmat=ZS)[1]
        for n in range(7,14):
            simv=np.mean([play_prefix(s,w["noise"],tuple(picks[:n]),season=w) for w in wl],axis=0)
            res[n].append((simv.sum(),heur(picks[:n]).sum(),np.abs(simv-heur(picks[:n])).mean()))
            if n<=12:
                cand=pool(s,tuple(picks[:n-1]))[:14]
                if len(cand)>=5:
                    sv=np.array([np.mean([play_prefix(s,w["noise"],tuple(picks[:n-1])+(p,),season=w).sum() for w in wl[:30]]) for p in cand])
                    hv=np.array([heur(tuple(picks[:n-1])+(p,)).sum() for p in cand])
                    rk[n].append(spear(sv,hv)); top[n].append(int(np.argmax(sv) in np.argsort(-hv)[:3]))
print("LATE ROUNDS  n picks | corr(sim, formula) cats won | mean abs diff (cats) | ranking of next pick: Spearman / sim#1 in formula top-3")
for n in range(7,14):
    a=np.array(res[n]); c=np.corrcoef(a[:,0],a[:,1])[0,1]
    print(f"  {n:2d} | {c:.3f} | {np.abs(a[:,0]-a[:,1]).mean():.2f} | "+(f"{np.nanmean(rk[n]):.2f} / {100*np.mean(top[n]):.0f}%" if n<=12 else "-"))
# ---- (c) chain estimate: greedy completion to 6 picks (formula) vs the tree's best chain ----
tree=json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results","tree_10team.json")))
def greedy(s,seq,upto=6):
    seq=tuple(seq)
    while len(seq)<upto:
        c=pool(s,seq)
        if not c: break
        seq+=(max(c,key=lambda p:heur(seq+(p,)).sum()),)
    return seq
def bestleaf(n): return (n["c"],) if not n["k"] else max((bestleaf(k) for k in n["k"]),key=lambda x:x[0])
rows=[]
def walk(s,node,seq):
    for ch in node["k"]:
        cs=seq+(IDX[ch["n"]],)
        rows.append((s,cs,bestleaf(ch)[0])); walk(s,ch,cs)
for s in range(1,11): walk(s,tree[str(s)]["tree"],(IDX[tree[str(s)]["tree"]["n"]],))
random.shuffle(rows); rows=[r for r in rows if len(r[1])<=5][:600]
g=[];t=[]
for s,seq,tc in rows: g.append(heur(greedy(s,seq)).sum()); t.append(tc)
g=np.array(g);t=np.array(t)
print(f"\nCHAIN: greedy-completion estimate vs tree's best chain over {len(rows)} nodes: corr {np.corrcoef(g,t)[0,1]:.3f}; mean (tree - greedy) = {np.mean(t-g):+.2f} cats; mean abs diff {np.mean(np.abs(t-g)):.2f}")
