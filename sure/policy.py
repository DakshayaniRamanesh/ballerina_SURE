"""Shared estimators and finite-horizon belief planner. No privileged inputs."""
from collections import Counter, defaultdict
import numpy as np

STATES=('unstarted','processing','completed')
ACTIONS=('retry','failover','reconcile','wait','stop')
FEATURES=('elapsed_ms','error','keyed','initial_timeout_ms')

def norm(x):
    x=np.asarray(x,dtype=float); return x/x.sum() if x.sum() else np.ones(3)/3

def safe_actions(keyed, remaining, budget):
    if remaining<=0: return ('stop',)
    # No authoritative terminal no-commit evidence in MVP; unknown != safe.
    out=['stop','wait']
    if budget>0:
        out.append('reconcile')
        if keyed: out.extend(['retry','failover'])
    return tuple(out)

def feature_key(f): return (f['error'],bool(f['keyed']),int(f['initial_timeout_ms']//50))

class EmpiricalBayes:
    def fit(self, rows):
        self.counts=defaultdict(Counter); self.total=Counter()
        for r in rows:
            self.counts[feature_key(r)][r['state']]+=1; self.total[r['state']]+=1
        return self
    def predict(self,f):
        c=self.counts.get(feature_key(f),self.total)
        return norm([c[s]+1 for s in STATES])

class Kernel:
    """Empirical joint model P(next state, observation, duration | state, action, keyed).
    Duration discretized upward into 25ms bins. Shared by all estimators/planners.
    A short timeout may censor an action; simplified model for initial prototype.
    """
    def fit(self,rows):
        self.counts=defaultdict(Counter)
        for r in rows:
            dt=max(25,int(np.ceil(r['duration_ms']/25))*25)
            self.counts[(r['state'],r['action'],bool(r['keyed']))][(r['next_state'],r['observation'],dt)]+=1
        return self
    def outcomes(self,state,action,keyed):
        c=self.counts.get((state,action,keyed))
        if not c: # conservative unknown outcome, no invented success
            return [(state,'unknown',100,1.)]
        total=sum(c.values()); return [(s,o,d,n/total) for (s,o,d),n in c.items()]
    def branches(self,b,action,keyed,remaining):
        groups=defaultdict(lambda:np.zeros(3))
        for i,s in enumerate(STATES):
            for ns,o,d,p in self.outcomes(s,action,keyed):
                # If action cannot finish within budget, it cannot confirm success.
                if d>remaining: o='timeout'; d=remaining
                groups[(o,d)][STATES.index(ns)]+=b[i]*p
        return [(o,d,float(v.sum()),norm(v)) for (o,d),v in groups.items() if v.sum()>0]
    def update(self,b,action,keyed,observation,duration):
        # Use observation marginal over duration; avoids brittle exact-bin matching.
        arr=np.zeros(3); predicted=np.zeros(3)
        for i,s in enumerate(STATES):
            for ns,o,d,p in self.outcomes(s,action,keyed):
                predicted[STATES.index(ns)]+=b[i]*p
                if o==observation: arr[STATES.index(ns)]+=b[i]*p
        return norm(arr if arr.sum() else predicted)

class Planner:
    def __init__(self,kernel,depth=2,call_cost=.025): self.kernel=kernel; self.depth=depth; self.call_cost=call_cost
    def choose(self,belief,keyed,remaining,budget):
        def search(b,d,r,h):
            best=(0.,'stop')
            if h<=0 or d<=0: return best
            for a in safe_actions(keyed,d,r):
                if a=='stop': continue
                cost=0 if a=='wait' else 1
                value=-self.call_cost*cost
                for o,dt,p,nb in self.kernel.branches(b,a,keyed,d):
                    reward=1. if o=='confirmed' else 0.
                    if not reward and h>1 and d>dt:
                        reward=search(nb,d-dt,r-cost,h-1)[0]
                    value+=p*reward
                if value>best[0]+1e-12: best=(value,a)
            return best
        return search(np.asarray(belief),remaining,budget,self.depth)[1]
