"""SURE V2.1/V4.1 development pilot. Run from project root.
Uses frozen B3 and B4.1 JSON; no reserved-data scoring. Synthetic fixture only.
"""
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sure.experiment import MODES, features
from sure.policy import STATES, FEATURES
from .evaluate_bayesian_planner import RecoveryClient, normalized

ACTIONS = ('wait', 'reconcile', 'retry', 'failover')
WAIT_MS, HTTP_CAP, MARGIN = 75, 180, 25
MAX_STEPS, CALL_BUDGET = 4, 3
CALL_COST = .025
POLICIES = ('safe_retry', 'fixed_reconcile', 'retry_backoff', 'reconcile_backoff',
            'v2.1', 'v4.1_no_time_context', 'v4.1')


def now():
    return time.perf_counter_ns() / 1e6


class TimeKernel:
    def __init__(self, saved):
        if tuple(saved['states']) != tuple(STATES):
            raise ValueError('Incompatible state order')
        self.edges = tuple(saved['time_edges_ms'])
        self.alpha = float(saved['alpha'])
        self.level = int(saved['selected_level'])
        if self.level not in (0,1,2) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('Invalid kernel settings')
        self.tables = []
        for entries in saved['tables']:
            table = {}
            for entry in entries:
                k = tuple(entry['key'])
                if k in table: raise ValueError('Duplicate kernel key')
                c = Counter()
                for ns,o,d,n in entry['outcomes']:
                    if ns not in STATES or d <= 0 or d % 25 or n <= 0 or (o=='confirmed' and ns!='completed'):
                        raise ValueError('Invalid kernel outcome')
                    if STATES.index(ns) < STATES.index(k[0]): raise ValueError('State regression')
                    c[ns,o,int(d)] += n
                if not c: raise ValueError('Empty kernel cell')
                table[k] = c
            self.tables.append(table)
        if len(self.tables)!=3: raise ValueError('Expected three backoff levels')
        self.reserves = {}
        for k in (False, True):
            for a in ACTIONS if k else ACTIONS[:2]:
                counts = [self.tables[0][s,a,k] for s in STATES]
                self.reserves[a,k] = max(WAIT_MS if a=='wait' else HTTP_CAP,
                    max(d for c in counts for _,_,d in c)) + MARGIN
        # Same admission reserves across all policies and contextual levels.

    def allowed(self, keyed, remaining, calls):
        return tuple(a for a in ACTIONS if (a,keyed) in self.reserves
                     and (a=='wait' or calls>0) and remaining>=self.reserves[a,keyed])

    @lru_cache(maxsize=8192)
    def _dist(self, level, s, a, k, tbin, pa, po):
        base=(s,a,k); keys=(base,base+(tbin,),base+(tbin,pa,po))
        c=self.tables[0][base]; total=sum(c.values()); p={o:n/total for o,n in c.items()}
        for i in range(1,level+1):
            c=self.tables[i].get(keys[i],{}); n=sum(c.values())
            p={o:(c.get(o,0)+self.alpha*v)/(n+self.alpha) for o,v in p.items()}
        if abs(sum(p.values())-1)>1e-9: raise ValueError('Bad probability mass')
        return tuple((STATES.index(ns),o,d,v) for (ns,o,d),v in p.items())

    def joint(self,b,a,k,t,pa,po,level):
        j=defaultdict(lambda:np.zeros(len(STATES)))
        tbin=int(np.searchsorted(self.edges,t,side='right'))
        for s,w in zip(STATES,b):
            if w>0:
                for i,o,d,p in self._dist(level,s,a,k,tbin,pa,po): j[o,d][i]+=float(w)*p
        return j

    @staticmethod
    def marginal(j):
        m=defaultdict(lambda:np.zeros(len(STATES)))
        for (o,d),v in j.items(): m[o]+=v
        return m

    def update(self,b,a,k,t,pa,po,obs,level):
        j=self.joint(b,a,k,t,pa,po,level); m=self.marginal(j)
        if obs in m and m[obs].sum()>0: return normalized(m[obs]),False
        # Unseen observation does not establish completion or permit unsafe retry.
        return normalized(sum(m.values(),np.zeros(len(STATES)))),True

    def choose(self,b,k,t,pa,po,deadline,calls,depth,level,time_cost):
        def search(b,t,pa,po,c,h):
            best=(0.0,'stop',0.0)
            if h<=0: return best
            remaining=deadline-t
            for a in self.allowed(k,remaining,c):
                j=self.joint(b,a,k,t,pa,po,level); m=self.marginal(j)
                cost=int(a!='wait'); expected_duration=sum(d*float(v.sum()) for (o,d),v in j.items())
                value=-CALL_COST*cost-time_cost*expected_duration/deadline
                for (o,d),v in j.items():
                    p=float(v.sum())
                    if p<=0 or d>remaining: continue
                    if o=='confirmed': value+=p
                    elif h>1:
                        # Match B4.1 replay: observation-only posterior; use duration
                        # to advance elapsed context, not as additional evidence.
                        value+=p*search(normalized(m[o]),t+d,a,o,c-cost,h-1)[0]
                # Prefer shorter expected duration on positive-utility ties.
                if value>best[0]+1e-12 or (value>0 and abs(value-best[0])<=1e-12 and expected_duration<best[2]):
                    best=(value,a,expected_duration)
            return best
        return search(normalized(b),t,pa,po,calls,depth)[:2]


def run_episode(client,kernel,bayes,policy,case,mode,keyed,cap,deadline,time_cost):
    ident=client.create(mode,keyed)
    start=now(); obs,_=client.action('retry',ident,keyed,deadline,cap); dt=now()-start
    initial_obs=obs; initial_elapsed=dt; pa='initial'; po=obs
    b=None; trace=[]; calls=steps=unseen=0; decision_ms=0.
    model_policy=policy.startswith('v'); level=0 if policy=='v4.1_no_time_context' else kernel.level
    if obs!='confirmed' and model_policy:
        tick=now(); b=normalized(bayes.predict(features(obs,dt,keyed,cap))); decision_ms+=now()-tick
    while obs!='confirmed' and steps<MAX_STEPS:
        t=now()-start; tick=now(); allowed=kernel.allowed(keyed,deadline-t,CALL_BUDGET-calls); utility=None
        if policy in ('safe_retry','fixed_reconcile','retry_backoff','reconcile_backoff'):
            target='retry' if policy in ('safe_retry','retry_backoff') else 'reconcile'
            # Fixed 75 ms delay after an unsuccessful network attempt, including the initial request.
            if target=='retry' and not keyed: a='stop'
            elif target not in allowed: a='stop'
            elif policy.endswith('_backoff') and pa!='wait' and 'wait' in allowed and steps+1<MAX_STEPS and deadline-t>=kernel.reserves['wait',keyed]+kernel.reserves[target,keyed]: a='wait'
            else: a=target
        else:
            utility,a=kernel.choose(b,keyed,t,pa,po,deadline,CALL_BUDGET-calls,min(1 if policy=='v2.1' else 2,MAX_STEPS-steps),level,time_cost)
        decision_ms+=now()-tick
        t=now()-start
        if a=='stop' or a not in kernel.allowed(keyed,deadline-t,CALL_BUDGET-calls): break
        if a in ('retry','failover') and not keyed: raise RuntimeError('Unsafe dispatch')
        before=b.tolist() if b is not None else None
        tick=now(); obs,_=client.action(a,ident,keyed,deadline-t); duration=now()-tick
        steps+=1; calls+=int(a!='wait')
        if b is not None and obs!='confirmed':
            tick=now(); b,unknown=kernel.update(b,a,keyed,t,pa,po,obs,level); unseen+=int(unknown); decision_ms+=now()-tick
        trace.append(dict(action=a,observation=obs,elapsed_before_ms=t,duration_ms=duration,expected_utility=utility,
                          belief_before=before,belief_after=b.tolist() if b is not None and obs!='confirmed' else None))
        pa,po=a,obs
    elapsed=now()-start
    # Truth is queried only after termination, never used as a policy input.
    time.sleep(.40); truth=client.truth(ident); effects=int(truth['effects']); confirmed=obs=='confirmed'
    return dict(case=case,episode=ident,policy=policy,mode=mode,keyed=keyed,initial_timeout_ms=cap,
        initial_observation=initial_obs,initial_elapsed_ms=initial_elapsed,deadline_ms=deadline,
        confirmed_on_time=int(confirmed and elapsed<=deadline and effects==1),duplicate=int(effects>1),effects=effects,
        final_observation=obs,unresolved=int(not confirmed),deadline_overrun=int(elapsed>deadline),latency_ms=elapsed,
        recovery_calls=calls,waits=sum(x['action']=='wait' for x in trace),steps=steps,decision_ms=decision_ms,
        unseen_observations=unseen,trace=json.dumps(trace))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=14); p.add_argument('--seed',type=int,default=4141)
    p.add_argument('--base',default='http://127.0.0.1:9090'); p.add_argument('--b3',default='bayesian/outputs/b3/bayesian_b3.joblib')
    p.add_argument('--kernel',default='bayesian/outputs/b41/kernel_tables.json'); p.add_argument('--out',default='bayesian/outputs/planner_v41_pilot')
    p.add_argument('--time-cost',type=float,default=.10); args=p.parse_args()
    if args.n<14 or args.n%14 or not math.isfinite(args.time_cost) or args.time_cost<0: p.error('Use a positive multiple of 14 and a finite nonnegative time cost')
    saved=joblib.load(args.b3)
    if tuple(saved['states'])!=tuple(STATES) or tuple(saved['features'])!=tuple(FEATURES): raise ValueError('Incompatible B3')
    kernel=TimeKernel(json.loads(Path(args.kernel).read_text(encoding='utf-8'))); client=RecoveryClient(args.base)
    if client.admin('/health').get('status')!='ok': raise RuntimeError('Service unhealthy')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(args.seed); cells=[(m,k) for m in MODES for k in (False,True)]*(args.n//14); rng.shuffle(cells)
    design=[dict(case=i,mode=m,keyed=k,cap=rng.choice([100,150,200]),deadline=rng.choice([500,750,1000])) for i,(m,k) in enumerate(cells)]
    pd.DataFrame(design).to_csv(out/'design.csv',index=False)
    (out/'config.json').write_text(json.dumps(dict(vars(args),selected_level=kernel.level,call_cost=CALL_COST,wait_ms=WAIT_MS,http_cap=HTTP_CAP,max_steps=MAX_STEPS,call_budget=CALL_BUDGET,admission_margin=MARGIN,policies=POLICIES),indent=2),encoding='utf-8')
    print('Admission reserves:',kernel.reserves,'; time cost:',args.time_cost,flush=True)
    rows=[]
    for cell in design:
        order=list(POLICIES); rng.shuffle(order)
        for policy in order:
            r=run_episode(client,kernel,saved['bayes'],policy,time_cost=args.time_cost,**cell); rows.append(r)
            pd.DataFrame([r]).to_csv(out/'runs.csv',mode='a',header=len(rows)==1,index=False)
            if r['duplicate'] or (r['final_observation']=='confirmed' and r['effects']!=1): raise RuntimeError('Safety/integrity violation; inspect runs.csv')
        print(f"Matched cases: {cell['case']+1}/{args.n}",flush=True)
    df=pd.DataFrame(rows)
    def summarize(group):
        return group.agg(n=('case','size'),success=('confirmed_on_time','mean'),duplicates=('duplicate','sum'),unresolved=('unresolved','mean'),deadline_overrun=('deadline_overrun','mean'),p95_ms=('latency_ms',lambda x:x.quantile(.95)),recovery_calls=('recovery_calls','mean'),waits=('waits','mean'),decision_ms=('decision_ms','mean'),unseen_observations=('unseen_observations','sum'))
    summary=summarize(df.groupby('policy')); summary.to_csv(out/'summary.csv'); print(summary.to_string())
    summarize(df.groupby(['policy','keyed'])).to_csv(out/'summary_by_keyed.csv')
    paired=df.pivot(index='case',columns='policy',values='confirmed_on_time')
    for comparator in ('v2.1','v4.1_no_time_context'):
        print(f'V4.1 vs {comparator}: wins / losses / ties:',int((paired['v4.1']>paired[comparator]).sum()),int((paired['v4.1']<paired[comparator]).sum()),int((paired['v4.1']==paired[comparator]).sum()))
    print('Development pilot only. Fixed-cap admission is not a hard deadline guarantee. No reserved CSV scoring.')

if __name__=='__main__': main()
