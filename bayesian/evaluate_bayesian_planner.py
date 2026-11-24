"""SURE V2/V4 pilot, frozen B3/B4 models; no held-out CSV evaluation.
Run from project root: python -m bayesian.evaluate_bayesian_planner --n 14
Local synthetic fixture only. HTTP timeouts are not hard wall-clock guarantees.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sure.experiment import Client, MODES, features, ms
from sure.policy import STATES, FEATURES

HTTP_CAP = 180
WAIT_MS = 75
MARGIN = 25  # shared pilot admission margin, not a proven upper bound
CALL_COST = 0.025
RECOVERY_CALLS = 3
MAX_STEPS = 4
POLICIES = ('safe_retry', 'fixed_reconcile', 'v2', 'v4')


def normalized(b):
    b = np.asarray(b, dtype=float)
    if b.shape != (len(STATES),) or not np.isfinite(b).all() or (b < 0).any() or b.sum() <= 0:
        raise ValueError('Invalid belief')
    return b / b.sum()


class RecoveryClient(Client):
    def action(self, a, ident, keyed, remaining, cap=HTTP_CAP):
        start = ms()
        if remaining <= 0:
            raise ValueError('Action dispatched after deadline')
        if a == 'wait':
            time.sleep(min(WAIT_MS, remaining) / 1000)
            return 'waited', ms() - start
        timeout = min(cap, remaining) / 1000
        try:
            if a == 'reconcile':
                res = self.session.get(f'{self.base}/primary/payments/{ident}', timeout=timeout, allow_redirects=False)
            elif a in ('retry', 'failover'):
                endpoint = 'backup' if a == 'failover' else 'primary'
                res = self.session.post(f'{self.base}/{endpoint}/payments', json={'id': ident, 'keyed': keyed}, timeout=timeout, allow_redirects=False)
            else:
                raise ValueError('Unknown action')
            body = res.json()
            if not res.ok or not isinstance(body, dict) or body.get('id') != ident:
                obs = 'error'
            else:
                obs = {'completed':'confirmed','pending':'pending','unknown':'unknown'}.get(body.get('status'), 'error')
        except requests.Timeout:
            obs = 'timeout'
        except (requests.RequestException, ValueError):
            obs = 'error'
        return obs, ms() - start


class Planner:
    def __init__(self, kernel):
        self.kernel = kernel
        self.tables = {}
        self.reserves = {}
        for keyed in (False, True):
            for a in ('wait','reconcile') + (('retry','failover') if keyed else ()):
                rows = []
                for s in STATES:
                    counts = kernel.counts.get((s,a,keyed))
                    if not counts or sum(counts.values()) <= 0:
                        raise ValueError(f'Missing kernel cell: {s}, {a}, {keyed}')
                    total = sum(counts.values())
                    for (ns,o,d),n in counts.items():
                        if ns not in STATES or d <= 0 or n <= 0:
                            raise ValueError('Invalid kernel outcome')
                        if o == 'confirmed' and ns != 'completed':
                            raise ValueError('Invalid confirmation in kernel')
                        rows.append((STATES.index(s),STATES.index(ns),o,int(d),n/total))
                self.tables[(a,keyed)] = rows
                self.reserves[(a,keyed)] = max(WAIT_MS if a=='wait' else HTTP_CAP, max(x[3] for x in rows)) + MARGIN

    def allowed(self, keyed, remaining, calls):
        # An unkeyed ambiguous operation can NEVER be retried/failover here.
        return tuple(a for a in ('wait','reconcile','retry','failover')
                     if (a,keyed) in self.tables and (a=='wait' or calls>0)
                     and remaining>=self.reserves[(a,keyed)])

    def distribution(self,b,a,keyed):
        joint=defaultdict(lambda:np.zeros(len(STATES)))
        for i,j,o,d,p in self.tables[(a,keyed)]:
            joint[o,d][j] += b[i]*p
        return joint

    def update(self,b,a,keyed,obs):
        joint=self.distribution(normalized(b),a,keyed)
        predicted=sum(joint.values(),np.zeros(len(STATES)))
        matching=sum((v for (o,d),v in joint.items() if o==obs),np.zeros(len(STATES)))
        # An unseen observation is flagged. Use the predictive marginal only,
        # never manufacture evidence of confirmation or allow unsafe actions.
        if matching.sum()<=0:
            return normalized(predicted), True
        return normalized(matching), False

    def choose(self,belief,keyed,remaining,calls,depth):
        def search(b,r,c,h):
            best=(0.0,'stop')
            if h<=0 or r<=0:
                return best
            for a in self.allowed(keyed,r,c):
                joint=self.distribution(b,a,keyed)
                marginals=defaultdict(lambda:np.zeros(len(STATES)))
                for (o,d),v in joint.items():
                    marginals[o]+=v
                cost=int(a!='wait'); value=-CALL_COST*cost
                for (o,d),v in joint.items():
                    p=float(v.sum())
                    if p<=0 or d>r:
                        continue  # no success credit or future state if action exceeds deadline
                    if o=='confirmed':
                        reward=1.0
                    elif h>1:
                        reward=search(normalized(marginals[o]),r-d,c-cost,h-1)[0]
                    else:
                        reward=0.0
                    value+=p*reward
                if value>best[0]+1e-12:
                    best=(value,a)
            return best
        return search(normalized(belief),remaining,calls,depth)[1]


def run_episode(client,planner,bayes,policy,case,mode,keyed,cap,deadline):
    ident=client.create(mode,keyed)
    start=ms(); obs,dt=client.action('retry',ident,keyed,deadline,cap)
    b=None; steps=0; calls=0; decision_ms=0.; unseen=0; trace=[]
    if obs!='confirmed' and policy in ('v2','v4'):
        ts=ms(); b=normalized(bayes.predict(features(obs,dt,keyed,cap))); decision_ms+=ms()-ts
    while obs!='confirmed' and steps<MAX_STEPS:
        remaining=deadline-(ms()-start)
        ts=ms(); allowed=planner.allowed(keyed,remaining,RECOVERY_CALLS-calls)
        if policy=='safe_retry': a='retry' if 'retry' in allowed else 'stop'
        elif policy=='fixed_reconcile': a='reconcile' if 'reconcile' in allowed else 'stop'
        else: a=planner.choose(b,keyed,remaining,RECOVERY_CALLS-calls,min(2 if policy=='v4' else 1,MAX_STEPS-steps))
        decision_ms+=ms()-ts
        remaining=deadline-(ms()-start)
        if a=='stop' or a not in planner.allowed(keyed,remaining,RECOVERY_CALLS-calls):
            break
        before=b.tolist() if b is not None else None
        obs,duration=client.action(a,ident,keyed,remaining)
        steps+=1; calls+=int(a!='wait')
        if b is not None and obs!='confirmed':
            ts=ms(); b,unknown=planner.update(b,a,keyed,obs); unseen+=int(unknown); decision_ms+=ms()-ts
        trace.append({'action':a,'observation':obs,'duration_ms':duration,'belief_before':before,
                      'belief_after':b.tolist() if b is not None and obs!='confirmed' else None})
    elapsed=ms()-start
    # Privileged truth only after policy terminates. Allow late fixture commits.
    time.sleep(.40); truth=client.truth(ident); effects=int(truth['effects'])
    confirmed=obs=='confirmed'
    return dict(case=case,episode=ident,policy=policy,mode=mode,keyed=keyed,initial_timeout_ms=cap,
                deadline_ms=deadline,confirmed_on_time=int(confirmed and elapsed<=deadline and effects==1),
                duplicate=int(effects>1),effects=effects,final_observation=obs,unresolved=int(not confirmed),
                deadline_overrun=int(elapsed>deadline),latency_ms=elapsed,recovery_calls=calls,
                total_client_calls=1+calls,steps=steps,decision_ms=decision_ms,unseen_observations=unseen,
                trace=json.dumps(trace))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=14,help='Number of matched scenario cases (multiple of 14)')
    p.add_argument('--seed',type=int,default=4041)
    p.add_argument('--base',default='http://127.0.0.1:9090')
    p.add_argument('--model',default='bayesian/outputs/b4/bayesian_b4.joblib')
    p.add_argument('--out',default='bayesian/outputs/planner_pilot')
    args=p.parse_args()
    if args.n<14 or args.n%14: p.error('--n must be a positive multiple of 14')
    out=Path(args.out)
    if out.exists(): raise FileExistsError(f'{out} exists; use a new --out to preserve earlier results')
    saved=joblib.load(args.model)
    if tuple(saved['states'])!=tuple(STATES) or tuple(saved['features'])!=tuple(FEATURES) or saved['duration_bin_ms']!=25:
        raise ValueError('Incompatible B4 model')
    planner=Planner(saved['kernel']); client=RecoveryClient(args.base)
    if client.admin('/health').get('status')!='ok': raise RuntimeError('Testbed unhealthy')
    rng=random.Random(args.seed)
    cells=[(m,k) for m in MODES for k in (False,True)]*(args.n//14)
    rng.shuffle(cells)
    out.mkdir(parents=True)
    design=[dict(case=i,mode=m,keyed=k,cap=rng.choice([100,150,200]),deadline=rng.choice([500,750,1000])) for i,(m,k) in enumerate(cells)]
    pd.DataFrame(design).to_csv(out/'design.csv',index=False)
    print('Admission reserves (ms):',planner.reserves,flush=True)
    rows=[]
    for cell in design:
        order=list(POLICIES); rng.shuffle(order)
        for policy in order:
            r=run_episode(client,planner,saved['bayes'],policy,**cell)
            rows.append(r)
            pd.DataFrame([r]).to_csv(out/'runs.csv',mode='a',header=len(rows)==1,index=False)
            if r['duplicate'] or (r['effects']!=1 and r['final_observation']=='confirmed'):
                raise RuntimeError('Safety/integrity violation; inspect runs.csv before continuing')
        print(f"Matched cases: {cell['case']+1}/{args.n}",flush=True)
    df=pd.DataFrame(rows)
    summary=df.groupby('policy').agg(n=('case','size'),success=('confirmed_on_time','mean'),duplicates=('duplicate','sum'),
        unresolved=('unresolved','mean'),deadline_overrun=('deadline_overrun','mean'),p95_ms=('latency_ms',lambda x:x.quantile(.95)),
        recovery_calls=('recovery_calls','mean'),decision_ms=('decision_ms','mean'),unseen_observations=('unseen_observations','sum'))
    summary.to_csv(out/'summary.csv'); print(summary.to_string())
    paired=df.pivot(index='case',columns='policy',values='confirmed_on_time')
    print('V4 wins / V2 wins / ties:',int((paired.v4>paired.v2).sum()),int((paired.v4<paired.v2).sum()),int((paired.v4==paired.v2).sum()))
    print('Pilot only: frozen models, new episodes, no held-out CSV scoring. Fixed-cap action admission; multi-step model not yet validated.')

if __name__=='__main__': main()
