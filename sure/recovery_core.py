"""Reusable SURE recovery core, extracted from frozen V4.1 implementation.
No fixture, ground truth, training, or experiment imports. Timing in ms.
Safety assumes a verified shared deduplication contract and trusted confirmations.
HTTP timeouts/admission reserves do not guarantee hard wall-clock deadlines.
"""
import math
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Protocol
import numpy as np
from .policy import STATES, FEATURES

ACTIONS = ('wait', 'reconcile', 'retry', 'failover')
WAIT_MS, HTTP_CAP, MARGIN = 75, 180, 25
MAX_STEPS, CALL_BUDGET = 4, 3
CALL_COST = .025

BASELINES = ('safe_retry','fixed_reconcile','retry_backoff','reconcile_backoff',
             'failover_backoff','hybrid_backoff')
SPECS = {**{p:None for p in BASELINES}, 'v2':(1,.10,2), 'v4':(2,.10,2),
         'v4_no_time':(2,.10,0), 'v4_time_only':(2,.10,1),
         'v3':(1,.10,2), 'v5':(2,.10,2)}

class Estimator(Protocol):
    def predict(self, f): ...

class Transport(Protocol):
    def __call__(self, action, remaining_ms, cap_ms=HTTP_CAP): ...


def features(obs, dt, keyed, cap):
    return dict(elapsed_ms=dt,error=obs,keyed=keyed,initial_timeout_ms=cap)


def normalized(b):
    b = np.asarray(b, dtype=float)
    if b.shape != (len(STATES),) or not np.isfinite(b).all() or (b < 0).any() or b.sum() <= 0:
        raise ValueError('Invalid belief')
    return b / b.sum()


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



def rule(policy,keyed,obs,pa,po,allowed,kernel,remaining,steps,last_network):
    if policy in ('safe_retry','retry_backoff'):
        target='retry' if keyed else 'stop'
    elif policy in ('fixed_reconcile','reconcile_backoff'):
        target='reconcile'
    elif policy=='failover_backoff':
        target='failover' if keyed else 'reconcile'
    elif policy=='hybrid_backoff':
        # Initial ambiguous response: status first, except explicit error on
        # keyed request. After status failure use backup; after backup failure
        # use primary. A pending response is never proof of nonexecution.
        if not keyed: target='reconcile'
        elif pa=='initial': target='failover' if obs=='error' else 'reconcile'
        elif last_network=='reconcile': target='failover' if po!='pending' and pa!='wait' else 'reconcile'
        elif last_network=='failover': target='retry' if po in ('error','timeout','unknown') and pa!='wait' else 'failover'
        else: target='reconcile' if po in ('error','timeout','unknown') and pa!='wait' else 'retry'
    else: raise ValueError(policy)
    if target not in allowed: return 'stop'
    # Wait before the next call only when an admitted network action can follow.
    backoff=policy in ('retry_backoff','reconcile_backoff','failover_backoff','hybrid_backoff')
    if policy=='hybrid_backoff': backoff=obs=='pending'
    if backoff and pa!='wait' and 'wait' in allowed and steps+1<MAX_STEPS and remaining>=kernel.reserves['wait',keyed]+kernel.reserves[target,keyed]:
        return 'wait'
    return target

def recover(transport: Transport, kernel: TimeKernel, estimator: Estimator,
            policy: str, *, keyed: bool, cap: int, deadline: int, budget: int,
            clock=now):
    """Execute one operation and its recovery. Caller supplies a request-bound transport.

    `keyed=True` requires verified deduplication across execution endpoints.
    The initial submission is not charged to the recovery-call budget.
    Transport returns (validated observation, duration_ms); measured clock is used.
    A confirmed observation must represent authoritative confirmation for this request.
    No server truth, simulated failure mode or future outcomes are accepted.
    The result is a client-side report; independently audit execution effects outside.
    V3/V5 use the caller's RF estimator; V2/V4 use its Bayesian estimator.
    """
    if policy not in SPECS:
        raise ValueError('Unknown policy')
    if type(keyed) is not bool:
        raise ValueError('keyed must be a verified boolean contract')
    if type(cap) is not int or cap <= 0 or type(deadline) is not int or deadline <= 0:
        raise ValueError('Positive integer cap and deadline required')
    if type(budget) is not int or budget < 0:
        raise ValueError('Nonnegative integer recovery budget required')
    spec = SPECS[policy]
    if spec is not None and estimator is None:
        raise ValueError('An estimator is required for probabilistic policies')
    start=clock()
    obs,_=transport('retry',deadline,cap); initial_elapsed=clock()-start
    initial_obs=obs; pa='initial'; po=obs; last_network='initial'; b=None
    trace=[]; calls=steps=unseen=0; decision=0.; reason='step_limit'
    if spec is not None and obs!='confirmed':
        tick=clock(); b=normalized(estimator.predict(features(obs,initial_elapsed,keyed,cap))); decision+=clock()-tick
    while obs!='confirmed' and steps<MAX_STEPS:
        t=clock()-start; remaining=deadline-t; tick=clock()
        allowed=kernel.allowed(keyed,remaining,budget-calls); utility=None
        if spec is None:
            a=rule(policy,keyed,obs,pa,po,allowed,kernel,remaining,steps,last_network)
        else:
            depth,cost,level=spec
            utility,a=kernel.choose(b,keyed,t,pa,po,deadline,budget-calls,min(depth,MAX_STEPS-steps),level,cost)
        decision+=clock()-tick
        if a=='stop': reason='call_budget' if calls>=budget else 'policy_stop'; break
        # Retain original runtime admission and safety gates; count planning time.
        t=clock()-start
        if a not in kernel.allowed(keyed,deadline-t,budget-calls): reason='admission_after_compute'; break
        if not keyed and a in ('retry','failover'): raise RuntimeError('Unsafe recovery dispatch')
        before=b.tolist() if b is not None else None
        tick=clock(); obs,_=transport(a,deadline-t); duration=clock()-tick
        steps+=1; calls+=int(a!='wait')
        if a!='wait': last_network=a
        if b is not None and obs!='confirmed':
            tick=clock(); b,unknown=kernel.update(b,a,keyed,t,pa,po,obs,level); unseen+=int(unknown); decision+=clock()-tick
        trace.append(dict(action=a,observation=obs,elapsed_before_ms=t,duration_ms=duration,expected_utility=utility,
                          belief_before=before,belief_after=b.tolist() if b is not None and obs!='confirmed' else None))
        pa,po=a,obs
    elapsed=clock()-start
    if obs=='confirmed': reason='confirmed'
    elif elapsed>deadline: reason='deadline'
    elif steps>=MAX_STEPS: reason='step_limit'
    if calls>budget or steps>MAX_STEPS: raise RuntimeError('Resource violation')
    return dict(policy=policy, initial_observation=initial_obs,
                initial_elapsed_ms=initial_elapsed, final_observation=obs,
                latency_ms=elapsed, client_confirmed_on_time=int(obs=='confirmed' and elapsed<=deadline),
                unresolved=int(obs!='confirmed'), deadline_overrun=int(elapsed>deadline),
                recovery_calls=calls, steps=steps, waits=sum(x['action']=='wait' for x in trace),
                decision_ms=decision, unseen_observations=unseen,
                termination_reason=reason, remaining_ms=deadline-elapsed, trace=trace)
