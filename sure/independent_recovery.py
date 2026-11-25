"""Shared evaluation adapter: uses existing frozen TimeKernel without changing its policy."""
import json
import time
import uuid
from typing import Protocol
from bayesian.evaluate_time_aware_planner import (TimeKernel, RecoveryClient, normalized,
    now, STATES, FEATURES, features, MAX_STEPS)
from bayesian.evaluate_planner_sensitivity import rule, BASELINES

class Estimator(Protocol):
    def predict(self, f): ...  # state order: unstarted, processing, completed

class ExtendedClient(RecoveryClient):
    def create_case(self, case):
        ident = uuid.uuid4().hex
        payload = {k:case[k] for k in ('keyed','process_ms','primary','backup','status')}
        if self.admin('/admin/scenario',dict(id=ident,**payload)).get('ok') is not True:
            raise RuntimeError('Scenario registration failed')
        return ident

    def settled_truth(self, ident, expected_max_requests):
        # All simulated handler delays are bounded by 350+1000+500 ms.
        # Check active handlers, then verify a quiet period. Local fixture only.
        end = time.monotonic()+4
        while time.monotonic()<end:
            t = self.truth(ident)
            if not t or 'active' not in t: raise RuntimeError('Missing truth ledger')
            if t['active']==0:
                time.sleep(.10)
                u=self.truth(ident)
                if t==u and u['active']==0:
                    if not 1<=u['attempts']+u['status_calls']<=expected_max_requests:
                        raise RuntimeError('Unexpected ledger request count')
                    return u
            time.sleep(.05)
        raise RuntimeError('Ledger did not settle; abort rather than report incomplete truth')

# Baseline names and original deterministic rules are shared with sensitivity experiment.
SPECS={**{p:None for p in BASELINES},'v2':(1,.10,2),'v4':(2,.10,2),
       'v4_no_time':(2,.10,0),'v4_time_only':(2,.10,1)}

def episode(client, kernel, estimator: Estimator, policy, spec, case):
    keyed=case['keyed']; cap=case['cap']; deadline=case['deadline']; budget=case['budget']
    ident=client.create_case(case); start=now()
    obs,_=client.action('retry',ident,keyed,deadline,cap); initial_elapsed=now()-start
    initial_obs=obs; pa='initial'; po=obs; last_network='initial'; b=None
    trace=[]; calls=steps=unseen=0; decision=0.; reason='step_limit'
    if spec is not None and obs!='confirmed':
        tick=now(); b=normalized(estimator.predict(features(obs,initial_elapsed,keyed,cap))); decision+=now()-tick
    while obs!='confirmed' and steps<MAX_STEPS:
        t=now()-start; remaining=deadline-t; tick=now()
        allowed=kernel.allowed(keyed,remaining,budget-calls); utility=None
        if spec is None:
            a=rule(policy,keyed,obs,pa,po,allowed,kernel,remaining,steps,last_network)
        else:
            depth,cost,level=spec
            utility,a=kernel.choose(b,keyed,t,pa,po,deadline,budget-calls,min(depth,MAX_STEPS-steps),level,cost)
        decision+=now()-tick
        if a=='stop': reason='call_budget' if calls>=budget else 'policy_stop'; break
        # Retain original runtime admission and safety gates; count planning time.
        t=now()-start
        if a not in kernel.allowed(keyed,deadline-t,budget-calls): reason='admission_after_compute'; break
        if not keyed and a in ('retry','failover'): raise RuntimeError('Unsafe recovery dispatch')
        before=b.tolist() if b is not None else None
        tick=now(); obs,_=client.action(a,ident,keyed,deadline-t); duration=now()-tick
        steps+=1; calls+=int(a!='wait')
        if a!='wait': last_network=a
        if b is not None and obs!='confirmed':
            tick=now(); b,unknown=kernel.update(b,a,keyed,t,pa,po,obs,level); unseen+=int(unknown); decision+=now()-tick
        trace.append(dict(action=a,observation=obs,elapsed_before_ms=t,duration_ms=duration,expected_utility=utility,
                          belief_before=before,belief_after=b.tolist() if b is not None and obs!='confirmed' else None))
        pa,po=a,obs
    elapsed=now()-start
    if obs=='confirmed': reason='confirmed'
    elif elapsed>deadline: reason='deadline'
    elif steps>=MAX_STEPS: reason='step_limit'
    if calls>budget or steps>MAX_STEPS: raise RuntimeError('Resource violation')
    # Independent ground truth after termination, never supplied to estimator/planner.
    truth=client.settled_truth(ident,1+calls); effects=int(truth['effects'])
    return dict(case=case['case'],family=case['family'],keyed=keyed,cap=cap,deadline_ms=deadline,budget=budget,
        process_ms=case['process_ms'],episode=ident,policy=policy,initial_observation=initial_obs,
        initial_elapsed_ms=initial_elapsed,final_observation=obs,latency_ms=elapsed,
        confirmed_on_time=int(obs=='confirmed' and elapsed<=deadline and effects==1),duplicate=int(effects>1),
        effects=effects,unresolved=int(obs!='confirmed'),deadline_overrun=int(elapsed>deadline),
        recovery_calls=calls,steps=steps,waits=sum(x['action']=='wait' for x in trace),decision_ms=decision,
        unseen_observations=unseen,termination_reason=reason,remaining_ms=deadline-elapsed,
        truth=json.dumps(truth),trace=json.dumps(trace))
