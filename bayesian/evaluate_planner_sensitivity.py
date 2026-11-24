"""Development-only sensitivity study for SURE. Run from project root.
Uses the existing frozen B3/B4.1 and TimeKernel without modifying them.
No holdout CSV scoring. No automatic winner selection.
"""
import argparse, hashlib, json, random, time
from pathlib import Path
import joblib
import pandas as pd
from .evaluate_time_aware_planner import (TimeKernel, RecoveryClient, normalized,
    now, MODES, STATES, FEATURES, features, MAX_STEPS, CALL_BUDGET, WAIT_MS,
    HTTP_CAP, MARGIN, CALL_COST)

BASELINES=('safe_retry','fixed_reconcile','retry_backoff','reconcile_backoff',
           'failover_backoff','hybrid_backoff')

def settings():
    specs={p:None for p in BASELINES}
    for depth in (1,2,3):
        for cost in (0.,.1,.2):
            specs[f'd{depth}_t{cost:.2f}_l2']=(depth,cost,2)
    for level in (0,1): specs[f'd2_t0.10_l{level}']=(2,.1,level)
    return specs

def design(n,seed):
    # Within each mode/keyed cell, every block of three uses each cap and deadline
    # once. Nine repeats cover the full cap x deadline grid.
    rng=random.Random(seed); rows=[]
    for mode in MODES:
        for keyed in (False,True):
            caps=[100,150,200]; deadlines=[500,750,1000]
            rng.shuffle(caps); rng.shuffle(deadlines)
            for rep in range(n//14):
                rows.append(dict(mode=mode,keyed=keyed,cap=caps[rep%3],
                    deadline=deadlines[(rep%3+rep//3)%3],rep=rep+1))
    rng.shuffle(rows)
    return [dict(case=i,**row) for i,row in enumerate(rows)]

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

def episode(client,kernel,bayes,policy,spec,case,mode,keyed,cap,deadline,rep):
    ident=client.create(mode,keyed); start=now(); obs,_=client.action('retry',ident,keyed,deadline,cap); initial_elapsed=now()-start
    initial_obs=obs; pa='initial'; po=obs; last_network='initial'; b=None; trace=[]; calls=steps=unseen=0; decision=0.; reason='confirmed' if obs=='confirmed' else 'step_limit'
    if spec is not None and obs!='confirmed':
        tick=now(); b=normalized(bayes.predict(features(obs,initial_elapsed,keyed,cap))); decision+=now()-tick
    while obs!='confirmed' and steps<MAX_STEPS:
        t=now()-start; remaining=deadline-t; tick=now(); allowed=kernel.allowed(keyed,remaining,CALL_BUDGET-calls); utility=None
        if spec is None: a=rule(policy,keyed,obs,pa,po,allowed,kernel,remaining,steps,last_network)
        else:
            depth,cost,level=spec
            utility,a=kernel.choose(b,keyed,t,pa,po,deadline,CALL_BUDGET-calls,min(depth,MAX_STEPS-steps),level,cost)
        decision+=now()-tick; t=now()-start
        if a=='stop': reason='policy_stop' if calls<CALL_BUDGET else 'call_budget'; break
        if a not in kernel.allowed(keyed,deadline-t,CALL_BUDGET-calls): reason='admission_after_compute'; break
        if not keyed and a in ('retry','failover'): raise RuntimeError('Unsafe dispatch')
        before=b.tolist() if b is not None else None
        tick=now(); obs,_=client.action(a,ident,keyed,deadline-t); duration=now()-tick
        steps+=1; calls+=int(a!='wait')
        if a!='wait': last_network=a
        if b is not None and obs!='confirmed':
            tick=now(); b,unknown=kernel.update(b,a,keyed,t,pa,po,obs,level); unseen+=int(unknown); decision+=now()-tick
        trace.append(dict(action=a,observation=obs,elapsed_before_ms=t,duration_ms=duration,expected_utility=utility,belief_before=before,belief_after=b.tolist() if b is not None and obs!='confirmed' else None))
        pa,po=a,obs
    elapsed=now()-start
    if obs=='confirmed': reason='confirmed'
    elif elapsed>deadline: reason='deadline'
    elif steps>=MAX_STEPS: reason='step_and_call_limit' if calls>=CALL_BUDGET else 'step_limit'
    # Independent truth read AFTER the client terminates. Never a runtime input.
    time.sleep(.40); truth=client.truth(ident); effects=int(truth['effects'])
    return dict(case=case,rep=rep,episode=ident,policy=policy,mode=mode,keyed=keyed,initial_timeout_ms=cap,initial_observation=initial_obs,initial_elapsed_ms=initial_elapsed,deadline_ms=deadline,confirmed_on_time=int(obs=='confirmed' and elapsed<=deadline and effects==1),duplicate=int(effects>1),effects=effects,final_observation=obs,unresolved=int(obs!='confirmed'),deadline_overrun=int(elapsed>deadline),latency_ms=elapsed,recovery_calls=calls,waits=sum(x['action']=='wait' for x in trace),steps=steps,decision_ms=decision,unseen_observations=unseen,termination_reason=reason,remaining_ms=deadline-elapsed,trace=json.dumps(trace))

def summarize(df,group):
    return df.groupby(group).agg(n=('case','size'),success=('confirmed_on_time','mean'),duplicates=('duplicate','sum'),unresolved=('unresolved','mean'),deadline_overrun=('deadline_overrun','mean'),p95_ms=('latency_ms',lambda x:x.quantile(.95)),mean_ms=('latency_ms','mean'),recovery_calls=('recovery_calls','mean'),waits=('waits','mean'),decision_ms=('decision_ms','mean'),unseen_observations=('unseen_observations','sum'))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--n',type=int,default=42); p.add_argument('--seed',type=int,default=4142)
    p.add_argument('--base',default='http://127.0.0.1:9090'); p.add_argument('--b3',default='bayesian/outputs/b3/bayesian_b3.joblib'); p.add_argument('--kernel',default='bayesian/outputs/b41/kernel_tables.json'); p.add_argument('--out',default='bayesian/outputs/planner_sensitivity')
    args=p.parse_args()
    if args.n<14 or args.n%14: p.error('n must be a positive multiple of 14')
    b3=joblib.load(args.b3)
    if tuple(b3['states'])!=tuple(STATES) or tuple(b3['features'])!=tuple(FEATURES): raise ValueError('Incompatible B3')
    kernel=TimeKernel(json.loads(Path(args.kernel).read_text(encoding='utf-8'))); client=RecoveryClient(args.base)
    if client.admin('/health').get('status')!='ok': raise RuntimeError('Service unhealthy')
    specs=settings(); rows=design(args.n,args.seed); rng=random.Random(args.seed+1)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'design.csv',index=False)
    config=dict(vars(args),specs=specs,max_steps=MAX_STEPS,call_budget=CALL_BUDGET,wait_ms=WAIT_MS,http_cap=HTTP_CAP,admission_margin=MARGIN,call_cost=CALL_COST,model_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('b3',args.b3),('kernel',args.kernel)]})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Cases:',len(rows),'policies:',len(specs),'episodes:',len(rows)*len(specs),flush=True)
    results=[]
    for cell in rows:
        order=list(specs); rng.shuffle(order)
        for policy in order:
            r=episode(client,kernel,b3['bayes'],policy,specs[policy],**cell); results.append(r)
            pd.DataFrame([r]).to_csv(out/'runs.csv',mode='a',header=len(results)==1,index=False)
            if r['duplicate'] or (r['final_observation']=='confirmed' and r['effects']!=1): raise RuntimeError('Safety/integrity violation; inspect partial runs.csv')
        print(f"Matched cases: {cell['case']+1}/{len(rows)}",flush=True)
    df=pd.DataFrame(results); summary=summarize(df,'policy'); summary.to_csv(out/'summary.csv'); print(summary.to_string())
    summarize(df,['policy','keyed']).to_csv(out/'summary_by_keyed.csv')
    summarize(df,['policy','mode']).to_csv(out/'summary_by_mode.csv')
    paired=df.pivot(index='case',columns='policy',values='confirmed_on_time'); ref='d2_t0.10_l2'
    comparisons=[]
    for name in specs:
        if name==ref: continue
        x=paired[ref]-paired[name]; comparisons.append(dict(reference=ref,comparator=name,wins=int((x>0).sum()),losses=int((x<0).sum()),ties=int((x==0).sum())))
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    print('Development sensitivity only; no automatic winner selection. No reserved CSV scoring.')

if __name__=='__main__': main()
