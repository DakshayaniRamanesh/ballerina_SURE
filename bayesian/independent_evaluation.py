"""Pre-register the design, verify fixture, run a frozen comparison, then analyse.
Run from project root. No model fitting or holdout access here.
"""
import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sure.independent_recovery import ExtendedClient, SPECS, episode, TimeKernel, STATES, FEATURES

ROOT=Path('bayesian/outputs/independent_v1')
B3=Path('bayesian/outputs/b3/bayesian_b3.joblib')
KERNEL=Path('bayesian/outputs/b41/kernel_tables.json')
EXPECTED={'b3':'348174d885c1cf679687eff034c6942394651eef30f7b6466d2a0446ec18d3d8',
          'kernel':'c71c371befcbfff09c751d8bdee65382b028cb594475f38006f517306367a3ad'}
CODE=['ballerina_service/extended_service.bal','sure/independent_recovery.py','sure/policy.py',
      'sure/experiment.py','bayesian/evaluate_bayesian_planner.py','bayesian/evaluate_time_aware_planner.py',
      'bayesian/evaluate_planner_sensitivity.py','bayesian/independent_evaluation.py','bayesian/score_frozen_holdouts.py',
      'bayesian/train_execution_bayes.py','bayesian/train_time_aware_kernel.py','bayesian/temporal_bayesian.py']
HOLDOUT_INPUTS=['bayesian/outputs/b3/recovery_validated.csv','bayesian/outputs/b3/episode_splits.csv',
                'bayesian/outputs/b3/validation_metrics.json','bayesian/outputs/b41/episode_splits.csv',
                'bayesian/outputs/b41/validation_metrics.json','data/recovery_multistep_v2.csv',
                'data/recovery_multistep_v2_episodes.csv','bayesian/outputs/rcaeval_features.csv',
                'bayesian/outputs/feature_columns.json','bayesian/outputs/gnb_tuned_validation_metrics.json',
                'bayesian/outputs/b2_validation_metrics.json']
FAMILIES=('normal','slow','response_delay','drop','endpoint','status','combined')
SEED=81442
N=504

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def check_models():
    got={'b3':sha(B3),'kernel':sha(KERNEL)}
    require(got==EXPECTED,f'Frozen model hash mismatch: {got}. Do not retrain or override; investigate.')
    saved=joblib.load(B3)
    require(tuple(saved['states'])==tuple(STATES) and tuple(saved['features'])==tuple(FEATURES),'B3 schema mismatch')
    kernel=TimeKernel(load(KERNEL)); require(kernel.level==2,'Unexpected selected kernel')
    return saved['bayes'],kernel

def make_fault(rng,pre=(0,15),post=(0,15),down=0.,drop=0.):
    return dict(before_ms=rng.randint(*pre),after_ms=rng.randint(*post),down=rng.random()<down,drop=rng.random()<drop)

def schedule(rng,family,keyed,cap,deadline,budget):
    # Configured fault injection; delay represents simulated network/service overhead.
    # Schedules are sampled before evaluation, not based on policy or results.
    proc=rng.randint(20,65)
    primary=[make_fault(rng) for _ in range(5)]
    backup=[make_fault(rng) for _ in range(5)]
    status=[make_fault(rng,(15,65),(0,20)) for _ in range(5)]
    if family=='slow':
        proc=rng.randint(220,1000)
    elif family=='response_delay':
        primary[0]['after_ms']=rng.randint(220,500)
        for f in primary[1:]+backup:
            f['after_ms']=rng.randint(0,160)
    elif family=='drop':
        primary[0].update(drop=True,after_ms=rng.randint(210,450))
        for f in primary[1:]+backup:
            f['drop']=rng.random()<.20
    elif family=='endpoint':
        primary[0]['down']=True
        for f in primary[1:]: f['down']=rng.random()<.55
        for f in backup: f['down']=rng.random()<.15
    elif family=='status':
        primary[0]['after_ms']=rng.randint(220,500)
        status=[make_fault(rng,(10,180),(0,200),down=.40) for _ in range(5)]
    elif family=='combined':
        proc=rng.randint(80,1000)
        primary=[make_fault(rng,(0,250),(0,500),.25,.20) for _ in range(5)]
        backup=[make_fault(rng,(0,200),(0,300),.15,.10) for _ in range(5)]
        status=[make_fault(rng,(0,200),(0,250),.30,.10) for _ in range(5)]
    return dict(family=family,keyed=keyed,cap=cap,deadline=deadline,budget=budget,
                process_ms=proc,primary=primary,backup=backup,status=status)

def design(seed,smoke=False):
    rng=random.Random(seed); cases=[]
    for family in FAMILIES:
        for keyed in (False,True):
            cells=[(c,d,b) for c in (100,150,200) for d in (350,750,1250) for b in (1,2,3,4)]
            if smoke: cells=[(150,750,3)]
            for cap,deadline,budget in cells:
                cases.append(schedule(rng,family,keyed,cap,deadline,budget))
    rng.shuffle(cases)
    for i,c in enumerate(cases):
        c['case']=i; order=list(SPECS); rng.shuffle(order); c['order']=order
    require(len(cases)==(14 if smoke else N),'Design size mismatch')
    return cases

def freeze():
    smoke=Path('bayesian/outputs/independent_smoke')
    require((smoke/'complete.json').is_file() and (smoke/'paired_analysis.json').is_file(),'Complete and audit the separate smoke run first')
    require(load(smoke/'complete.json')['rows']==140,'Unexpected smoke run size')
    check_models(); code_hashes={p:sha(p) for p in CODE}
    input_hashes={p:sha(p) for p in HOLDOUT_INPUTS if Path(p).is_file()}
    missing=sorted(set(HOLDOUT_INPUTS)-set(input_hashes))
    if missing: print('Missing holdout inputs (live evaluation unaffected; resolve before scoring):',missing)
    ROOT.mkdir(parents=True,exist_ok=False)
    dump(ROOT/'design.json',design(SEED))
    dump(ROOT/'protocol.json',dict(seed=SEED,n=N,model_sha256=EXPECTED,code_sha256=code_hashes,holdout_input_sha256=input_hashes,missing_holdout_inputs=missing,
         design_sha256=sha(ROOT/'design.json'),policies=SPECS,max_steps=4,wait_ms=75,http_cap_ms=180,
         admission_margin_ms=25,call_cost=.025,time_cost=.10,
         primary_comparator='failover_backoff',primary_endpoint='confirmed_on_time',
         primary_latency_subset='both policies confirmed on time AND both initially unconfirmed',
         resampling='paired stratified case bootstrap (family/keyed), 10000 draws; descriptive intervals',
         secondary=['v4 vs v2','v4 vs v4_no_time','v4 vs v4_time_only','per-family and budget summaries'],
         limitations=['504 is an exploratory fixed design size, not a powered sample size',
                      'synthetic delay and availability schedules, not actual packet loss or independent hosts',
                      'same pre-sampled endpoint ordinal schedules, not identical wall-clock timings',
                      'fixed admission caps; no hard deadline proof','conditional latency subset is descriptive']))
    print('Frozen design and source/model hashes:',ROOT)

def verify_freeze():
    p=load(ROOT/'protocol.json'); require(p['model_sha256']==EXPECTED,'Model hash protocol mismatch')
    require(p['design_sha256']==sha(ROOT/'design.json'),'Design changed')
    require(all(sha(path)==h for path,h in p['code_sha256'].items()),'Source changed after freeze')
    return p

def client_at(base):
    c=ExtendedClient(base)
    require(c.admin('/health')=={'status':'ok','fixture':'sure_extended_v1'},'Wrong fixture; use extended service on port 9091')
    return c

def fixture_check(base):
    c=client_at(base); rng=random.Random(111)
    def case(keyed=True):
        d=schedule(rng,'normal',keyed,150,750,3)
        d['process_ms']=30
        for k in ('primary','backup','status'): d[k]=[dict(before_ms=0,after_ms=0,down=False,drop=False) for _ in range(5)]
        return d
    def act(a,i,k,cap=180): return c.action(a,i,k,1000,cap)[0]
    def truth(i,maxcalls): return c.settled_truth(i,maxcalls)
    d=case(); i=c.create_case(d); require(act('retry',i,True)=='confirmed','Normal execution')
    require(act('failover',i,True)=='confirmed' and truth(i,2)['effects']==1,'Completed cross-endpoint dedup')
    d=case(); d['process_ms']=400; i=c.create_case(d)
    # Simultaneous calls against the shared ledger using independent HTTP sessions.
    def submit(a): return client_at(base).action(a,i,True,1000,100)[0]
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(submit,('retry','failover')))
    t=truth(i,2); require(t['attempts']==2 and t['effects']==1,'Concurrent keyed dedup')
    d=case(False); d['primary'][0]['drop']=True; i=c.create_case(d)
    require(act('retry',i,False)=='unknown' and act('reconcile',i,False)=='unknown' and truth(i,2)['effects']==0,'Drop before execution')
    d=case(False); d['primary'][0]['after_ms']=350; i=c.create_case(d)
    require(act('retry',i,False,100)=='timeout','Delayed response did not time out')
    require(act('reconcile',i,False)=='confirmed' and truth(i,2)['effects']==1,'Delayed response after commit')
    d=case(); d['primary'][0]['down']=True; d['status'][0]['down']=True; i=c.create_case(d)
    require(act('retry',i,True)=='error' and act('reconcile',i,True)=='error','Availability injection')
    require(act('failover',i,True)=='confirmed' and act('reconcile',i,True)=='confirmed' and truth(i,4)['effects']==1,'Intermittent status and backup')
    print('Fixture checks passed: commit, drop, delayed response, endpoint/status faults, concurrent shared dedup.')

def run(base,smoke=False):
    bayes,kernel=check_models(); c=client_at(base)
    if smoke:
        out=Path('bayesian/outputs/independent_smoke'); cases=design(SEED-1,True)
    else:
        verify_freeze(); cases=load(ROOT/'design.json'); out=ROOT/'live'
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'design.json',cases)
    count=0
    for cell in cases:
        for policy in cell['order']:
            r=episode(c,kernel,bayes,policy,SPECS[policy],cell)
            pd.DataFrame([r]).to_csv(out/'runs.csv',mode='a',header=count==0,index=False); count+=1
            require(not r['duplicate'] and (r['final_observation']!='confirmed' or r['effects']==1),'Safety/integrity violation; inspect saved row')
        print(f"Matched cases {cell['case']+1}/{len(cases)}",flush=True)
    dump(out/'complete.json',dict(rows=count,model_sha256=EXPECTED,design_sha256=sha(out/'design.json')))
    print('Finished:',out,'rows:',count)

def summarize(d,groups):
    return d.groupby(groups).agg(n=('case','size'),success=('confirmed_on_time','mean'),duplicates=('duplicate','sum'),
        deadline_overrun=('deadline_overrun','mean'),unresolved=('unresolved','mean'),mean_ms=('latency_ms','mean'),p95_ms=('latency_ms',lambda s:s.quantile(.95)),
        calls=('recovery_calls','mean'),waits=('waits','mean'),decision_ms=('decision_ms','mean'),unseen=('unseen_observations','sum'))

def bootstrap(values,strata,seed=441):
    rng=np.random.default_rng(seed); v=np.asarray(values,float); labels=np.asarray(strata)
    if len(v)==0: return [None,None]
    parts=[v[labels==g] for g in sorted(set(labels))]; totals=np.zeros(10000)
    for p in parts: totals+=rng.choice(p,size=(10000,len(p)),replace=True).sum(axis=1)
    return np.quantile(totals/len(v),[.025,.975]).tolist()

def analyze(smoke=False):
    if not smoke: verify_freeze()
    out=Path('bayesian/outputs/independent_smoke') if smoke else ROOT/'live'
    marker=load(out/'complete.json'); cases=load(out/'design.json'); d=pd.read_csv(out/'runs.csv')
    require(marker['rows']==len(cases)*len(SPECS)==len(d) and marker['design_sha256']==sha(out/'design.json') and marker['model_sha256']==EXPECTED,'Incomplete or incompatible run')
    if not smoke: require(cases==load(ROOT/'design.json'),'Run design differs from frozen design')
    require(d.episode.is_unique and not d.duplicated(['case','policy']).any(),'Duplicate records')
    require(set(d.policy)==set(SPECS) and d.groupby('case').policy.nunique().eq(len(SPECS)).all(),'Unmatched records')
    for x in cases:
        g=d[d.case==x['case']]; require(len(g)==len(SPECS) and all(g[k].eq(x[j]).all() for k,j in [('family','family'),('keyed','keyed'),('cap','cap'),('deadline_ms','deadline'),('budget','budget'),('process_ms','process_ms')]),'Design mismatch')
    for r in d.itertuples():
        tr=json.loads(r.trace); t=json.loads(r.truth)
        require(len(tr)==r.steps<=4 and sum(x['action']!='wait' for x in tr)==r.recovery_calls<=r.budget,'Trace resource mismatch')
        require(r.keyed or not any(x['action'] in ('retry','failover') for x in tr),'Unsafe trace')
        require(t['active']==0 and t['effects']==r.effects and r.duplicate==int(r.effects>1),'Ledger mismatch')
        require(r.confirmed_on_time==int(r.final_observation=='confirmed' and r.latency_ms<=r.deadline_ms and r.effects==1),'Outcome mismatch')
    for name,keys in [('summary',['policy']),('by_family',['policy','family']),('by_budget',['policy','budget']),('by_keyed',['policy','keyed']),('by_deadline',['policy','deadline_ms']),('by_initial_timeout',['policy','cap'])]:
        summarize(d,keys).to_csv(out/(name+'.csv'))
    d['processing_range']=pd.cut(d.process_ms,[-1,100,250,500,750,1000],include_lowest=True)
    summarize(d,['policy','processing_range']).to_csv(out/'by_processing_range.csv')
    summarize(d[d.initial_observation!='confirmed'],['policy']).to_csv(out/'initially_unconfirmed_summary.csv')
    # Pair at scenario level, not episode level. Smoke is diagnostic only.
    a=d[d.policy=='v4'].set_index('case').sort_index(); reports={}
    for comp in ('failover_backoff','v2','v4_no_time','v4_time_only','hybrid_backoff'):
        b=d[d.policy==comp].set_index('case').reindex(a.index)
        delta=a.confirmed_on_time-b.confirmed_on_time; strata=a.family+'_'+a.keyed.astype(str)
        mask=(a.confirmed_on_time==1)&(b.confirmed_on_time==1)&(a.initial_observation!='confirmed')&(b.initial_observation!='confirmed')
        latency=b.loc[mask,'latency_ms']-a.loc[mask,'latency_ms']; w=int((delta==1).sum()); l=int((delta==-1).sum())
        reports[comp]=dict(n=len(delta),wins=w,losses=l,ties=int((delta==0).sum()),success_difference=float(delta.mean()),
            success_difference_ci95=None if smoke else bootstrap(delta,strata),
            mcnemar_exact_p=None if smoke else (float(binomtest(w,w+l).pvalue) if w+l else 1.),
            commonly_recovered_n=int(mask.sum()),mean_latency_saved_ms=float(latency.mean()) if len(latency) else None,
            latency_saved_ci95=None if smoke else bootstrap(latency,strata[mask]),
            v4_mean_latency_ms=float(a.loc[mask,'latency_ms'].mean()) if len(latency) else None,
            comparator_mean_latency_ms=float(b.loc[mask,'latency_ms'].mean()) if len(latency) else None,
            recovery_calls_difference=float((a.recovery_calls-b.recovery_calls).mean()))
    dump(out/'paired_analysis.json',reports); print(summarize(d,['policy']).to_string()); print(json.dumps(reports,indent=2))
    print('Smoke only' if smoke else 'Independent synthetic evaluation; conditional latency is descriptive; no equivalence claim.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['verify','smoke','analyze-smoke','freeze','run','analyze']);p.add_argument('--base',default='http://127.0.0.1:9091');a=p.parse_args()
    if a.stage=='verify': fixture_check(a.base)
    elif a.stage=='smoke': run(a.base,True)
    elif a.stage=='analyze-smoke': analyze(True)
    elif a.stage=='freeze': freeze()
    elif a.stage=='run': run(a.base)
    else: analyze()
if __name__=='__main__': main()
