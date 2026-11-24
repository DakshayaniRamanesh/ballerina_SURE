"""Collect actual HTTP traces, train both estimators, and compare policies."""
import argparse, csv, json, random, time, uuid
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from .policy import STATES, FEATURES, EmpiricalBayes, Kernel, Planner, safe_actions, norm

MODES=('normal','lost','drop','slow','status_slow','status_down','primary_down')

def ms(): return time.monotonic()*1000

def epoch(): return int(time.time()*1000)

def label(row,t):
    if row['commit_ms'] and row['commit_ms']<=t: return 'completed'
    if row['start_ms'] and row['start_ms']<=t: return 'processing'
    return 'unstarted'

class Client:
    def __init__(self,base): self.base=base.rstrip('/'); self.session=requests.Session()
    def admin(self,path,data=None):
        res=self.session.get(self.base+path,timeout=5) if data is None else self.session.post(self.base+path,json=data,timeout=5)
        res.raise_for_status(); return res.json()
    def create(self,mode,keyed):
        ident=uuid.uuid4().hex; self.admin('/admin/scenario',{'id':ident,'mode':mode,'keyed':keyed}); return ident
    def action(self,a,ident,keyed,remaining,cap=180):
        start=ms(); timeout=min(remaining,cap)/1000
        if timeout<=0: return 'timeout',0.
        if a=='wait': time.sleep(min(.075,timeout)); return 'waited',ms()-start
        try:
            if a=='reconcile': res=self.session.get(f'{self.base}/primary/payments/{ident}',timeout=timeout,allow_redirects=False)
            else:
                endpoint='backup' if a=='failover' else 'primary'
                res=self.session.post(f'{self.base}/{endpoint}/payments',json={'id':ident,'keyed':keyed},timeout=timeout,allow_redirects=False)
            status=res.json().get('status')
            obs={'completed':'confirmed','pending':'pending','unknown':'unknown'}.get(status,'error') if res.ok else 'error'
        except requests.Timeout: obs='timeout'
        except (requests.RequestException,ValueError): obs='error'
        return obs,ms()-start
    def truth(self,ident): return self.admin('/admin/truth/'+ident)

def features(obs,dt,keyed,cap): return dict(elapsed_ms=dt,error=obs,keyed=keyed,initial_timeout_ms=cap)

def collect(args):
    rng=random.Random(args.seed); c=Client(args.base); rows=[]
    for i in range(args.n):
        mode=rng.choice(MODES); keyed=rng.choice([True,False]); cap=rng.choice([100,150,200]); ident=c.create(mode,keyed)
        obs,dt=c.action('retry',ident,keyed,cap,cap); t0=epoch()
        if obs=='confirmed': continue
        f=features(obs,dt,keyed,cap)
        a=rng.choice([x for x in safe_actions(keyed,500,3) if x!='stop'])
        o,d=c.action(a,ident,keyed,500); t1=epoch()
        # Allow late commits before retrieving the event trace. No admin data reaches policy.
        time.sleep(.36); truth=c.truth(ident)
        rows.append(dict(f,episode=ident,mode=mode,state=label(truth,t0),action=a,observation=o,duration_ms=d,next_state=label(truth,t1)))
        if (i+1)%25==0: print(f'Collected {i+1}/{args.n} episodes',flush=True)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(f'{len(rows)} recovery samples -> {out}')

def rf_probs(model,f):
    probs=model.predict_proba(pd.DataFrame([f],columns=FEATURES))[0]
    return norm([probs[list(model.classes_).index(s)] if s in model.classes_ else 0 for s in STATES])

def train(args):
    data=pd.read_csv(args.data); data['keyed']=data.keyed.astype(bool)
    if len(data)<15 or data.state.nunique()<3: raise SystemExit('Need more diverse recovery samples (all 3 states).')
    # One sample per independently generated episode; split by episode, not future observations.
    shuffled=data.sample(frac=1,random_state=args.seed); split=int(.8*len(shuffled)); tr,te=shuffled.iloc[:split],shuffled.iloc[split:]
    if tr.state.nunique()<3: raise SystemExit('Training split lacks a state; collect more samples.')
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['error']),('numeric','passthrough',['elapsed_ms','keyed','initial_timeout_ms'])])
    rf_base=make_pipeline(pre,RandomForestClassifier(n_estimators=100,min_samples_leaf=4,random_state=args.seed))
    # Calibrate on training folds only; keep the held-out episodes untouched.
    rf=CalibratedClassifierCV(rf_base,method='sigmoid',cv=3) if tr.state.value_counts().min()>=3 else rf_base
    rf.fit(tr[list(FEATURES)],tr.state); bayes=EmpiricalBayes().fit(tr.to_dict('records')); kernel=Kernel().fit(tr.to_dict('records'))
    for name,fn in [('RF',lambda f:rf_probs(rf,f)),('Bayes',bayes.predict)]:
        pr=np.array([fn(f) for f in te.to_dict('records')]); one=np.array([[int(s==y) for s in STATES] for y in te.state])
        print(name,'holdout log loss',round(log_loss(te.state,pr[:,[STATES.index(s) for s in sorted(STATES)]],labels=sorted(STATES)),4),'Brier',round(np.mean(np.sum((pr-one)**2,axis=1)),4))
    Path(args.model).parent.mkdir(parents=True,exist_ok=True); joblib.dump({'rf':rf,'bayes':bayes,'kernel':kernel},args.model); print('Saved',args.model)

def select(policy,model,b,keyed,remaining,budget,step):
    if policy=='safe_retry': return 'retry' if keyed and budget else 'stop'
    if policy=='fixed_reconcile': return 'reconcile' if budget else 'stop'
    return Planner(model['kernel'],depth=2 if policy in ('v4','v5') else 1).choose(b,keyed,remaining,budget)

def evaluate(args):
    model=joblib.load(args.model); c=Client(args.base); rng=random.Random(args.seed)
    trace=[(rng.choice(MODES),rng.choice([True,False]),rng.choice([100,150,200]),rng.choice([350,500,750])) for _ in range(args.n)]
    policies=['safe_retry','fixed_reconcile','v2','v3','v4','v5']; results=[]
    for i,(mode,keyed,cap,deadline) in enumerate(trace):
        order=policies[:]; rng.shuffle(order)
        for policy in order:
            ident=c.create(mode,keyed); start=ms(); obs,dt=c.action('retry',ident,keyed,deadline,cap); calls=0; decision_ms=0.; steps=0
            f=features(obs,dt,keyed,cap); b=None
            if obs!='confirmed' and policy.startswith('v'):
                ts=ms(); b=rf_probs(model['rf'],f) if policy in ('v3','v5') else model['bayes'].predict(f); decision_ms+=ms()-ts
            while obs!='confirmed' and steps<4 and ms()-start<deadline:
                remain=deadline-(ms()-start); ts=ms(); a=select(policy,model,b,keyed,remain,3-calls,steps); decision_ms+=ms()-ts
                if a=='stop' or a not in safe_actions(keyed,deadline-(ms()-start),3-calls): break
                obs,duration=c.action(a,ident,keyed,deadline-(ms()-start)); steps+=1; calls+=int(a!='wait')
                if b is not None and obs!='confirmed':
                    ts=ms(); b=model['kernel'].update(b,a,keyed,obs,duration); decision_ms+=ms()-ts
            elapsed=ms()-start; confirmed=obs=='confirmed' and elapsed<=deadline
            time.sleep(.36); truth=c.truth(ident); effects=truth['effects']
            results.append(dict(case=i,policy=policy,mode=mode,keyed=keyed,deadline_ms=deadline,confirmed_on_time=int(confirmed and effects==1),duplicate=int(effects>1),effects=effects,unresolved=int(obs!='confirmed'),deadline_violation=int(elapsed>deadline),latency_ms=elapsed,recovery_calls=calls,decision_ms=decision_ms))
        if (i+1)%10==0: print(f'Evaluated {i+1}/{args.n} cases',flush=True)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(results); df.to_csv(out,index=False)
    summary=df.groupby('policy').agg(n=('case','size'),success=('confirmed_on_time','mean'),duplicates=('duplicate','mean'),unresolved=('unresolved','mean'),deadline_violations=('deadline_violation','mean'),p95_ms=('latency_ms',lambda x:x.quantile(.95)),p99_ms=('latency_ms',lambda x:x.quantile(.99)),calls=('recovery_calls','mean'),decision_ms=('decision_ms','mean'))
    summary.to_csv(out.with_name('summary.csv')); print(summary.to_string())

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    for name,fn in [('collect',collect),('train',train),('evaluate',evaluate)]:
        q=sub.add_parser(name); q.set_defaults(fn=fn); q.add_argument('--seed',type=int,default=42 if name!='evaluate' else 91)
        if name!='train': q.add_argument('--base',default='http://127.0.0.1:9090'); q.add_argument('--n',type=int,default=120 if name=='collect' else 30); q.add_argument('--out',default='data/recovery.csv' if name=='collect' else 'results/runs.csv')
        if name!='collect': q.add_argument('--model',default='models/sure.joblib')
        if name=='train': q.add_argument('--data',default='data/recovery.csv')
    a=p.parse_args(); a.fn(a)
if __name__=='__main__': main()
