"""B4.1 development audit and model comparison. Run from SURE project root.
No server needed. Repeat 4 is reserved; no test predictions are produced.
JSON model stores train-only joint distributions for future planner integration.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

STATES=('unstarted','processing','completed')
ACTIONS=('wait','reconcile','retry','failover')
EDGES=(150,225,275,325)
ALPHA=10.0
BOUNDARY=2


def require(ok,msg):
    if not bool(ok): raise ValueError(msg)


def audit(r,e):
    require(not r.isna().any().any() and not e.isna().any().any(),'Missing data')
    require(e.episode.is_unique and not r.duplicated(['episode','step']).any(),'Duplicate IDs')
    require(set(r.episode)<=set(e.episode),'Unknown episode')
    require(e.repeat.isin([1,2,3,4]).all(),'Unexpected repeat')
    cells=['mode','keyed','initial_timeout_ms','sequence']
    require(len(e)==756 and (e.groupby(cells).repeat.agg(lambda v: set(v)=={1,2,3,4})).all() and not e.duplicated(cells+['repeat']).any(),'Incomplete repeated design')
    for d in (r,e):
        require(d.keyed.isin([True,False]).all(),'Invalid keyed')
        require(d.effects.isin([0,1]).all(),'Duplicate or invalid effect count')
        require(((d.commit_ms>0)==(d.effects==1)).all(),'Commit/effect mismatch')
        require(((d.commit_ms==0)|((d.start_ms>0)&(d.commit_ms>=d.start_ms))).all(),'Invalid commit order')
    require((e.deadline_overrun==0).all(),'Collection overrun')
    require(r.action.isin(ACTIONS).all() and not ((~r.keyed)&r.action.isin(['retry','failover'])).any(),'Unsafe action')
    require((r.duration_ms>0).all() and np.isfinite(r[['duration_ms','elapsed_before_ms','elapsed_after_ms']]).all().all(),'Invalid duration')
    require(np.allclose(r.elapsed_after_ms-r.elapsed_before_ms,r.duration_ms,atol=1e-6,rtol=0),'Duration mismatch')
    require((r.action_end_epoch_ms>=r.decision_epoch_ms).all(),'Wall clock regression')
    require((r.elapsed_after_ms<=1000).all(),'Step deadline overrun')
    def label(t):
        return np.where((r.commit_ms>0)&(r.commit_ms<=t),'completed',np.where((r.start_ms>0)&(r.start_ms<=t),'processing','unstarted'))
    def near(t):
        return ((r.start_ms>0)&((r.start_ms-t).abs()<=BOUNDARY))|((r.commit_ms>0)&((r.commit_ms-t).abs()<=BOUNDARY))
    start,end=near(r.decision_epoch_ms),near(r.action_end_epoch_ms)
    confirmed=r.observation=='confirmed'
    require((r.clock_state==label(r.decision_epoch_ms)).all() and (r.clock_next_state==label(r.action_end_epoch_ms)).all(),'Clock label mismatch')
    require((r.state==r.clock_state).all() and (r.next_state==np.where(confirmed,'completed',r.clock_next_state)).all(),'Corrected label mismatch')
    require((r.boundary_uncertain==(start|end).astype(int)).all(),'Boundary flag mismatch')
    require((r.usable==(~start&(~end|confirmed)).astype(int)).all(),'Usability mismatch')
    require((r.confirmation_resolved==(confirmed&end&~start).astype(int)).all(),'Confirmation resolution mismatch')
    require(((~confirmed)|((r.effects==1)&(r.commit_ms<=r.action_end_epoch_ms+BOUNDARY))).all(),'Unverified confirmation')
    require(r.state.isin(STATES).all() and r.next_state.isin(STATES).all(),'Unknown state')
    u=r[r.usable==1]; order={s:i for i,s in enumerate(STATES)}
    require((u.state.map(order)<=u.next_state.map(order)).all(),'State regression')
    groups={k:g.sort_values('step') for k,g in r.groupby('episode')}
    nonterminal=0; prev_actions={}
    for ep in e.itertuples(index=False):
        g=groups.get(ep.episode,r.iloc[:0]); require(len(g)==ep.steps,'Step count mismatch')
        require(ep.calls==int((g.action!='wait').sum()) and ep.calls<=3,'Call count mismatch')
        require((ep.initial_observation=='confirmed')==(ep.steps==0),'Initial confirmation mismatch')
        if ep.final_observation=='confirmed': require(ep.effects==1,'Confirmation without effect')
        prev=ep.initial_observation; last_action='initial'; history=[]; last_end=ep.initial_elapsed_ms; last_epoch=None; last_row=None
        for i,x in enumerate(g.itertuples(index=False),1):
            require(x.step==i and x.action==ep.sequence.split('>')[i-1],'Sequence mismatch')
            for c in cells+['repeat','initial_observation','initial_elapsed_ms','effects','attempts','start_ms','commit_ms']:
                require(getattr(x,c)==getattr(ep,c),f'Episode mismatch: {c}')
            require(x.previous_observation==prev and prev!='confirmed','Previous observation mismatch')
            h=json.loads(x.history_json); require(len(h)==len(history),'History length mismatch')
            for a,b in zip(h,history):
                require(a['action']==b['action'] and a['observation']==b['observation'] and abs(a['duration_ms']-b['duration_ms'])<1e-6,'History content mismatch')
            require(x.elapsed_before_ms+1e-6>=last_end,'Elapsed order mismatch')
            if last_epoch is not None: require(x.decision_epoch_ms>=last_epoch,'Epoch order mismatch')
            if last_row is not None and x.usable and last_row.usable:
                require(order[x.state]>=order[last_row.next_state],'Consecutive state regression')
            if not x.usable and i<ep.steps: nonterminal+=1
            prev_actions[(x.episode,x.step)]=last_action
            history.append(dict(action=x.action,observation=x.observation,duration_ms=x.duration_ms))
            prev=x.observation; last_action=x.action; last_end=x.elapsed_after_ms;last_epoch=x.action_end_epoch_ms;last_row=x
        require(prev==ep.final_observation,'Final observation mismatch')
    r=r.copy(); r['previous_action']=[prev_actions[(x.episode,x.step)] for x in r.itertuples()]
    print('Audit passed:',len(e),'episodes;',len(r),'steps;',int(r.usable.sum()),'usable;',nonterminal,'nonterminal quarantined')
    return r


def key(s,a,k,t,pa,po):
    return (s,a,bool(k)),(s,a,bool(k),int(np.searchsorted(EDGES,t,side='right'))),(s,a,bool(k),int(np.searchsorted(EDGES,t,side='right')),pa,po)


def outcome(r): return (r.next_state,r.observation,max(25,int(math.ceil(r.duration_ms/25))*25))


def fit(train):
    tables=[defaultdict(Counter) for _ in range(3)]
    for r in train.itertuples():
        for tab,k in zip(tables,key(r.state,r.action,r.keyed,r.elapsed_before_ms,r.previous_action,r.previous_observation)):
            tab[k][outcome(r)]+=1
    for s in STATES:
        for k in (False,True):
            for a in ACTIONS if k else ACTIONS[:2]: require(tables[0].get((s,a,k)),f'Uncovered cell {s,a,k}')
    return tables


def predict(tables,level,s,a,k,t,pa,po):
    keys=key(s,a,k,t,pa,po); c=tables[0][keys[0]]; n=sum(c.values()); p={o:v/n for o,v in c.items()}
    for i in range(1,level+1):
        c=tables[i].get(keys[i],{}); n=sum(c.values()); p={o:(c.get(o,0)+ALPHA*v)/(n+ALPHA) for o,v in p.items()}
    require(abs(sum(p.values())-1)<1e-9,'Bad probability mass')
    return p


def mixture(tables,level,b,r):
    j=defaultdict(float)
    for s,w in zip(STATES,b):
        if w>0:
            for o,p in predict(tables,level,s,r.action,r.keyed,r.elapsed_before_ms,r.previous_action,r.previous_observation).items():j[o]+=float(w)*p
    return j


def score(j,r):
    ns=Counter(); obs=Counter()
    for (s,o,d),p in j.items(): ns[s]+=p; obs[o]+=p
    p=j.get(outcome(r),0); ps=ns[r.next_state]; po=obs[r.observation]
    return dict(joint_nll=-math.log(max(p,1e-15)),unseen_joint=int(p==0),next_state_nll=-math.log(max(ps,1e-15)),observation_nll=-math.log(max(po,1e-15)),next_state_brier=sum((ns[s]-(s==r.next_state))**2 for s in STATES),confirmation_brier=(obs['confirmed']-(r.observation=='confirmed'))**2,duration_mae=abs(sum(d*p for (s,o,d),p in j.items())-r.duration_ms),next_state_correct=int(max(ns,key=ns.get)==r.next_state),observation_correct=int(max(obs,key=obs.get)==r.observation))


def prior_fit(train):
    counts=defaultdict(Counter); global_counts=Counter()
    for r in train[train.step==1].itertuples():
        counts[(r.initial_observation,r.keyed,r.initial_timeout_ms)][r.state]+=1; global_counts[r.state]+=1
    return counts,global_counts


def initial(prior,r,b3):
    if b3 is not None:
        b=np.asarray(b3.predict(dict(error=r.initial_observation,keyed=r.keyed,initial_timeout_ms=r.initial_timeout_ms,elapsed_ms=r.initial_elapsed_ms)),float)
    else:
        c=prior[0].get((r.initial_observation,r.keyed,r.initial_timeout_ms),prior[1]); b=np.array([c[s]+1 for s in STATES],float)
    require(b.shape==(3,) and np.isfinite(b).all() and (b>=0).all() and b.sum()>0,'Invalid initial belief')
    return b/b.sum()


def evaluate(tables,level,val_raw,prior,b3):
    records=[]; unseen=0
    for _,g in val_raw.groupby('episode'):
        b=None
        for r in g.sort_values('step').itertuples():
            if b is None: b=initial(prior,r,b3)
            j=mixture(tables,level,b,r)
            if r.usable:
                oracle=mixture(tables,level,[float(s==r.state) for s in STATES],r)
                records.append(dict(episode=r.episode,step=r.step,action=r.action,**{'oracle_'+k:v for k,v in score(oracle,r).items()},**{'replay_'+k:v for k,v in score(j,r).items()}))
            # No true state is used during replay, even at quarantined steps.
            # Update on the observable response; timing is used as context for the next action.
            m=np.array([sum(p for (s,o,d),p in j.items() if s==state and o==r.observation) for state in STATES]);
            if m.sum()==0:
                unseen+=1; m=np.array([sum(p for (s,o,d),p in j.items() if s==state) for state in STATES])
            b=m/m.sum()
    df=pd.DataFrame(records); metrics=df.select_dtypes('number').drop(columns='step').mean().to_dict()
    for prefix in ('oracle','replay'): metrics[prefix+'_unseen_joint_count']=int(df[prefix+'_unseen_joint'].sum())
    metrics['replay_unseen_observation_updates']=unseen
    # Episode-balanced scores to avoid overweighting longer trajectories.
    for prefix in ('oracle','replay'):
        metrics[prefix+'_episode_mean_joint_nll']=float(df.groupby('episode')[prefix+'_joint_nll'].mean().mean())
    return metrics,df


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/recovery_multistep_v2.csv');p.add_argument('--episodes',default='data/recovery_multistep_v2_episodes.csv');p.add_argument('--out',default='bayesian/outputs/b41');p.add_argument('--b3',default=None,help='Optional frozen B3 joblib path');a=p.parse_args()
    r=audit(pd.read_csv(a.data),pd.read_csv(a.episodes));e=pd.read_csv(a.episodes); split={1:'train',2:'train',3:'validation',4:'test'};e['split']=e.repeat.map(split);r['split']=r.repeat.map(split)
    train=r[(r.split=='train')&(r.usable==1)]; val=r[r.split=='validation']; tables=fit(train); prior=prior_fit(train); b3=None
    if a.b3:
        import joblib
        saved=joblib.load(a.b3);require(tuple(saved['states'])==STATES,'B3 state order mismatch');b3=saved['bayes']
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True); e[['episode','repeat','split']].to_csv(out/'episode_splits.csv',index=False)
    coverage=train.groupby(['keyed','state','action']).size().rename('samples').reset_index();coverage.to_csv(out/'training_coverage.csv',index=False)
    print('Training coverage:\n',coverage.to_string(index=False))
    report={'split_episodes':e.split.value_counts().to_dict(),'usable_steps':r[r.usable==1].split.value_counts().to_dict(),'test_evaluated':False,'time_edges_ms':list(EDGES),'backoff_strength':ALPHA,'replay_initial_estimator':'frozen B3' if b3 else 'development-only conditional prior (not frozen B3)','models':{}}
    names=['B4.0_matched','B4.1a_time','B4.1b_time_history']
    for i,name in enumerate(names):
        m,df=evaluate(tables,i,val,prior,b3);report['models'][name]=m;df.to_csv(out/(name+'_validation.csv'),index=False)
    # Select on oracle joint probability quality, separately from policy evaluation.
    # Use episode-balanced validation loss; no tuning on reserved repeat 4.
    report['selected_by_oracle_episode_joint_nll']=min(names,key=lambda n: report['models'][n]['oracle_episode_mean_joint_nll'])
    model={'states':list(STATES),'time_edges_ms':list(EDGES),'alpha':ALPHA,'selected_level':names.index(report['selected_by_oracle_episode_joint_nll']),'tables':[[{'key':list(k),'outcomes':[[*o,n] for o,n in c.items()]} for k,c in tab.items()] for tab in tables]}
    (out/'kernel_tables.json').write_text(json.dumps(model,indent=2),encoding='utf-8');(out/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));print('Saved to',out,'; repeat 4 not scored.')

if __name__=='__main__':main()
