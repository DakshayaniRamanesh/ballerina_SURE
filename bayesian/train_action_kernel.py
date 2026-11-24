"""B4: train and validate SURE's one-step empirical action outcome model.
Run from project root: python -m bayesian.train_action_kernel
Uses frozen B3 partitions. The final test set is never evaluated.
"""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import joblib
import numpy as np
import pandas as pd
from sure.policy import STATES, FEATURES, Kernel, safe_actions


def require(ok, msg):
    if not bool(ok):
        raise ValueError(msg)


def fit_kernel(train):
    kernel = Kernel().fit(train.to_dict('records'))
    coverage = []
    for keyed in (False, True):
        for state in STATES:
            for action in safe_actions(keyed, 500, 3):
                if action == 'stop':
                    continue
                counts = kernel.counts.get((state, action, keyed), {})
                coverage.append(dict(keyed=keyed, state=state, action=action,
                                     samples=sum(counts.values()), distinct_outcomes=len(counts)))
    coverage = pd.DataFrame(coverage)
    require((coverage.samples > 0).all(), 'Missing training coverage; inspect state/action/keyed cells before fitting planner')
    rows = []
    for (state, action, keyed), counts in kernel.counts.items():
        for (ns, obs, dt), n in counts.items():
            rows.append(dict(state=state, action=action, keyed=keyed, next_state=ns,
                             observation=obs, duration_bin_ms=dt, count=n,
                             probability=n/sum(counts.values())))
    outcomes = pd.DataFrame(rows)
    require(np.allclose(outcomes.groupby(['state','action','keyed']).probability.sum(), 1), 'Invalid kernel normalization')
    return kernel, coverage, outcomes


def distribution(kernel, belief, action, keyed):
    require(np.isfinite(belief).all() and np.min(belief)>=0 and np.isclose(sum(belief),1), 'Invalid initial belief')
    joint = defaultdict(float)
    for s, w in zip(STATES, belief):
        if w == 0:
            continue
        require((s,action,keyed) in kernel.counts, 'Unseen action context')
        for ns, o, dt, p in kernel.outcomes(s, action, keyed):
            joint[(ns,o,dt)] += w*p
    require(np.isclose(sum(joint.values()),1), 'Invalid predictive distribution')
    return joint


def assess(kernel, bayes, val, oracle=False):
    rows = []
    for r in val.to_dict('records'):
        b = np.array([float(s==r['state']) for s in STATES]) if oracle else np.asarray(bayes.predict({f:r[f] for f in FEATURES}),float)
        joint = distribution(kernel,b,r['action'],r['keyed'])
        ns = Counter(); obs = Counter()
        for (s,o,d),p in joint.items():
            ns[s] += p; obs[o] += p
        actual = (r['next_state'],r['observation'],max(25,int(np.ceil(r['duration_ms']/25))*25))
        p_joint = joint.get(actual,0.)
        p_state = ns[r['next_state']]; p_obs = obs[r['observation']]
        expected_d = sum(d*p for (s,o,d),p in joint.items())
        row = dict(episode=r['episode'],action=r['action'],keyed=r['keyed'],
                   actual_state=r['state'],actual_next_state=r['next_state'],actual_observation=r['observation'],
                   actual_duration_ms=r['duration_ms'],expected_binned_duration_ms=expected_d,
                   predicted_next_state=max(ns,key=ns.get),predicted_observation=max(obs,key=obs.get),
                   p_true_joint=p_joint,p_true_next_state=p_state,p_true_observation=p_obs,
                   p_confirmed=obs['confirmed'],
                   next_state_brier=sum((ns[s]-float(s==r['next_state']))**2 for s in STATES),
                   confirmation_brier=(obs['confirmed']-float(r['observation']=='confirmed'))**2)
        for s in STATES:
            row['p_next_'+s]=ns[s]
        rows.append(row)
    df = pd.DataFrame(rows)
    def nll(col):
        # Clip only for reporting; never replace zero model probabilities silently.
        return float(-np.log(df[col].clip(lower=1e-15)).mean())
    metrics = dict(samples=len(df),next_state_accuracy=float((df.predicted_next_state==df.actual_next_state).mean()),
                   observation_accuracy=float((df.predicted_observation==df.actual_observation).mean()),
                   next_state_log_loss_clipped=nll('p_true_next_state'),observation_log_loss_clipped=nll('p_true_observation'),
                   joint_log_loss_clipped=nll('p_true_joint'),unseen_joint_outcomes=int((df.p_true_joint==0).sum()),
                   unseen_observations=int((df.p_true_observation==0).sum()),
                   unseen_next_states=int((df.p_true_next_state==0).sum()),
                   next_state_brier=float(df.next_state_brier.mean()),confirmation_brier=float(df.confirmation_brier.mean()),
                   duration_mae_ms=float((df.expected_binned_duration_ms-df.actual_duration_ms).abs().mean()))
    return metrics,df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b3',default='bayesian/outputs/b3')
    p.add_argument('--out',default='bayesian/outputs/b4')
    a=p.parse_args(); root=Path(a.b3); out=Path(a.out)
    data=pd.read_csv(root/'recovery_validated.csv'); splits=pd.read_csv(root/'episode_splits.csv')
    require(data.episode.is_unique and splits.episode.is_unique,'Duplicate episode identifiers')
    require(set(data.episode)<=set(splits.episode),'Missing split assignments')
    require(splits.split.isin(['train','validation','test']).all(),'Invalid split label')
    data=data.merge(splits,on='episode',validate='one_to_one')
    # Test rows are not used in fitting, selection or scoring.
    train=data[data.split=='train'].copy(); val=data[data.split=='validation'].copy()
    for d in (train,val):
        require(len(d)>0,'Empty development partition')
        d['keyed']=d.keyed.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
        require(d.keyed.notna().all(),'Invalid keyed flag')
        d['keyed']=d.keyed.astype(bool)
        require(d.state.isin(STATES).all() and d.next_state.isin(STATES).all(),'Unknown state label')
        require(d.observation.notna().all() and np.isfinite(d.duration_ms).all() and (d.duration_ms>=0).all(),'Invalid action outcome')
        require(all(r.action!='stop' and r.action in safe_actions(r.keyed,500,3) for r in d.itertuples()),'Unsafe action in development data')
        require((d.loc[d.state=='completed','next_state']=='completed').all(),'Completed state regressed')
        require((d.loc[d.state=='processing','next_state']!='unstarted').all(),'Processing state regressed')
        require((d.loc[d.observation=='confirmed','next_state']=='completed').all(),'Confirmation inconsistent with validated state')
    saved=joblib.load(root/'bayesian_b3.joblib')
    require(tuple(saved['states'])==tuple(STATES) and tuple(saved['features'])==tuple(FEATURES),'Incompatible B3 model')
    kernel,coverage,outcomes=fit_kernel(train)
    oracle,oracle_df=assess(kernel,saved['bayes'],val,oracle=True)
    belief,belief_df=assess(kernel,saved['bayes'],val,oracle=False)
    report={'train_samples':len(train),'validation_samples':len(val),'test_evaluated':False,
            'duration_bin_ms':25,'collection_http_cap_ms':180,'collection_wait_target_ms':75,
            'minimum_training_cell_count':int(coverage.samples.min()),'eligible_cells':len(coverage),
            'oracle_state_diagnostic':oracle,'b3_belief_prediction':belief,
            'limitations':['One-step empirical joint probabilities; no smoothing of unseen outcomes.',
                           'Oracle diagnostic uses true state only for offline assessment, never for runtime inference.',
                           'Collection used fixed action timeout and wait target; outcomes at shorter remaining deadlines are not validated.',
                           'Multi-step history, latent fault dependence, and action-censoring effects are not established.',
                           'Log losses are clipped at 1e-15; zero-probability counts are reported separately.']}
    out.mkdir(parents=True,exist_ok=True)
    coverage.to_csv(out/'training_coverage.csv',index=False)
    outcomes.to_csv(out/'action_outcomes.csv',index=False)
    oracle_df.to_csv(out/'validation_oracle_predictions.csv',index=False)
    belief_df.to_csv(out/'validation_belief_predictions.csv',index=False)
    (out/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    joblib.dump({'kernel':kernel,'bayes':saved['bayes'],'states':STATES,'features':FEATURES,'duration_bin_ms':25},out/'bayesian_b4.joblib')
    print('Training state-action coverage:'); print(coverage.to_string(index=False))
    print('\n'+json.dumps(report,indent=2)); print('\nSaved B4 to',out,'; test split untouched.')

if __name__=='__main__':
    main()
