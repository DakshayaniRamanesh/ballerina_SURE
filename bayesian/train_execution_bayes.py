"""SURE B3: audit request data, freeze episode splits and fit Bayesian baseline.
Run from project root: python -m bayesian.train_execution_bayes
The final test partition is exported, but never scored by this script.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sure.policy import STATES, FEATURES, EmpiricalBayes, safe_actions


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def bool_column(series):
    result = series.astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    require(result.notna().all(), 'Invalid boolean value')
    return result.astype(bool)


def truth_at(df, time_column):
    return np.where((df.commit_ms > 0) & (df.commit_ms <= df[time_column]), 'completed',
                    np.where((df.start_ms > 0) & (df.start_ms <= df[time_column]), 'processing', 'unstarted'))


# Conservative screening threshold for the observed millisecond discrepancy.
# This is not a measured upper bound on clock error.
BOUNDARY_MS = 2

def audit_data(data, audit):
    data = data.copy(); audit = audit.copy()
    require(data.episode.notna().all() and audit.episode.notna().all(), 'Missing episode ID')
    require(data.episode.is_unique and audit.episode.is_unique, 'Duplicate episode ID')
    for df in (data, audit):
        df['keyed'] = bool_column(df.keyed)
        require(df.initial_timeout_ms.isin([100, 150, 200]).all(), 'Unexpected initial timeout')
    require(audit.recovery_sample.isin([0, 1]).all(), 'Invalid recovery flag')
    require(set(data.episode) == set(audit.loc[audit.recovery_sample == 1, 'episode']), 'Recovery/audit episode mismatch')
    require((audit.recovery_sample == (audit.initial_observation != 'confirmed').astype(int)).all(), 'Recovery flags disagree with initial observations')
    for col in ['start_ms','commit_ms','effects','attempts','decision_epoch_ms','action_end_epoch_ms']:
        require(np.isfinite(audit[col]).all() and (audit[col] >= 0).all(), f'Invalid audit column: {col}')
    require((audit.action_end_epoch_ms >= audit.decision_epoch_ms).all(), 'Action ended before decision')
    require(((audit.commit_ms == 0) | ((audit.start_ms > 0) & (audit.commit_ms >= audit.start_ms))).all(), 'Invalid start/commit ordering')
    require((truth_at(audit, 'decision_epoch_ms') == audit.state).all(), 'Initial state label mismatch')
    require((truth_at(audit, 'action_end_epoch_ms') == audit.next_state).all(), 'Next state label mismatch')
    require(((audit.effects > 0) == (audit.commit_ms > 0)).all(), 'Commit/effects disagree')
    require((audit.effects <= 1).all(), 'Duplicate execution detected: inspect safety before training')
    # The controlled service sends completed only after committing. The Python
    # clock truncates ms; the Ballerina decimal-to-int conversion can round.
    # Preserve and check the original timestamp-derived labels above.
    initial_conflict = (audit.initial_observation == 'confirmed') & (audit.state != 'completed')
    initial_delta = audit.commit_ms - audit.decision_epoch_ms
    acceptable_initial = ((audit.state == 'processing') & (audit.effects == 1) &
                          (audit.start_ms > 0) & (audit.start_ms <= audit.decision_epoch_ms) &
                          (initial_delta > 0) & (initial_delta <= BOUNDARY_MS) &
                          (audit.recovery_sample == 0))
    require((~initial_conflict | acceptable_initial).all(),
            'Initial confirmation conflict outside the allowed boundary; inspect audit')
    require((audit.loc[audit.initial_observation == 'confirmed', 'effects'] == 1).all(),
            'Initial confirmation without exactly one recorded effect')
    print('Initial confirmation timestamp exceptions (excluded from recovery by design):', int(initial_conflict.sum()))
    j = data.merge(audit, on='episode', suffixes=('', '_audit'), validate='one_to_one')
    for left, right in [('mode','mode_audit'),('keyed','keyed_audit'),('state','state_audit'),('next_state','next_state_audit'),('action','planned_action'),('error','initial_observation'),('initial_timeout_ms','initial_timeout_ms_audit')]:
        require((j[left] == j[right]).all(), f'Recovery/audit mismatch: {left}')
    require(np.allclose(j.elapsed_ms, j.initial_elapsed_ms), 'Initial timing mismatch')
    for col in ['elapsed_ms','duration_ms']:
        require(np.isfinite(data[col]).all() and (data[col] >= 0).all(), f'Invalid duration: {col}')
    require(data.state.isin(STATES).all() and data.next_state.isin(STATES).all(), 'Unknown state label')
    require(data.error.isin(['timeout','error','unknown','pending']).all(), 'Unexpected initial observation')
    require(all(r.action in safe_actions(r.keyed, 500, 3) and r.action != 'stop' for r in data.itertuples()), 'Ineligible recovery action')
    recovery_conflict = (j.observation == 'confirmed') & (j.next_state != 'completed')
    recovery_delta = j.commit_ms - j.action_end_epoch_ms
    acceptable_recovery = ((j.next_state == 'processing') & (j.effects == 1) &
                           (j.start_ms > 0) & (j.start_ms <= j.action_end_epoch_ms) &
                           (recovery_delta > 0) & (recovery_delta <= BOUNDARY_MS))
    require((~recovery_conflict | acceptable_recovery).all(),
            'Recovery confirmation conflict outside the allowed boundary; inspect audit')
    require((j.loc[j.observation == 'confirmed', 'effects'] == 1).all(),
            'Recovery confirmation without exactly one recorded effect')
    print('Recovery confirmation timestamp exceptions:', int(recovery_conflict.sum()))
    near = np.zeros(len(audit), dtype=bool)
    for event in ['start_ms','commit_ms']:
        for point in ['decision_epoch_ms','action_end_epoch_ms']:
            near |= ((audit[event] > 0) & ((audit[event] - audit[point]).abs() <= BOUNDARY_MS)).to_numpy()
    # Keep the full audit for reproducible stratified splits; exclude uncertain
    # recovery rows from all estimator/outcome-model partitions consistently.
    quarantine = audit.loc[near & (audit.recovery_sample == 1)].copy()
    quarantine['reason'] = 'event timestamp within 2 ms of decision or action-end boundary'
    clean = data.loc[~data.episode.isin(quarantine.episode)].copy()
    print('Audit passed:', len(audit), 'initial episodes;', len(data), 'raw recovery samples; zero recorded duplicates')
    print('Episodes with event timestamps within 2 ms of a labeling boundary:', int(near.sum()))
    print('Quarantined recovery samples:', len(quarantine))
    print('Usable recovery samples:', len(clean))
    return clean, audit, quarantine


def make_splits(audit, seed):
    rng = np.random.default_rng(seed)
    rows = []
    # Design metadata is only used to distribute experiments, not to predict states.
    for _, group in audit.groupby(['mode','keyed','initial_timeout_ms','planned_action'], sort=True):
        ids = np.array(sorted(group.episode))
        require(len(ids) >= 4, 'Too few repeats in an experimental cell')
        rng.shuffle(ids)
        n_hold = max(1, len(ids)//4)
        splits = ['test']*n_hold + ['validation']*n_hold + ['train']*(len(ids)-2*n_hold)
        rows.extend(zip(ids, splits))
    return pd.DataFrame(rows, columns=['episode','split'])


def metrics(y, probs):
    idx = np.array([STATES.index(s) for s in y])
    pred = np.array(STATES)[probs.argmax(axis=1)]
    onehot = np.eye(len(STATES))[idx]
    return {'accuracy': float(accuracy_score(y,pred)), 'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
            'macro_f1': float(f1_score(y,pred,labels=STATES,average='macro',zero_division=0)),
            'log_loss': float(-np.log(np.clip(probs[np.arange(len(idx)),idx],1e-15,1)).mean()),
            'multiclass_brier': float(np.square(probs-onehot).sum(axis=1).mean())}, pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/recovery_balanced.csv')
    p.add_argument('--audit', default='data/recovery_balanced_episodes.csv')
    p.add_argument('--out', default='bayesian/outputs/b3')
    p.add_argument('--seed', type=int, default=2027)
    a = p.parse_args()
    data, audit, quarantine = audit_data(pd.read_csv(a.data), pd.read_csv(a.audit))
    out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
    splits = make_splits(audit,a.seed)
    split_path = out/'episode_splits.csv'
    if split_path.exists():
        old = pd.read_csv(split_path).sort_values('episode').reset_index(drop=True)
        require(old.equals(splits.sort_values('episode').reset_index(drop=True)), 'Existing split differs; use original seed/data or new output folder')
    splits.to_csv(split_path,index=False)
    quarantine.to_csv(out/'quarantined_episodes.csv', index=False)
    data.to_csv(out/'recovery_validated.csv', index=False)
    data = data.merge(splits,on='episode',validate='one_to_one')
    print('\nRecovery samples by split and state:')
    print(pd.crosstab(data.split,data.state))
    train = data[data.split == 'train']; val = data[data.split == 'validation']
    for d in (train,val):
        require(set(d.state) == set(STATES),'A development split lacks a state')
    records = train[list(FEATURES)+['state']].to_dict('records')
    model = EmpiricalBayes().fit(records)
    counts = train.state.value_counts()
    prior = np.array([counts.get(s,0)+1 for s in STATES],dtype=float); prior /= prior.sum()
    inputs = val[list(FEATURES)].to_dict('records')
    probs = np.array([model.predict(f) for f in inputs])
    require(np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1),'Invalid predicted probabilities')
    prior_metrics,_ = metrics(val.state,np.tile(prior,(len(val),1)))
    conditional_metrics,pred = metrics(val.state,probs)
    report = {'states': list(STATES),'runtime_inputs_available': list(FEATURES),
              'actual_conditioning_features':['error','keyed','initial_timeout_ms'],
              'elapsed_ms_used':False,'smoothing_alpha':1,'boundary_screen_ms':BOUNDARY_MS,
              'quarantined_recovery_samples':len(quarantine),'train_samples':len(train),
              'validation_samples':len(val),'test_evaluated':False,
              'global_prior_validation':prior_metrics,'B3_conditional_validation':conditional_metrics}
    print('\n'+json.dumps(report,indent=2))
    print('\nB3 validation confusion matrix:')
    print(pd.DataFrame(confusion_matrix(val.state,pred,labels=STATES),index=STATES,columns=STATES))
    predictions = val[['episode','state']].copy(); predictions['prediction'] = pred
    for i,s in enumerate(STATES): predictions['p_'+s] = probs[:,i]
    predictions.to_csv(out/'validation_predictions.csv',index=False)
    (out/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.crosstab([train.error,train.keyed,train.initial_timeout_ms],train.state).to_csv(out/'training_state_counts.csv')
    joblib.dump({'bayes':model,'states':STATES,'features':FEATURES,'prior':prior},out/'bayesian_b3.joblib')
    print('\nSaved to',out,'; test split not evaluated.')
    print('Share episode_splits.csv and recovery_validated.csv with the RF developer. Priors reflect the experimental mixture, not deployment prevalence.')

if __name__ == '__main__':
    main()
