"""B2: causal fault belief filtering, training-fold selection, validation only.
Run from project root: python bayesian/temporal_bayesian.py
Requires prepare_rcaeval.py and tune_bayesian.py outputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix

OUT = Path('bayesian/outputs')
TARGET_RECALL = .90  # exploratory operating target, not a safety guarantee


def soften(logp, t):
    z = logp / t
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit(train, features, classes, config):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
        GaussianNB(priors=np.full(len(classes), 1/len(classes)) if config['prior']=='uniform' else None,
                   var_smoothing=config['var_smoothing']))
    model.fit(train[features], train.label)
    assert np.array_equal(model.classes_, classes)
    prior = train.label.value_counts().reindex(classes).to_numpy(dtype=float)
    prior /= prior.sum()
    return model, prior


def transition(classes, onset, recovery):
    """Hypothesized per-60s Markov transitions; no fault identity or injection time inputs."""
    n = len(classes); normal = list(classes).index('normal')
    t = np.zeros((n, n))
    for i in range(n):
        if i == normal:
            t[i, :] = onset/(n-1)
            t[i, i] = 1-onset
        else:
            t[i, i] = 1-recovery
            t[i, normal] = recovery
    return t


def causal_filter(frame, posteriors, priors, classes, onset, recovery, strength):
    """Approximate emission likelihood ratio p(class|x)/training prior.
    strength tempers correlated evidence; never uses labels or future telemetry.
    Each independent case starts from the training prior, not a known normal state.
    """
    result = np.empty_like(posteriors)
    T = transition(classes, onset, recovery)
    for _, case in frame.groupby('case_id', sort=False):
        ids = case.sort_values('window_start_s').index.to_numpy()
        belief = priors[ids[0]].copy()
        previous = None
        for idx in ids:
            ts = frame.loc[idx, 'window_start_s']
            if previous is not None:
                gap = (ts-previous)/60
                if gap < 1 or not np.isclose(gap, round(gap)):
                    raise ValueError('Expected positive multiples of 60s between windows')
                belief = belief @ np.linalg.matrix_power(T, int(round(gap)))
            # No double-counting the classifier's training prior.
            logw = np.log(np.clip(belief, 1e-15, 1)) + strength*(
                np.log(np.clip(posteriors[idx], 1e-15, 1)) - np.log(priors[idx]))
            weights = np.exp(logw-logw.max())
            belief = weights/weights.sum()
            result[idx] = belief
            previous = ts
    return result


def predict(p, classes, threshold=None):
    if threshold is None:
        return classes[p.argmax(axis=1)]
    normal = list(classes).index('normal')
    fault_indices = np.flatnonzero(classes!='normal')
    best_fault = fault_indices[p[:,fault_indices].argmax(axis=1)]
    return np.where(1-p[:,normal]>=threshold, classes[best_fault], 'normal')


def scores(y, p, classes, threshold=None):
    y=np.asarray(y); pred=predict(p,classes,threshold); normal=y=='normal'; fault=~normal
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro')),log_loss=float(log_loss(y,p,labels=classes)),
        multiclass_brier=float(np.mean(np.sum((p-(y[:,None]==classes[None,:]))**2,axis=1))),
        nominal_false_alarm_rate=float(np.mean(pred[normal]!='normal')),
        fault_detection_recall=float(np.mean(pred[fault]!='normal')))


def choose_threshold(y,p,classes):
    # Choose lowest false-alarm rate subject to exploratory recall target, training OOF only.
    y=np.asarray(y); q=1-p[:,list(classes).index('normal')]; normal=y=='normal'; fault=~normal
    options=[]
    for threshold in np.unique(np.r_[0.,q,1.]):
        detected=q>=threshold
        recall=float(detected[fault].mean()); far=float(detected[normal].mean())
        if recall>=TARGET_RECALL:
            options.append((far,-recall,-float(threshold),float(threshold)))
    return min(options)[-1]


def main():
    df=pd.read_csv(OUT/'rcaeval_features.csv')
    features=json.loads((OUT/'feature_columns.json').read_text(encoding='utf-8'))
    config=json.loads((OUT/'gnb_tuned_validation_metrics.json').read_text(encoding='utf-8'))['selected_on_training_cv']
    if df.groupby('case_id').split.nunique().max()!=1: raise ValueError('Case leakage')
    train=df[df.split=='train'].reset_index(drop=True)
    val=df[df.split=='validation'].reset_index(drop=True)
    classes=np.sort(train.label.unique())
    if set(classes)!=set(val.label) or 'normal' not in classes: raise ValueError('Missing class')
    if len(train.run.unique())!=3 or train.groupby('case_id').run.nunique().max()!=1:
        raise ValueError('Expected three training repetition groups')
    p_oof=np.empty((len(train),len(classes))); prior_oof=np.empty_like(p_oof)
    for run in sorted(train.run.unique()):
        hold=train.run==run
        model,prior=fit(train.loc[~hold],features,classes,config)
        p_oof[hold]=soften(model.predict_log_proba(train.loc[hold,features]),config['temperature'])
        prior_oof[hold]=prior
    rows=[]
    for onset in (.02,.05,.1,.2,.35):
        for recovery in (.02,.1,.25):
            for strength in (.25,.5,1.):
                p=causal_filter(train,p_oof,prior_oof,classes,onset,recovery,strength)
                rows.append(dict(onset=onset,recovery=recovery,strength=strength,**scores(train.label,p,classes)))
    tuning=pd.DataFrame(rows).sort_values(['log_loss','multiclass_brier'])
    best={k:float(tuning.iloc[0][k]) for k in ('onset','recovery','strength')}
    temporal_oof=causal_filter(train,p_oof,prior_oof,classes,**best)
    t_static=choose_threshold(train.label,p_oof,classes)
    t_temporal=choose_threshold(train.label,temporal_oof,classes)
    model,prior=fit(train,features,classes,config)
    p_val=soften(model.predict_log_proba(val[features]),config['temperature'])
    temporal_val=causal_filter(val,p_val,np.broadcast_to(prior,p_val.shape),classes,**best)
    results={
        'B1.1_argmax':scores(val.label,p_val,classes),
        'B1.1_threshold':scores(val.label,p_val,classes,t_static),
        'B2_argmax':scores(val.label,temporal_val,classes),
        'B2_threshold':scores(val.label,temporal_val,classes,t_temporal)}
    report=dict(transition_parameters=best,training_oof_recall_target=TARGET_RECALL,
        static_threshold=t_static,temporal_threshold=t_temporal,
        training_cases=int(train.case_id.nunique()),validation_cases=int(val.case_id.nunique()),
        validation_results=results,test_evaluated=False,
        limitations='Approximate likelihood ratios and assumed transition family; injection-relative offsets used only for elapsed time and ordering. Training OOF is for tuning, not an independent estimate.')
    print(json.dumps(report,indent=2))
    print('\nB2 threshold confusion matrix:')
    print(pd.DataFrame(confusion_matrix(val.label,predict(temporal_val,classes,t_temporal),labels=classes),index=classes,columns=classes))
    tuning.to_csv(OUT/'b2_train_cv.csv',index=False)
    (OUT/'b2_validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    predictions=val[['case_id','window_start_s','label']].copy()
    predictions['b1_prediction']=predict(p_val,classes,t_static)
    predictions['b2_prediction']=predict(temporal_val,classes,t_temporal)
    for j,c in enumerate(classes):
        predictions['b1_p_'+c]=p_val[:,j]; predictions['b2_p_'+c]=temporal_val[:,j]
    predictions.to_csv(OUT/'b2_validation_predictions.csv',index=False)
    print('\nSaved B2 validation results. Test split untouched.')

if __name__=='__main__': main()
