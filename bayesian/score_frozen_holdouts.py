"""Score the previously reserved partitions once, without changing saved B3/B4.1.
B2 was not serialized by its original script; reconstruct from training-only data
and verify reproduction of validation metrics before scoring its reserved partition.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from . import train_execution_bayes as b3mod
from . import train_time_aware_kernel as b41mod
from . import temporal_bayesian as b2mod
from .independent_evaluation import ROOT,B3,KERNEL,check_models,verify_freeze,sha,dump,load,require

OUT=Path('bayesian/outputs')

def same_metrics(actual,expected,keys):
    for k in keys:
        require(np.isclose(actual[k],expected[k],rtol=1e-6,atol=1e-8),f'Validation reproduction failed: {k}: {actual[k]} vs {expected[k]}')

def main():
    protocol=verify_freeze(); bayes,kernel=check_models(); dest=ROOT/'holdouts'
    require(not protocol['missing_holdout_inputs'],'Holdout inputs missing at freeze; resolve before scoring; do not tune on test')
    require(all(sha(p)==h for p,h in protocol['holdout_input_sha256'].items()),'Holdout inputs changed after freeze')
    require(not dest.exists(),'Holdout output exists. Do not repeat test scoring for tuning.')
    # Verify the frozen estimator and saved split definitions first.
    b3path=OUT/'b3'; b41path=OUT/'b41'
    data=pd.read_csv(b3path/'recovery_validated.csv'); data['keyed']=b3mod.bool_column(data.keyed)
    split=pd.read_csv(b3path/'episode_splits.csv')
    require(split.episode.is_unique and data.episode.is_unique and set(data.episode)<=set(split.episode),'Invalid B3 split')
    data=data.merge(split,on='episode',validate='one_to_one')
    require(data.split.value_counts().to_dict()=={'train':405,'validation':201,'test':204},'B3 split counts changed')
    def b3score(d):
        probs=np.array([bayes.predict(x) for x in d[list(b3mod.FEATURES)].to_dict('records')])
        require(np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(axis=1),1),'Invalid B3 probabilities')
        met,pred=b3mod.metrics(d.state,probs)
        pp=d[['episode','state']].copy();pp['prediction']=pred
        for j,s in enumerate(b3mod.STATES): pp['p_'+s]=probs[:,j]
        met['confusion_matrix']=confusion_matrix(d.state,pred,labels=b3mod.STATES).tolist()
        return met,pp
    val,_=b3score(data[data.split=='validation']); old=load(b3path/'validation_metrics.json')['B3_conditional_validation']
    same_metrics(val,old,old.keys())
    raw=pd.read_csv('data/recovery_multistep_v2.csv'); eps=pd.read_csv('data/recovery_multistep_v2_episodes.csv')
    raw=b41mod.audit(raw,eps); splits=pd.read_csv(b41path/'episode_splits.csv')
    expected=eps[['episode','repeat']].copy();expected['split']=expected.repeat.map({1:'train',2:'train',3:'validation',4:'test'})
    require(splits.sort_values('episode').reset_index(drop=True).equals(expected.sort_values('episode').reset_index(drop=True)),'B4.1 split changed')
    require(raw[raw.usable==1].groupby('repeat').size().reindex([3,4]).tolist()==[481,487],'B4.1 counts changed')
    # Use original exported training counts, never fit on holdout. TimeKernel parses them.
    tables=kernel.tables
    require(tuple(kernel.edges)==b41mod.EDGES and kernel.alpha==b41mod.ALPHA,'Kernel context mismatch')
    names=['B4.0_matched','B4.1a_time','B4.1b_time_history']; original=load(b41path/'validation_metrics.json')['models']
    for level,name in enumerate(names):
        m,_=b41mod.evaluate(tables,level,raw[raw.repeat==3],None,bayes)
        same_metrics(m,original[name],('oracle_joint_nll','replay_joint_nll','replay_next_state_brier','replay_episode_mean_joint_nll'))
    # B2: original code fitted a training-only GNB and saved settings, not a model.
    frame=pd.read_csv(OUT/'rcaeval_features.csv'); feats=load(OUT/'feature_columns.json')
    config=load(OUT/'gnb_tuned_validation_metrics.json')['selected_on_training_cv']; b2report=load(OUT/'b2_validation_metrics.json')
    require(frame.groupby('case_id').split.nunique().max()==1,'RCAEval case leakage')
    require(frame.groupby('split').case_id.nunique().to_dict()=={'train':75,'validation':25,'test':25},'RCAEval split changed')
    train=frame[frame.split=='train'].reset_index(drop=True);classes=np.sort(train.label.unique())
    model,prior=b2mod.fit(train,feats,classes,config)
    def b2score(f):
        f=f.reset_index(drop=True); p=b2mod.soften(model.predict_log_proba(f[feats]),config['temperature'])
        temporal=b2mod.causal_filter(f,p,np.broadcast_to(prior,p.shape),classes,**b2report['transition_parameters'])
        result={'B1.1_argmax':b2mod.scores(f.label,p,classes),
                'B1.1_threshold':b2mod.scores(f.label,p,classes,b2report['static_threshold']),
                'B2_argmax':b2mod.scores(f.label,temporal,classes),
                'B2_threshold':b2mod.scores(f.label,temporal,classes,b2report['temporal_threshold'])}
        pred=f[['case_id','window_start_s','label']].copy()
        pred['prediction']=b2mod.predict(temporal,classes,b2report['temporal_threshold'])
        for j,c in enumerate(classes):pred['p_'+c]=temporal[:,j]
        return result,pred
    reproduction,_=b2score(frame[frame.split=='validation'])
    for name,m in reproduction.items(): same_metrics(m,b2report['validation_results'][name],m.keys())
    # Only now create the one-time output directory; all validation reproduction checks passed.
    dest.mkdir(parents=True,exist_ok=False)
    m3,p3=b3score(data[data.split=='test']);p3.to_csv(dest/'b3_predictions.csv',index=False)
    baseline,_=b3mod.metrics(data[data.split=='test'].state,np.tile(joblib.load(B3)['prior'],(204,1)))
    m41={}
    for level,name in enumerate(names):
        m41[name],pred=b41mod.evaluate(tables,level,raw[raw.repeat==4],None,bayes)
        pred.to_csv(dest/(name+'_predictions.csv'),index=False)
    m2,p2=b2score(frame[frame.split=='test']);p2.to_csv(dest/'b2_predictions.csv',index=False)
    report=dict(b3=m3,b3_global_prior=baseline,b41=m41,b2=m2,b2_classes=classes.tolist(),
                b2_reconstructed_training_only=True,holdout_counts=dict(b3=204,b41=487,b2=len(p2)),
                hashes={str(p):sha(p) for p in [B3,KERNEL,b3path/'recovery_validated.csv',b3path/'episode_splits.csv',b41path/'episode_splits.csv',Path('data/recovery_multistep_v2.csv'),Path('data/recovery_multistep_v2_episodes.csv'),OUT/'rcaeval_features.csv']},
                limitations=['Reserved records have previously been included in descriptive data audits; not fully blinded.','B2 supporting fault model is not part of live recovery.','B4.1 uses recorded actions for replay; not counterfactual policy evaluation.'])
    dump(dest/'test_metrics.json',report); print(json.dumps(report,indent=2));print('Saved one-time reserved scores to',dest)
if __name__=='__main__':main()
