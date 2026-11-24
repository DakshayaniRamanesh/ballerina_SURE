"""Tune GNB on training repetitions only; evaluate on validation, never test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

OUT = Path('bayesian/outputs')
SMOOTH = (1e-9, 1e-6, 1e-4, 1e-3, 1e-2, .1, 1.)
TEMPS = (1., 1.5, 2., 3., 5., 10., 20., 50., 100.)

def model_for(prior, smoothing, k):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
        GaussianNB(priors=np.full(k,1/k) if prior=='uniform' else None,
                   var_smoothing=smoothing))

def soften(logp, temperature):
    z = logp / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

def metrics(y, p, classes):
    y = np.asarray(y)
    pred = classes[p.argmax(axis=1)]
    normal = y=='normal'
    fault = ~normal
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'log_loss':float(log_loss(y,p,labels=classes)),
            'multiclass_brier':float(np.mean(np.sum((p-(y[:,None]==classes[None,:]))**2,axis=1))),
            'nominal_false_alarm_rate':float(np.mean(pred[normal]!='normal')),
            'fault_detection_recall':float(np.mean(pred[fault]!='normal'))}

def main():
    df = pd.read_csv(OUT/'rcaeval_features.csv')
    columns = json.loads((OUT/'feature_columns.json').read_text(encoding='utf-8'))
    if df.groupby('case_id').split.nunique().max()!=1:
        raise ValueError('Case leakage across splits')
    train = df[df.split=='train'].reset_index(drop=True)
    val = df[df.split=='validation'].reset_index(drop=True)
    classes = np.sort(train.label.unique())
    if set(classes)!=set(val.label) or len(classes)!=6:
        raise ValueError('Expected six classes in train and validation')
    runs = sorted(train.run.unique())
    if len(runs)!=3 or train.groupby('case_id').run.nunique().max()!=1:
        raise ValueError('Expected three independent training repetition groups')
    X=train[columns]; y=train.label.to_numpy()
    if X.isna().all().any(): raise ValueError('Entirely missing training feature')
    records=[]
    for prior in ('uniform','empirical'):
        for smoothing in SMOOTH:
            oof=np.empty((len(train),len(classes)))
            for run in runs:
                held=train.run.to_numpy()==run
                if set(y[~held])!=set(classes): raise ValueError('Missing fold class')
                model=model_for(prior,smoothing,len(classes))
                model.fit(X.loc[~held],y[~held])
                oof[held]=model.predict_log_proba(X.loc[held])
            for temp in TEMPS:
                m=metrics(y,soften(oof,temp),classes)
                records.append({'prior':prior,'var_smoothing':smoothing,'temperature':temp,**m})
    scores=pd.DataFrame(records).sort_values(['log_loss','multiclass_brier']).reset_index(drop=True)
    scores.to_csv(OUT/'gnb_tuning_train_cv.csv',index=False)
    best=scores.iloc[0]
    chosen={'prior':str(best.prior),'var_smoothing':float(best.var_smoothing),'temperature':float(best.temperature)}
    model=model_for(chosen['prior'],chosen['var_smoothing'],len(classes))
    model.fit(X,y)
    p=soften(model.predict_log_proba(val[columns]),chosen['temperature'])
    pred=classes[p.argmax(axis=1)]
    baseline=model_for('uniform',1e-9,len(classes)).fit(X,y)
    baseline_p=baseline.predict_proba(val[columns])
    report={'selected_on_training_cv':chosen,'baseline_validation':metrics(val.label,baseline_p,classes),'tuned_validation':metrics(val.label,p,classes),'train_cases':int(train.case_id.nunique()),'validation_cases':int(val.case_id.nunique()),'test_evaluated':False}
    print(json.dumps(report,indent=2))
    print(classification_report(val.label,pred,labels=classes,zero_division=0))
    print(pd.DataFrame(confusion_matrix(val.label,pred,labels=classes),index=classes,columns=classes))
    (OUT/'gnb_tuned_validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    result=val[['case_id','window_start_s','label']].copy()
    result['prediction']=pred
    for i,c in enumerate(classes): result[f'p_{c}']=p[:,i]
    result.to_csv(OUT/'gnb_tuned_validation_predictions.csv',index=False)
    joblib.dump({'model':model,'features':columns,'classes':classes.tolist(),'temperature':chosen['temperature']},OUT/'gnb_tuned.joblib')
    print('Saved tuning results. Test split untouched. Use soften(model.predict_log_proba(X), temperature) for calibrated predictions.')

if __name__=='__main__': main()
