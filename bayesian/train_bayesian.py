"""Train Gaussian Naive Bayes on the shared RCAEval window data."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, confusion_matrix, classification_report)

OUT = Path('bayesian/outputs')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--fault-only',action='store_true',help='Five-class diagnosis given a fault is present')
    args=parser.parse_args()
    df=pd.read_csv(OUT/'rcaeval_features.csv')
    columns=json.loads((OUT/'feature_columns.json').read_text(encoding='utf-8'))
    if args.fault_only:
        df=df[df.label!='normal'].copy()
    train=df[df.split=='train'].copy()
    evaluation=df[df.split==args.split].copy()
    if set(train.case_id)&set(evaluation.case_id):
        raise ValueError('Case leakage')
    X_train=train[columns]; y_train=train.label
    X_eval=evaluation[columns]; y_eval=evaluation.label
    if X_train.isna().all().any():
        raise ValueError('A feature is entirely missing in training; revise shared features')
    if set(y_train)!=set(y_eval):
        raise ValueError('Missing class in training or evaluation')
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),GaussianNB())
    # Equal class prior for this balanced-fault benchmark; not a production fault-rate estimate.
    classes=np.sort(y_train.unique())
    model.set_params(gaussiannb__priors=np.full(len(classes),1/len(classes)))
    model.fit(X_train,y_train)
    classes=model.classes_
    probs=model.predict_proba(X_eval)
    pred=model.predict(X_eval)
    onehot=(y_eval.to_numpy()[:,None]==classes[None,:]).astype(float)
    scores={'accuracy':accuracy_score(y_eval,pred),
            'balanced_accuracy':balanced_accuracy_score(y_eval,pred),
            'macro_f1':f1_score(y_eval,pred,average='macro'),
            'log_loss':log_loss(y_eval,probs,labels=classes),
            'multiclass_brier':float(np.mean(np.sum((probs-onehot)**2,axis=1))),
            'train_cases':int(train.case_id.nunique()),'evaluation_cases':int(evaluation.case_id.nunique()),
            'train_windows':len(train),'evaluation_windows':len(evaluation)}
    tag=('fault5' if args.fault_only else 'condition6')+'_'+args.split
    print('Classes:',classes.tolist())
    print(json.dumps(scores,indent=2))
    print(classification_report(y_eval,pred,labels=classes,zero_division=0))
    print('Confusion matrix (rows actual; columns predicted):')
    cm=pd.DataFrame(confusion_matrix(y_eval,pred,labels=classes),index=classes,columns=classes)
    print(cm.to_string())
    cm.to_csv(OUT/f'gnb_{tag}_confusion.csv')
    output=evaluation[['case_id','window_start_s','label']].copy()
    output['prediction']=pred
    for i,c in enumerate(classes): output[f'p_{c}']=probs[:,i]
    output.to_csv(OUT/f'gnb_{tag}_predictions.csv',index=False)
    (OUT/f'gnb_{tag}_metrics.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    joblib.dump({'model':model,'features':columns,'classes':classes.tolist()},OUT/f'gnb_{tag}.joblib')
    print('Saved model and metrics to',OUT)

if __name__=='__main__': main()
