"""Shared RCAEval RE1-OB preprocessing for SURE, run from project root."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path('data')
OUT = Path('bayesian/outputs')
FAULTS = {'cpu', 'delay', 'disk', 'loss', 'mem'}
SERVICES = ('adservice','cartservice','checkoutservice','currencyservice',
            'emailservice','frontend','main','paymentservice',
            'productcatalogservice','recommendationservice','redis','shippingservice')
METRICS = [f'{s}_{m}' for s in SERVICES for m in ('cpu','mem')]
WINDOW = 60
# Three nonoverlapping windows on each side, excluding 30 s around injection.
WINDOWS = [(a, a+WINDOW, 'nominal') for a in (-210,-150,-90)] + [
    (a,a+WINDOW,'injected') for a in (30,90,150)]
FEATURES = [f'{m}__{stat}' for m in METRICS for stat in ('mean','std')]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cases, rejected = [], [], []
    dirs = sorted(p for p in DATA.glob('re1ob_*') if p.is_dir())
    if len(dirs) != 125:
        raise ValueError(f'Expected 125 case directories; found {len(dirs)}')
    for case in dirs:
        _, service, fault, run_text = case.name.split('_')
        run = int(run_text)
        if fault not in FAULTS or run not in range(1,6):
            raise ValueError(f'Unexpected case: {case.name}')
        split = 'train' if run <= 3 else ('validation' if run == 4 else 'test')
        injection = int((case/'inject_time.txt').read_text().strip())
        df = pd.read_parquet(case/'metrics.parquet', columns=['time']+METRICS)
        df = df.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
        df = df.dropna(subset=['time']).sort_values('time')
        if df['time'].duplicated().any():
            raise ValueError(f'Duplicate timestamps in {case.name}; inspect before proceeding')
        times = df['time'].to_numpy()
        if len(times)<2 or not times[0] <= injection <= times[-1]:
            raise ValueError(f'Invalid time range in {case.name}')
        cadence = float(np.median(np.diff(times)))
        if not 0 < cadence <= WINDOW/4:
            raise ValueError(f'Unexpected sampling interval {cadence} in {case.name}; verify time units')
        expected = WINDOW/cadence
        case_rows = []
        for lo, hi, phase in WINDOWS:
            window = df[(df.time >= injection+lo) & (df.time < injection+hi)]
            # Check temporal coverage as well as row count; do not interpolate across fault boundary.
            if len(window) < max(2, .8*expected) or window.time.max()-window.time.min() < .75*WINDOW:
                rejected.append({'case_id':case.name,'window_start_s':lo,'reason':'insufficient time coverage'})
                continue
            vals = window[METRICS]
            valid = vals.count() >= max(2, .8*expected)
            means = vals.mean().where(valid)
            stds = vals.std(ddof=0).where(valid)
            row = {'case_id':case.name,'root_service':service,'injected_fault':fault,
                   'run':run,'split':split,'window_start_s':lo,'phase':phase,
                   'label':'normal' if phase=='nominal' else fault}
            for m in METRICS:
                row[f'{m}__mean'] = means[m]
                row[f'{m}__std'] = stds[m]
            if all(pd.isna(row[f]) for f in FEATURES):
                rejected.append({'case_id':case.name,'window_start_s':lo,'reason':'no usable metrics'})
                continue
            case_rows.append(row)
        # Equal windows across all included cases. Fail closed rather than silently skew class balance.
        if len(case_rows) == len(WINDOWS):
            rows.extend(case_rows)
        cases.append({'case_id':case.name,'root_service':service,'injected_fault':fault,
                      'run':run,'split':split,'cadence_seconds':cadence,
                      'valid_windows':len(case_rows)})
    manifest = pd.DataFrame(cases)
    manifest.to_csv(OUT/'case_splits.csv',index=False)
    pd.DataFrame(rejected, columns=['case_id','window_start_s','reason']).to_csv(OUT/'rejected_windows.csv',index=False)
    if manifest.groupby(['root_service','injected_fault']).size().shape[0] != 25 or not (manifest.groupby(['root_service','injected_fault']).size()==5).all():
        raise ValueError('Expected 5 repetitions per service/fault combination')
    if not (manifest.valid_windows==len(WINDOWS)).all():
        raise ValueError('Some cases lack usable windows. See rejected_windows.csv and case_splits.csv; adjust shared window protocol before training.')
    features = pd.DataFrame(rows)
    assert features.groupby('case_id').split.nunique().max()==1
    features.to_csv(OUT/'rcaeval_features.csv',index=False)
    (OUT/'feature_columns.json').write_text(json.dumps(FEATURES,indent=2),encoding='utf-8')
    print('\nCases per split:\n',manifest.groupby('split').size().to_string())
    print('\nWindows per split and label:\n',pd.crosstab(features.split,features.label).to_string())
    print('\nTotal windows:',len(features),'| Model features:',len(FEATURES))
    print('Missing feature values before imputation:',int(features[FEATURES].isna().sum().sum()))
    print('Saved:',OUT/'rcaeval_features.csv')
    print('NOTE: normal = nominal pre-injection; post-injection labels = experimental conditions, not independently verified instantaneous faults.')

if __name__=='__main__':
    build()
