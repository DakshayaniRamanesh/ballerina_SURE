"""Adapters for the frozen Bayesian estimator and a separately trained RF."""
import hashlib
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .recovery_core import TimeKernel, STATES, FEATURES, normalized

B3_SHA = '348174d885c1cf679687eff034c6942394651eef30f7b6466d2a0446ec18d3d8'
KERNEL_SHA = 'c71c371befcbfff09c751d8bdee65382b028cb594475f38006f517306367a3ad'
RF_FEATURES = ('error', 'keyed', 'initial_timeout_ms')

def load_frozen(b3='bayesian/outputs/b3/bayesian_b3.joblib',
                kernel='bayesian/outputs/b41/kernel_tables.json'):
    """Only load trusted local artifacts. Check exact hashes before deserialization."""
    for path, expected in ((b3,B3_SHA),(kernel,KERNEL_SHA)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen hash mismatch: {path}')
    saved=joblib.load(b3)
    if tuple(saved['states'])!=tuple(STATES) or tuple(saved['features'])!=tuple(FEATURES):
        raise ValueError('Incompatible B3 schema')
    transition=TimeKernel(json.loads(Path(kernel).read_text(encoding='utf-8')))
    if transition.level!=2:
        raise ValueError('Wrong selected transition level')
    return saved['bayes'],transition

class RandomForestAdapter:
    """Accept a fitted sklearn classifier/pipeline using the three B3 inputs.
    Training, calibration and split provenance must be verified separately.
    """
    def __init__(self, model):
        self.model=model
        self.classes=tuple(model.classes_)
        if len(self.classes)!=3 or set(self.classes)!=set(STATES):
            raise ValueError('RF must contain all three execution states')
        if hasattr(model,'feature_names_in_') and tuple(model.feature_names_in_)!=RF_FEATURES:
            raise ValueError('RF input columns/order do not match primary comparison')

    def predict(self, f):
        x=pd.DataFrame([{k:f[k] for k in RF_FEATURES}],columns=RF_FEATURES)
        p=np.asarray(self.model.predict_proba(x),dtype=float)
        if p.shape!=(1,3) or not np.isclose(p.sum(),1.,atol=1e-8,rtol=0):
            raise ValueError('Invalid RF probabilities')
        return normalized([p[0,self.classes.index(s)] for s in STATES])
