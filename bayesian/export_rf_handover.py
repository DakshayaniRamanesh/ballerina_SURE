"""Export development-only RF data from the frozen B3 split. No model fitting."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path('bayesian/outputs/b3')
OUT = Path('bayesian/outputs/rf_handover_v1')
EXPECTED = {
    'recovery_validated.csv': '2876545d2b056a32f2b02c5347375ba3954ed30747499162a548d6051c1471d7',
    'episode_splits.csv': 'e34426aca80a7d3189e2f7c47304cabb608d59f50b68b8a642ad58c43afc2569',
}
FEATURES = ['error', 'keyed', 'initial_timeout_ms']
STATES = ['unstarted', 'processing', 'completed']
COUNTS = {'train': 405, 'validation': 201, 'test': 204}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def prepare(data, splits):
    require(data['episode'].notna().all() and data['episode'].is_unique, 'Invalid data episode IDs')
    require(splits['episode'].notna().all() and splits['episode'].is_unique, 'Invalid split episode IDs')
    cols = ['episode', *FEATURES, 'state']
    joined = data[cols].merge(splits[['episode', 'split']], on='episode', how='left', validate='one_to_one')
    require(not joined.isna().any().any(), 'Missing values or split assignments')
    require(len(joined) == 810 and joined['split'].value_counts().to_dict() == COUNTS, 'Unexpected split counts')
    require(set(joined['state']) == set(STATES), 'Unexpected state labels')
    require(pd.api.types.is_bool_dtype(joined['keyed']), 'keyed must contain boolean values')
    require(set(joined['error']) <= {'timeout', 'error'}, 'Unexpected error labels')
    require(set(joined['initial_timeout_ms']) == {100, 150, 200}, 'Unexpected timeout values')
    for part in COUNTS:
        require(set(joined.loc[joined['split'] == part, 'state']) == set(STATES), 'Missing class in ' + part)
    return {part: joined.loc[joined['split'] == part, cols].copy() for part in ('train', 'validation')}

def main():
    for name, expected in EXPECTED.items():
        require(sha(ROOT / name) == expected, 'Frozen input hash mismatch: ' + name)
    files = prepare(pd.read_csv(ROOT / 'recovery_validated.csv'), pd.read_csv(ROOT / 'episode_splits.csv'))
    OUT.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for part, frame in files.items():
        path = OUT / f'{part}.csv'
        frame.to_csv(path, index=False)
        outputs[path.name] = {'rows': len(frame), 'sha256': sha(path)}
    manifest = {'schema_version': 1, 'features_in_order': FEATURES, 'target': 'state',
                'metadata_not_a_feature': ['episode'], 'states_in_engine_order': STATES,
                'source_sha256': EXPECTED, 'exports': outputs,
                'reserved_test_rows_not_exported': 204,
                'instructions': 'Train on train.csv only; select and calibrate using development data only. Freeze model before reserved test scoring.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('RF handover created:', OUT)
    print('Training: 405 | Validation: 201 | Reserved test: 204 (not exported)')

if __name__ == '__main__':
    main()
