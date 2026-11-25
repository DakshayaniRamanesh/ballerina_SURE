# SURE — Random Forest integration handover

## Goal
Compare initial execution-state probability estimators while holding the transition kernel, belief update, safety, transport, resource restrictions and planner implementation constant.

| Version | Initial estimator | Planning depth |
|---|---|---:|
| V2 | Frozen B3 Bayesian | 1 |
| V3 | Random Forest | 1 |
| V4 | Frozen B3 Bayesian | 2 |
| V5 | Random Forest | 2 |

V5 is not an ensemble of RF and Bayesian initial estimators. It uses RF for the initial belief, followed by the shared Bayesian transition/update mechanism.

## Export development data
From the project root, with the existing virtual environment activated:

```powershell
python -m bayesian.export_rf_handover
Compress-Archive -Path bayesian/outputs/rf_handover_v1/* -DestinationPath RF_development_data.zip
```

Send `RF_development_data.zip` and this guide to the RF developer. The export checks the original frozen input hashes, preserves the episode split and excludes test labels and privileged ground truth other than the training/validation target. Do not regenerate the splits or edit the source CSVs to bypass a hash error.

## Training contract
- Training: 405 labelled rows. Validation: 201 labelled rows. Reserved test: 204 rows, retained by the Bayesian developer for final scoring.
- Inputs, in exact order: `error` (categorical string), `keyed` (boolean), `initial_timeout_ms` (integer).
- Target: `state`, with string labels `unstarted`, `processing`, `completed`.
- `episode` is metadata only; do not feed it to the estimator. Do not use mode, action, observation, future state, elapsed time, or ledger fields in the primary comparison.
- Fit preprocessing and RF on training data only; use training-only cross-validation for model selection/calibration where feasible and validation for development assessment. Do not fit on the reserved test set. Record the calibration approach and seeds; if using validation for calibration, do not treat that same validation score as independent.
- Save a fitted scikit-learn pipeline (including categorical preprocessing) with `predict_proba`, `classes_` and preferably `feature_names_in_`. Keep all three string classes. Provide the fitted artifact as `rf_execution.joblib`, the reproducible training script, package versions, hyperparameters, input hashes, validation predictions and metrics.
- Only deserialize artifacts from trusted collaborators. Match the original environment's Python/scikit-learn/joblib versions.

## Evaluation metrics
Report accuracy, balanced accuracy, macro F1, per-state recall/confusion matrix, multiclass log loss and multiclass Brier score (mean sum of squared errors across all three classes; not divided by 3). Use the full probability vector, not just the hard prediction. The Bayesian reserved test reference is 68.63% accuracy, 51.11% balanced accuracy, 48.90% macro F1, 0.7946 log loss and 0.4559 Brier score. Do not tune RF against these reserved-test results. The RF model must be frozen before the same 204 rows are scored.

## Runtime interface
```python
import joblib
from sure.recovery_estimators import load_frozen, RandomForestAdapter
from sure.recovery_core import recover
from sure.recovery_http import PaymentTransport

bayes, kernel = load_frozen()
rf = RandomForestAdapter(joblib.load('rf_execution.joblib'))
features = {'error': 'timeout', 'keyed': True, 'initial_timeout_ms': 150}
print(rf.predict(features))  # [P(unstarted), P(processing), P(completed)]
```

For live integration, register a fresh case in the fixture before constructing `PaymentTransport` and invoking `recover`. Use `policy='v3'` or `policy='v5'` with the RF adapter; use `v2` or `v4` with `bayes`. Never pass an already executed request to the engine as a new one. The transport must implement verified shared deduplication if keyed=True.

The adapter only enforces the probability/class interface; it does not validate provenance or train/calibrate RF. Its existence does not mean RF integration has already passed.

## Next shared experiment
1. Receive and freeze RF artifact and training metadata.
2. Verify adapter and reserved-test scoring; do not adjust RF based on test results.
3. Run V2/V3/V4/V5 integration smoke against matching configured conditions and check independent execution ledger.
4. Predefine a fresh matched factorial design and primary comparisons before evaluation. Include strong conventional baselines and shared resource/safety restrictions. Reuse the frozen transition model without refitting for one estimator.
5. Report initial-estimator effect, planner-depth effect and their interaction, with paired estimates and uncertainty. Keep prior independent Bayesian results separate. A classifier accuracy improvement need not translate into higher confirmed completion.
6. Complete native Ballerina implementation and testing separately. Current engine is Python with a Ballerina test fixture.
