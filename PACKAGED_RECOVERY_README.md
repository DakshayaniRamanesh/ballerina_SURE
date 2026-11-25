# SURE reusable Python recovery core

This is an additive extraction of the frozen V4.1 runtime logic. It does not change the original experiment scripts, model artifacts, protocol, or results. Do not refreeze or rerun the independent experiment merely to install it.

## Modules
- `sure/recovery_core.py`: time-aware transition kernel, belief updating, greedy/lookahead planning, safety admission, conventional policies and execution loop. No experiment or ledger imports.
- `sure/recovery_estimators.py`: SHA-256 checked B3/B4.1 loading; RF probability adapter with exact three-feature schema (`error`, `keyed`, `initial_timeout_ms`) and explicit class reordering.
- `sure/recovery_http.py`: request-bound HTTP transport for the SURE fixture. Not a general production transport.
- `tests/test_recovery_core.py`: deterministic regression parity against frozen experiment implementation, RF interface, input and confirmation checks.
- `bayesian/check_packaged_recovery.py`: 28-episode live diagnostic using existing extended testbed, ledger audit after termination. Output: `bayesian/outputs/packaged_smoke` (does not overwrite).

## Verify (project root, existing environment)

```powershell
python -m unittest discover -s tests -p "test_recovery_core.py" -v
```

Then start `bal run ballerina_service/extended_service.bal` in another terminal (port 9091), and run:

```powershell
python -m bayesian.check_packaged_recovery
```

The parity tests check frozen source and model hashes and compare complete traces using identical scripted observations and a deterministic clock. Local author-side tests passed using a reconstructed transition table from the original training partition and a stub initial estimator; original binary artifacts and live Ballerina must be checked on the project machine.

## API example (fixture)

```python
from sure.recovery_core import recover
from sure.recovery_estimators import load_frozen
from sure.recovery_http import PaymentTransport

bayes, kernel = load_frozen()
# request_id must already be registered with the fixture; keyed must match
# its verified shared deduplication contract.
transport = PaymentTransport('http://127.0.0.1:9091', request_id, keyed=True)
result = recover(transport, kernel, bayes, 'v4', keyed=True,
                 cap=150, deadline=750, budget=3)
```

The core owns initial submission and recovery; do not pass an already executed operation as a new one. The caller must bind transport to the same verified deduplication contract and request identifier. All time values are milliseconds. The initial submission is not charged to the recovery-call budget. `client_confirmed_on_time` is a client-side outcome; correct execution is separately verified by the experimental ledger. HTTP timeouts/admission are not hard wall-clock guarantees.

## RF integration

Use `RandomForestAdapter(fitted_model)` for V3/V5 and the frozen Bayesian estimator for V2/V4. Fit the RF on the shared training partition only, use the fixed three feature columns in the listed order, validate probability calibration, and return all three states. The adapter reorders classifier classes into `(unstarted, processing, completed)`. It does not train or calibrate RF or verify split provenance. No RF performance claims are made here.

This package remains a Python prototype, not a native Ballerina module. Live integration smoke is not a new independent performance evaluation.
