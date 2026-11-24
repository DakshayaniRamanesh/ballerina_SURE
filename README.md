# SURE — research starter prototype

Semantics-Constrained Uncertainty-Aware Recovery Engine. One shared client-side policy framework; interchangeable initial estimators and planners.

**Status:** runnable Python research prototype and draft Ballerina service/client examples. This is not a complete Ballerina library, production fault-tolerance system, or validated novel algorithm. The Python reference server and policy tests can be run without Ballerina. The Ballerina source requires compilation and integration testing with your installed distribution. Do not use the synthetic payment service for real transactions.

## 1. Install

Install Python 3.10+ (3.11 or 3.12 recommended), Git, VS Code, and Ballerina Swan Lake from https://ballerina.io/downloads/. Open this folder in VS Code. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
bal version
python -m unittest discover -s tests -v
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in every command. On Linux/macOS use `source .venv/bin/activate`.

## 2. Start the test service (terminal 1)

Preferred for the research target:

```powershell
bal run ballerina_service/service.bal
```

Or run the Python reference service, which implements the same synthetic contract and is useful for checking the experiment code independently:

```powershell
python -m sure.reference_service
```

Run only ONE of the two services at a time; both listen on port 9090. Keep that terminal open. The Ballerina service is a source example; it was not compiled in the preparation environment. If it has a compilation or runtime issue, use the Python server to unblock model development and report the Ballerina error separately.

## 3. Understand the problem first (terminal 2)

The Ballerina client example configures a synthetic operation to commit but delay the response, then uses fixed reconciliation and prints the independent ledger:

```powershell
bal run ballerina_client/client.bal
```

Expected concept: `status: completed` and `effects: 1`; the original response exceeds the first attempt timeout, but the status query confirms the commit. The client example is a small wrapper, not the V3–V5 implementation. It uses wall-clock time for illustration; the Python harness uses a monotonic local clock for the deadline.

You can also inspect the service with PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:9090/health
```

## 4. Collect a dataset (terminal 2)

```powershell
python -m sure.experiment collect --n 120
```

120 independent initial requests is only a pipeline smoke-test, not a sufficient dataset for a paper. The code samples synthetic scenarios and performs a random admissible recovery action after an ambiguous initial result. Only failed/ambiguous initial requests become rows. It waits for possible late commits before retrieving the privileged ledger. The initial hidden state is reconstructed at the recovery decision point using recorded start/commit timestamps. The same host clock is required for this local fixture. Timestamp granularity and races around state transitions need further validation.

Output: `data/recovery.csv`. Increase sample count, diversify workloads, and improve exploration coverage later. Example: `--n 2000`. `--seed` controls the sampled scenarios and exploration, not OS scheduling or network jitter.

Features: elapsed_ms, error, keyed, initial_timeout_ms. `mode`, ledger fields, and hidden labels are never model inputs. The `state` column is a supervised training target. `action`, `observation`, `duration_ms` and `next_state` train the shared empirical transition/observation model. The initial fault mode is intentionally hidden. Some timeout scenarios are observationally indistinguishable with these limited features; do not expect RF to infer information the client does not have. Add client-observable history, health probes, and load signals later.

## 5. Train both estimators

```powershell
python -m sure.experiment train
```

Output: `models/sure.joblib`. The script prints held-out multiclass log loss and Brier score; lower is better. Random Forest uses 100 trees with sigmoid calibration when the training classes have enough samples for 3-fold calibration. The empirical Bayesian estimator uses smoothed class frequencies conditioned on a small feature bucket. The same learned transition/observation/duration kernel is used for Bayesian and RF policies; after each recovery observation, both update their belief via that kernel. The joblib model contains Python objects; load only models you created and trust.

## 6. Compare all recovery policies

```powershell
python -m sure.experiment evaluate --n 10
```

`--n` is the number of paired scenarios, each repeated for six policies (fresh IDs, same fault mode, contract, initial timeout and deadline); policy order is randomized. The default evaluation seed differs from training. Output: `results/runs.csv` and `results/summary.csv`. This is a small functional benchmark; not a statistically powered experiment. For a paper, use independent workloads/fault schedules, repeated runs, confidence intervals, sensitivity tests, and held-out fault regimes. The script measures only local, sequential requests.

| Policy | Initial belief | Decision |
|---|---|---|
| safe_retry | none | Retry only for deduplicated operations |
| fixed_reconcile | none | Always query status after an ambiguous result until confirmed or budget exhausted |
| v2 | empirical Bayes | depth 1 |
| v3 | calibrated RF (if feasible) | depth 1 |
| v4 | empirical Bayes | depth 2 |
| v5 | calibrated RF (if feasible) | depth 2 |

All use the same safe action filter and three additional HTTP call budget, maximum four recovery actions, and deadline. A wait does not consume a network call. The initial request counts against the deadline, not the additional-call budget. Policies stop on confirmation or unresolved outcome. The planner maximizes predicted probability of confirmed completion before deadline minus 0.025 per extra HTTP call. Depth-2 branches on possible observations and action durations. It is a bounded belief-space search, not an exact POMDP solver. Duration bins and transition assumptions are coarse; the kernel is not yet conditioned on load, remaining timeout, or time already spent processing. Missing transition groups conservatively predict no confirmation. This limitation is important for the next iteration.

The primary metric `confirmed_on_time` requires client confirmation before the deadline and exactly one actual effect after allowing the fixture's late commit. Secondary metrics: duplicate effects, unresolved outcomes, deadline violations, P95/P99, recovery calls, planner time. `requests` timeouts are not a strict total wall-clock guarantee under arbitrary network behavior: the harness uses a monotonic deadline, caps each call, and separately records overruns. Do not claim hard real-time enforcement.

## 7. File map

```
ballerina_service/service.bal  Ballerina synthetic payment fixture
ballerina_client/client.bal    Ballerina first-milestone wrapper
sure/reference_service.py    runnable Python reference fixture
sure/policy.py               safety, Bayes, transition model, lookahead
sure/experiment.py           collect, train, evaluate commands
tests/test_policy.py         core unit tests
```

## 8. Safety and limitations

- A stable logical operation ID refers to a single synthetic payment. For `keyed=true`, both logical replica routes share one in-memory ledger and atomically check/reserve the operation under a lock before applying an effect. This is a **process-local** deduplication demonstration, not a durable exactly-once guarantee. Production needs atomic durable effect/result recording, key scoping, payload binding, retention, authorization and replica consistency.
- For `keyed=false`, the filter forbids retry and failover after the initial ambiguous call. Status `unknown` does not prove the operation cannot commit; no probability can override this rule. This MVP does not support a definitive no-commit/fencing protocol.
- `normal`, `lost`, `drop`, `slow`, `status_slow`, `status_down`, `primary_down` are service-injected synthetic behaviours. A delayed response emulates a timeout, not a genuine transport packet loss. Primary and backup are paths in the same process, not independent replicas. No Docker, Toxiproxy, load generator, or external network fault injection has been integrated yet.
- The admin ledger is evaluation-only. Never feed it into the policy. This fixture binds to localhost and has no authentication. Do not expose it publicly.
- RF is used only for initial-state prediction. It cannot identify the best action alone; the shared kernel and planner evaluate actions. Stronger estimation requires labelled multi-step trajectories, probability calibration on representative held-out conditions, and richer available telemetry.
- V2–V5 currently execute in Python using HTTP, not in the Ballerina client. The next integration milestone is a versioned predictor/planner interface and Ballerina wrapper with shared safety, monotonic budget enforcement, and tested timeout handling; either port inference to Ballerina or measure the overhead of a Python planner service.
- Do not conclude novelty or superiority from these small experiments. Include strong adaptive safe retry/endpoint selection, tuned fixed reconciliation, and an oracle (upper bound only) in the eventual research benchmark.

## 9. Next work by team

1. Reproduce the lost-response scenario in Ballerina and confirm the ledger trace. Validate compilation and API parity with the Python reference service.
2. Replace in-memory ledger with durable transactional storage; run true independent replicas and a fault proxy. Add late-commit/fencing tests.
3. Collect richer multi-step trajectories using only client-visible predictors; separate training, calibration and unseen-regime test sets. Compare calibrated RF against Bayesian estimator.
4. Improve the transition model and planning algorithm; compare depth 1/2/3 under strict time budgets. Port or integrate in Ballerina. Establish novelty from the closest existing literature in parallel.

Official references: https://ballerina.io/learn/by-example/http-basic-rest-service/ ; https://ballerina.io/learn/by-example/http-timeout/ ; https://ballerina.io/spec/http/ ; https://scikit-learn.org/stable/modules/calibration.html
