# SURE — Bayesian Recovery Component

**Status:** Bayesian research prototype implemented, independently evaluated, packaged and regression/live integration tested. RF model integration, factorial evaluation and native Ballerina library are pending.

## Purpose
SURE (Semantics-Constrained Uncertainty-Aware Recovery Engine) investigates distributed request recovery when a client cannot determine whether an operation has started, is processing, or has completed. A timeout does not prove non-execution. Recovery must respect operation semantics, an end-to-end deadline and a limited recovery-call budget.

The runtime combines an initial execution-state probability estimator (B3), a time-aware joint transition/observation/duration model (B4.1b), sequential belief updates and greedy or bounded lookahead planning. Supported actions are wait, reconcile authoritative status, retry, failover and stop. Retry/failover for ambiguous unkeyed requests are blocked; keyed operations require verified shared deduplication. Probabilities cannot override semantic restrictions. Unexpected observations fall back to the predicted next-state distribution rather than being treated as execution confirmation.

## Development history
| Stage | Work completed |
|---|---|
| B1 | RCAEval audit, common CPU/memory features, 750 observation windows across 125 cases, Gaussian naive Bayes |
| B1.1 | Prior/smoothing selection and temperature calibration |
| B2 | Causal temporal filtering and nominal/fault thresholding; supporting telemetry study, not part of live recovery |
| B3 | Balanced Ballerina collection (1,008 episodes, 864 recovery rows, 810 validated after boundary screening); Dirichlet-smoothed conditional probabilities using error, keyed and initial timeout |
| B4 | Joint state/action/observation/duration model, initially single-step |
| B4.1 | Multi-step pilot (189 episodes, 481 usable steps), expanded collection (756 episodes, 1,935 usable steps), hierarchical backoff, elapsed time and previous action/observation context |
| V2/V4 | Greedy and depth-two lookahead, semantic restrictions and resource-aware utility |
| Development experiments | Initial 56-episode pilot; improved 98-episode pilot; 714-episode sensitivity analysis across 42 scenarios and 17 configurations |
| Independent experiments | Extended Ballerina fixture, 140-episode smoke, frozen protocol, reserved-test scores, 5,040 independent live episodes |
| Packaging | Reusable Python core, Bayesian/RF estimator interface, HTTP fixture transport, regression tests and live integration smoke |

B3 estimates a three-state distribution using conditional counts with additive smoothing (alpha=1). B4.1b predicts a joint distribution over next execution state, observation and duration; the belief update conditions on the observed response, while actual action duration advances elapsed time. The planner evaluates expected confirmed-on-time reward minus call and time costs under hard action-admission rules.

## Frozen configuration
| Parameter | Value |
|---|---|
| Estimator | B3 |
| Transition | B4.1b (time + history) |
| Greedy / lookahead depth | 1 / 2 |
| Time / call cost | 0.10 / 0.025 |
| Wait / HTTP action cap / admission margin | 75 / 180 / 25 ms |
| Maximum recovery steps | 4 |
| Independent scenario call budget | 1–4 |
| Independent scenario deadline | 350, 750, 1,250 ms |
| Initial timeout | 100, 150, 200 ms |

Frozen model SHA-256:
- B3: `348174d885c1cf679687eff034c6942394651eef30f7b6466d2a0446ec18d3d8`
- B4.1: `c71c371befcbfff09c751d8bdee65382b028cb594475f38006f517306367a3ad`

Do not overwrite original models, frozen source files, designs or outputs when extending the project.

## Reserved-test prediction performance
| Metric | B2 threshold | B3 | B4.1b replay |
|---|---:|---:|---:|
| Samples | 150 windows / 25 cases | 204 requests | 487 steps |
| Accuracy | 90.67% | 68.63% | 81.72% next state |
| Macro F1 | 87.57% | 48.90% | — |
| Balanced accuracy | 87.11% | 51.11% | — |
| Log loss | 0.8921 | 0.7946 | 1.0740 joint |
| Brier | 0.3868 | 0.4559 | 0.2777 next state |

B2 threshold nominal false-alarm rate is 4%, fault recall 94.67%; corresponding static threshold accuracy is 68% and false-alarm rate 46.67%. B2 was reconstructed from training data only with validation reproduction before reserved scoring. B3 global prior accuracy is 52.94%; B3 recall is 53.33% unstarted, 0% processing, 100% completed. B4.1b observation accuracy is 83.37%, duration MAE 32.36 ms and confirmation Brier 0.0570. Its time-independent matched reference next-state Brier is 0.3020 (about 8% relative reduction). These are different prediction tasks and their scores must not be combined as one system accuracy. B4.1 replay uses recorded actions, not counterfactual policy evaluation; reserved records were previously descriptively audited.

## Independent recovery evaluation
504 matched configurations (7 fault families × 2 deduplication settings × 3 initial timeouts × 3 deadlines × 4 call budgets) × 10 policies = 5,040 request episodes. Fault families: normal, slow, response delay, drop, endpoint, status, combined. Seeded fault schedules and randomized policy order; separate runs can have different actual timings. Single-process simulated testbed, not production deployment.

| Policy | On-time confirmed / 504 | Rate |
|---|---:|---:|
| V4 time + history (selected) | 232 | 46.03% |
| V2 greedy | 228 | 45.24% |
| V4 no time | 235 | 46.63% |
| V4 time only | 235 | 46.63% |
| Failover backoff | 230 | 45.63% |
| Hybrid backoff | 209 | 41.47% |
| Retry backoff | 178 | 35.32% |
| Safe retry | 177 | 35.12% |
| Reconcile backoff | 148 | 29.37% |
| Fixed reconcile | 142 | 28.17% |

Among initially unconfirmed requests V4 recovered 169/441 (38.32%), V2 163/439 (37.13%) and failover backoff 164/438 (37.44%). Initial outcomes vary slightly between policies.

### Paired results (V4 minus comparator)
| Comparator | Success difference (percentage points) | Exact McNemar p | Mean latency saved (ms) in commonly recovered subset |
|---|---:|---:|---:|
| Failover backoff | +0.40 | 0.7539 | 27.99 [15.42, 40.15], n=160 |
| V2 greedy | +0.79 | 0.3877 | −37.33 [−49.43, −25.87], n=159 |
| Hybrid backoff | +4.56 | 0.000117 | −0.46 [−11.68, 10.20], n=137 |
| V4 no time | −0.60 | 0.4531 | 14.46 [4.66, 24.64], n=162 |
| V4 time only | −0.60 | 0.3750 | 11.04 [3.95, 19.07], n=166 |

Brackets are reported 95% paired bootstrap intervals. Success difference CI versus failover is −0.79 to +1.59 percentage points; versus hybrid is +2.58 to +6.75. Conditional latency reduction vs failover is 9.83% (256.59 vs 284.58 ms), not an unconditional speed or equivalence claim. Lookahead costs more decision time and is slower than greedy on their common recoveries. The hybrid comparison is positive but does not isolate the contribution of lookahead. Secondary comparisons and subgroup findings are exploratory.

Zero recorded duplicate effects and deadline overruns across all policies in this experiment do not constitute a formal safety proof. V4 incurred 57 unexpected observation updates. The selected V4 configuration is retained; no post-test tuning or replacement based on these results.

### Limitations
- B3 cannot identify processing as its top class on the reserved partition; client observations are often ambiguous.
- Fixed waits/step limits and training distribution restrict performance at long execution delays and under combined faults.
- Conditional latency comparisons select requests successfully recovered by both methods; cannot infer population-wide latency benefits.
- Confirmed completion is distinct from execution effects and from estimator accuracy. Unconfirmed operations may already have executed.
- Synthetic faults, finite sample, shared process and deduplication ledger; no real network fault injection or distributed deployment claim.
- No formal safety proof, hard wall-clock guarantee or established superiority of lookahead over greedy/failover in success.

## Key files
- `ballerina_service/service.bal`, `ballerina_client/client.bal`: original fixture and client.
- `ballerina_service/extended_service.bal`: extended fixture (9091).
- `sure/policy.py`, `sure/experiment.py`, `sure/independent_recovery.py`: original experimental implementation.
- `bayesian/audit_dataset.py`, `prepare_rcaeval.py`, `train_bayesian.py`, `tune_bayesian.py`, `temporal_bayesian.py`: supporting fault estimation (under `bayesian/`).
- `bayesian/collect_balanced.py`, `train_execution_bayes.py`, `train_action_kernel.py`, `collect_multistep.py`, `collect_multistep_v2.py`, `train_time_aware_kernel.py`: dataset/model pipeline (under `bayesian/`).
- `bayesian/evaluate_bayesian_planner.py`, `evaluate_time_aware_planner.py`, `evaluate_planner_sensitivity.py`, `independent_evaluation.py`, `score_frozen_holdouts.py`: original evaluation scripts (under `bayesian/`).
- `sure/recovery_core.py`, `sure/recovery_estimators.py`, `sure/recovery_http.py`: reusable engine and transport.
- `tests/test_frozen_recovery.py`, `tests/test_recovery_core.py`, `bayesian/check_packaged_recovery.py`: tests.
- `bayesian/outputs/b3/`, `bayesian/outputs/b41/`: trained artifacts and splits.
- `bayesian/outputs/planner_sensitivity/`: development results.
- `bayesian/outputs/independent_v1/`: frozen protocol, design, holdouts and `live/` results.
- `bayesian/outputs/packaged_smoke/`: integration traces (not independent performance evidence).
- `PACKAGED_RECOVERY_README.md`, `INDEPENDENT_README.txt`, `RF_HANDOVER.md`: usage/reproducibility guides.

## Verification and usage
From the Windows project root, activate the existing environment:
```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -p "test_frozen_recovery.py" -v
python -m unittest discover -s tests -p "test_recovery_core.py" -v
```
Eight frozen tests and six packaged tests reported passed on the project machine; the latter includes 1,500 scripted parity comparisons. Live packaged smoke reported 28 episodes, 7/14 completions for each planner, with its integration checks passed. This is not a new performance comparison.

For fixture use, start `bal run ballerina_service/extended_service.bal` in a separate terminal. See `PACKAGED_RECOVERY_README.md` for API details. Existing smoke, freeze, holdout and independent output directories are intentionally protected against overwrite; do not rerun these commands as routine installation tests. The original independent run used `python -m bayesian.independent_evaluation freeze`, `python -m bayesian.score_frozen_holdouts`, `python -m bayesian.independent_evaluation run`, and `python -m bayesian.independent_evaluation analyze` after fixture verification and smoke checks.

## Remaining project work
- Export frozen RF development data using `python -m bayesian.export_rf_handover`.
- Obtain RF training script, frozen model, calibration and provenance; score the shared reserved partition after freeze.
- Run V2/V3/V4/V5 factorial comparison and conventional baselines on a fresh common design.
- Implement native Ballerina recovery library, further safety and integration testing, research publication.

Future adaptive wait/depth or richer observation models are new research versions that require new training and evaluation data. They must not replace the reported frozen V4 results.
