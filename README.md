# SURE — Semantics-Constrained Uncertainty-Aware Recovery Engine

## Project Overview

SURE is a research-oriented adaptive fault-recovery framework for distributed systems, with Ballerina as the target integration environment.

The project investigates how probabilistic execution-state estimation, sequential belief updating, semantic safety constraints and bounded lookahead planning can improve recovery decisions when the outcome of a distributed request is uncertain.

Unlike conventional recovery mechanisms that rely on predefined retry and backoff strategies, SURE dynamically evaluates possible recovery actions based on observed failures, estimated execution states, elapsed time, remaining deadlines and available recovery resources.

The current implementation consists of a Ballerina experimental service and a Python-based probabilistic recovery engine.

**Current status:** Advanced research prototype — Bayesian implementation and development experiments completed; Random Forest integration and independent evaluation pending.

---

## 1. Problem Statement

In distributed systems, a communication failure does not necessarily indicate that the requested operation failed.

When a client experiences a timeout, the operation may be:

- **Unstarted:** The request has not begun execution.
- **Processing:** The request is currently being executed.
- **Completed:** The operation has completed, but the client has not received confirmation.

Conventional retry mechanisms may introduce duplicate execution, unnecessary waiting, excessive network traffic or unresolved operations.

SURE addresses this problem by maintaining a probabilistic representation of the underlying execution state and selecting appropriate recovery actions while respecting semantic safety and resource constraints.

## 2. Research Objectives

- Estimate hidden execution states using observable client-side information.
- Investigate Bayesian and Random Forest probability estimators.
- Update execution-state beliefs sequentially using recovery observations.
- Model execution-state transitions and recovery-action outcomes.
- Incorporate elapsed time and recovery history into transition modelling.
- Enforce operation-specific semantic safety constraints.
- Optimize recovery decisions under deadline and network-call constraints.
- Compare greedy and bounded lookahead recovery strategies.
- Evaluate recovery effectiveness against conventional recovery baselines.
- Investigate the trade-offs between recovery success, latency, network usage and computational overhead.
- Integrate the resulting recovery framework into Ballerina.

---

## 3. System Architecture

The current recovery workflow is:

1. A Ballerina service receives a request.
2. The client encounters an ambiguous failure or timeout.
3. The execution-state estimator calculates the initial probability distribution.
4. The semantic safety layer determines permissible recovery actions.
5. The transition model predicts possible action outcomes, execution states and durations.
6. The recovery planner evaluates the expected utility of permissible actions.
7. The selected action is executed.
8. The execution-state belief is updated using the observed response.
9. Recovery continues until confirmation, resource exhaustion, deadline termination or an explicit stop decision.

### Core Components

| Component | Function | Status |
|---|---|---|
| Ballerina testbed | Simulates distributed request failures | Implemented |
| Bayesian fault estimator | Estimates telemetry-based fault categories | Implemented |
| Temporal Bayesian filter | Updates fault probabilities over time | Implemented |
| Bayesian execution estimator | Estimates hidden request execution states | Implemented |
| Random Forest estimator | Alternative execution-state probability estimator | Integration pending |
| Transition model | Predicts recovery-action outcomes | Implemented |
| Sequential belief updater | Updates probabilities after recovery observations | Implemented |
| Semantic safety layer | Prevents inadmissible recovery actions | Implemented |
| Greedy planner | Selects actions using immediate expected utility | Implemented |
| Lookahead planner | Evaluates possible future recovery sequences | Implemented |
| Native Ballerina library | Reusable recovery framework | Pending |

---

## 4. Ballerina Experimental Environment

A controlled Ballerina service was implemented to simulate different distributed-system failure conditions.

| Failure Mode | Description |
|---|---|
| `normal` | Successful normal execution |
| `lost` | Operation completes but the response is lost |
| `drop` | Request is dropped before execution |
| `slow` | Operation requires additional processing time |
| `status_slow` | Status service responds slowly |
| `status_down` | Status service is unavailable |
| `primary_down` | Primary execution endpoint is unavailable |

The environment supports operations with and without deduplication protection.

An independent server-side ledger records execution start times, completion times, execution attempts and actual execution effects.

These records provide ground-truth labels for model training and experimental verification. Privileged server-side execution information is not supplied directly to the runtime recovery estimator.

---

## 5. Bayesian Fault Estimation Using RCAEval

### B1 — Gaussian Naive Bayes

The initial Bayesian fault-estimation experiments were conducted using the RCAEval dataset.

The selected dataset contains 125 experimental cases covering:

- CPU faults
- Memory faults
- Network delay
- Network packet loss
- Disk faults
- Nominal operation observations

### Dataset Preparation

The raw telemetry dataset was audited for missing values, inconsistent feature availability, timestamp validity and differences in experimental observation lengths.

Common CPU and memory metrics were extracted and converted into statistical observation windows.

| Dataset Property | Value |
|---|---:|
| Experimental cases | 125 |
| Raw telemetry rows | 262,603 |
| Observation windows | 750 |
| Features per window | 48 |
| Training cases | 75 |
| Validation cases | 25 |
| Reserved test cases | 25 |

Complete experimental cases were assigned to individual partitions to prevent leakage between observation windows from the same experiment.

### B1.1 — Bayesian Model Tuning

Gaussian Naive Bayes was improved through:

- Variance-smoothing parameter tuning.
- Class-prior investigation.
- Probability calibration using temperature scaling.
- Fault-detection threshold selection.
- Evaluation of probability quality and nominal false alarms.

### B2 — Temporal Bayesian Filtering

The original classifier processed observation windows independently, resulting in a high nominal false-alarm rate.

Temporal Bayesian filtering was introduced to incorporate previous fault beliefs when interpreting new telemetry observations.

### Fault-Estimation Validation Results

| Model | Validation Accuracy |
|---|---:|
| Initial Gaussian Naive Bayes | 61.33% |
| Tuned Bayesian model | 65.33% |
| Temporal Bayesian model | 91.33% |
| Temporal model with decision threshold | 93.33% |

The nominal false-alarm rate decreased from 65.33% in the original classifier to 2.67% in the temporal threshold configuration.

These are development validation results. The reserved test partition has not been scored.

### Transition to Request-Level Estimation

Although RCAEval supports system-level fault classification, it does not provide the request-level execution-state labels and recovery-action outcomes required for SURE.

Identifying a system fault does not necessarily establish whether a particular operation has already executed.

Therefore, the RCAEval models remain supporting research components. They are not currently used directly by the live recovery planner.

---

## 6. Request-Level Dataset Generation

A custom experimental dataset was generated using the Ballerina service.

The collection process recorded:

- Client-side error observations.
- Configured initial request timeouts.
- Elapsed execution time.
- Deduplication support.
- Ground-truth execution states.
- Selected recovery actions.
- Observed recovery responses.
- Recovery-action durations.
- Subsequent execution states.
- Independent execution-effect counts.

### Single-Step Collection

| Property | Count |
|---|---:|
| Initial experimental episodes | 1,008 |
| Raw recovery samples | 864 |
| Quarantined recovery samples | 54 |
| Validated recovery samples | 810 |

Timestamp inconsistencies between the Python client and Ballerina server were investigated. A conservative boundary-screening procedure was introduced to quarantine uncertain state labels rather than silently modifying the original dataset.

### Dataset Partitioning

| Partition | Recovery Samples |
|---|---:|
| Training | 405 |
| Validation | 201 |
| Reserved testing | 204 |

The dataset and episode-level split definitions are shared resources for the Bayesian and Random Forest estimator comparison.

---

## 7. B3 — Bayesian Execution-State Estimator

The B3 model estimates the probability of an ambiguous request being unstarted, processing or completed.

The current estimator uses Dirichlet-smoothed conditional frequency estimates.

### Runtime Conditioning Features

- `error`
- `keyed`
- `initial_timeout_ms`

Elapsed time is available as an observable feature but is not used as a conditioning variable in the frozen B3 estimator.

Ground-truth execution states, injected failure modes and future recovery outcomes are excluded from runtime estimator inputs.

The model outputs a probability distribution rather than only a predicted execution-state class.

Example output:

```json
{
  "unstarted": 0.1316,
  "processing": 0.2237,
  "completed": 0.6447
}
```

### Validation Results

| Metric | Global Prior Baseline | B3 Bayesian |
|---|---:|---:|
| Accuracy | 53.73% | 68.66% |
| Balanced accuracy | 33.33% | 50.88% |
| Macro F1 | 23.30% | 48.79% |
| Log loss | 0.9993 | 0.7957 |
| Multiclass Brier score | 0.5989 | 0.4567 |

The conditional Bayesian estimator improved probability prediction over the global prior baseline.

However, the model did not correctly classify processing requests as the most probable state in the validation confusion matrix.

This limitation motivates maintaining probabilistic beliefs and obtaining additional information through recovery actions instead of relying exclusively on hard classifications.

---

## 8. B4 — Single-Step Action Transition Model

The action outcome model estimates the joint probability distribution of:

- The next execution state.
- The recovery observation.
- The recovery-action duration.

The original transition model is conditioned on the current execution state, selected action and deduplication support.

### Supported Recovery Actions

| Action | Description |
|---|---|
| Wait | Allow execution to progress |
| Reconcile | Query authoritative execution status |
| Retry | Resubmit the operation under a verified safety contract |
| Failover | Contact an alternative endpoint under a shared safety contract |
| Stop | Terminate recovery without further actions |

### Validation Results

| Metric | Known-State Diagnostic | B3 Belief Prediction |
|---|---:|---:|
| Next-state accuracy | 99.50% | 70.15% |
| Observation accuracy | 87.56% | 70.15% |
| Joint log loss | 1.2704 | 2.0351 |
| Confirmation Brier score | 0.0449 | 0.0974 |
| Duration MAE | 28.85 ms | 31.51 ms |

The known-state diagnostic uses true execution states for offline assessment only.

The initial model was limited by sparse state-action combinations, fixed action durations and the absence of multi-step training trajectories.

---

## 9. Sequential Bayesian Belief Updating

SURE maintains a probability distribution over the hidden execution states throughout the recovery process.

After each recovery action, the belief distribution is updated using the observed response and the learned state-transition and observation probabilities.

This allows the recovery engine to adapt its decisions as new evidence becomes available.

For example:

1. An initial timeout produces an uncertain execution-state belief.
2. A reconciliation action returns a pending response.
3. The probability of the processing state increases.
4. The planner evaluates whether waiting or another safe recovery action is appropriate.
5. Subsequent observations further update the belief.

Action duration is currently used to advance the elapsed-time context. Fully conditioning runtime belief updates on observed action duration remains a possible extension.

---

## 10. Semantic Safety Constraints

Semantic safety is enforced independently of probabilistic predictions.

The recovery engine must not authorize an unsafe action merely because the predicted probability of successful execution is high.

| Recovery Action | Without Deduplication | With Verified Deduplication |
|---|---|---|
| Wait | Allowed | Allowed |
| Reconcile | Allowed | Allowed |
| Stop | Allowed | Allowed |
| Retry ambiguous operation | Blocked | Conditionally allowed |
| Failover ambiguous operation | Blocked | Conditionally allowed |

Retry and failover require appropriate operation-level execution guarantees. Failover requires deduplication protection across the relevant endpoints.

A timeout, an ordinary not-found response or a high predicted probability of non-execution is not sufficient evidence to authorize an otherwise unsafe retry.

The experimental system additionally applies:

- End-to-end deadline constraints.
- Recovery network-call budgets.
- Maximum recovery-step limits.
- Conservative action-admission rules.

The current implementation is an experimental safety mechanism rather than a formally verified production-grade execution guarantee.

---

## 11. Greedy and Bounded Lookahead Planning

### V2 — Bayesian Greedy Recovery

The greedy planner evaluates the immediate expected utility of each permissible recovery action.

After executing the selected action, it updates the execution belief and repeats the decision process.

### V4 — Bayesian Bounded Lookahead Recovery

The bounded lookahead planner evaluates possible future recovery actions and observations within a limited planning horizon.

This allows the planner to account for the future benefit of waiting, reconciliation and preserving recovery resources.

The planner performs sequential decision-making rather than relying on a fixed recovery-action sequence.

### Recovery Objective

The improved planner maximizes expected utility based on:

- Probability of confirmed on-time completion.
- Recovery network-call cost.
- Time consumed relative to the end-to-end deadline.

Semantic safety and resource constraints are mandatory restrictions rather than optional reward penalties.

---

## 12. Initial Live Recovery Experiments

The initial Bayesian planner was evaluated against conventional recovery policies using the live Ballerina service.

| Policy | Confirmed On-Time Completion |
|---|---:|
| Fixed reconciliation | 35.71% |
| Safe retry | 42.86% |
| Bayesian greedy | 57.14% |
| Bayesian lookahead | 64.29% |

The initial pilot contained 14 matched scenarios and 56 request episodes.

No duplicate execution effects or deadline overruns were recorded.

### Identified Limitations

Detailed action-trace analysis revealed:

- Excessive waiting in the initial lookahead implementation.
- Missing natural execution-completion transitions in the single-step model.
- Insufficient modelling of elapsed time.
- Premature exhaustion of recovery-call budgets.
- Absence of stronger failover-capable conventional baselines.

These observations motivated multi-step data collection and time-aware transition modelling.

---

## 13. Multi-Step Recovery Data Collection

A sequential recovery collector was developed to capture execution-state changes across multiple recovery actions.

Example recovery sequences include:

- Wait → Wait → Wait → Reconcile
- Reconcile → Wait → Reconcile
- Wait → Reconcile → Wait → Reconcile
- Wait → Wait → Wait → Failover
- Failover → Wait → Retry
- Retry → Wait → Reconcile

Retry and failover sequences are restricted by semantic safety requirements.

### Dataset Summary

| Dataset | Experimental Episodes | Usable Recovery Steps |
|---|---:|---:|
| Original single-step collection | 1,008 | 810 |
| Multi-step pilot | 189 | 481 |
| Expanded multi-step collection | 756 | 1,935 |

The expanded collector introduced:

- High-resolution action-duration measurements.
- Sequential action-history recording.
- Chronological trajectory auditing.
- Evidence-based confirmation-boundary resolution.
- Quarantine of uncertain state labels.
- Preservation of observable histories from quarantined steps.

### Natural Completion During Waiting

The expanded dataset contained 177 usable waiting transitions beginning in the processing state.

| Outcome | Count |
|---|---:|
| Processing → Processing | 129 |
| Processing → Completed | 48 |

### Effect of Elapsed Time

| Elapsed Time Before Waiting | Observed Completions | Completion Rate |
|---|---:|---:|
| Below 150 ms | 0/32 | 0% |
| 150–224 ms | 0/79 | 0% |
| 225–274 ms | 15/33 | 45.45% |
| 275–324 ms | 33/33 | 100% |

These findings demonstrate time-dependent execution dynamics within the synthetic experimental environment.

They are not assumed to represent universal completion probabilities for real distributed applications.

---

## 14. B4.1 — Time-Aware Bayesian Transition Model

The transition model was extended to incorporate elapsed time and previous recovery actions and observations.

Three alternative transition configurations were implemented.

| Model | Conditioning Information |
|---|---|
| B4.0 | Execution state, action and deduplication |
| B4.1a | B4.0 information and elapsed time |
| B4.1b | B4.1a information and previous action/observation |

Hierarchical Bayesian backoff is used to handle sparse combinations by combining detailed contextual estimates with more general transition distributions.

### Dataset Partitioning

| Partition | Episodes | Usable Recovery Steps |
|---|---:|---:|
| Training | 378 | 967 |
| Validation | 189 | 481 |
| Reserved testing | 189 | 487 |

Entire trajectories are assigned to individual partitions to prevent episode-level leakage.

### Sequential Validation Using Frozen B3

| Metric | B4.0 | B4.1a | B4.1b |
|---|---:|---:|---:|
| Next-state accuracy | 80.04% | 81.50% | 81.50% |
| Observation accuracy | 81.29% | 83.16% | 83.37% |
| Joint log loss | 1.0547 | 1.0229 | 1.0155 |
| Next-state Brier score | 0.3027 | 0.2856 | 0.2814 |
| Confirmation Brier score | 0.0619 | 0.0587 | 0.0571 |
| Duration MAE | 32.69 ms | 32.09 ms | 31.82 ms |

The inclusion of elapsed time improved probability prediction.

B4.1b was provisionally selected as the primary transition model. B4.1a is retained for ablation analysis because the additional benefit of action history is relatively small.

---

## 15. V4.1 — Improved Time-Aware Recovery Planner

V4.1 extends the original recovery planner with:

- Time-aware Bayesian transition probabilities.
- Previous recovery-action and observation context.
- Explicit time-consumption penalties.
- Improved sequential belief updating.
- High-resolution decision-overhead measurement.
- Conservative deadline-aware action admission.
- Recovery-call and maximum-step budgets.

The explicit time penalty was introduced to reduce unnecessary waiting while preserving waiting actions that provide useful future recovery opportunities.

### Development Pilot Results

The updated planner was evaluated across 14 matched scenarios and seven recovery policies, producing 98 request episodes.

| Policy | Overall Completion |
|---|---:|
| Fixed reconciliation | 35.71% |
| Safe retry | 42.86% |
| Retry backoff | 42.86% |
| Reconciliation backoff | 42.86% |
| Bayesian greedy V2.1 | 64.29% |
| Bayesian lookahead V4.1 | 64.29% |
| Time-independent lookahead | 71.43% |

The pilot demonstrated that improved probability prediction does not necessarily guarantee improved recovery decisions.

Compared with the earlier V4.0 pilot, V4.1 maintained the same observed completion rate while reducing total waiting actions from 14 to 1 and P95 latency from 593.55 ms to 400.84 ms.

These were separate development experiments involving multiple implementation changes.

---

## 16. Conventional Recovery Baselines

The following conventional recovery policies were implemented for comparison:

| Baseline | Description |
|---|---|
| Safe retry | Immediate retry when semantically permissible |
| Fixed reconciliation | Repeated execution-status queries |
| Retry backoff | Retry following a fixed waiting interval |
| Reconciliation backoff | Status querying following fixed waiting intervals |
| Failover backoff | Delayed failover under verified deduplication |
| Hybrid backoff | Deterministic selection of recovery actions using observed responses |

All policies operate under common semantic safety and resource constraints where applicable.

The failover-capable baselines were introduced to avoid unfair comparisons between conventional policies and Bayesian planners with access to additional recovery actions.

---

## 17. Planner Sensitivity Analysis

A controlled development sensitivity experiment was conducted to investigate the effects of planning depth, time-cost weight and transition-model context.

| Experimental Property | Value |
|---|---:|
| Matched scenarios | 42 |
| Recovery configurations | 17 |
| Total request episodes | 714 |
| Recorded duplicate executions | 0 |
| Recorded deadline overruns | 0 |
| Unseen observation updates | 0 |

### Investigated Parameters

- Lookahead depths: 1, 2 and 3.
- Time-cost weights: 0.00, 0.10 and 0.20.
- Transition context: time-independent, time-aware and time-and-history-aware.

### Main Results

| Recovery Policy | Overall Completion |
|---|---:|
| Bayesian lookahead — Depth 2 | 66.67% |
| Bayesian lookahead — Depth 3 | 66.67% |
| Failover backoff | 66.67% |
| Hybrid backoff | 64.29% |
| Bayesian greedy | 61.90% |
| Retry backoff | 47.62% |
| Safe retry | 42.86% |
| Reconciliation backoff | 38.10% |
| Fixed reconciliation | 33.33% |

### Planning-Depth Comparison

The following configurations use a time-cost weight of 0.10 and time-and-history-aware transitions.

| Metric | Depth 1 | Depth 2 | Depth 3 |
|---|---:|---:|---:|
| Overall completion | 61.90% | 66.67% | 66.67% |
| Successful requests | 26 | 28 | 28 |
| Mean latency | 189.90 ms | 200.95 ms | 213.74 ms |
| Average recovery calls | 1.36 | 1.40 | 1.31 |
| Decision overhead | 0.94 ms | 3.31 ms | 10.17 ms |

Depth 2 was provisionally retained as a trade-off between recovery performance and computational overhead.

### Transition Context Ablation

| Configuration | Overall Completion |
|---|---:|
| Time-independent lookahead | 61.90% |
| Time-aware lookahead | 66.67% |
| Time-and-history-aware lookahead | 66.67% |

Elapsed time provided preliminary improvements in recovery outcomes. Previous action history did not independently improve aggregate completion rates.

---

## 18. Detailed Paired Recovery Analysis

Bayesian lookahead and failover backoff succeeded in exactly the same 28 matched scenarios and failed in the remaining 14 scenarios.

After excluding initially successful requests, both methods successfully recovered 22 of 36 initially unsuccessful requests.

Latency and resource usage were compared on the 22 commonly recovered requests.

| Metric | Bayesian Lookahead | Failover Backoff |
|---|---:|---:|
| Recovery success | 61.11% | 61.11% |
| Mean successful-recovery latency | 211.13 ms | 255.35 ms |
| Average waiting actions | 0.09 | 1.00 |
| Average recovery network calls | 1.45 | 1.14 |
| Decision overhead | 3.79 ms | 0.057 ms |

The Bayesian lookahead planner achieved approximately **17.31% lower mean end-to-end latency** among commonly recovered requests and was faster in 17 of the 22 matched cases.

However, it consumed more recovery network calls and computational resources.

The results provide preliminary evidence of adaptive recovery efficiency rather than superior recovery-success rates against the strongest conventional baseline.

### Greedy Versus Lookahead

Lookahead recovered two additional requests compared with greedy.

Detailed trace analysis identified one case where lookahead preserved its final recovery call by waiting before reconciliation. The other difference occurred despite identical action sequences and was associated with timing variation between runs.

This provides preliminary evidence of useful sequential resource allocation, but does not establish statistically significant superiority.

---

## 19. Current Frozen Bayesian Configuration

The following configuration has been provisionally selected for independent evaluation.

| Parameter | Selected Value |
|---|---|
| Initial estimator | B3 Bayesian |
| Transition model | B4.1b |
| Lookahead depth | 2 |
| Time-cost weight | 0.10 |
| Network-call cost | 0.025 |
| Recovery-call budget | 3 |
| Maximum recovery steps | 4 |
| Waiting interval | 75 ms |
| HTTP action timeout | 180 ms |
| Action-admission margin | 25 ms |
| Primary conventional comparator | Failover backoff |

Sensitivity experiment configuration identifier:

```text
d2_t0.10_l2
```

Model SHA-256 hashes are recorded in the sensitivity experiment configuration for reproducibility.

Further hyperparameter tuning should be separated from independent evaluation.

---

## 20. Planned Random Forest Integration

The Random Forest estimator will serve as an alternative initial execution-state probability estimator.

It should use the same validated request-level dataset, episode partitions and observable input features as the Bayesian estimator for the primary controlled comparison.

### Shared Estimator Features

```text
error
keyed
initial_timeout_ms
```

Both estimators must return probabilities for:

```text
unstarted
processing
completed
```

Probability calibration should be evaluated because the recovery planner relies on the full execution-state probability distribution.

### Planned Experimental Configurations

| Version | Initial Estimator | Recovery Planner |
|---|---|---|
| V2 | Bayesian | Greedy |
| V3 | Random Forest | Greedy |
| V4 | Bayesian | Bounded lookahead |
| V5 | Random Forest | Bounded lookahead |

All four configurations will share the transition model, semantic safety layer, execution environment, resource constraints and evaluation procedures.

This factorial comparison is intended to separate the effect of the initial probability estimator from the effect of the recovery-planning strategy.

V5 does not require combining Bayesian and Random Forest into a single classifier.

### Shared Integration Files

```text
bayesian/outputs/b3/recovery_validated.csv
bayesian/outputs/b3/episode_splits.csv
bayesian/outputs/b3/validation_metrics.json
bayesian/outputs/b41/kernel_tables.json
bayesian/outputs/b41/validation_metrics.json
```

---

## 21. Repository Structure

Relevant development files:

```text
SURE_starter/
├── ballerina_service/
│   └── service.bal
├── ballerina_client/
│   └── client.bal
├── sure/
│   ├── __init__.py
│   ├── reference_service.py
│   ├── policy.py
│   └── experiment.py
├── bayesian/
│   ├── audit_dataset.py
│   ├── prepare_rcaeval.py
│   ├── train_bayesian.py
│   ├── tune_bayesian.py
│   ├── temporal_bayesian.py
│   ├── collect_balanced.py
│   ├── train_execution_bayes.py
│   ├── train_action_kernel.py
│   ├── evaluate_bayesian_planner.py
│   ├── collect_multistep.py
│   ├── collect_multistep_v2.py
│   ├── train_time_aware_kernel.py
│   ├── evaluate_time_aware_planner.py
│   ├── evaluate_planner_sensitivity.py
│   └── outputs/
├── data/
├── tests/
│   └── test_policy.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Bayesian Development Scripts

| Script | Purpose |
|---|---|
| `audit_dataset.py` | RCAEval dataset integrity analysis |
| `prepare_rcaeval.py` | Telemetry window preparation |
| `train_bayesian.py` | Initial Gaussian Naive Bayes training |
| `tune_bayesian.py` | Bayesian tuning and probability calibration |
| `temporal_bayesian.py` | Temporal fault-belief filtering |
| `collect_balanced.py` | Balanced request-level data collection |
| `train_execution_bayes.py` | B3 execution-state estimator |
| `train_action_kernel.py` | B4 single-step outcome model |
| `evaluate_bayesian_planner.py` | Initial greedy and lookahead evaluation |
| `collect_multistep.py` | Multi-step pilot collection |
| `collect_multistep_v2.py` | Expanded high-resolution trajectory collection |
| `train_time_aware_kernel.py` | Hierarchical time-aware transition modelling |
| `evaluate_time_aware_planner.py` | V4.1 evaluation and context ablation |
| `evaluate_planner_sensitivity.py` | Planning-depth and reward sensitivity analysis |

---

## 22. Installation and Execution

### Requirements

- Python 3
- Ballerina
- Python virtual environment
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Requests
- Other dependencies specified in `requirements.txt`

### Set Up the Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start the Ballerina Service

Open a separate terminal:

```powershell
bal run ballerina_service/service.bal
```

Check service availability:

```powershell
Invoke-RestMethod -Uri "http://localhost:9090/health"
```

### Generate the Single-Step Recovery Dataset

```powershell
python -m bayesian.collect_balanced --repeats 8
```

### Train the Bayesian Execution Estimator

```powershell
python -m bayesian.train_execution_bayes
```

### Train the Original Action Outcome Model

```powershell
python -m bayesian.train_action_kernel
```

### Run the Initial Recovery Pilot

```powershell
python -m bayesian.evaluate_bayesian_planner --n 14
```

### Generate Multi-Step Pilot Data

```powershell
python -m bayesian.collect_multistep --repeats 1
```

### Generate the Expanded Multi-Step Dataset

```powershell
python -m bayesian.collect_multistep_v2 --repeats 4
```

### Train Time-Aware Transition Models

```powershell
python -m bayesian.train_time_aware_kernel --b3 bayesian/outputs/b3/bayesian_b3.joblib
```

### Evaluate the Improved Planner

```powershell
python -m bayesian.evaluate_time_aware_planner --n 14
```

### Conduct Planner Sensitivity Analysis

```powershell
python -m bayesian.evaluate_planner_sensitivity --n 42
```

Live collection and recovery evaluation require the experimental service to be running. Offline model training does not require the service.

Existing experimental datasets, model artifacts and results should be preserved before rerunning scripts that write to the same output locations.

---

## 23. Experimental Outputs

```text
bayesian/outputs/
├── b3/
│   ├── bayesian_b3.joblib
│   ├── episode_splits.csv
│   ├── recovery_validated.csv
│   ├── training_state_counts.csv
│   ├── validation_metrics.json
│   └── validation_predictions.csv
├── b4/
│   └── action_outcomes.csv
├── b41/
│   ├── episode_splits.csv
│   ├── training_coverage.csv
│   ├── validation_metrics.json
│   ├── kernel_tables.json
│   ├── B4.0_matched_validation.csv
│   ├── B4.1a_time_validation.csv
│   └── B4.1b_time_history_validation.csv
├── planner_pilot/
├── planner_v41_pilot/
└── planner_sensitivity/
    ├── config.json
    ├── design.csv
    ├── runs.csv
    ├── summary.csv
    ├── summary_by_keyed.csv
    ├── summary_by_mode.csv
    └── paired_comparisons.csv
```

### Evaluation Metrics

The framework records:

- Confirmed on-time completion rate.
- Recovery success among initially unsuccessful requests.
- Duplicate execution effects.
- Unresolved request rate.
- Deadline-overrun rate.
- Mean and P95 end-to-end latency.
- Recovery network-call usage.
- Waiting-action count.
- Planner computational overhead.
- Unexpected recovery observations.
- Paired wins, losses and ties.
- Probability-prediction accuracy, log loss and Brier scores.

---

## 24. Current Limitations

The current implementation remains a research prototype.

Identified limitations include:

- Synthetic and relatively predictable execution-delay conditions.
- Limited diversity of service and network environments.
- Sparse transition observations for certain recovery actions.
- Imperfect initial execution-state probability estimates.
- Dependence on the experimental failure distribution.
- Limited evidence for the additional benefit of action-history conditioning.
- Higher computational overhead than conventional recovery policies.
- Increased network-call usage in some Bayesian recovery scenarios.
- Fixed recovery-action durations and HTTP timeout settings.
- Incomplete modelling of observed action duration as Bayesian evidence.
- Conservative action admission rather than guaranteed hard wall-clock deadlines.
- No formal proof of semantic safety.
- Random Forest integration not yet completed.
- Native Ballerina library implementation pending.

The current development experiments do not establish statistically significant superiority, universal semantic safety or generalization to real-world distributed systems.

Reserved dataset partitions and fresh independent experiments are required for final evaluation.

---

## 25. Development Roadmap

### Completed

- [x] Ballerina experimental service and client.
- [x] Synthetic distributed failure simulation.
- [x] Independent execution-ledger verification.
- [x] RCAEval dataset auditing and preprocessing.
- [x] Gaussian Naive Bayes fault classification.
- [x] Bayesian parameter tuning and probability calibration.
- [x] Temporal Bayesian fault filtering.
- [x] Balanced request-level dataset generation.
- [x] Execution-state label validation.
- [x] Bayesian execution-state probability estimation.
- [x] Single-step action outcome modelling.
- [x] Sequential Bayesian belief updating.
- [x] Semantic safety constraints.
- [x] Deadline and recovery-call resource constraints.
- [x] Bayesian greedy recovery planner.
- [x] Bayesian bounded lookahead recovery planner.
- [x] Initial live comparative experiments.
- [x] Multi-step recovery trajectory collection.
- [x] High-resolution timing measurements.
- [x] Hierarchical time-aware transition modelling.
- [x] Time-aware recovery-objective optimization.
- [x] Conventional backoff and failover baselines.
- [x] Transition-context ablation experiments.
- [x] Planning-depth sensitivity analysis.
- [x] Reward-weight sensitivity analysis.
- [x] Development experiments across 714 request episodes.
- [x] Detailed paired recovery analysis.
- [x] Provisional Bayesian configuration freeze.

### Remaining

- [ ] Extend the experimental environment with variable execution delays.
- [ ] Introduce network latency variation and jitter.
- [ ] Simulate combined faults and intermittent service availability.
- [ ] Integrate the Random Forest execution-state estimator.
- [ ] Conduct estimator probability-calibration comparisons.
- [ ] Evaluate V2, V3, V4 and V5 under identical conditions.
- [ ] Conduct independent generalization experiments.
- [ ] Evaluate reserved dataset partitions.
- [ ] Perform statistical significance and uncertainty analysis.
- [ ] Investigate additional semantic safety conditions.
- [ ] Implement the native Ballerina recovery library.
- [ ] Complete integration testing and technical documentation.
- [ ] Prepare the final research publication.

---

## 26. Research Contribution

SURE investigates probabilistic, semantics-constrained and resource-aware recovery for ambiguous distributed request failures.

The proposed approach combines:

- Hidden execution-state probability estimation.
- Sequential belief updating.
- Time-aware execution-transition modelling.
- Information-gathering recovery actions.
- Mandatory semantic safety constraints.
- Deadline and network-call resource management.
- Bounded sequential recovery planning.

The current development results indicate that Bayesian lookahead can provide useful adaptive recovery behaviour, particularly by reducing unnecessary waiting and improving successful-recovery latency compared with a fixed failover-backoff strategy.

The final research evaluation will determine whether these advantages remain reliable across independent, more diverse distributed-system environments and whether the additional computational and network costs are justified.

---

## Project Status

**Bayesian recovery prototype implemented and evaluated.**

**Current phase:** Random Forest integration, independent experimental environment expansion and final comparative evaluation.