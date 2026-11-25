SURE independent evaluation patch (new files only)

Place the four source files in their matching project folders. Do not replace the original service or existing training/evaluation scripts. Existing B3 and B4.1 model artifacts must match the hashes recorded by the development sensitivity experiment. Python scripts reuse the original planner and deterministic baseline rule; no new model fitting occurs in live evaluation. B2 is reconstructed from training only because the original script did not serialize the classifier. Its validation scores must reproduce before holdout scoring.

From the project root, in an activated venv:
Terminal 1: bal run ballerina_service/extended_service.bal
Terminal 2:
  python -m bayesian.independent_evaluation verify
  python -m bayesian.independent_evaluation smoke
  python -m bayesian.independent_evaluation analyze-smoke

Stop here if the service fails to compile, the verification fails, or the smoke run reveals an integrity problem. The smoke uses a separate seed and output directory. Do not select policies by smoke success rates. If checks pass:
  python -m bayesian.independent_evaluation freeze
  python -m bayesian.score_frozen_holdouts
  python -m bayesian.independent_evaluation run
  python -m bayesian.independent_evaluation analyze

The final live design has 7 families x 2 keyed settings x 3 initial caps x 3 deadlines x 4 call budgets = 504 matched cases, and 10 policies (5040 requests). This is an exploratory fixed size, not a formal power calculation. The main comparison is V4 vs failover_backoff; latency on the jointly recovered subset is descriptive. Use independent model validation without tuning based on held-out scores. If holdout inputs are missing at freeze, the holdout command stops (live evaluation can still proceed). Preserve logs; output directories will not overwrite existing runs. An interrupted live run has no resume support. Keep the previous development service and results unchanged.

The extension simulates request/response delays, drops and intermittent availability using pre-sampled endpoint-call schedules, with a shared in-memory ledger. It does not implement real packet loss, separate hosts or hard deadlines. Schedule draws are matched, not wall-clock execution times. The truth read is outside the measured recovery latency. No formal semantic safety guarantee is asserted. Native Ballerina integration and the RF estimator are not implemented by this patch.

Validation performed before delivery: Python syntax checks, actual expanded training trajectory audit, reproducible balanced design checks, simulated adapter runs (all ten policies), and the smoke-result trace audit and summary pipeline. The Ballerina CLI and the original frozen joblib are not available in the authoring environment; compile and live smoke checks on the user's computer are mandatory.
