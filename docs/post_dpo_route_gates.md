# Post-DPO Route Gates

This document fixes the current Phase08 decision after DPO v1/v2 did not meet the business target.

## Decision

- M2 remains the business baseline.
- M3v2 is a partial DPO recovery result, not a replacement for M2.
- DPO v1/v2 should be reported as preference-objective convergence without stable High-risk Miss improvement.
- The next low-cost path is High-risk Repair SFT plus error attribution.
- GRPO is optional online RL and must not start automatically after a DPO failure.

## Required Order

```text
M2 baseline
-> High-risk Repair SFT dry-run
-> Repair fast gate on train_decode_dev
-> Repair main validation gate
-> Error attribution on residual cases
-> RL decision gate
-> GRPO compatibility smoke only if READY_FOR_RL
```

## Gates

Gate thresholds live in `configs/gates/post_dpo_route_v1.yaml`.

- `repair_fast_gate` catches catastrophic regression on train_decode_dev.
- `repair_main_gate` requires Audit Accuracy protection, High-risk Miss improvement, and Evidence Support protection.
- `rl_decision` requires residual errors to be mainly decision candidates before RL is allowed.

The route gate CLI is:

```bash
python -m mv_audit.analysis.post_dpo_route \
  --mode repair_main \
  --baseline-metrics docs/experiments/phase08_high_risk_repair_pack_20260813/metric_snapshot.csv \
  --baseline-model-id m2_sft \
  --baseline-scope sample500_mean \
  --candidate-metrics <repair_metrics.json-or-csv> \
  --output docs/experiments/phase09_repair_v4/repair_main_gate.json
```

For RL decision:

```bash
python -m mv_audit.analysis.post_dpo_route \
  --mode rl_decision \
  --baseline-metrics <m2_validation_metrics.json-or-csv> \
  --candidate-metrics <best_repair_validation_metrics.json-or-csv> \
  --attribution-jsonl docs/experiments/phase09_repair_v4/best_repair_error_attribution.jsonl \
  --output docs/experiments/phase09_repair_v4/rl_decision.json
```

## Safety Locks

- `scripts/11_run_high_risk_repair_sft_r1_server.sh` runs its dry-run first, then requires `ALLOW_TRAINING=1` before formal training.
- `scripts/06_train_grpo.sh` allows dry-run, but formal GRPO requires `ALLOW_RL=1` and a `READY_FOR_RL` decision file.
- sample500 remains a historical benchmark and must not be used for model selection or reward tuning.