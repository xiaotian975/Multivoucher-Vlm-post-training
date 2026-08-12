

## Phase08 Loss Ablation Baseline Partial（20260812_5gpu_ablation_r3）

已先行归档第一个候选 `dpo_v2_baseline` 的训练结果，完整目录见 `docs/experiments/phase08_loss_ablation_baseline_partial_20260812_5gpu_ablation_r3/phase08_loss_ablation_baseline_partial_report.md`。

| variant | status | loss_type | lambda_sft | global_step | loss | preference_loss | sft_nll_loss | preference_margin | holdout_pair_accuracy | holdout_preference_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dpo_v2_baseline | training_done_eval_not_run | dpo | 0.1000 | 40.0000 | 0.0121 | 0.0121 | 0.0000 | 44.0543 | 1.0000 | 25.7227 |

说明：该候选目前只完成 DPO 训练，尚未运行 Train decode dev/sample500 推理评测，因此暂无 error cases、sample500 metrics 和错误迁移分析。服务器上的后续候选训练仍在继续。
