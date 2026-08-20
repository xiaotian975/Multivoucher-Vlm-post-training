# Model Lineage Archive Audit

- Source manifest: multivoucher-audit-model-lineage-20260820
- Verified: 6/6
- All verified: true

| Model | Role | Parent | Status | Weight bytes |
| --- | --- | --- | --- | ---: |
| m2_sft | HISTORICAL_SAMPLE500_BASELINE | - | VERIFIED | 174663096 |
| repair_sft_r1 | HIGH_RISK_REPAIR_STAGE_1 | m2_sft | VERIFIED | 174663096 |
| repair_sft_r2 | ORDER_ID_REPAIR_STAGE_2 | repair_sft_r1 | VERIFIED | 174663096 |
| repair_sft_r3 | PRODUCTION_CANDIDATE_NOT_DEPLOYED | repair_sft_r2 | VERIFIED | 174663096 |
| dpo_v3_weak_checkpoint40 | DPO_STRONG_INITIALIZATION | repair_sft_r3 | VERIFIED | 174663096 |
| dpo_v3_strong_checkpoint15 | ALIGNMENT_RESEARCH_CANDIDATE | dpo_v3_weak_checkpoint40 | VERIFIED | 174663096 |
