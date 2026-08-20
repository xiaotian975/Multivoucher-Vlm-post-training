# VLM 后训练项目简历材料

## 项目标题

**多凭证审计 VLM 后训练：Structured SFT 与 Model-Mined DPO 偏好对齐**

技术栈：`Qwen3-VL-8B`、`PyTorch`、`PEFT/LoRA`、`DPO`、多卡推理、Evidence-Grounded JSON、自动评测与发布门禁。

## 一页简历版

- 基于 Qwen3-VL-8B 搭建企业多凭证审计 VLM 后训练链路，完成 Train-only 数据构造、Structured Repair SFT、LoRA adapter 训练、5 卡并行推理和结构化评测；SFT production candidate 在 152 条开发门禁上达到 Audit Accuracy `96.71%`、JSON/Schema 合规率 `100%`、Evidence Support `98.76%`。
- 针对人工 rejected 过易和 preference margin 快速饱和问题，设计 Model-Error-Mined DPO：从当前 SFT 模型采样真实错误输出，构造困难偏好对，引入 assistant-token 平均 log-prob、risk/evidence-aware reward 和 case-disjoint alignment probe。
- DPO alignment probe reward 提升 `0.167`、order-id 双侧证据命中率提升 `11.1pp`；通过 152-case 全量业务门禁和 paired error attribution 识别局部偏好过拟合，阻止不满足 Audit/HRM 约束的 checkpoint 进入部署候选。

## 60 秒面试介绍

这个项目研究的是企业报销场景下的多图一致性审计。模型需要同时读取发票、订单、支付截图和报销单，输出带字段、风险结论和 bbox 证据的固定 JSON。

我先通过 Structured Repair SFT 建立稳定业务基线，然后完成了三版 DPO。早期 DPO 使用人工构造的简单 rejected，训练 loss 和 preference margin 很好看，但业务行为没有同步改善。为解决这个问题，我在 v3 中让当前 SFT 模型对 Train-only case 进行多次采样，从真实生成错误中挖掘 hard preference pairs，并把长 JSON 的 sequence-sum log-prob 改为 assistant-token 平均 log-prob。

DPO v3 在独立 alignment probe 上取得 reward `+0.167` 和 order-id 双侧证据命中率 `+11.1pp`。但我没有用局部 probe 包装最终效果，而是继续运行 152 条全量门禁，通过 paired attribution 发现它会把部分已识别的 amount mismatch 从 reject 降为 manual review。因此最终保留 SFT 作为 production candidate，将 DPO checkpoint 标为 research candidate。这个过程让我真正理解了偏好数据覆盖、reward 饱和、probe overfitting 和模型发布门禁之间的关系。

## 技术追问口径

### 为什么不用 DPO loss 选 checkpoint？

DPO loss 只衡量训练 pair 上 chosen 相对 rejected 的概率优势。DPO v1 的 loss 已接近 0、margin 很大，但 sample500 Audit 下降，因此 checkpoint 必须同时通过 case-disjoint probe 和业务 fast gate。

### 为什么使用 model-mined pairs？

人工 rejected 容易形成标签捷径。模型真实生成错误更接近推理分布，能够暴露“缺证据但通过”“识别风险却放行”等真实偏好边界。

### 为什么改成 mean-token log-prob？

Evidence-Grounded JSON 很长，sequence-sum 会让长度主导 margin。对 assistant token 求平均后，不同 completion 的偏好差异更接近单位 token 的行为变化。

### DPO 最终是否超过 SFT？

没有。它在局部 probe 上产生了可测量对齐信号，但未通过完整业务门禁，因此没有作为 production candidate。这也是项目建立双层评测和发布拦截的原因。

## 表述边界

- SFT 的 `96.71%` 不归因于 DPO。
- DPO 的 `+0.167/+11.1pp` 只属于 24 条 alignment probe。
- 项目应表述为 `SFT + Preference Alignment (DPO)`，不冒充已经完成 PPO/GRPO 式在线强化学习。
- `PRODUCTION_CANDIDATE` 表示开发门禁选中，不等于已部署或已运行 final holdout。
