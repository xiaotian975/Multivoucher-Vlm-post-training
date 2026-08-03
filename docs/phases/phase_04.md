# Phase 04: 凭证图片渲染、bbox 记录和视觉扰动

## 阶段目标

把结构化 case 渲染成四类凭证图片，并在渲染阶段自动记录字段 bbox。随后加入轻度和强视觉扰动，并为 `duplicate_in_batch` 和 `unreadable_image` 生成对应图片状态。

## 允许修改范围

- `src/mv_audit/rendering/`
- `src/mv_audit/perturbation/`
- `scripts/02_render_images.sh`
- `scripts/visualize_bbox.py`
- `data/mv_audit/images/`
- `data/mv_audit/annotations/`
- `outputs/eval_reports/figures/bbox_samples/`

## 禁止事项

- 不修改 risk rule engine。
- 不修改 case schema，除非有显式 schema migration。
- 不构造训练数据。
- 不训练模型。
- 不提交版权不清楚的字体文件。
- 如果不实现 bbox 几何同步变换，不允许使用会改变图像坐标系的扰动作为第一版训练标注。

## 输入

- `train_cases.jsonl`
- `val_*_cases.jsonl`
- `test_*_cases.jsonl`
- 模板配置。
- 字段位置配置。
- 用户本地指定的字体文件路径。

## 输出

- `src/mv_audit/rendering/render_invoice.py`
- `src/mv_audit/rendering/render_payment.py`
- `src/mv_audit/rendering/render_reimbursement_form.py`
- `src/mv_audit/rendering/render_order.py`
- `src/mv_audit/rendering/render_all.py`
- `src/mv_audit/rendering/bbox_recorder.py`
- `src/mv_audit/perturbation/visual_augment.py`
- `src/mv_audit/perturbation/robust_augment.py`
- `src/mv_audit/perturbation/unreadable_generator.py`
- `src/mv_audit/perturbation/duplicate_generator.py`
- `data/mv_audit/images/<split>/`
- `data/mv_audit/annotations/field_bboxes_<split>.jsonl`

每条 bbox record 至少包含：

- `case_id`
- `image_id`
- `doc_type`
- `image_path`
- `field`
- `value`
- `bbox_abs`
- `bbox_norm`
- `evidence_text`
- `readable`

## 测试方式

- 随机读取图片和 bbox records。
- 在图片上画出 bbox。
- 保存可视化结果到 `outputs/eval_reports/figures/bbox_samples/`。
- 人工检查至少 50 张图片，确认字段框与文字对齐。

## 完成定义

- 四类凭证都能按 case 渲染。
- 每个 `image_id` 唯一。
- 模型输出使用的 bbox 坐标为 0 到 1000 归一化坐标。
- `duplicate_in_batch` 不覆盖原图，并记录 duplicate pair。
- `unreadable_image` 不删除原始 bbox 标注，但把对应 annotation 标记为 `readable=false`。
- 若 bbox 明显偏移，必须先修 bbox，不进入下一阶段。

## 下一阶段依赖

phase 05 依赖本阶段输出的图片路径、`image_id`、`source_doc_type`、bbox records 和 readable 状态。

## 待确认问题

- 本地中文字体文件路径和授权方式需要确认。
