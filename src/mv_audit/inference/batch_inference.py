"""Batch inference for Phase 07 M0/M1/M2 comparisons."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from mv_audit.converters.common import build_audit_output, build_prompt, existing_image_items, group_records_by_case
from mv_audit.inference.qwen3vl_common import (
    generate_text,
    image_to_message_uri,
    load_qwen3vl_model_and_processor,
    move_inputs_to_model,
    process_messages,
)
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, read_jsonl, write_jsonl


MODEL_IDS = {"m0_zero_shot", "m1_few_shot", "m2_sft"}
TEST_SPLITS = {"test_clean", "test_robust", "test_unseen_template", "test_hard_negative"}
DEFAULT_CONFIG = "configs/train/sft_lora_qwen3vl_8b.yaml"


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _split_file_name(split: str) -> str:
    return f"{split}_cases.jsonl"


def _annotation_file_name(split: str) -> str:
    return f"field_bboxes_{split}.jsonl"


def _read_few_shot_examples(train_file: str | Path, *, count: int, seed: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    rows = list(iter_jsonl(train_file))
    rng = random.Random(seed)
    selected = rng.sample(rows, min(count, len(rows)))
    examples: list[dict[str, str]] = []
    for row in selected:
        messages = row.get("messages") or []
        if len(messages) < 2:
            continue
        examples.append(
            {
                "case_id": str(row["case_id"]),
                "prompt": str(messages[0].get("content") or ""),
                "answer": str(messages[-1].get("content") or ""),
            }
        )
    return examples


def _few_shot_prefix(examples: list[dict[str, str]]) -> str:
    if not examples:
        return ""
    blocks = ["以下示例全部来自训练集，仅用于说明输出格式和审计偏好，不得复制示例中的 case_id 或字段值。"]
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"示例{index} case_id={example['case_id']}\n"
            f"用户任务：\n{example['prompt']}\n"
            f"助手输出：\n{example['answer']}"
        )
    return "\n\n".join(blocks) + "\n\n现在请处理新的输入 case。\n"


def build_eval_rows(
    *,
    config: dict[str, Any],
    split: str,
    model_id: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if split not in TEST_SPLITS:
        raise ValueError(f"Unsupported split {split!r}. Expected one of {sorted(TEST_SPLITS)}.")
    data_config = _section(config, "data")
    inference_config = _section(config, "inference")

    raw_cases_dir = Path(str(data_config.get("raw_cases_dir", "data/mv_audit/raw_cases/main")))
    annotations_dir = Path(str(data_config.get("annotations_dir", "data/mv_audit/annotations_main")))
    cases_path = raw_cases_dir / _split_file_name(split)
    annotations_path = annotations_dir / _annotation_file_name(split)
    cases = read_jsonl(cases_path)
    if limit is not None:
        cases = cases[:limit]
    records_by_case = group_records_by_case(read_jsonl(annotations_path))

    few_shot_examples = []
    if model_id == "m1_few_shot":
        few_shot_examples = _read_few_shot_examples(
            data_config["train_file"],
            count=int(inference_config.get("few_shot_count", 2)),
            seed=int(config.get("seed", 42)),
        )
    prefix = _few_shot_prefix(few_shot_examples)
    rng = random.Random(int(config.get("seed", 42)))

    rows: list[dict[str, Any]] = []
    for case in cases:
        records = records_by_case[case["case_id"]]
        image_items = existing_image_items(records, rng=rng)
        output = build_audit_output(case, records)
        prompt = prefix + build_prompt(
            case,
            image_items,
            task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。",
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "split": split,
                "model_id": model_id,
                "images": image_items,
                "prompt": prompt,
                "answer": output,
                "source_split": split,
            }
        )
    return rows


def _messages_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for item in row["images"]:
        image_id = str(item.get("image_id") or "")
        doc_type = str(item.get("doc_type") or "")
        image_path = Path(str(item["image_path"])).resolve()
        content.append({"type": "text", "text": f"{image_id}: {doc_type}"})
        content.append({"type": "image", "image": image_to_message_uri(image_path)})
    content.append({"type": "text", "text": str(row["prompt"])})
    return [{"role": "user", "content": content}]


def _prediction_path(config: dict[str, Any], *, model_id: str, split: str) -> Path:
    output_dir = Path(str(_section(config, "inference").get("predictions_dir", "outputs/predictions")))
    return output_dir / model_id / f"{split}.jsonl"


def _ground_truth_path(config: dict[str, Any], *, split: str) -> Path:
    output_dir = Path(str(_section(config, "inference").get("ground_truth_dir", "data/mv_audit/eval_sets_main")))
    return output_dir / f"{split}.jsonl"


def _dry_run(config: dict[str, Any], *, model_id: str, split: str, limit: int | None) -> None:
    rows = build_eval_rows(config=config, split=split, model_id=model_id, limit=limit)
    if not rows:
        raise ValueError(f"No rows built for split {split}.")
    for row in rows[: min(len(rows), 2)]:
        if not row["images"]:
            raise ValueError(f"Eval row {row['case_id']} has no images.")
        for item in row["images"]:
            image_path = Path(str(item["image_path"]))
            if not image_path.exists():
                raise FileNotFoundError(f"Image does not exist: {image_path}")
    print("phase07_inference_dry_run=ok")
    print(f"model_id={model_id}")
    print(f"split={split}")
    print(f"rows_checked={len(rows)}")
    print(f"prediction_output={_prediction_path(config, model_id=model_id, split=split)}")
    print(f"ground_truth_output={_ground_truth_path(config, split=split)}")


def _load_model_for_inference(config: dict[str, Any], *, model_id: str):
    model_config = dict(_section(config, "model"))
    model, processor, model_path = load_qwen3vl_model_and_processor(model_config)
    if model_id == "m2_sft":
        adapter_dir = _section(config, "inference").get("sft_adapter_dir")
        if not adapter_dir:
            raise ValueError("inference.sft_adapter_dir is required for m2_sft.")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load the M2 SFT adapter.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model, processor, model_path


def run_inference(config: dict[str, Any], *, model_id: str, split: str, limit: int | None) -> None:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported model_id {model_id!r}. Expected one of {sorted(MODEL_IDS)}.")
    rows = build_eval_rows(config=config, split=split, model_id=model_id, limit=limit)
    prediction_path = _prediction_path(config, model_id=model_id, split=split)
    ground_truth_path = _ground_truth_path(config, split=split)
    ensure_dir(prediction_path.parent)
    ensure_dir(ground_truth_path.parent)
    write_jsonl(rows, ground_truth_path)

    model, processor, model_path = _load_model_for_inference(config, model_id=model_id)
    inference_config = _section(config, "inference")
    predictions: list[dict[str, Any]] = []
    for row in rows:
        messages = _messages_for_row(row)
        inputs = process_messages(processor, messages)
        inputs = move_inputs_to_model(inputs, model)
        started = time.perf_counter()
        raw_output = generate_text(
            model,
            processor,
            inputs,
            max_new_tokens=int(inference_config.get("max_new_tokens", 1536)),
            temperature=float(inference_config.get("temperature", 0.0)),
            top_p=float(inference_config.get("top_p", 0.9)),
        )
        predictions.append(
            {
                "case_id": row["case_id"],
                "model_id": model_id,
                "split": split,
                "images": row["images"],
                "raw_output": raw_output,
                "elapsed_seconds": time.perf_counter() - started,
                "model_path": model_path,
            }
        )
        if len(predictions) % int(inference_config.get("flush_every", 20)) == 0:
            write_jsonl(predictions, prediction_path)
    write_jsonl(predictions, prediction_path)
    print(f"predictions={len(predictions)}")
    print(f"prediction_output={prediction_path}")
    print(f"ground_truth_output={ground_truth_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 07 batch inference for M0/M1/M2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model_id", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--split", required=True, choices=sorted(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        _dry_run(config, model_id=args.model_id, split=args.split, limit=args.limit)
        return
    run_inference(config, model_id=args.model_id, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
