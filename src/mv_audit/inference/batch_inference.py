"""Batch inference for M0/M1/M2 and Phase 08 M3 comparisons."""

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
from mv_audit.inference.schema_guard import guard_raw_output
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, read_jsonl, read_yaml, write_jsonl


MODEL_IDS = {
    "m0_zero_shot",
    "m1_few_shot",
    "m2_sft",
    "m3_dpo",
    "m3v2_dpo",
    "repair_sft_r1",
    "repair_sft_r2",
    "repair_sft_r3",
    "dpo_v3_model_mined",
    "dpo_v3_model_mined_strong",
}
TEST_SPLITS = {"test_clean", "test_robust", "test_unseen_template", "test_hard_negative", "train_decode_dev"}
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


def _sample_manifest_path(config: dict[str, Any], split: str) -> Path | None:
    inference_config = _section(config, "inference")
    manifest_dir = inference_config.get("sample_manifest_dir")
    if not manifest_dir:
        return None
    return Path(str(manifest_dir)) / f"{split}_case_ids.jsonl"


def _sample_case_ids(config: dict[str, Any], split: str) -> list[str] | None:
    manifest_path = _sample_manifest_path(config, split)
    if manifest_path is None:
        return None
    ids: list[str] = []
    for row in iter_jsonl(manifest_path):
        case_id = row.get("case_id")
        if not case_id:
            raise ValueError(f"Missing case_id in sample manifest row: {manifest_path}")
        ids.append(str(case_id))
    if not ids:
        raise ValueError(f"Sample manifest is empty: {manifest_path}")
    return ids


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
    if split == "train_decode_dev":
        decode_dev_file = data_config.get("decode_dev_file") or inference_config.get("train_decode_dev_file")
        if not decode_dev_file:
            raise ValueError("data.decode_dev_file or inference.train_decode_dev_file is required for train_decode_dev.")
        rows = read_jsonl(decode_dev_file)
        if limit is not None:
            rows = rows[:limit]
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized = dict(row)
            normalized["split"] = "train_decode_dev"
            normalized["model_id"] = model_id
            normalized_rows.append(normalized)
        return normalized_rows

    raw_cases_dir = Path(str(data_config.get("raw_cases_dir", "data/mv_audit/raw_cases/main")))
    annotations_dir = Path(str(data_config.get("annotations_dir", "data/mv_audit/annotations_main")))
    cases_path = raw_cases_dir / _split_file_name(split)
    annotations_path = annotations_dir / _annotation_file_name(split)
    cases = read_jsonl(cases_path)
    sample_ids = _sample_case_ids(config, split)
    if sample_ids is not None:
        cases_by_id = {str(case["case_id"]): case for case in cases}
        missing_ids = [case_id for case_id in sample_ids if case_id not in cases_by_id]
        if missing_ids:
            raise ValueError(f"Sample manifest references missing cases for {split}: {missing_ids[:5]}")
        cases = [cases_by_id[case_id] for case_id in sample_ids]
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


def _messages_for_row(row: dict[str, Any], inference_config: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_max_pixels = inference_config.get("image_max_pixels")
    image_min_pixels = inference_config.get("image_min_pixels")
    for item in row["images"]:
        image_id = str(item.get("image_id") or "")
        doc_type = str(item.get("doc_type") or "")
        image_path = Path(str(item["image_path"])).resolve()
        content.append({"type": "text", "text": f"{image_id}: {doc_type}"})
        image_payload: dict[str, Any] = {"type": "image", "image": image_to_message_uri(image_path)}
        if image_max_pixels is not None:
            image_payload["max_pixels"] = int(image_max_pixels)
        if image_min_pixels is not None:
            image_payload["min_pixels"] = int(image_min_pixels)
        content.append(image_payload)
    content.append({"type": "text", "text": str(row["prompt"])})
    return [{"role": "user", "content": content}]


def _sharded_file_name(split: str, *, shard_index: int | None, num_shards: int | None) -> str:
    if shard_index is None or num_shards is None:
        return f"{split}.jsonl"
    return f"{split}.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def _prediction_path(
    config: dict[str, Any],
    *,
    model_id: str,
    split: str,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Path:
    output_dir = Path(str(_section(config, "inference").get("predictions_dir", "outputs/predictions")))
    return output_dir / model_id / _sharded_file_name(split, shard_index=shard_index, num_shards=num_shards)


def _ground_truth_path(
    config: dict[str, Any],
    *,
    split: str,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Path:
    inference_config = _section(config, "inference")
    if split == "train_decode_dev" and inference_config.get("train_decode_dev_ground_truth_dir"):
        output_dir = Path(str(inference_config["train_decode_dev_ground_truth_dir"]))
    else:
        output_dir = Path(str(inference_config.get("ground_truth_dir", "data/mv_audit/eval_sets_main")))
    return output_dir / _sharded_file_name(split, shard_index=shard_index, num_shards=num_shards)


def _ordered_predictions(
    rows: list[dict[str, Any]],
    predictions_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [predictions_by_case[row["case_id"]] for row in rows if row["case_id"] in predictions_by_case]


def _existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    predictions: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        case_id = row.get("case_id")
        if case_id:
            predictions[str(case_id)] = row
    return predictions


def _select_shard(
    rows: list[dict[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int | None,
) -> list[dict[str, Any]]:
    if shard_index is None and num_shards is None:
        return rows
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together.")
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}.")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _dry_run(
    config: dict[str, Any],
    *,
    model_id: str,
    split: str,
    limit: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> None:
    rows = build_eval_rows(config=config, split=split, model_id=model_id, limit=limit)
    rows = _select_shard(rows, shard_index=shard_index, num_shards=num_shards)
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
    print(f"prediction_output={_prediction_path(config, model_id=model_id, split=split, shard_index=shard_index, num_shards=num_shards)}")
    print(f"ground_truth_output={_ground_truth_path(config, split=split, shard_index=shard_index, num_shards=num_shards)}")


def _load_model_for_inference(config: dict[str, Any], *, model_id: str):
    model_config = dict(_section(config, "model"))
    model, processor, model_path = load_qwen3vl_model_and_processor(model_config)
    adapter_key_by_model = {
        "m2_sft": "sft_adapter_dir",
        "repair_sft_r1": "sft_adapter_dir",
        "repair_sft_r2": "sft_adapter_dir",
        "repair_sft_r3": "sft_adapter_dir",
        "dpo_v3_model_mined": "dpo_adapter_dir",
        "dpo_v3_model_mined_strong": "dpo_adapter_dir",
        "m3_dpo": "dpo_adapter_dir",
        "m3v2_dpo": "dpo_adapter_dir",
    }
    adapter_key = adapter_key_by_model.get(model_id)
    if adapter_key is not None:
        adapter_dir = _section(config, "inference").get(adapter_key)
        if not adapter_dir:
            raise ValueError(f"inference.{adapter_key} is required for {model_id}.")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(f"peft is required to load the {model_id} adapter.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model, processor, model_path

def _schema_guard_schema(config: dict[str, Any]) -> dict[str, Any] | None:
    inference_config = _section(config, "inference")
    if not inference_config.get("schema_guard", False):
        return None
    data_config = _section(config, "data")
    schema_path = (
        inference_config.get("output_schema")
        or data_config.get("output_schema")
        or "configs/schema/output_schema.json"
    )
    return read_yaml(schema_path)

def run_inference(
    config: dict[str, Any],
    *,
    model_id: str,
    split: str,
    limit: int | None,
    resume: bool,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> None:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported model_id {model_id!r}. Expected one of {sorted(MODEL_IDS)}.")
    rows = build_eval_rows(config=config, split=split, model_id=model_id, limit=limit)
    rows = _select_shard(rows, shard_index=shard_index, num_shards=num_shards)
    prediction_path = _prediction_path(
        config,
        model_id=model_id,
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    ground_truth_path = _ground_truth_path(
        config,
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    ensure_dir(prediction_path.parent)
    ensure_dir(ground_truth_path.parent)
    write_jsonl(rows, ground_truth_path)

    predictions_by_case = _existing_predictions(prediction_path) if resume else {}
    pending_rows = [row for row in rows if row["case_id"] not in predictions_by_case]
    if not pending_rows:
        write_jsonl(_ordered_predictions(rows, predictions_by_case), prediction_path)
        print(f"predictions={len(predictions_by_case)}")
        print(f"skipped_existing={len(rows)}")
        print(f"prediction_output={prediction_path}")
        print(f"ground_truth_output={ground_truth_path}")
        return

    model, processor, model_path = _load_model_for_inference(config, model_id=model_id)
    inference_config = _section(config, "inference")
    schema_guard_schema = _schema_guard_schema(config)
    new_predictions = 0
    total_rows = len(rows)
    skipped_existing = len(rows) - len(pending_rows)
    flush_every = int(inference_config.get("flush_every", 20))
    for row in pending_rows:
        messages = _messages_for_row(row, inference_config)
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
        prediction_row = {
            "case_id": row["case_id"],
            "model_id": model_id,
            "split": split,
            "images": row["images"],
            "raw_output": raw_output,
            "elapsed_seconds": time.perf_counter() - started,
            "model_path": model_path,
        }
        if schema_guard_schema is not None:
            guarded_output, guard_meta = guard_raw_output(raw_output, schema_guard_schema)
            if guard_meta.get("changed"):
                prediction_row["raw_output_original"] = raw_output
                prediction_row["raw_output"] = guarded_output
            prediction_row["schema_guard"] = guard_meta
        predictions_by_case[row["case_id"]] = prediction_row
        new_predictions += 1
        completed = skipped_existing + new_predictions
        print(
            json.dumps(
                {
                    "event": "prediction_complete",
                    "model_id": model_id,
                    "split": split,
                    "case_id": row["case_id"],
                    "completed": completed,
                    "total": total_rows,
                    "elapsed_seconds": round(predictions_by_case[row["case_id"]]["elapsed_seconds"], 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if new_predictions % flush_every == 0:
            write_jsonl(_ordered_predictions(rows, predictions_by_case), prediction_path)
    write_jsonl(_ordered_predictions(rows, predictions_by_case), prediction_path)
    print(f"predictions={len(predictions_by_case)}")
    print(f"new_predictions={new_predictions}")
    print(f"skipped_existing={len(rows) - len(pending_rows)}")
    print(f"prediction_output={prediction_path}")
    print(f"ground_truth_output={ground_truth_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for M0/M1/M2 and Phase 08 M3.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model_id", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--split", required=True, choices=sorted(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        _dry_run(
            config,
            model_id=args.model_id,
            split=args.split,
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        return
    run_inference(
        config,
        model_id=args.model_id,
        split=args.split,
        limit=args.limit,
        resume=args.resume,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
