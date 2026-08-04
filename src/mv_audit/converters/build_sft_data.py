"""Build phase 05 SFT JSONL data from MV-Train only."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from mv_audit.converters.common import (
    build_audit_output,
    build_prompt,
    existing_image_items,
    group_records_by_case,
    json_answer,
    make_messages,
    output_validator,
    validate_output,
    add_common_args,
)
from mv_audit.utils import read_jsonl, write_jsonl


TASK_MIX = [
    ("full_audit", 0.60, "完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。"),
    ("field_evidence", 0.25, "重点完成字段抽取和证据定位，同时输出完整 Evidence-Grounded JSON。"),
    ("consistency_check", 0.15, "重点判断金额、商户、人员、日期、订单号和材料完整性，同时输出完整 Evidence-Grounded JSON。"),
]


def _assign_task(index: int, total: int) -> tuple[str, str]:
    ratio = index / max(total, 1)
    cumulative = 0.0
    for task_name, weight, instruction in TASK_MIX:
        cumulative += weight
        if ratio < cumulative:
            return task_name, instruction
    return TASK_MIX[-1][0], TASK_MIX[-1][2]


def build_examples(
    *,
    cases: list[dict[str, Any]],
    records_by_case: dict[str, list[dict[str, Any]]],
    schema_path: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    validator = output_validator(schema_path)
    shuffled = list(cases)
    rng.shuffle(shuffled)

    examples: list[dict[str, Any]] = []
    for index, case in enumerate(shuffled):
        records = records_by_case[case["case_id"]]
        image_items = existing_image_items(records, rng=rng)
        task_type, instruction = _assign_task(index, len(shuffled))
        output = build_audit_output(case, records)
        validate_output(output, validator)
        prompt = build_prompt(case, image_items, task_instruction=instruction)
        answer = json_answer(output)
        examples.append(
            {
                "id": f"{case['case_id']}_{task_type}",
                "case_id": case["case_id"],
                "task_type": task_type,
                "images": image_items,
                "messages": make_messages(prompt, answer),
                "answer": output,
                "source_split": "MV-Train",
            }
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase 05 SFT train/val JSONL from MV-Train.")
    add_common_args(parser)
    parser.add_argument("--train_output", default="data/mv_audit/sft/train.jsonl")
    parser.add_argument("--val_output", default="data/mv_audit/sft/val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.cases)
    records_by_case = group_records_by_case(read_jsonl(args.annotations))
    examples = build_examples(cases=cases, records_by_case=records_by_case, schema_path=args.output_schema, seed=args.seed)

    val_count = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]
    write_jsonl(train_examples, Path(args.train_output))
    write_jsonl(val_examples, Path(args.val_output))
    print(f"sft_train={len(train_examples)}")
    print(f"sft_val={len(val_examples)}")


if __name__ == "__main__":
    main()
