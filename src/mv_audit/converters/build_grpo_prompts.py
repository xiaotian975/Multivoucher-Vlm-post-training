"""Build phase 05 GRPO prompt data from MV-Train only."""

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
    output_validator,
    validate_output,
    add_common_args,
)
from mv_audit.utils import read_jsonl, write_jsonl


PRIORITY_ANOMALIES = {
    "over_reimbursement",
    "merchant_mismatch",
    "applicant_mismatch",
    "order_id_mismatch",
    "missing_document",
    "unreadable_image",
}


def _priority(case: dict[str, Any]) -> tuple[int, str]:
    anomaly_types = set(case.get("anomaly_types") or [])
    score = 0
    if case["risk_level"] == "high":
        score += 100
    if anomaly_types & PRIORITY_ANOMALIES:
        score += 50
    if case["audit_result"] in {"missing_info", "reject_recommendation", "manual_review"}:
        score += 20
    return (-score, case["case_id"])


def build_prompts(
    *,
    cases: list[dict[str, Any]],
    records_by_case: dict[str, list[dict[str, Any]]],
    schema_path: str,
    seed: int,
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 700)
    validator = output_validator(schema_path)
    ordered = sorted(cases, key=_priority)
    if max_prompts is not None:
        ordered = ordered[:max_prompts]
    prompts: list[dict[str, Any]] = []

    for case in ordered:
        records = records_by_case[case["case_id"]]
        output = build_audit_output(case, records)
        validate_output(output, validator)
        image_items = existing_image_items(records, rng=rng)
        prompt = build_prompt(case, image_items, task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。")
        prompts.append(
            {
                "id": f"{case['case_id']}_grpo",
                "case_id": case["case_id"],
                "images": image_items,
                "prompt": prompt,
                "ground_truth": {
                    "output": output,
                    "risk_level": case["risk_level"],
                    "audit_result": case["audit_result"],
                    "anomaly_types": list(case.get("anomaly_types") or []),
                    "evidence": output["evidence"],
                    "uncertainty": output["uncertainty"],
                    "bbox_records": [
                        record
                        for record in records
                        if record.get("readable", True) and not record.get("duplicate_of_image_id")
                    ],
                },
                "reward_tags": {
                    "high_risk": case["risk_level"] == "high",
                    "priority_anomaly": bool(set(case.get("anomaly_types") or []) & PRIORITY_ANOMALIES),
                    "missing_document": "missing_document" in set(case.get("anomaly_types") or []),
                    "unreadable_image": "unreadable_image" in set(case.get("anomaly_types") or []),
                },
                "source_split": "MV-Train",
            }
        )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase 05 GRPO prompt JSONL from MV-Train.")
    add_common_args(parser)
    parser.add_argument("--output", default="data/mv_audit/grpo/prompts_train.jsonl")
    parser.add_argument("--max_prompts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.cases)
    records_by_case = group_records_by_case(read_jsonl(args.annotations))
    prompts = build_prompts(
        cases=cases,
        records_by_case=records_by_case,
        schema_path=args.output_schema,
        seed=args.seed,
        max_prompts=args.max_prompts,
    )
    write_jsonl(prompts, Path(args.output))
    print(f"grpo_prompts={len(prompts)}")


if __name__ == "__main__":
    main()
