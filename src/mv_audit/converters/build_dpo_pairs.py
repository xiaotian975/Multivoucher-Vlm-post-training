"""Build phase 05 DPO preference pairs from MV-Train only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mv_audit.converters.common import (
    build_audit_output,
    build_prompt,
    existing_image_items,
    group_records_by_case,
    json_answer,
    output_validator,
    validate_output,
    add_common_args,
)
from mv_audit.utils import read_jsonl, write_jsonl


REJECTED_TYPES = [
    "high_risk_pass",
    "wrong_anomaly_type",
    "wrong_risk_level",
    "wrong_evidence_source",
    "wrong_bbox",
    "unsupported_reason",
    "json_invalid",
    "hallucinated_field",
    "unreadable_but_guess",
    "missing_doc_but_pass",
]


def _invalid_json(_chosen: dict[str, Any], _case: dict[str, Any]) -> str:
    return '{"case_id": "broken", "audit_result": pass'


def _mutate_output(chosen: dict[str, Any], case: dict[str, Any], rejected_type: str) -> str:
    rejected = json.loads(json.dumps(chosen, ensure_ascii=False))
    if rejected_type == "high_risk_pass":
        rejected["risk_level"] = "low"
        rejected["audit_result"] = "pass"
        rejected["reason"] = "未发现异常，建议通过。"
    elif rejected_type == "wrong_anomaly_type":
        rejected["anomaly_types"] = ["merchant_mismatch"] if "merchant_mismatch" not in rejected["anomaly_types"] else ["amount_mismatch"]
    elif rejected_type == "wrong_risk_level":
        rejected["risk_level"] = "low" if chosen["risk_level"] != "low" else "high"
    elif rejected_type == "wrong_evidence_source":
        if rejected["evidence"]:
            rejected["evidence"][0]["source_image_id"] = "wrong_image_id"
            rejected["evidence"][0]["source_doc_type"] = "invoice"
    elif rejected_type == "wrong_bbox":
        if rejected["evidence"]:
            rejected["evidence"][0]["bbox"] = [0, 0, 10, 10]
    elif rejected_type == "unsupported_reason":
        rejected["reason"] = "因为系统直觉判断异常，所以给出该结论。"
    elif rejected_type == "hallucinated_field":
        rejected["field_extraction"]["payment_id"] = "PAY99999999999999"
        rejected["reason"] = "补全了未可靠出现的支付流水号。"
    elif rejected_type == "unreadable_but_guess":
        for field in (case.get("metadata") or {}).get("unreadable_fields") or ["invoice_amount"]:
            if field in rejected["field_extraction"]:
                rejected["field_extraction"][field] = str(case.get(field) or "UNKNOWN")
        rejected["uncertainty"] = {"has_uncertain_fields": False, "uncertain_fields": [], "requires_manual_review": False}
    elif rejected_type == "missing_doc_but_pass":
        rejected["consistency_check"]["document_complete"] = True
        rejected["risk_level"] = "low"
        rejected["audit_result"] = "pass"
        rejected["reason"] = "材料完整，建议通过。"
    else:
        raise ValueError(f"Unsupported rejected_type: {rejected_type}")
    return json_answer(rejected)


def _eligible_rejected_types(case: dict[str, Any]) -> list[str]:
    anomaly_types = set(case.get("anomaly_types") or [])
    eligible = list(REJECTED_TYPES[:8])
    if "unreadable_image" in anomaly_types:
        eligible.append("unreadable_but_guess")
    if "missing_document" in anomaly_types:
        eligible.append("missing_doc_but_pass")
    if case["risk_level"] == "high":
        eligible.append("high_risk_pass")
    return list(dict.fromkeys(eligible))


def build_pairs(
    *,
    cases: list[dict[str, Any]],
    records_by_case: dict[str, list[dict[str, Any]]],
    schema_path: str,
    seed: int,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 500)
    validator = output_validator(schema_path)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    pairs: list[dict[str, Any]] = []

    coverage_counts = {name: 0 for name in REJECTED_TYPES}
    for case in shuffled:
        records = records_by_case[case["case_id"]]
        output = build_audit_output(case, records)
        validate_output(output, validator)
        image_items = existing_image_items(records, rng=rng)
        prompt = build_prompt(case, image_items, task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。")

        eligible = _eligible_rejected_types(case)
        rejected_type = min(eligible, key=lambda name: coverage_counts.get(name, 0))
        rejected = _invalid_json(output, case) if rejected_type == "json_invalid" else _mutate_output(output, case, rejected_type)
        coverage_counts[rejected_type] += 1
        pairs.append(
            {
                "id": f"{case['case_id']}_{rejected_type}",
                "case_id": case["case_id"],
                "rejected_type": rejected_type,
                "images": image_items,
                "prompt": prompt,
                "chosen": json_answer(output),
                "rejected": rejected,
                "source_split": "MV-Train",
            }
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase 05 DPO preference pairs from MV-Train.")
    add_common_args(parser)
    parser.add_argument("--output", default="data/mv_audit/dpo/pairs_train.jsonl")
    parser.add_argument("--max_pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.cases)
    records_by_case = group_records_by_case(read_jsonl(args.annotations))
    pairs = build_pairs(
        cases=cases,
        records_by_case=records_by_case,
        schema_path=args.output_schema,
        seed=args.seed,
        max_pairs=args.max_pairs,
    )
    write_jsonl(pairs, Path(args.output))
    print(f"dpo_pairs={len(pairs)}")


if __name__ == "__main__":
    main()
