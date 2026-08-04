"""Generate fake predictions for phase 06 evaluator validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mv_audit.utils import read_jsonl, write_jsonl


BROKEN_MODES = [
    "json_invalid",
    "schema_invalid",
    "high_risk_pass",
    "false_manual_review",
    "wrong_evidence_source",
    "wrong_bbox",
    "unsupported_conclusion",
    "hallucinated_field",
    "missing_doc_fake_bbox",
    "unreadable_but_guess",
]


def _extract_answer(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and isinstance(ground_truth.get("output"), dict):
        return ground_truth["output"]
    if isinstance(row.get("output"), dict):
        return row["output"]
    raise ValueError(f"No answer found for {row.get('case_id')}")


def _perfect_raw(answer: dict[str, Any], *, index: int) -> str:
    payload = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    if index % 3 == 0:
        return f"```json\n{payload}\n```"
    if index % 3 == 1:
        return f"审核结果如下：\n{payload}\n请以 JSON 为准。"
    return payload


def _mutate(answer: dict[str, Any], row: dict[str, Any], broken_mode: str) -> str:
    output = json.loads(json.dumps(answer, ensure_ascii=False))
    if broken_mode == "json_invalid":
        return '{"case_id": "broken", "audit_result": pass'
    if broken_mode == "schema_invalid":
        output["primary_anomaly_type"] = "none"
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    if broken_mode == "high_risk_pass":
        output["risk_level"] = "low"
        output["audit_result"] = "pass"
        output["reason"] = "未发现异常，建议通过。"
    elif broken_mode == "false_manual_review":
        output["audit_result"] = "manual_review"
        output["reason"] = "证据已充分但仍要求人工复核。"
    elif broken_mode == "wrong_evidence_source":
        if output.get("evidence"):
            output["evidence"][0]["source_image_id"] = "MV_DEBUG_FAKE_IMAGE"
    elif broken_mode == "wrong_bbox":
        if output.get("evidence"):
            output["evidence"][0]["bbox"] = [0, 0, 10, 10]
    elif broken_mode == "unsupported_conclusion":
        output["reason"] = "根据模型直觉判断该审核结论正确，无法提供证据。"
    elif broken_mode == "hallucinated_field":
        output["field_extraction"]["payment_id"] = "PAY99999999999999"
        output.setdefault("evidence", []).append(
            {
                "source_image_id": "MV_DEBUG_FAKE_IMAGE",
                "source_doc_type": "payment",
                "field": "payment_id",
                "value": "PAY99999999999999",
                "bbox": [1, 1, 20, 20],
                "evidence_text": "伪造流水号",
            }
        )
    elif broken_mode == "missing_doc_fake_bbox":
        output["consistency_check"]["document_complete"] = True
        output.setdefault("evidence", []).append(
            {
                "source_image_id": "MISSING_DOC_IMAGE",
                "source_doc_type": "reimbursement_form",
                "field": "reimbursement_amount",
                "value": "999.99",
                "bbox": [100, 100, 200, 140],
                "evidence_text": "缺失材料伪证据",
            }
        )
    elif broken_mode == "unreadable_but_guess":
        uncertain_fields = output.get("uncertainty", {}).get("uncertain_fields") or ["invoice_amount"]
        for field in uncertain_fields:
            if field in output["field_extraction"]:
                output["field_extraction"][field] = "999.99"
        output["uncertainty"] = {"has_uncertain_fields": False, "uncertain_fields": [], "requires_manual_review": False}
    else:
        raise ValueError(f"Unsupported broken mode: {broken_mode}")
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def build_predictions(rows: list[dict[str, Any]], *, mode: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = _extract_answer(row)
        if mode == "perfect":
            raw_output = _perfect_raw(answer, index=index)
        elif mode == "broken":
            broken_mode = BROKEN_MODES[index % len(BROKEN_MODES)]
            if broken_mode == "high_risk_pass" and answer.get("risk_level") != "high":
                broken_mode = "wrong_bbox"
            if broken_mode == "false_manual_review" and answer.get("audit_result") != "pass":
                broken_mode = "wrong_evidence_source"
            if broken_mode == "unreadable_but_guess" and not answer.get("uncertainty", {}).get("uncertain_fields"):
                broken_mode = rng.choice(["wrong_evidence_source", "hallucinated_field"])
            raw_output = _mutate(answer, row, broken_mode)
        else:
            raise ValueError("--mode must be perfect or broken")
        predictions.append({"case_id": row["case_id"], "raw_output": raw_output})
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate phase 06 fake predictions.")
    parser.add_argument("--ground_truth", default="data/mv_audit/sft/val.jsonl")
    parser.add_argument("--mode", choices=["perfect", "broken"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = build_predictions(read_jsonl(args.ground_truth), mode=args.mode, seed=args.seed)
    write_jsonl(predictions, Path(args.output))
    print(f"fake_predictions={len(predictions)}")
    print(f"mode={args.mode}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
