"""Evaluate MultiVoucher-Audit predictions against evidence-grounded ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.audit_metrics import (
    RISK_LEVELS,
    audit_accuracy,
    false_manual_review_rate,
    high_risk_miss_rate,
    macro_f1,
)
from mv_audit.evaluation.consistency_metrics import consistency_exact_counts
from mv_audit.evaluation.evidence_metrics import evidence_counts
from mv_audit.evaluation.field_metrics import field_exact_counts
from mv_audit.evaluation.hallucination_metrics import hallucination_count
from mv_audit.evaluation.json_parser import parse_json_output
from mv_audit.utils import read_jsonl, read_yaml, write_jsonl


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _extract_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and isinstance(ground_truth.get("output"), dict):
        return ground_truth["output"]
    if isinstance(row.get("output"), dict):
        return row["output"]
    raise ValueError(f"Cannot find ground-truth output in row: {row.get('id') or row.get('case_id')}")


def _image_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("images") or [])


def _schema_ok(output: dict[str, Any] | None, validator: Draft202012Validator) -> tuple[int, str | None]:
    if output is None:
        return 0, "no_output"
    errors = sorted(validator.iter_errors(output), key=lambda err: err.path)
    if errors:
        err = errors[0]
        path = ".".join(str(part) for part in err.path) or "<root>"
        return 0, f"{path}: {err.message}"
    return 1, None


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate(
    *,
    ground_truth_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validator = Draft202012Validator(output_schema)
    predictions = {row["case_id"]: row for row in prediction_rows}

    json_valid = 0
    schema_valid = 0
    field_correct = field_total = 0
    consistency_correct = consistency_total = 0
    evidence_accumulator = {
        "support_correct": 0,
        "support_total": 0,
        "value_correct": 0,
        "value_total": 0,
        "source_correct": 0,
        "source_total": 0,
        "bbox_strict_correct": 0,
        "bbox_relaxed_correct": 0,
        "bbox_total": 0,
    }
    hallucinated = hallucination_total = 0
    truth_outputs: list[dict[str, Any]] = []
    pred_outputs_for_business: list[dict[str, Any] | None] = []
    error_cases: list[dict[str, Any]] = []

    for truth_row in ground_truth_rows:
        case_id = truth_row["case_id"]
        truth_output = _extract_ground_truth(truth_row)
        truth_outputs.append(truth_output)
        prediction = predictions.get(case_id)
        parse_result = parse_json_output(prediction.get("raw_output", "") if prediction else "")
        json_valid += parse_result.json_validity

        schema_flag, schema_error = _schema_ok(parse_result.output, validator) if parse_result.json_validity else (0, "parse_failed")
        schema_valid += schema_flag
        business_output = parse_result.output if schema_flag else None
        pred_outputs_for_business.append(business_output)

        fc, ft = field_exact_counts(truth_output, business_output, schema_ok=bool(schema_flag))
        cc, ct = consistency_exact_counts(truth_output, business_output, schema_ok=bool(schema_flag))
        field_correct += fc
        field_total += ft
        consistency_correct += cc
        consistency_total += ct

        evidence_case = evidence_counts(truth_output, business_output, schema_ok=bool(schema_flag))
        for key, value in evidence_case.items():
            evidence_accumulator[key] += value

        hallu_count, hallu_total = hallucination_count(
            truth_output,
            business_output,
            schema_ok=bool(schema_flag),
            image_items=_image_items(truth_row),
        )
        hallucinated += hallu_count
        hallucination_total += hallu_total

        issue_codes: list[str] = []
        if not parse_result.json_validity:
            issue_codes.append("json_invalid")
        if not schema_flag:
            issue_codes.append("schema_invalid")
        if business_output is None:
            issue_codes.append("business_metrics_zeroed")
        else:
            if truth_output["risk_level"] == "high" and (
                business_output.get("risk_level") != "high" or business_output.get("audit_result") == "pass"
            ):
                issue_codes.append("high_risk_miss")
            if business_output.get("audit_result") != truth_output.get("audit_result"):
                issue_codes.append("audit_mismatch")
            reason = str(business_output.get("reason") or "")
            if "直觉" in reason or "无法提供证据" in reason:
                issue_codes.append("unsupported_conclusion")
            if hallu_count:
                issue_codes.append("hallucination")
            if evidence_case["support_correct"] < evidence_case["support_total"]:
                issue_codes.append("unsupported_evidence")
            if evidence_case["bbox_strict_correct"] < evidence_case["bbox_total"]:
                issue_codes.append("bbox_strict_error")
        if issue_codes:
            error_cases.append(
                {
                    "case_id": case_id,
                    "issues": issue_codes,
                    "parse_error": parse_result.error,
                    "schema_error": schema_error,
                    "truth_risk_level": truth_output.get("risk_level"),
                    "pred_risk_level": business_output.get("risk_level") if business_output else None,
                    "truth_audit_result": truth_output.get("audit_result"),
                    "pred_audit_result": business_output.get("audit_result") if business_output else None,
                }
            )

    truth_risk = [output["risk_level"] for output in truth_outputs]
    pred_risk = [output.get("risk_level") if output else None for output in pred_outputs_for_business]
    total_cases = len(ground_truth_rows)
    metrics = {
        "total_cases": total_cases,
        "json_validity": _rate(json_valid, total_cases),
        "schema_compliance": _rate(schema_valid, total_cases),
        "field_em": _rate(field_correct, field_total),
        "consistency_em": _rate(consistency_correct, consistency_total),
        "risk_type_macro_f1": macro_f1(truth_risk, pred_risk, RISK_LEVELS),
        "audit_accuracy": audit_accuracy(truth_outputs, pred_outputs_for_business),
        "high_risk_miss_rate": high_risk_miss_rate(truth_outputs, pred_outputs_for_business),
        "false_manual_review_rate": false_manual_review_rate(truth_outputs, pred_outputs_for_business),
        "evidence_support_rate": _rate(evidence_accumulator["support_correct"], evidence_accumulator["support_total"]),
        "evidence_value_accuracy": _rate(evidence_accumulator["value_correct"], evidence_accumulator["value_total"]),
        "evidence_source_accuracy": _rate(evidence_accumulator["source_correct"], evidence_accumulator["source_total"]),
        "evidence_bbox_accuracy_strict": _rate(
            evidence_accumulator["bbox_strict_correct"], evidence_accumulator["bbox_total"]
        ),
        "evidence_bbox_accuracy_relaxed": _rate(
            evidence_accumulator["bbox_relaxed_correct"], evidence_accumulator["bbox_total"]
        ),
        "hallucination_rate": _rate(hallucinated, hallucination_total),
        "error_cases": len(error_cases),
    }
    return metrics, error_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MultiVoucher-Audit model outputs.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--metrics_output", required=True)
    parser.add_argument("--errors_output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, error_cases = evaluate(
        ground_truth_rows=read_jsonl(args.ground_truth),
        prediction_rows=read_jsonl(args.predictions),
        output_schema=read_yaml(args.output_schema),
    )
    _write_json(metrics, args.metrics_output)
    write_jsonl(error_cases, args.errors_output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
