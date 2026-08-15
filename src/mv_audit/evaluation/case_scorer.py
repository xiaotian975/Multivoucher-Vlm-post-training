"""Case-level scoring built from the existing evaluation metric functions."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    json_valid: bool
    schema_valid: bool
    field_correct: int
    field_total: int
    consistency_correct: int
    consistency_total: int
    evidence: dict[str, int]
    hallucinated: int
    hallucination_total: int
    truth_output: dict[str, Any]
    pred_output: dict[str, Any] | None
    parse_error: str | None
    schema_error: str | None
    issue_codes: list[str]

    @property
    def field_score(self) -> float:
        return _rate(self.field_correct, self.field_total)

    @property
    def consistency_score(self) -> float:
        return _rate(self.consistency_correct, self.consistency_total)

    @property
    def audit_correct(self) -> bool:
        return self.pred_output is not None and self.pred_output.get("audit_result") == self.truth_output.get("audit_result")

    @property
    def risk_correct(self) -> bool:
        return self.pred_output is not None and self.pred_output.get("risk_level") == self.truth_output.get("risk_level")

    @property
    def high_risk_miss(self) -> bool:
        if self.truth_output.get("risk_level") != "high":
            return False
        return self.pred_output is None or self.pred_output.get("risk_level") != "high" or self.pred_output.get("audit_result") == "pass"

    @property
    def false_manual_review(self) -> bool:
        return (
            self.truth_output.get("audit_result") == "pass"
            and self.pred_output is not None
            and self.pred_output.get("audit_result") == "manual_review"
        )

    @property
    def false_escalation(self) -> bool:
        return (
            self.truth_output.get("audit_result") == "pass"
            and self.pred_output is not None
            and self.pred_output.get("audit_result") != "pass"
        )

    @property
    def anomaly_score(self) -> float:
        truth = set(str(item) for item in self.truth_output.get("anomaly_types") or [])
        pred = set(str(item) for item in (self.pred_output or {}).get("anomaly_types") or [])
        if not truth and not pred:
            return 1.0
        if not truth or not pred:
            return 0.0
        return len(truth.intersection(pred)) / len(truth.union(pred))

    @property
    def evidence_support(self) -> float:
        return _rate(self.evidence.get("support_correct", 0), self.evidence.get("support_total", 0))

    @property
    def hallucination(self) -> float:
        return _rate(self.hallucinated, self.hallucination_total)

    @property
    def bbox_score(self) -> float | None:
        total = self.evidence.get("bbox_total", 0)
        if total == 0:
            return None
        return _rate(self.evidence.get("bbox_relaxed_correct", 0), total)


def extract_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and isinstance(ground_truth.get("output"), dict):
        return ground_truth["output"]
    if isinstance(row.get("output"), dict):
        return row["output"]
    raise ValueError(f"Cannot find ground-truth output in row: {row.get('id') or row.get('case_id')}")


def image_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("images") or [])


def schema_ok(output: dict[str, Any] | None, validator: Draft202012Validator) -> tuple[int, str | None]:
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


def score_case(
    truth_row: dict[str, Any],
    prediction_row: dict[str, Any] | None,
    *,
    validator: Draft202012Validator,
) -> CaseScore:
    case_id = str(truth_row["case_id"])
    truth_output = extract_ground_truth(truth_row)
    parse_result = parse_json_output(prediction_row.get("raw_output", "") if prediction_row else "")
    schema_flag, schema_error = schema_ok(parse_result.output, validator) if parse_result.json_validity else (0, "parse_failed")
    business_output = parse_result.output if schema_flag else None

    field_correct, field_total = field_exact_counts(truth_output, business_output, schema_ok=bool(schema_flag))
    consistency_correct, consistency_total = consistency_exact_counts(truth_output, business_output, schema_ok=bool(schema_flag))
    evidence_case = evidence_counts(truth_output, business_output, schema_ok=bool(schema_flag))
    hallucinated, hallucination_total = hallucination_count(
        truth_output,
        business_output,
        schema_ok=bool(schema_flag),
        image_items=image_items(truth_row),
    )

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
        if hallucinated:
            issue_codes.append("hallucination")
        if evidence_case["support_correct"] < evidence_case["support_total"]:
            issue_codes.append("unsupported_evidence")
        if evidence_case["bbox_strict_correct"] < evidence_case["bbox_total"]:
            issue_codes.append("bbox_strict_error")

    return CaseScore(
        case_id=case_id,
        json_valid=bool(parse_result.json_validity),
        schema_valid=bool(schema_flag),
        field_correct=field_correct,
        field_total=field_total,
        consistency_correct=consistency_correct,
        consistency_total=consistency_total,
        evidence=evidence_case,
        hallucinated=hallucinated,
        hallucination_total=hallucination_total,
        truth_output=truth_output,
        pred_output=business_output,
        parse_error=parse_result.error,
        schema_error=schema_error,
        issue_codes=issue_codes,
    )


def error_case_record(score: CaseScore) -> dict[str, Any]:
    return {
        "case_id": score.case_id,
        "issues": score.issue_codes,
        "parse_error": score.parse_error,
        "schema_error": score.schema_error,
        "truth_risk_level": score.truth_output.get("risk_level"),
        "pred_risk_level": score.pred_output.get("risk_level") if score.pred_output else None,
        "truth_audit_result": score.truth_output.get("audit_result"),
        "pred_audit_result": score.pred_output.get("audit_result") if score.pred_output else None,
    }


def aggregate_case_scores(scores: list[CaseScore]) -> dict[str, Any]:
    total_cases = len(scores)
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
    for score in scores:
        for key, value in score.evidence.items():
            evidence_accumulator[key] += value

    truth_outputs = [score.truth_output for score in scores]
    pred_outputs = [score.pred_output for score in scores]
    truth_risk = [output["risk_level"] for output in truth_outputs]
    pred_risk = [output.get("risk_level") if output else None for output in pred_outputs]
    return {
        "total_cases": total_cases,
        "json_validity": _rate(sum(int(score.json_valid) for score in scores), total_cases),
        "schema_compliance": _rate(sum(int(score.schema_valid) for score in scores), total_cases),
        "field_em": _rate(sum(score.field_correct for score in scores), sum(score.field_total for score in scores)),
        "consistency_em": _rate(
            sum(score.consistency_correct for score in scores),
            sum(score.consistency_total for score in scores),
        ),
        "risk_type_macro_f1": macro_f1(truth_risk, pred_risk, RISK_LEVELS),
        "audit_accuracy": audit_accuracy(truth_outputs, pred_outputs),
        "high_risk_miss_rate": high_risk_miss_rate(truth_outputs, pred_outputs),
        "false_manual_review_rate": false_manual_review_rate(truth_outputs, pred_outputs),
        "evidence_support_rate": _rate(evidence_accumulator["support_correct"], evidence_accumulator["support_total"]),
        "evidence_value_accuracy": _rate(evidence_accumulator["value_correct"], evidence_accumulator["value_total"]),
        "evidence_source_accuracy": _rate(evidence_accumulator["source_correct"], evidence_accumulator["source_total"]),
        "evidence_bbox_accuracy_strict": _rate(
            evidence_accumulator["bbox_strict_correct"], evidence_accumulator["bbox_total"]
        ),
        "evidence_bbox_accuracy_relaxed": _rate(
            evidence_accumulator["bbox_relaxed_correct"], evidence_accumulator["bbox_total"]
        ),
        "hallucination_rate": _rate(
            sum(score.hallucinated for score in scores),
            sum(score.hallucination_total for score in scores),
        ),
        "error_cases": sum(1 for score in scores if score.issue_codes),
    }
