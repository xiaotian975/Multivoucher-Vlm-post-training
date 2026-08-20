"""Error attribution helpers for post-training v4 preparation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.case_scorer import CaseScore, score_case
from mv_audit.utils import read_jsonl, read_yaml, write_jsonl


DEFAULT_THRESHOLDS = {
    "field_score_min": 0.95,
    "anomaly_score_min": 0.90,
    "evidence_support_min": 1.0,
    "bbox_score_min": 1.0,
}

ERROR_PRIORITY = [
    "output_contract_error",
    "perception_error",
    "consistency_error",
    "decision_error",
    "evidence_error",
]

PROBLEM_CLASS_PRIORITY = [
    "schema_contract_failure",
    "model_missed_high_risk",
    "recognized_risk_but_decision_released",
    "evidence_not_trustworthy",
]


@dataclass(frozen=True)
class ErrorAttribution:
    case_id: str
    error_tags: list[str]
    primary_error_source: str | None
    problem_classes: list[str]
    primary_problem_class: str | None
    decision_candidate: bool
    scores: dict[str, float | bool | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "error_tags": self.error_tags,
            "primary_error_source": self.primary_error_source,
            "problem_classes": self.problem_classes,
            "primary_problem_class": self.primary_problem_class,
            "decision_candidate": self.decision_candidate,
            "scores": self.scores,
        }


def load_thresholds(config_path: str | Path | None = None) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if config_path is None:
        return thresholds
    path = Path(config_path)
    if not path.exists():
        return thresholds
    loaded = read_yaml(path)
    for key in DEFAULT_THRESHOLDS:
        if key in loaded:
            thresholds[key] = float(loaded[key])
    return thresholds


def attribute_score(score: CaseScore, *, thresholds: dict[str, float] | None = None) -> ErrorAttribution:
    cfg = thresholds or DEFAULT_THRESHOLDS
    tags: list[str] = []
    if not score.json_valid or not score.schema_valid:
        tags.append("output_contract_error")

    if score.schema_valid and score.field_score < float(cfg["field_score_min"]):
        tags.append("perception_error")

    if (
        score.schema_valid
        and score.field_score >= float(cfg["field_score_min"])
        and (score.anomaly_score < float(cfg["anomaly_score_min"]) or score.consistency_score < 1.0)
    ):
        tags.append("consistency_error")

    if (
        score.schema_valid
        and score.field_score >= float(cfg["field_score_min"])
        and score.anomaly_score >= float(cfg["anomaly_score_min"])
        and (not score.audit_correct or score.high_risk_miss)
    ):
        tags.append("decision_error")

    if score.audit_correct and score.evidence_support < float(cfg["evidence_support_min"]):
        tags.append("evidence_error")

    primary = next((tag for tag in ERROR_PRIORITY if tag in tags), None)
    problem_classes = classify_problem(score, thresholds=cfg)
    primary_problem_class = next((tag for tag in PROBLEM_CLASS_PRIORITY if tag in problem_classes), None)
    decision_candidate = (
        score.json_valid
        and score.schema_valid
        and score.field_score >= float(cfg["field_score_min"])
        and score.anomaly_score >= float(cfg["anomaly_score_min"])
        and (not score.audit_correct or score.high_risk_miss)
    )
    return ErrorAttribution(
        case_id=score.case_id,
        error_tags=tags,
        primary_error_source=primary,
        problem_classes=problem_classes,
        primary_problem_class=primary_problem_class,
        decision_candidate=decision_candidate,
        scores={
            "json_valid": score.json_valid,
            "schema_valid": score.schema_valid,
            "field_score": score.field_score,
            "anomaly_score": score.anomaly_score,
            "consistency_score": score.consistency_score,
            "audit_correct": score.audit_correct,
            "risk_correct": score.risk_correct,
            "high_risk_miss": score.high_risk_miss,
            "false_manual_review": score.false_manual_review,
            "false_escalation": score.false_escalation,
            "evidence_support": score.evidence_support,
            "hallucination": score.hallucination,
            "bbox_score": score.bbox_score,
        },
    )


def classify_problem(score: CaseScore, *, thresholds: dict[str, float] | None = None) -> list[str]:
    cfg = thresholds or DEFAULT_THRESHOLDS
    classes: list[str] = []
    if not score.json_valid or not score.schema_valid:
        classes.append("schema_contract_failure")

    if score.high_risk_miss:
        classes.append("model_missed_high_risk")

    truth_audit = score.truth_output.get("audit_result")
    pred_audit = score.pred_output.get("audit_result") if score.pred_output else None
    pred_risk = score.pred_output.get("risk_level") if score.pred_output else None
    if (
        score.schema_valid
        and score.truth_output.get("risk_level") == "high"
        and pred_risk == "high"
        and truth_audit == "reject_recommendation"
        and pred_audit in {"pass", "manual_review"}
    ):
        classes.append("recognized_risk_but_decision_released")

    bbox_score = score.bbox_score
    if (
        score.schema_valid
        and (
            score.evidence_support < float(cfg["evidence_support_min"])
            or (bbox_score is not None and bbox_score < float(cfg["bbox_score_min"]))
        )
    ):
        classes.append("evidence_not_trustworthy")

    return classes


def attribute_rows(
    *,
    ground_truth_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_schema: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    validator = Draft202012Validator(output_schema)
    predictions = {row["case_id"]: row for row in prediction_rows}
    return [
        attribute_score(score_case(row, predictions.get(row["case_id"]), validator=validator), thresholds=thresholds).to_dict()
        for row in ground_truth_rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute MultiVoucher-Audit errors on dev predictions.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--config", default="configs/analysis/error_attribution_v1.yaml")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = attribute_rows(
        ground_truth_rows=read_jsonl(args.ground_truth),
        prediction_rows=read_jsonl(args.predictions),
        output_schema=read_yaml(args.output_schema),
        thresholds=load_thresholds(args.config),
    )
    write_jsonl(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
