"""Rule-based reward function for Phase 08 GRPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.consistency_metrics import consistency_exact_counts
from mv_audit.evaluation.evidence_metrics import evidence_counts
from mv_audit.evaluation.field_metrics import field_exact_counts
from mv_audit.evaluation.hallucination_metrics import hallucination_count
from mv_audit.evaluation.json_parser import parse_json_output
from mv_audit.utils import read_yaml


DETAIL_KEYS = [
    "r_field",
    "r_consistency",
    "r_anomaly",
    "r_audit",
    "r_evidence",
    "r_json",
    "r_uncertainty",
    "p_hallucination",
    "p_high_risk_miss",
]


def _empty_details() -> dict[str, float]:
    return {key: 0.0 for key in DETAIL_KEYS}


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _schema_from_input(output_schema: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(output_schema, dict):
        return output_schema
    return read_yaml(output_schema)


def _schema_ok(output: dict[str, Any] | None, output_schema: dict[str, Any]) -> bool:
    if output is None:
        return False
    validator = Draft202012Validator(output_schema)
    return not list(validator.iter_errors(output))


def _ground_truth_output(ground_truth: dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth.get("output"), dict):
        return ground_truth["output"]
    return ground_truth


def _set_score(truth_items: list[Any], pred_items: list[Any]) -> float:
    return 1.0 if set(truth_items) == set(pred_items) else 0.0


def _uncertainty_score(truth: dict[str, Any], pred: dict[str, Any]) -> float:
    truth_uncertainty = truth.get("uncertainty") or {}
    pred_uncertainty = pred.get("uncertainty") or {}
    checks = [
        truth_uncertainty.get("has_uncertain_fields") == pred_uncertainty.get("has_uncertain_fields"),
        set(truth_uncertainty.get("uncertain_fields") or []) == set(pred_uncertainty.get("uncertain_fields") or []),
        truth_uncertainty.get("requires_manual_review") == pred_uncertainty.get("requires_manual_review"),
    ]
    return sum(1 for item in checks if item) / len(checks)


def _high_risk_miss(truth: dict[str, Any], pred: dict[str, Any] | None) -> float:
    if truth.get("risk_level") != "high":
        return 0.0
    if pred is None:
        return 1.0
    if pred.get("risk_level") != "high" or pred.get("audit_result") == "pass":
        return 1.0
    return 0.0


def _weighted_reward(details: dict[str, float]) -> float:
    raw = (
        0.15 * details["r_field"]
        + 0.15 * details["r_consistency"]
        + 0.20 * details["r_anomaly"]
        + 0.15 * details["r_audit"]
        + 0.15 * details["r_evidence"]
        + 0.10 * details["r_json"]
        + 0.10 * details["r_uncertainty"]
        - 0.20 * details["p_hallucination"]
        - 0.40 * details["p_high_risk_miss"]
    )
    return _clip(raw)


def score_output(
    raw_output: str,
    ground_truth: dict[str, Any],
    image_items: list[dict[str, Any]],
    output_schema: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Score one raw model completion against the evidence-grounded answer."""

    schema = _schema_from_input(output_schema)
    truth = _ground_truth_output(ground_truth)
    details = _empty_details()
    parse_result = parse_json_output(raw_output)
    if not parse_result.json_validity:
        return {"reward": -1.0, "details": details}

    pred = parse_result.output
    details["r_json"] = 1.0
    if not _schema_ok(pred, schema):
        return {"reward": -1.0, "details": details}
    assert pred is not None

    if truth.get("risk_level") == "high" and pred.get("audit_result") == "pass":
        details["p_high_risk_miss"] = 1.0
        return {"reward": -1.0, "details": details}

    if "missing_document" in set(truth.get("anomaly_types") or []) and pred.get("audit_result") == "pass":
        details["p_high_risk_miss"] = 1.0 if truth.get("risk_level") == "high" else 0.0
        return {"reward": -1.0, "details": details}

    field_correct, field_total = field_exact_counts(truth, pred, schema_ok=True)
    consistency_correct, consistency_total = consistency_exact_counts(truth, pred, schema_ok=True)
    evidence_case = evidence_counts(truth, pred, schema_ok=True)
    hallucinated, hallucination_total = hallucination_count(
        truth,
        pred,
        schema_ok=True,
        image_items=image_items,
    )
    evidence_subscores = [
        _rate(evidence_case["support_correct"], evidence_case["support_total"]),
        _rate(evidence_case["value_correct"], evidence_case["value_total"]),
        _rate(evidence_case["source_correct"], evidence_case["source_total"]),
        _rate(evidence_case["bbox_relaxed_correct"], evidence_case["bbox_total"]),
    ]

    details.update(
        {
            "r_field": _rate(field_correct, field_total),
            "r_consistency": _rate(consistency_correct, consistency_total),
            "r_anomaly": _set_score(truth.get("anomaly_types") or [], pred.get("anomaly_types") or []),
            "r_audit": 1.0
            if truth.get("risk_level") == pred.get("risk_level")
            and truth.get("audit_result") == pred.get("audit_result")
            else 0.0,
            "r_evidence": sum(evidence_subscores) / len(evidence_subscores),
            "r_uncertainty": _uncertainty_score(truth, pred),
            "p_hallucination": _rate(hallucinated, hallucination_total),
            "p_high_risk_miss": _high_risk_miss(truth, pred),
        }
    )
    return {"reward": _weighted_reward(details), "details": details}


def reward_for_grpo(
    *,
    completions: list[str],
    ground_truths: list[dict[str, Any]],
    image_items: list[list[dict[str, Any]]],
    output_schema: dict[str, Any] | str | Path,
) -> list[float]:
    """Return scalar rewards for a TRL-style batch."""

    return [
        float(score_output(completion, truth, images, output_schema)["reward"])
        for completion, truth, images in zip(completions, ground_truths, image_items, strict=True)
    ]
