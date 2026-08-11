"""Error migration analysis for DPO variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.evidence_metrics import evidence_counts
from mv_audit.evaluation.hallucination_metrics import hallucination_count
from mv_audit.evaluation.json_parser import parse_json_output
from mv_audit.utils import ensure_dir, read_jsonl, read_yaml


def _extract_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and isinstance(ground_truth.get("output"), dict):
        return ground_truth["output"]
    if isinstance(row.get("output"), dict):
        return row["output"]
    raise ValueError(f"Cannot find ground truth for {row.get('case_id')}")


def _schema_ok(output: dict[str, Any] | None, validator: Draft202012Validator) -> bool:
    return output is not None and not list(validator.iter_errors(output))


def _parsed_prediction(row: dict[str, Any] | None, validator: Draft202012Validator) -> tuple[dict[str, Any] | None, list[str]]:
    parse = parse_json_output(row.get("raw_output", "") if row else "")
    issues: list[str] = []
    if not parse.json_validity:
        issues.append("json_invalid")
    if not _schema_ok(parse.output, validator):
        issues.append("schema_invalid")
        return None, issues
    return parse.output, issues


def _case_issues(
    *,
    truth: dict[str, Any],
    pred: dict[str, Any] | None,
    base_issues: list[str],
    images: list[dict[str, Any]],
) -> list[str]:
    issues = list(base_issues)
    if pred is None:
        issues.append("business_metrics_zeroed")
        return sorted(set(issues))
    if pred.get("audit_result") != truth.get("audit_result"):
        issues.append("audit_mismatch")
    if pred.get("risk_level") != truth.get("risk_level"):
        issues.append("risk_mismatch")
    if truth.get("risk_level") == "high" and (pred.get("risk_level") != "high" or pred.get("audit_result") == "pass"):
        issues.append("high_risk_miss")
    evidence = evidence_counts(truth, pred, schema_ok=True)
    if evidence["support_correct"] < evidence["support_total"]:
        issues.append("unsupported_evidence")
    if evidence["bbox_strict_correct"] < evidence["bbox_total"]:
        issues.append("bbox_strict_error")
    hallu_count, _ = hallucination_count(truth, pred, schema_ok=True, image_items=images)
    if hallu_count:
        issues.append("hallucination")
    return sorted(set(issues))


def _audit_correct(truth: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    return pred is not None and pred.get("audit_result") == truth.get("audit_result")


def _transition(base_correct: bool, candidate_correct: bool) -> str:
    if base_correct and candidate_correct:
        return "both_correct"
    if base_correct and not candidate_correct:
        return "baseline_correct_candidate_wrong"
    if not base_correct and candidate_correct:
        return "baseline_wrong_candidate_correct"
    return "both_wrong"


def _rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_se(values: list[int], *, samples: int, seed: int) -> float:
    if not values or samples <= 0:
        return 0.0
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(_rate(draw))
    mean = sum(estimates) / len(estimates)
    variance = sum((value - mean) ** 2 for value in estimates) / max(1, len(estimates) - 1)
    return variance**0.5


def analyze(
    *,
    ground_truth_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    output_schema: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validator = Draft202012Validator(output_schema)
    baseline_by_case = {str(row["case_id"]): row for row in baseline_rows}
    candidate_by_case = {str(row["case_id"]): row for row in candidate_rows}
    case_rows: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    issue_delta: Counter[str] = Counter()
    base_correct_values: list[int] = []
    candidate_correct_values: list[int] = []
    base_high_risk_miss_values: list[int] = []
    candidate_high_risk_miss_values: list[int] = []

    for row in ground_truth_rows:
        case_id = str(row["case_id"])
        truth = _extract_ground_truth(row)
        images = list(row.get("images") or [])
        baseline_pred, baseline_base_issues = _parsed_prediction(baseline_by_case.get(case_id), validator)
        candidate_pred, candidate_base_issues = _parsed_prediction(candidate_by_case.get(case_id), validator)
        baseline_issues = _case_issues(truth=truth, pred=baseline_pred, base_issues=baseline_base_issues, images=images)
        candidate_issues = _case_issues(truth=truth, pred=candidate_pred, base_issues=candidate_base_issues, images=images)
        base_correct = _audit_correct(truth, baseline_pred)
        candidate_correct = _audit_correct(truth, candidate_pred)
        transition = _transition(base_correct, candidate_correct)
        transition_counts[transition] += 1
        for issue in candidate_issues:
            issue_delta[issue] += 1
        for issue in baseline_issues:
            issue_delta[issue] -= 1
        is_high = truth.get("risk_level") == "high"
        base_high_miss = int(is_high and (baseline_pred is None or baseline_pred.get("risk_level") != "high" or baseline_pred.get("audit_result") == "pass"))
        candidate_high_miss = int(
            is_high and (candidate_pred is None or candidate_pred.get("risk_level") != "high" or candidate_pred.get("audit_result") == "pass")
        )
        base_correct_values.append(int(base_correct))
        candidate_correct_values.append(int(candidate_correct))
        if is_high:
            base_high_risk_miss_values.append(base_high_miss)
            candidate_high_risk_miss_values.append(candidate_high_miss)
        case_rows.append(
            {
                "case_id": case_id,
                "primary_anomaly_type": row.get("primary_anomaly_type")
                or truth.get("primary_anomaly_type")
                or "unknown",
                "risk_level_gt": truth.get("risk_level"),
                "audit_result_gt": truth.get("audit_result"),
                "baseline_correct": int(base_correct),
                "candidate_correct": int(candidate_correct),
                "transition": transition,
                "baseline_issues": "|".join(baseline_issues),
                "candidate_issues": "|".join(candidate_issues),
                "baseline_high_risk_miss": base_high_miss,
                "candidate_high_risk_miss": candidate_high_miss,
            }
        )

    baseline_accuracy = _rate(base_correct_values)
    candidate_accuracy = _rate(candidate_correct_values)
    baseline_high_miss = _rate(base_high_risk_miss_values)
    candidate_high_miss = _rate(candidate_high_risk_miss_values)
    summary = {
        "total_cases": len(case_rows),
        "transition_counts": dict(transition_counts),
        "issue_delta_candidate_minus_baseline": dict(sorted(issue_delta.items())),
        "audit_accuracy": {
            "baseline": baseline_accuracy,
            "candidate": candidate_accuracy,
            "delta": candidate_accuracy - baseline_accuracy,
            "baseline_bootstrap_se": _bootstrap_se(base_correct_values, samples=bootstrap_samples, seed=seed),
            "candidate_bootstrap_se": _bootstrap_se(candidate_correct_values, samples=bootstrap_samples, seed=seed + 1),
        },
        "high_risk_miss_rate": {
            "baseline": baseline_high_miss,
            "candidate": candidate_high_miss,
            "delta": candidate_high_miss - baseline_high_miss,
            "baseline_bootstrap_se": _bootstrap_se(base_high_risk_miss_values, samples=bootstrap_samples, seed=seed + 2),
            "candidate_bootstrap_se": _bootstrap_se(candidate_high_risk_miss_values, samples=bootstrap_samples, seed=seed + 3),
        },
        "leakage_note": "This script is for frozen evaluation outputs only; do not use Val/Test transitions to build DPO v2 training pairs.",
    }
    return case_rows, summary


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    fields = [
        "case_id",
        "primary_anomaly_type",
        "risk_level_gt",
        "audit_result_gt",
        "baseline_correct",
        "candidate_correct",
        "transition",
        "baseline_issues",
        "candidate_issues",
        "baseline_high_risk_miss",
        "candidate_high_risk_miss",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DPO error migration between baseline and candidate predictions.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--baseline_predictions", required=True)
    parser.add_argument("--candidate_predictions", required=True)
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = analyze(
        ground_truth_rows=read_jsonl(args.ground_truth),
        baseline_rows=read_jsonl(args.baseline_predictions),
        candidate_rows=read_jsonl(args.candidate_predictions),
        output_schema=read_yaml(args.output_schema),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_csv(rows, args.output_csv)
    summary_path = Path(args.summary_output)
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
