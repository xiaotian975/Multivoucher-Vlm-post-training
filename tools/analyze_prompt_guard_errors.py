"""Analyze prompt-guard Repair SFT v2 train_decode_dev errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OLD_ORDER_ID_CASES = {
    "MV_MAIN_003978",
    "MV_MAIN_016083",
    "MV_MAIN_019752",
    "MV_MAIN_025856",
    "MV_MAIN_030493",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_raw_output(row: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(str(row["raw_output"]))
    if not isinstance(output, dict):
        raise ValueError(f"Prediction output is not an object: {row.get('case_id')}")
    return output


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return answer
    raise ValueError(f"Ground truth row is missing answer object: {row.get('case_id')}")


def _order_id_evidence(output: dict[str, Any]) -> list[tuple[str, str]]:
    evidence = output.get("evidence") or []
    pairs: list[tuple[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("field") == "order_id":
            pairs.append((str(item.get("source_doc_type") or ""), str(item.get("value") or "")))
    return pairs


def _classify(row: dict[str, Any], truth: dict[str, Any], pred: dict[str, Any]) -> str:
    issues = set(row.get("issues") or [])
    if issues == {"bbox_strict_error"}:
        return "bbox_only"
    if "high_risk_miss" not in issues:
        return "other"
    pred_anomalies = set(pred.get("anomaly_types") or [])
    pred_consistency = pred.get("consistency_check") or {}
    if not _order_id_evidence(pred) and "order_id_mismatch" in set(truth.get("anomaly_types") or []):
        return "missing_order_id_evidence"
    if "order_id_mismatch" in pred_anomalies or pred_consistency.get("order_id_consistent") is False:
        return "recognized_risk_but_decision_released"
    return "model_missed_high_risk"


def analyze(
    *,
    metrics_path: Path,
    errors_path: Path,
    predictions_path: Path,
    ground_truth_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors = _read_jsonl(errors_path)
    predictions = {row["case_id"]: _parse_raw_output(row) for row in _read_jsonl(predictions_path)}
    ground_truth = {row["case_id"]: _answer(row) for row in _read_jsonl(ground_truth_path)}

    records: list[dict[str, Any]] = []
    for error in errors:
        case_id = str(error["case_id"])
        truth = ground_truth[case_id]
        pred = predictions[case_id]
        record = {
            "case_id": case_id,
            "issues": list(error.get("issues") or []),
            "problem_class": _classify(error, truth, pred),
            "is_old_order_id_case": case_id in OLD_ORDER_ID_CASES,
            "truth_anomaly_types": truth.get("anomaly_types") or [],
            "pred_anomaly_types": pred.get("anomaly_types") or [],
            "truth_risk_level": truth.get("risk_level"),
            "pred_risk_level": pred.get("risk_level"),
            "truth_audit_result": truth.get("audit_result"),
            "pred_audit_result": pred.get("audit_result"),
            "truth_order_id_consistent": (truth.get("consistency_check") or {}).get("order_id_consistent"),
            "pred_order_id_consistent": (pred.get("consistency_check") or {}).get("order_id_consistent"),
            "truth_order_id_evidence": _order_id_evidence(truth),
            "pred_order_id_evidence": _order_id_evidence(pred),
            "pred_reason": pred.get("reason"),
        }
        records.append(record)

    class_counter = Counter(record["problem_class"] for record in records)
    issue_counter = Counter(issue for row in errors for issue in row.get("issues") or [])
    high_risk_misses = [record for record in records if "high_risk_miss" in set(record["issues"])]
    decision_candidates = [
        record for record in high_risk_misses if record["problem_class"] == "recognized_risk_but_decision_released"
    ]
    rl_recommended = bool(high_risk_misses) and len(decision_candidates) / len(high_risk_misses) >= 0.60

    return {
        "metrics": metrics,
        "total_error_rows": len(errors),
        "issue_counter": dict(issue_counter),
        "problem_class_counter": dict(class_counter),
        "high_risk_miss_count": len(high_risk_misses),
        "decision_candidate_count": len(decision_candidates),
        "rl_recommended": rl_recommended,
        "rl_decision_status": "READY_FOR_RL" if rl_recommended else "NOT_READY_FOR_RL",
        "old_order_id_case_status": [record for record in records if record["case_id"] in OLD_ORDER_ID_CASES],
        "error_records": records,
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Prompt-Guard Repair SFT v2 Error Attribution",
        "",
        "## Metrics",
        "",
        f"- total_cases: {metrics.get('total_cases')}",
        f"- audit_accuracy: {metrics.get('audit_accuracy')}",
        f"- high_risk_miss_rate: {metrics.get('high_risk_miss_rate')}",
        f"- evidence_support_rate: {metrics.get('evidence_support_rate')}",
        f"- error_cases: {metrics.get('error_cases')}",
        "",
        "## Attribution",
        "",
        f"- issue_counter: {report['issue_counter']}",
        f"- problem_class_counter: {report['problem_class_counter']}",
        f"- high_risk_miss_count: {report['high_risk_miss_count']}",
        f"- decision_candidate_count: {report['decision_candidate_count']}",
        f"- rl_recommended: {str(report['rl_recommended']).lower()}",
        f"- rl_decision_status: {report['rl_decision_status']}",
        "",
        "## High-Risk Misses",
        "",
        "| case_id | old_order_id | class | truth | pred | pred_order_id_evidence |",
        "|---|---:|---|---|---|---|",
    ]
    for record in report["error_records"]:
        if "high_risk_miss" not in set(record["issues"]):
            continue
        lines.append(
            "| {case_id} | {old} | {problem_class} | {truth_risk}/{truth_audit}/{truth_anomaly} | "
            "{pred_risk}/{pred_audit}/{pred_anomaly} | {pred_evidence} |".format(
                case_id=record["case_id"],
                old=record["is_old_order_id_case"],
                problem_class=record["problem_class"],
                truth_risk=record["truth_risk_level"],
                truth_audit=record["truth_audit_result"],
                truth_anomaly=record["truth_anomaly_types"],
                pred_risk=record["pred_risk_level"],
                pred_audit=record["pred_audit_result"],
                pred_anomaly=record["pred_anomaly_types"],
                pred_evidence=record["pred_order_id_evidence"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Prompt guard did not change the business metrics versus Repair SFT v2.",
            "- Residual high-risk misses are missing-order-id-evidence failures, not decision-preference failures.",
            "- Formal GRPO is not recommended from this state; use Order-ID Structured Repair v3 first.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt-guard Repair SFT v2 errors.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(
        metrics_path=Path(args.metrics),
        errors_path=Path(args.errors),
        predictions_path=Path(args.predictions),
        ground_truth_path=Path(args.ground_truth),
    )
    (output_dir / "prompt_guard_error_attribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_guard_error_attribution.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["problem_class_counter", "rl_recommended", "rl_decision_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
