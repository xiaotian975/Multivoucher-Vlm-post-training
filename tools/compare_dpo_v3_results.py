"""Compare SFT v3 and model-mined DPO v3 train_decode_dev results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mv_audit.utils import read_jsonl


OLD_ORDER_ID_MISSES = {
    "MV_MAIN_025856",
    "MV_MAIN_003978",
    "MV_MAIN_019752",
    "MV_MAIN_016083",
    "MV_MAIN_030493",
}


def _high_risk_misses(path: str) -> set[str]:
    misses = set()
    for row in read_jsonl(path):
        issues = row.get("issues")
        if issues is None:
            issues = row.get("issue_codes")
        if "high_risk_miss" in set(issues or []):
            misses.add(str(row["case_id"]))
    return misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--baseline_errors", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--candidate_errors", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_metrics).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    baseline_misses = _high_risk_misses(args.baseline_errors)
    candidate_misses = _high_risk_misses(args.candidate_errors)
    fixed = sorted(baseline_misses - candidate_misses)
    introduced = sorted(candidate_misses - baseline_misses)
    metrics = {
        "json_validity_delta": float(candidate["json_validity"]) - float(baseline["json_validity"]),
        "audit_accuracy_delta": float(candidate["audit_accuracy"]) - float(baseline["audit_accuracy"]),
        "high_risk_miss_rate_delta": float(candidate["high_risk_miss_rate"]) - float(baseline["high_risk_miss_rate"]),
        "false_manual_review_rate_delta": float(candidate["false_manual_review_rate"])
        - float(baseline["false_manual_review_rate"]),
        "evidence_support_rate_delta": float(candidate["evidence_support_rate"]) - float(baseline["evidence_support_rate"]),
        "schema_compliance_delta": float(candidate["schema_compliance"]) - float(baseline["schema_compliance"]),
    }
    effective = (
        selection.get("status") == "ELIGIBLE"
        and len(fixed) >= 2
        and not introduced
        and float(candidate["json_validity"]) == 1.0
        and float(candidate["schema_compliance"]) == 1.0
        and metrics["audit_accuracy_delta"] >= 0.0
        and metrics["false_manual_review_rate_delta"] <= (1.0 / float(candidate["total_cases"]))
        and metrics["evidence_support_rate_delta"] >= -0.01
        and (
            metrics["high_risk_miss_rate_delta"] <= -0.02
            or len(fixed) >= 2
        )
    )
    payload = {
        "status": "EFFECTIVE_ALIGNMENT" if effective else "ALIGNMENT_GATE_NOT_MET",
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "deltas": metrics,
        "baseline_high_risk_misses": sorted(baseline_misses),
        "candidate_high_risk_misses": sorted(candidate_misses),
        "fixed_high_risk_misses": fixed,
        "introduced_high_risk_misses": introduced,
        "old_order_id_misses_fixed": sorted(OLD_ORDER_ID_MISSES & set(fixed)),
        "checkpoint_selection": selection,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
