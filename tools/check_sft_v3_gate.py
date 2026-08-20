from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE = {
    "json_validity": 1.0,
    "schema_compliance": 1.0,
    "audit_accuracy": 0.9605263157894737,
    "high_risk_miss_rate": 0.06896551724137931,
    "evidence_support_rate": 0.988527724665392,
}
OLD_ORDER_ID_MISSES = {
    "MV_MAIN_025856",
    "MV_MAIN_003978",
    "MV_MAIN_019752",
    "MV_MAIN_016083",
    "MV_MAIN_030493",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate SFT v3 before model mining.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    errors = _read_jsonl(args.errors)
    remaining = {
        str(row.get("case_id"))
        for row in errors
        if str(row.get("case_id")) in OLD_ORDER_ID_MISSES
        and "high_risk_miss" in set(row.get("issues") or [])
    }
    fixed = sorted(OLD_ORDER_ID_MISSES - remaining)
    safety_checks = {
        "json_validity_is_one": metrics.get("json_validity") == 1.0,
        "schema_compliance_is_one": metrics.get("schema_compliance") == 1.0,
        "audit_drop_within_0_01": metrics.get("audit_accuracy", 0.0)
        >= BASELINE["audit_accuracy"] - 0.01,
        "evidence_drop_within_0_01": metrics.get("evidence_support_rate", 0.0)
        >= BASELINE["evidence_support_rate"] - 0.01,
    }
    target_checks = {"old_order_id_fixed_at_least_three": len(fixed) >= 3}
    dpo_allowed = (
        all(safety_checks.values())
        and metrics.get("high_risk_miss_rate", 1.0) < BASELINE["high_risk_miss_rate"]
    )
    if dpo_allowed and all(target_checks.values()):
        status = "PASS"
    elif dpo_allowed:
        status = "PASS_WITH_TARGET_WARNING"
    else:
        status = "FAIL"
    report = {
        "status": status,
        "dpo_allowed": dpo_allowed,
        "safety_checks": safety_checks,
        "target_checks": target_checks,
        "baseline": BASELINE,
        "candidate": metrics,
        "old_order_id_fixed": fixed,
        "old_order_id_remaining": sorted(remaining),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if dpo_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())