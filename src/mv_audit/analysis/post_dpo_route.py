"""Route gates after DPO v2 and high-risk repair experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mv_audit.utils import ensure_dir, iter_jsonl, read_yaml

DEFAULT_CONFIG = "configs/gates/post_dpo_route_v1.yaml"


METRIC_KEYS = [
    "json_validity",
    "schema_compliance",
    "audit_accuracy",
    "high_risk_miss_rate",
    "evidence_support_rate",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def _to_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("Missing numeric metric value")
    return float(value)


def load_metric_row(path: str | Path, *, model_id: str | None = None, scope: str | None = None) -> dict[str, Any]:
    metric_path = Path(path)
    if metric_path.suffix.lower() == ".json":
        row = _read_json(metric_path)
    elif metric_path.suffix.lower() == ".csv":
        with metric_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No metric rows found in {metric_path}")
        candidates = rows
        if model_id is not None:
            candidates = [row for row in candidates if row.get("model_id") == model_id]
        if scope is not None:
            candidates = [row for row in candidates if row.get("scope") == scope or row.get("split") == scope]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one metric row in {metric_path}, got {len(candidates)}; "
                "pass --model-id and/or --scope to disambiguate."
            )
        row = dict(candidates[0])
    else:
        raise ValueError(f"Unsupported metric file suffix: {metric_path}")

    normalized = dict(row)
    for key in METRIC_KEYS:
        if key in row:
            normalized[key] = _to_float(row[key])
    if "total_cases" in row:
        normalized["total_cases"] = int(float(row["total_cases"]))
    if "error_cases" in row:
        normalized["error_cases"] = float(row["error_cases"])
    return normalized


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return float(candidate[key]) - float(baseline[key])


def evaluate_repair_gate(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
    gate_name: str,
) -> dict[str, Any]:
    cfg = dict(config[gate_name])
    deltas = {key: _delta(candidate, baseline, key) for key in METRIC_KEYS if key in baseline and key in candidate}
    checks = {
        "json_validity_delta": deltas.get("json_validity", 0.0) >= float(cfg.get("json_validity_delta_min", -1.0)),
        "schema_compliance_delta": deltas.get("schema_compliance", 0.0)
        >= float(cfg.get("schema_compliance_delta_min", -1.0)),
        "audit_accuracy_delta": deltas.get("audit_accuracy", 0.0) >= float(cfg.get("audit_accuracy_delta_min", -1.0)),
        "evidence_support_delta": deltas.get("evidence_support_rate", 0.0)
        >= float(cfg.get("evidence_support_delta_min", -1.0)),
    }
    if "high_risk_miss_improvement_min" in cfg:
        improvement = float(baseline["high_risk_miss_rate"]) - float(candidate["high_risk_miss_rate"])
        checks["high_risk_miss_improvement"] = improvement >= float(cfg["high_risk_miss_improvement_min"])
    else:
        checks["high_risk_miss_delta"] = deltas.get("high_risk_miss_rate", 0.0) <= float(
            cfg.get("high_risk_miss_delta_max", 1.0)
        )
    passed = all(checks.values())
    return {
        "gate": gate_name,
        "status": f"{gate_name.upper()}_PASS" if passed else f"{gate_name.upper()}_FAIL",
        "passed": passed,
        "checks": checks,
        "deltas": deltas,
        "baseline": {key: baseline.get(key) for key in METRIC_KEYS if key in baseline},
        "candidate": {key: candidate.get(key) for key in METRIC_KEYS if key in candidate},
    }


def summarize_attributions(path: str | Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    error_rows = [row for row in rows if row.get("error_tags") or row.get("problem_classes")]
    decision_candidates = [row for row in rows if row.get("decision_candidate")]
    primary_counts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}
    for row in rows:
        primary = row.get("primary_error_source")
        if primary:
            primary_counts[str(primary)] = primary_counts.get(str(primary), 0) + 1
        problem = row.get("primary_problem_class")
        if problem:
            problem_counts[str(problem)] = problem_counts.get(str(problem), 0) + 1
    total_errors = len(error_rows)
    return {
        "total_rows": len(rows),
        "residual_error_cases": total_errors,
        "decision_candidate_cases": len(decision_candidates),
        "decision_candidate_ratio": len(decision_candidates) / total_errors if total_errors else 0.0,
        "primary_error_source_counts": primary_counts,
        "primary_problem_class_counts": problem_counts,
        "decision_primary_ratio": primary_counts.get("decision_error", 0) / total_errors if total_errors else 0.0,
    }


def evaluate_rl_decision(*, main_gate: dict[str, Any], attribution_summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("rl_decision", {}))
    min_error_cases = int(cfg.get("min_residual_error_cases", 1))
    checks = {
        "repair_main_gate_passed": bool(main_gate["passed"]),
        "enough_residual_errors": int(attribution_summary["residual_error_cases"]) >= min_error_cases,
        "decision_candidate_ratio": float(attribution_summary["decision_candidate_ratio"])
        >= float(cfg.get("min_decision_candidate_ratio", 0.5)),
        "decision_primary_ratio": float(attribution_summary["decision_primary_ratio"])
        >= float(cfg.get("min_decision_primary_ratio", 0.5)),
    }
    recommended = all(checks.values())
    return {
        "status": "READY_FOR_RL" if recommended else "RL_NOT_RECOMMENDED",
        "rl_recommended": recommended,
        "checks": checks,
        "main_gate": main_gate,
        "attribution_summary": attribution_summary,
        "next_step": "GRPO_COMPATIBILITY_SMOKE" if recommended else "STOP_OR_REPAIR_DIAGNOSIS",
    }


def assert_ready_for_rl(path: str | Path) -> None:
    decision = _read_json(path)
    if decision.get("status") != "READY_FOR_RL" or decision.get("rl_recommended") is not True:
        raise SystemExit(
            f"RL is not ready: {path} has status={decision.get('status')} "
            f"rl_recommended={decision.get('rl_recommended')}"
        )
    print(json.dumps({"status": "READY_FOR_RL", "decision_file": str(path)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--candidate-metrics")
    parser.add_argument("--baseline-model-id")
    parser.add_argument("--candidate-model-id")
    parser.add_argument("--baseline-scope")
    parser.add_argument("--candidate-scope")
    parser.add_argument("--attribution-jsonl")
    parser.add_argument("--mode", choices=["repair_fast", "repair_main", "rl_decision"], default="repair_main")
    parser.add_argument("--output")
    parser.add_argument("--assert-ready-for-rl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.assert_ready_for_rl:
        assert_ready_for_rl(args.assert_ready_for_rl)
        return
    if not args.baseline_metrics or not args.candidate_metrics:
        raise SystemExit("--baseline-metrics and --candidate-metrics are required unless --assert-ready-for-rl is used")
    config = read_yaml(args.config)
    baseline = load_metric_row(args.baseline_metrics, model_id=args.baseline_model_id, scope=args.baseline_scope)
    candidate = load_metric_row(args.candidate_metrics, model_id=args.candidate_model_id, scope=args.candidate_scope)
    gate_name = "repair_fast_gate" if args.mode == "repair_fast" else "repair_main_gate"
    gate = evaluate_repair_gate(baseline=baseline, candidate=candidate, config=config, gate_name=gate_name)
    payload = gate
    if args.mode == "rl_decision":
        if not args.attribution_jsonl:
            raise SystemExit("--attribution-jsonl is required for rl_decision")
        payload = evaluate_rl_decision(
            main_gate=gate,
            attribution_summary=summarize_attributions(args.attribution_jsonl),
            config=config,
        )
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()