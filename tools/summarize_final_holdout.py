"""Summarize final holdout metrics and mark the holdout as consumed."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _problem_classes(row: dict[str, Any]) -> list[str]:
    issues = set(row.get("issues") or [])
    classes: list[str] = []
    if {"json_invalid", "schema_invalid", "business_metrics_zeroed"} & issues:
        classes.append("schema_contract_failure")
    pred_risk = row.get("pred_risk_level")
    pred_audit = row.get("pred_audit_result")
    truth_risk = row.get("truth_risk_level")
    truth_audit = row.get("truth_audit_result")
    if truth_risk == "high" and (pred_risk != "high" or pred_audit == "pass"):
        classes.append("model_missed_high_risk")
    if truth_risk == "high" and pred_risk == "high" and truth_audit != pred_audit:
        classes.append("recognized_risk_but_decision_released")
    if {"unsupported_evidence", "bbox_strict_error", "hallucination"} & issues:
        classes.append("evidence_not_trustworthy")
    if {"audit_mismatch", "risk_mismatch"} & issues:
        classes.append("decision_or_risk_mismatch")
    return classes or ["other"]


def _error_attribution(rows: list[dict[str, str]], errors_dir: Path) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    risk_transitions: Counter[str] = Counter()
    audit_transitions: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    per_split: dict[str, Any] = {}

    for row in rows:
        model_id = row["model_id"]
        split = row["split"]
        error_path = errors_dir / f"{model_id}_{split}_errors.jsonl"
        if not error_path.exists():
            raise FileNotFoundError(f"Missing final holdout error cases: {error_path}")
        error_rows = _read_jsonl(error_path)
        per_split[split] = {"error_file": str(error_path), "error_rows": len(error_rows)}
        for error in error_rows:
            case_id = str(error.get("case_id") or "")
            for issue in error.get("issues") or []:
                issue_counts[str(issue)] += 1
            risk_transitions[f"{error.get('truth_risk_level')} -> {error.get('pred_risk_level')}"] += 1
            audit_transitions[f"{error.get('truth_audit_result')} -> {error.get('pred_audit_result')}"] += 1
            for problem_class in _problem_classes(error):
                class_counts[problem_class] += 1
                if case_id and len(examples[problem_class]) < 10:
                    examples[problem_class].append(case_id)

    return {
        "per_split": per_split,
        "issue_counts": dict(issue_counts.most_common()),
        "problem_class_counts": dict(class_counts.most_common()),
        "truth_to_pred_risk_counts": dict(risk_transitions.most_common()),
        "truth_to_pred_audit_counts": dict(audit_transitions.most_common()),
        "examples_by_problem_class": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final_holdout_v1 metrics.")
    parser.add_argument("--manifest", default="data/mv_audit/final_holdout_v1/final_holdout_v1_manifest.json")
    parser.add_argument("--metrics_summary", default="outputs/eval_reports/final_holdout_v1/repair_sft_r3/metrics_summary.csv")
    parser.add_argument("--errors_dir", default=None)
    parser.add_argument("--output_dir", default="docs/experiments/final_holdout_v1")
    parser.add_argument("--consume_marker", default="data/mv_audit/final_holdout_v1/FINAL_HOLDOUT_CONSUMED")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    metrics_path = Path(args.metrics_summary)
    errors_dir = Path(args.errors_dir) if args.errors_dir else metrics_path.parent
    output_dir = Path(args.output_dir)
    rows = _read_summary(metrics_path)
    if len(rows) != 4:
        raise ValueError(f"Expected 4 split metric rows, found {len(rows)} in {metrics_path}.")

    fields = [
        "total_cases",
        "json_validity",
        "schema_compliance",
        "audit_accuracy",
        "high_risk_miss_rate",
        "evidence_support_rate",
        "error_cases",
    ]
    total_cases = sum(_float(row, "total_cases") for row in rows)
    if int(total_cases) != 1000:
        raise ValueError(f"Expected 1000 final holdout cases, found {total_cases}.")

    weighted = {
        key: sum(_float(row, key) * _float(row, "total_cases") for row in rows) / total_cases
        for key in fields
        if key not in {"total_cases", "error_cases"}
    }
    error_cases = sum(_float(row, "error_cases") for row in rows)
    error_attribution = _error_attribution(rows, errors_dir)
    result = {
        "created_at": _now(),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "split_metrics": rows,
        "aggregate": {
            "total_cases": int(total_cases),
            **weighted,
            "error_cases": int(error_cases),
        },
        "error_attribution": error_attribution,
        "policy": "Final holdout consumed. Do not use these results to continue training or retune model selection.",
    }
    _write_json(output_dir / "final_holdout_result.json", result)
    _write_json(output_dir / "error_attribution_summary.json", error_attribution)

    lines = [
        "# Final Holdout v1 Summary",
        "",
        "This final holdout has been consumed. Results must not be used for further training or reward tuning.",
        "",
        "## Aggregate",
        "",
        "| total_cases | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {int(total_cases)} | {weighted['json_validity']:.4f} | {weighted['schema_compliance']:.4f} | "
            f"{weighted['audit_accuracy']:.4f} | {weighted['high_risk_miss_rate']:.4f} | "
            f"{weighted['evidence_support_rate']:.4f} | {int(error_cases)} |"
        ),
        "",
        "## Split Metrics",
        "",
        "| split | total_cases | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {int(float(row['total_cases']))} | {float(row['json_validity']):.4f} | "
            f"{float(row['schema_compliance']):.4f} | {float(row['audit_accuracy']):.4f} | "
            f"{float(row['high_risk_miss_rate']):.4f} | {float(row['evidence_support_rate']):.4f} | "
            f"{int(float(row['error_cases']))} |"
        )
    lines.extend(
        [
            "",
            "## Error Attribution",
            "",
            "Machine-readable attribution is written to `error_attribution_summary.json`.",
            "",
            "| problem_class | cases |",
            "| --- | ---: |",
        ]
    )
    for key, value in error_attribution["problem_class_counts"].items():
        lines.append(f"| {key} | {value} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    marker = Path(args.consume_marker)
    marker.write_text(f"consumed_at={_now()}\nmetrics_summary={metrics_path}\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
