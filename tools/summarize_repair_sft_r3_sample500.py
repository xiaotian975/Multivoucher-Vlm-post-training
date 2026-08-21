"""Summarize the repair_sft_r3 diagnostic sample500 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


CORE_FIELDS = [
    "json_validity",
    "schema_compliance",
    "field_em",
    "risk_type_macro_f1",
    "audit_accuracy",
    "high_risk_miss_rate",
    "false_manual_review_rate",
    "evidence_support_rate",
    "hallucination_rate",
    "evidence_bbox_accuracy_relaxed",
    "error_cases",
]
SPLITS = ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]
HISTORICAL_MEANS = Path("docs/experiments/phase08_m3v2_sample500/metrics_by_model.csv")
HISTORICAL_SPLITS = Path("docs/experiments/phase08_m3v2_sample500/m2_m3_m3v2_split_metrics.csv")
FINAL_RESULT = Path("docs/experiments/final_holdout_v1/final_holdout_result.json")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_csv(source: Path, target: Path) -> None:
    rows = _read_csv(source)
    if not rows:
        raise ValueError(f"CSV is empty: {source}")
    _write_csv(target, rows, list(rows[0]))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _format(value: float) -> str:
    return f"{value:.4f}"


def _mean_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValueError(f"Expected 4 repair_sft_r3 split rows, found {len(rows)}")
    splits = {row["split"] for row in rows}
    if splits != set(SPLITS):
        raise ValueError(f"Unexpected split set: {sorted(splits)}")
    for row in rows:
        if row["model_id"] != "repair_sft_r3":
            raise ValueError(f"Unexpected model_id in diagnostic metrics: {row['model_id']}")
        if int(float(row["total_cases"])) != 500:
            raise ValueError(f"Expected 500 cases for {row['split']}, got {row['total_cases']}")

    mean: dict[str, Any] = {"model_id": "repair_sft_r3"}
    for field in CORE_FIELDS:
        mean[field] = sum(_float(row, field) for row in rows) / len(rows)
    return mean


def _delta_rows(r3_mean: dict[str, Any], historical: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in historical:
        rows.append(
            {
                "baseline_model_id": baseline["model_id"],
                "candidate_model_id": "repair_sft_r3",
                "audit_accuracy_delta": _float(r3_mean, "audit_accuracy") - _float(baseline, "audit_accuracy"),
                "high_risk_miss_rate_delta": _float(r3_mean, "high_risk_miss_rate")
                - _float(baseline, "high_risk_miss_rate"),
                "evidence_support_rate_delta": _float(r3_mean, "evidence_support_rate")
                - _float(baseline, "evidence_support_rate"),
                "schema_compliance_delta": _float(r3_mean, "schema_compliance")
                - _float(baseline, "schema_compliance"),
                "error_cases_delta": _float(r3_mean, "error_cases") - _float(baseline, "error_cases"),
            }
        )
    return rows


def _final_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in payload:
        return payload["aggregate"]
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": str(path.as_posix()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "experiment": "repair_sft_r3_sample500_diagnostic",
        "model_id": "repair_sft_r3",
        "protocol": "historical_sample500",
        "scope": "diagnostic_only_after_final_holdout_failure",
        "files": files,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append(_format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    output_dir: Path,
    split_rows: list[dict[str, str]],
    means: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    final: dict[str, Any] | None,
) -> None:
    r3_mean = next(row for row in means if row["model_id"] == "repair_sft_r3")
    m2 = next(row for row in means if row["model_id"] == "m2_sft")
    direction = "改善" if _float(r3_mean, "audit_accuracy") >= _float(m2, "audit_accuracy") else "退化"
    final_line = "final_holdout_v1 result is not available in this checkout."
    if final:
        final_line = (
            "final_holdout_v1 已消耗且失败："
            f"Audit Accuracy={_format(_float(final, 'audit_accuracy'))}, "
            f"High-risk Miss={_format(_float(final, 'high_risk_miss_rate'))}, "
            f"Schema={_format(_float(final, 'schema_compliance'))}, "
            f"Error Cases={int(_float(final, 'error_cases'))}。"
        )

    lines = [
        "# repair_sft_r3 sample500 历史口径诊断",
        "",
        "## 实验定位",
        "",
        "- 本轮只补跑 `repair_sft_r3` 在历史 `sample500` 四 split 上的推理和评测。",
        "- 该结果仅用于诊断和补表，不用于训练、调参、checkpoint 选择或 final holdout 重试。",
        "- DPO V3 checkpoint-15 未运行。",
        f"- {final_line}",
        "",
        "## R3 split 指标",
        "",
        *_markdown_table(
            split_rows,
            [
                "split",
                "total_cases",
                "json_validity",
                "schema_compliance",
                "audit_accuracy",
                "high_risk_miss_rate",
                "evidence_support_rate",
                "error_cases",
            ],
        ),
        "",
        "## 历史均值对比",
        "",
        *_markdown_table(
            means,
            [
                "model_id",
                "json_validity",
                "schema_compliance",
                "audit_accuracy",
                "high_risk_miss_rate",
                "evidence_support_rate",
                "error_cases",
            ],
        ),
        "",
        "## Delta vs 历史模型",
        "",
        *_markdown_table(
            deltas,
            [
                "baseline_model_id",
                "candidate_model_id",
                "audit_accuracy_delta",
                "high_risk_miss_rate_delta",
                "evidence_support_rate_delta",
                "schema_compliance_delta",
                "error_cases_delta",
            ],
        ),
        "",
        "## 诊断结论",
        "",
        f"- 相对 M2 历史 sample500 baseline，`repair_sft_r3` 的 Audit Accuracy 方向为：{direction}。",
        "- sample500 是 reporting-only 历史 benchmark；本轮结果只用于诊断，不能抵消已消耗 final_holdout_v1 的失败结论。",
        "- 后续若继续改进，应建立新的开发/验证闭环和新的 final holdout v2，不能把本轮 sample500 error cases 回流训练或选择。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_summary", default="outputs/eval_reports/repair_sft_r3_sample500_diagnostic/metrics_summary.csv")
    parser.add_argument("--report_dir", default="outputs/eval_reports/repair_sft_r3_sample500_diagnostic")
    parser.add_argument("--output_dir", default="docs/experiments/repair_sft_r3_sample500_diagnostic")
    parser.add_argument("--historical_means", default=str(HISTORICAL_MEANS))
    parser.add_argument("--historical_splits", default=str(HISTORICAL_SPLITS))
    parser.add_argument("--final_result", default=str(FINAL_RESULT))
    args = parser.parse_args()

    metrics_summary = Path(args.metrics_summary)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = _read_csv(metrics_summary)
    r3_mean = _mean_rows(split_rows)
    historical_means = _read_csv(Path(args.historical_means))
    historical_splits = _read_csv(Path(args.historical_splits))
    means = [*historical_means, r3_mean]
    deltas = _delta_rows(r3_mean, historical_means)

    _copy_csv(metrics_summary, output_dir / "metrics_summary.csv")
    _write_csv(output_dir / "metrics_by_model.csv", means, ["model_id", *CORE_FIELDS])
    _write_csv(
        output_dir / "metrics_delta_vs_history.csv",
        deltas,
        [
            "baseline_model_id",
            "candidate_model_id",
            "audit_accuracy_delta",
            "high_risk_miss_rate_delta",
            "evidence_support_rate_delta",
            "schema_compliance_delta",
            "error_cases_delta",
        ],
    )
    _write_csv(output_dir / "split_metrics_with_history.csv", [*historical_splits, *split_rows], list(split_rows[0]))

    error_dir = output_dir / "error_cases"
    error_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        source = report_dir / f"repair_sft_r3_{split}_errors.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing error cases file: {source}")
        target = error_dir / source.name
        target.write_bytes(source.read_bytes())

    final = _final_summary(Path(args.final_result))
    _write_report(output_dir, split_rows, means, deltas, final)
    _write_manifest(output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "repair_sft_r3_audit_accuracy": r3_mean["audit_accuracy"],
                "repair_sft_r3_high_risk_miss_rate": r3_mean["high_risk_miss_rate"],
                "repair_sft_r3_evidence_support_rate": r3_mean["evidence_support_rate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
