from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mv_audit.analysis.build_final_holdout import case_ids_sha256, stratified_manifest


def _row(index: int, anomaly: str, risk: str = "high", audit: str = "reject_recommendation") -> dict:
    return {
        "case_id": f"MV_MAIN_{index:06d}",
        "primary_anomaly_type": anomaly,
        "risk_level": risk,
        "audit_result": audit,
    }


def test_case_ids_sha256_is_order_independent() -> None:
    assert case_ids_sha256(["MV_MAIN_000002", "MV_MAIN_000001"]) == case_ids_sha256(
        ["MV_MAIN_000001", "MV_MAIN_000002"]
    )


def test_stratified_manifest_excludes_none_for_hard_negative() -> None:
    rows = [_row(index, "none", "low", "pass") for index in range(10)]
    rows.extend(_row(100 + index, "amount_mismatch") for index in range(10))
    rows.extend(_row(200 + index, "order_id_mismatch") for index in range(10))

    manifest = stratified_manifest(rows, split="test_hard_negative", sample_size=10, seed=42)

    assert len(manifest) == 10
    assert {row["primary_anomaly_type"] for row in manifest} == {"amount_mismatch", "order_id_mismatch"}
    assert len({row["case_id"] for row in manifest}) == 10


def test_stratified_manifest_is_jsonl_serializable() -> None:
    rows = [_row(index, "none", "low", "pass") for index in range(10)]
    rows.extend(_row(100 + index, "date_mismatch") for index in range(10))

    manifest = stratified_manifest(rows, split="test_clean", sample_size=8, seed=7)

    encoded = "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest)
    assert "MV_MAIN_" in encoded
    assert len(manifest) == 8


def test_final_holdout_summary_requires_and_writes_error_attribution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"num_cases": 1000}), encoding="utf-8")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    metrics = report_dir / "metrics_summary.csv"
    metrics.write_text(
        "\n".join(
            [
                "model_id,split,total_cases,json_validity,schema_compliance,audit_accuracy,high_risk_miss_rate,evidence_support_rate,error_cases",
                "repair_sft_r3,test_clean,250,1,1,0.95,0.02,0.9,1",
                "repair_sft_r3,test_robust,250,1,1,0.94,0.03,0.9,1",
                "repair_sft_r3,test_unseen_template,250,1,1,0.93,0.04,0.9,1",
                "repair_sft_r3,test_hard_negative,250,1,1,0.92,0.05,0.9,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for split in ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]:
        error_row = {
            "case_id": f"MV_MAIN_{len(split):06d}",
            "issues": ["high_risk_miss", "audit_mismatch"],
            "truth_risk_level": "high",
            "pred_risk_level": "medium",
            "truth_audit_result": "reject_recommendation",
            "pred_audit_result": "manual_review",
        }
        (report_dir / f"repair_sft_r3_{split}_errors.jsonl").write_text(
            json.dumps(error_row) + "\n",
            encoding="utf-8",
        )

    out_dir = tmp_path / "docs"
    marker = tmp_path / "FINAL_HOLDOUT_CONSUMED"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/summarize_final_holdout.py",
            "--manifest",
            str(manifest),
            "--metrics_summary",
            str(metrics),
            "--errors_dir",
            str(report_dir),
            "--output_dir",
            str(out_dir),
            "--consume_marker",
            str(marker),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    result = json.loads((out_dir / "final_holdout_result.json").read_text(encoding="utf-8"))
    attribution = json.loads((out_dir / "error_attribution_summary.json").read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert marker.exists()
    assert result["aggregate"]["total_cases"] == 1000
    assert result["error_attribution"]["problem_class_counts"]["model_missed_high_risk"] == 4
    assert attribution["problem_class_counts"]["decision_or_risk_mismatch"] == 4
