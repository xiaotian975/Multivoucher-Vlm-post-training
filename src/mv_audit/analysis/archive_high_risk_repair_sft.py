"""Archive Phase08 high-risk repair SFT validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from mv_audit.utils import ensure_dir


BASELINE_PACK_DIR = Path("docs/experiments/phase08_high_risk_repair_pack_20260813")
TWO_CANDIDATE_DIR = Path("docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3")
REPAIR_REPORT_DIR = Path("outputs/eval_reports/phase08_high_risk_repair_train_decode_dev/repair_sft_r1")
REPAIR_PRED_DIR = Path("outputs/predictions/phase08_high_risk_repair_train_decode_dev/repair_sft_r1")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _copy_tree_files(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    copied = 0
    for path in src.rglob("*"):
        if path.is_file():
            target = dst / path.relative_to(src)
            ensure_dir(target.parent)
            shutil.copy2(path, target)
            copied += 1
    return copied


def _prediction_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0}
    rows = 0
    first_case = None
    last_case = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            row = json.loads(line)
            case_id = row.get("case_id")
            first_case = first_case or case_id
            last_case = case_id
    return {
        "exists": True,
        "rows": rows,
        "first_case_id": first_case,
        "last_case_id": last_case,
        "sha256": _sha256(path),
        "path": str(path),
    }


def _comparison_rows() -> list[dict[str, Any]]:
    fields = [
        "json_validity",
        "schema_compliance",
        "audit_accuracy",
        "high_risk_miss_rate",
        "evidence_support_rate",
        "error_cases",
    ]
    rows: list[dict[str, Any]] = []
    for row in _read_csv(BASELINE_PACK_DIR / "metric_snapshot.csv"):
        if row.get("model_id") in {"m2_sft", "dpo_v2_baseline", "auxdpo_v2_strong"}:
            rows.append({"model_id": row.get("model_id"), "scope": row.get("scope"), **{field: row.get(field) for field in fields}})
    repair_rows = _read_csv(REPAIR_REPORT_DIR / "metrics_summary.csv")
    if repair_rows:
        row = repair_rows[0]
        rows.append({"model_id": "repair_sft_r1", "scope": "train_decode_dev", **{field: row.get(field) for field in fields}})
    return rows


def _write_readme(path: Path, comparison: list[dict[str, Any]], prediction_summary: dict[str, Any]) -> None:
    lines = [
        "# Phase08 High-risk Repair SFT r1 Archive",
        "",
        "This archive stores the low-cost Train decode dev validation artifacts for the high-risk repair loop.",
        "",
        "## Comparison",
        "",
        "| model_id | scope | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(key, ""))
                for key in [
                    "model_id",
                    "scope",
                    "json_validity",
                    "schema_compliance",
                    "audit_accuracy",
                    "high_risk_miss_rate",
                    "evidence_support_rate",
                    "error_cases",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- JSON Validity must remain 1.0.",
            "- Audit Accuracy must not fall below M2, or may drop by at most 0.01.",
            "- High-risk Miss Rate must improve by at least 0.03 versus M2.",
            "- Evidence Support Rate may drop by at most 0.01.",
            "- If High-risk Miss Rate does not improve, stop this training line and report Phase08 as a DPO negative result.",
            "",
            "## Prediction Summary",
            "",
            "```json",
            json.dumps(prediction_summary, ensure_ascii=False, indent=2),
            "```",
            "",
            "Adapter checkpoint files are intentionally excluded by default.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def archive(
    *,
    run_id: str,
    run_root: Path,
    config: Path,
    archive_dir: Path,
    include_adapter: bool,
    adapter_dir: Path,
) -> dict[str, Any]:
    ensure_dir(archive_dir)
    copied = {
        "config": _copy_if_exists(config, archive_dir / "configs" / config.name),
        "logs": _copy_tree_files(run_root / "logs", archive_dir / "logs"),
        "metrics": _copy_tree_files(REPAIR_REPORT_DIR, archive_dir / "train_decode_dev" / "repair_sft_r1"),
        "repair_pack_manifest": _copy_if_exists(BASELINE_PACK_DIR / "repair_pack_manifest.json", archive_dir / "repair_pack_manifest.json"),
        "repair_mix_manifest": _copy_if_exists(BASELINE_PACK_DIR / "repair_sft_train_mix_manifest.json", archive_dir / "repair_sft_train_mix_manifest.json"),
        "leakage_check": _copy_if_exists(BASELINE_PACK_DIR / "leakage_check.json", archive_dir / "leakage_check.json"),
    }
    if include_adapter:
        copied["adapter_files"] = _copy_tree_files(adapter_dir, archive_dir / "adapter" / "repair_sft_r1")
    comparison = _comparison_rows()
    _write_csv(
        archive_dir / "repair_sft_r1_comparison.csv",
        comparison,
        [
            "model_id",
            "scope",
            "json_validity",
            "schema_compliance",
            "audit_accuracy",
            "high_risk_miss_rate",
            "evidence_support_rate",
            "error_cases",
        ],
    )
    prediction_summary = _prediction_summary(REPAIR_PRED_DIR / "train_decode_dev.jsonl")
    (archive_dir / "prediction_summary.json").write_text(
        json.dumps(prediction_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(archive_dir / "README.md", comparison, prediction_summary)

    manifest_files = []
    for path in sorted(archive_dir.rglob("*")):
        if path.is_file():
            manifest_files.append(
                {
                    "path": str(path.relative_to(archive_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "name": f"phase08_high_risk_repair_sft_r1_{run_id}",
        "run_id": run_id,
        "run_root": str(run_root),
        "include_adapter": include_adapter,
        "copied": copied,
        "files": manifest_files,
    }
    (archive_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tar_path = archive_dir.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(archive_dir, arcname=archive_dir.name)
    manifest["tar_path"] = str(tar_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml"))
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--include-adapter", action="store_true")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir or Path(f"docs/experiments/phase08_high_risk_repair_sft_r1_{args.run_id}")
    manifest = archive(
        run_id=args.run_id,
        run_root=args.run_root,
        config=args.config,
        archive_dir=archive_dir,
        include_adapter=args.include_adapter,
        adapter_dir=args.adapter_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
