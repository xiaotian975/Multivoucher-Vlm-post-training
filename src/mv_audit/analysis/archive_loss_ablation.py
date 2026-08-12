"""Archive DPO v2 loss-ablation artifacts for local reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from mv_audit.utils import ensure_dir


CORE_METRICS = [
    "json_validity",
    "schema_compliance",
    "field_em",
    "audit_accuracy",
    "high_risk_miss_rate",
    "evidence_support_rate",
    "error_cases",
]
SPLITS = ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    value = row.get(key, "")
    try:
        return float(value)
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


def _metric_means(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in CORE_METRICS}
    means: dict[str, float] = {}
    for metric in CORE_METRICS:
        means[metric] = sum(_float(row, metric) for row in rows) / len(rows)
    return means


def _plot_bar(path: Path, title: str, labels: list[str], series: dict[str, list[float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ensure_dir(path.parent)
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for index, (name, values) in enumerate(series.items()):
        offsets = [value + (index - (len(series) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join([header, sep, *body])


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 10:
        return f"{number:.1f}"
    return f"{number:.4f}"


def _training_summary(project_root: Path, variants: list[str], report_dirs: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        report_path = project_root / "outputs" / "eval_reports" / report_dirs[variant] / "dpo_v2_reward_audit.json"
        if not report_path.exists():
            rows.append({"variant": variant, "status": "missing"})
            continue
        payload = _read_json(report_path)
        history = payload.get("training_history") or []
        last = history[-1] if history else {}
        rows.append(
            {
                "variant": variant,
                "status": "done",
                "loss_type": payload.get("loss_type", last.get("loss_type", "")),
                "lambda_sft": payload.get("lambda_sft", ""),
                "global_step": payload.get("global_step", last.get("global_step", "")),
                "loss": _fmt(last.get("loss", "")),
                "preference_loss": _fmt(last.get("preference_loss", "")),
                "sft_nll_loss": _fmt(last.get("sft_nll_loss", "")),
                "preference_margin": _fmt(last.get("preference_margin", "")),
            }
        )
    return rows


def _decode_summary(project_root: Path, variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics_path = project_root / "outputs" / "eval_reports" / "phase08_loss_ablation_train_decode_dev" / variant / "metrics_summary.csv"
        metrics_rows = _read_csv(metrics_path)
        if not metrics_rows:
            rows.append({"variant": variant, "status": "missing"})
            continue
        row = metrics_rows[0]
        rows.append(
            {
                "variant": variant,
                "status": "done",
                "audit_accuracy": _fmt(row.get("audit_accuracy")),
                "high_risk_miss_rate": _fmt(row.get("high_risk_miss_rate")),
                "evidence_support_rate": _fmt(row.get("evidence_support_rate")),
                "schema_compliance": _fmt(row.get("schema_compliance")),
                "error_cases": _fmt(row.get("error_cases")),
            }
        )
    return rows


def _selected_sample_rows(project_root: Path, selected_variant: str) -> list[dict[str, str]]:
    return _read_csv(project_root / "outputs" / "eval_reports" / "phase08_loss_ablation_sample500" / selected_variant / "metrics_summary.csv")


def _copy_artifacts(
    *,
    project_root: Path,
    archive_dir: Path,
    run_root: Path,
    selected_variant: str,
    variants: list[str],
    report_dirs: dict[str, str],
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for variant in variants:
        src = project_root / "outputs" / "eval_reports" / report_dirs[variant] / "dpo_v2_reward_audit.json"
        dst = archive_dir / "training" / variant / "dpo_v2_reward_audit.json"
        if _copy_if_exists(src, dst):
            copied.append({"source": str(src), "path": str(dst.relative_to(archive_dir))})
        decode_dir = project_root / "outputs" / "eval_reports" / "phase08_loss_ablation_train_decode_dev" / variant
        for file in decode_dir.glob("*") if decode_dir.exists() else []:
            dst = archive_dir / "train_decode_dev" / variant / file.name
            if _copy_if_exists(file, dst):
                copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})

    sample_dir = project_root / "outputs" / "eval_reports" / "phase08_loss_ablation_sample500" / selected_variant
    for file in sample_dir.glob("*.csv") if sample_dir.exists() else []:
        dst = archive_dir / file.name
        if _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})
    for file in sample_dir.glob("*_metrics.json") if sample_dir.exists() else []:
        dst = archive_dir / "metrics_json" / file.name
        if _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})
    for file in sample_dir.glob("*_errors.jsonl") if sample_dir.exists() else []:
        dst = archive_dir / "error_cases" / file.name
        if _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})

    migration_dir = project_root / "outputs" / "eval_reports" / "phase08_loss_ablation_error_migration" / selected_variant
    for file in migration_dir.glob("*") if migration_dir.exists() else []:
        dst = archive_dir / "error_migration" / file.name
        if file.is_file() and _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})

    for file in run_root.glob("*.json") if run_root.exists() else []:
        dst = archive_dir / "runtime" / file.name
        if _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})
    log_dir = run_root / "logs"
    for file in log_dir.glob("*.log") if log_dir.exists() else []:
        dst = archive_dir / "logs" / file.name
        if _copy_if_exists(file, dst):
            copied.append({"source": str(file), "path": str(dst.relative_to(archive_dir))})
    pair_report = project_root / "data" / "mv_audit" / "dpo_v2" / "pair_report.json"
    if _copy_if_exists(pair_report, archive_dir / "dpo_v2_pair_report.json"):
        copied.append({"source": str(pair_report), "path": "dpo_v2_pair_report.json"})
    return copied


def _write_report(
    *,
    archive_dir: Path,
    run_id: str,
    selected_variant: str,
    training_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, str]],
    figures: list[str],
) -> None:
    sample_table_rows = []
    for row in sample_rows:
        sample_table_rows.append({"split": row.get("split", ""), **{metric: _fmt(row.get(metric, "")) for metric in CORE_METRICS}})
    means = _metric_means(sample_rows)
    mean_row = {"variant": selected_variant, **{metric: _fmt(value) for metric, value in means.items()}}
    report = [
        f"# Phase08 DPO v2 Loss Ablation 结果归档（{run_id}）",
        "",
        "## 实验设置",
        "",
        "- 本轮只比较 DPO v2、AuxDPO 和 IPO 类候选，不启动 GRPO/M4。",
        "- DPO pair 仍遵守 Train-only 原则，训练不使用 Val/Test/sample500 case。",
        "- 先用 Train decode dev 做低成本筛选，再仅对最优候选运行 sample500。",
        f"- 进入 sample500 的候选：`{selected_variant}`。",
        "",
        "## DPO/AuxDPO/IPO 训练摘要",
        "",
        _markdown_table(
            training_rows,
            ["variant", "status", "loss_type", "lambda_sft", "global_step", "loss", "preference_loss", "sft_nll_loss", "preference_margin"],
        ),
        "",
        "## Train Decode Dev 筛选结果",
        "",
        _markdown_table(
            decode_rows,
            ["variant", "status", "audit_accuracy", "high_risk_miss_rate", "evidence_support_rate", "schema_compliance", "error_cases"],
        ),
        "",
        "## 最优候选 Sample500 结果",
        "",
        _markdown_table(sample_table_rows, ["split", *CORE_METRICS]),
        "",
        "## 最优候选均值",
        "",
        _markdown_table([mean_row], ["variant", *CORE_METRICS]),
        "",
        "## 图表",
        "",
        *[f"- [{Path(name).name}](figures/{Path(name).name})" for name in figures],
        "",
        "## 解释口径",
        "",
        "- 若 High-risk Miss Rate 未较 M2 明显下降，说明仅更换偏好损失不足以解决高风险漏检问题。",
        "- 若 Audit Accuracy 或 Evidence Support Rate 下降，应优先复盘 pair 质量和 hard rejected 构造，而不是继续扩大训练。",
        "- 本报告是 DPO v2 修正实验结果，不包含 GRPO 正式结论。",
        "",
    ]
    (archive_dir / "phase08_loss_ablation_report.md").write_text("\n".join(report), encoding="utf-8")

    readme_append = [
        f"\n## Phase08 DPO v2 / AuxDPO / IPO 修正实验（{run_id}）",
        "",
        "本轮在 DPO v2 的 Train-only 数据约束下，比较了 DPO、AuxDPO 和 IPO 变体。训练阶段以 5 个候选并发方式提高 5 卡利用率，随后用 Train decode dev 筛选，只让最优候选进入 sample500 推理评测。",
        "",
        f"- 最优候选：`{selected_variant}`",
        f"- 完整归档：`docs/experiments/phase08_loss_ablation_{run_id}/phase08_loss_ablation_report.md`",
        "- 本轮不包含 GRPO/M4 正式训练。",
        "",
        "### Train Decode Dev 筛选",
        "",
        _markdown_table(
            decode_rows,
            ["variant", "status", "audit_accuracy", "high_risk_miss_rate", "evidence_support_rate", "schema_compliance", "error_cases"],
        ),
        "",
        "### 最优候选 Sample500 均值",
        "",
        _markdown_table([mean_row], ["variant", *CORE_METRICS]),
        "",
        "### 结果解释",
        "",
        "本轮优先判断 High-risk Miss Rate 是否相对 M2 产生实质下降，同时要求 Audit Accuracy 和 Evidence Support Rate 不明显退化。如果最优候选仍不能满足该标准，后续应继续优化 DPO pair 构造和审计规则样本质量，而不是直接恢复 GRPO。",
        "",
    ]
    (archive_dir / "README_APPEND.md").write_text("\n".join(readme_append), encoding="utf-8")


def archive(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_root = (project_root / args.run_root).resolve()
    archive_dir = (project_root / args.archive_dir).resolve()
    selected_variant = args.selected_variant.strip()
    variants = args.variants.split()
    report_dirs = dict(item.split("=", 1) for item in args.report_dirs.split())
    ensure_dir(archive_dir)

    copied = _copy_artifacts(
        project_root=project_root,
        archive_dir=archive_dir,
        run_root=run_root,
        selected_variant=selected_variant,
        variants=variants,
        report_dirs=report_dirs,
    )
    training_rows = _training_summary(project_root, variants, report_dirs)
    decode_rows = _decode_summary(project_root, variants)
    sample_rows = _selected_sample_rows(project_root, selected_variant)

    _write_csv(
        archive_dir / "variant_training_summary.csv",
        training_rows,
        ["variant", "status", "loss_type", "lambda_sft", "global_step", "loss", "preference_loss", "sft_nll_loss", "preference_margin"],
    )
    _write_csv(
        archive_dir / "variant_train_decode_dev_summary.csv",
        decode_rows,
        ["variant", "status", "audit_accuracy", "high_risk_miss_rate", "evidence_support_rate", "schema_compliance", "error_cases"],
    )
    if sample_rows:
        _write_csv(archive_dir / "selected_sample500_metrics_summary.csv", sample_rows, list(sample_rows[0].keys()))

    figures: list[str] = []
    done_decode_rows = [row for row in decode_rows if row.get("status") == "done"]
    if done_decode_rows:
        labels = [str(row["variant"]) for row in done_decode_rows]
        figure = archive_dir / "figures" / "train_decode_dev_core_metrics.png"
        if _plot_bar(
            figure,
            "Train Decode Dev Core Metrics",
            labels,
            {
                "Audit Accuracy": [_float(row, "audit_accuracy") for row in done_decode_rows],
                "High-risk Miss Rate": [_float(row, "high_risk_miss_rate") for row in done_decode_rows],
                "Evidence Support": [_float(row, "evidence_support_rate") for row in done_decode_rows],
            },
        ):
            figures.append(str(figure.relative_to(archive_dir)))
    if sample_rows:
        labels = [str(row["split"]) for row in sample_rows]
        figure = archive_dir / "figures" / "selected_sample500_core_metrics.png"
        if _plot_bar(
            figure,
            f"{selected_variant} Sample500 Core Metrics",
            labels,
            {
                "Audit Accuracy": [_float(row, "audit_accuracy") for row in sample_rows],
                "High-risk Miss Rate": [_float(row, "high_risk_miss_rate") for row in sample_rows],
                "Evidence Support": [_float(row, "evidence_support_rate") for row in sample_rows],
            },
        ):
            figures.append(str(figure.relative_to(archive_dir)))
        figure = archive_dir / "figures" / "selected_sample500_error_cases.png"
        if _plot_bar(figure, f"{selected_variant} Error Cases", labels, {"Error Cases": [_float(row, "error_cases") for row in sample_rows]}):
            figures.append(str(figure.relative_to(archive_dir)))

    _write_report(
        archive_dir=archive_dir,
        run_id=args.run_id,
        selected_variant=selected_variant,
        training_rows=training_rows,
        decode_rows=decode_rows,
        sample_rows=sample_rows,
        figures=figures,
    )

    manifest_files = []
    for file in sorted(path for path in archive_dir.rglob("*") if path.is_file()):
        manifest_files.append(
            {
                "path": str(file.relative_to(archive_dir)).replace("\\", "/"),
                "bytes": file.stat().st_size,
                "sha256": _sha256(file),
            }
        )
    manifest = {
        "run_id": args.run_id,
        "selected_variant": selected_variant,
        "source_run_root": str(run_root),
        "copied_sources": copied,
        "files": manifest_files,
        "leakage_note": "Archive only. DPO v2 training data remains Train-only; sample500 outputs are for reporting/evaluation only.",
    }
    (archive_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_dir": str(archive_dir), "selected_variant": selected_variant, "files": len(manifest_files)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Phase08 DPO v2 loss-ablation outputs.")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--selected_variant", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--report_dirs", required=True)
    return parser.parse_args()


def main() -> None:
    archive(parse_args())


if __name__ == "__main__":
    main()
