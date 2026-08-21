"""Build reproducible Phase 10 metrics, curves, and case-study artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POSITIVE_CASES = ("MV_MAIN_004522", "MV_MAIN_020454")
OVERFIT_CASES = ("MV_MAIN_023069", "MV_MAIN_015818")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_training_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stage") in {"dpo_train", "dpo_holdout"}:
            records.append(row)
    if not records:
        raise ValueError(f"No DPO training records found in {path}")
    return records


def _issues(row: dict[str, Any]) -> set[str]:
    values = row.get("issues")
    if values is None:
        values = row.get("issue_codes")
    return set(values or [])


def _output(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["completions"][0]["raw_output"])


def _has_order_pair(output: dict[str, Any]) -> bool:
    doc_types = {
        item.get("source_doc_type")
        for item in output.get("evidence", [])
        if item.get("field") == "order_id"
    }
    return {"order", "reimbursement_form"}.issubset(doc_types)


def _decision_view(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_level": output.get("risk_level"),
        "audit_result": output.get("audit_result"),
        "anomaly_types": output.get("anomaly_types") or [],
        "has_two_sided_order_id_evidence": _has_order_pair(output),
        "reason": output.get("reason"),
        "evidence": output.get("evidence") or [],
    }


def _metric_rows(root: Path, artifact_root: Path, sft_run: str, full_run: str) -> list[dict[str, Any]]:
    historical = _read_csv(root / "docs/experiments/phase08_m3v2_sample500/metrics_by_model.csv")
    by_id = {row["model_id"]: row for row in historical}
    sft_metrics = _read_json(
        artifact_root / sft_run / "outputs/runtime/model_mined_dpo_v3" / sft_run / "eval/sft_v3/metrics.json"
    )
    dpo_metrics = _read_json(
        artifact_root / full_run / "outputs/runtime/model_mined_dpo_v3" / full_run
        / "eval/dpo_v3_selected/metrics.json"
    )
    rows: list[dict[str, Any]] = []
    for model_id, method, role in [
        ("m2_sft", "LoRA SFT", "HISTORICAL_SAMPLE500_BASELINE"),
        ("m3_dpo", "DPO v1", "RESEARCH_ABLATION"),
        ("m3v2_dpo", "DPO v2", "RESEARCH_ABLATION"),
    ]:
        source = by_id[model_id]
        rows.append(
            {
                "benchmark_group": "sample500",
                "evaluation_protocol": "four_split_average_500_per_split",
                "model_id": model_id,
                "method": method,
                "cases": "4x500",
                "json_validity": source["json_validity"],
                "schema_compliance": source["schema_compliance"],
                "audit_accuracy": source["audit_accuracy"],
                "high_risk_miss_rate": source["high_risk_miss_rate"],
                "evidence_support_rate": source["evidence_support_rate"],
                "error_cases": source["error_cases"],
                "selection_role": role,
                "source": "docs/experiments/phase08_m3v2_sample500/metrics_by_model.csv",
            }
        )
    diagnostic_path = root / "docs/experiments/repair_sft_r3_sample500_diagnostic/metrics_by_model.csv"
    if diagnostic_path.exists():
        diagnostic = {row["model_id"]: row for row in _read_csv(diagnostic_path)}
        source = diagnostic["repair_sft_r3"]
        rows.append(
            {
                "benchmark_group": "sample500",
                "evaluation_protocol": "four_split_average_500_per_split_historical",
                "model_id": "repair_sft_r3",
                "method": "Structured Repair SFT v3",
                "cases": "4x500",
                "json_validity": source["json_validity"],
                "schema_compliance": source["schema_compliance"],
                "audit_accuracy": source["audit_accuracy"],
                "high_risk_miss_rate": source["high_risk_miss_rate"],
                "evidence_support_rate": source["evidence_support_rate"],
                "error_cases": source["error_cases"],
                "selection_role": "DIAGNOSTIC_AFTER_FINAL_FAILURE",
                "source": "docs/experiments/repair_sft_r3_sample500_diagnostic/metrics_by_model.csv",
            }
        )
    for model_id, method, role, metrics, source in [
        (
            "repair_sft_r3",
            "Structured Repair SFT v3",
            "PRODUCTION_CANDIDATE",
            sft_metrics,
            f"outputs/remote_artifacts/model_mined_dpo_v3/{sft_run}/.../eval/sft_v3/metrics.json",
        ),
        (
            "dpo_v3_model_mined_strong_checkpoint15",
            "Model-Mined DPO v3",
            "ALIGNMENT_RESEARCH_CANDIDATE",
            dpo_metrics,
            f"outputs/remote_artifacts/model_mined_dpo_v3/{full_run}/.../eval/dpo_v3_selected/metrics.json",
        ),
    ]:
        rows.append(
            {
                "benchmark_group": "train_decode_dev",
                "evaluation_protocol": "train_only_fast_gate_152",
                "model_id": model_id,
                "method": method,
                "cases": metrics["total_cases"],
                "json_validity": metrics["json_validity"],
                "schema_compliance": metrics["schema_compliance"],
                "audit_accuracy": metrics["audit_accuracy"],
                "high_risk_miss_rate": metrics["high_risk_miss_rate"],
                "evidence_support_rate": metrics["evidence_support_rate"],
                "error_cases": metrics["error_cases"],
                "selection_role": role,
                "source": source,
            }
        )
    return rows


def _probe_rows(probe_root: Path) -> list[dict[str, Any]]:
    selection = _read_json(probe_root / "checkpoint_selection.json")
    rows = [
        {
            "checkpoint": "baseline",
            "step": 0,
            "eligible": False,
            **selection["baseline"],
            "reward_delta": 0.0,
            "order_id_pair_delta": 0.0,
        }
    ]
    for candidate in sorted(selection["candidates"], key=lambda item: item["step"]):
        rows.append(
            {
                "checkpoint": Path(candidate["checkpoint"]).name,
                "step": candidate["step"],
                "eligible": candidate["eligible"],
                **candidate["metrics"],
                "reward_delta": candidate["reward_delta"],
                "order_id_pair_delta": candidate["order_id_pair_delta"],
            }
        )
    assert selection["selected"]["step"] == 15
    return rows


def _case_studies(
    probe_root: Path,
    artifact_root: Path,
    sft_run: str,
    full_run: str,
) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in _read_jsonl(probe_root / "probes/baseline/rollouts.jsonl")}
    checkpoint = {
        row["case_id"]: row for row in _read_jsonl(probe_root / "probes/checkpoint-15/rollouts.jsonl")
    }
    positive = []
    for case_id in POSITIVE_CASES:
        base_output = _output(baseline[case_id])
        dpo_output = _output(checkpoint[case_id])
        truth = baseline[case_id]["ground_truth"]["output"]
        assert base_output["audit_result"] == "pass" and not _has_order_pair(base_output)
        assert dpo_output["audit_result"] == "reject_recommendation" and _has_order_pair(dpo_output)
        positive.append(
            {
                "case_id": case_id,
                "kind": "positive_alignment_probe",
                "ground_truth": _decision_view(truth),
                "sft_v3_probe_baseline": _decision_view(base_output),
                "dpo_v3_checkpoint15": _decision_view(dpo_output),
            }
        )

    baseline_errors = {
        row["case_id"]: row
        for row in _read_jsonl(
            artifact_root / sft_run / "outputs/runtime/model_mined_dpo_v3" / sft_run / "eval/sft_v3/errors.jsonl"
        )
    }
    full_root = artifact_root / full_run
    candidate_errors = {
        row["case_id"]: row
        for row in _read_jsonl(
            full_root / "outputs/runtime/model_mined_dpo_v3" / full_run / "eval/dpo_v3_selected/errors.jsonl"
        )
    }
    predictions = {
        row["case_id"]: json.loads(row["raw_output"])
        for row in _read_jsonl(
            full_root
            / "outputs/predictions/phase10_dpo_v3_strong_selected_train_decode_dev"
            / "dpo_v3_model_mined_strong/train_decode_dev.jsonl"
        )
    }
    ground_truth = {
        row["case_id"]: row["answer"]
        for row in _read_jsonl(
            full_root
            / "outputs/eval_sets/phase10_dpo_v3_strong_selected_train_decode_dev/train_decode_dev.jsonl"
        )
    }
    assert len(ground_truth) == 152
    overfit = []
    for case_id in OVERFIT_CASES:
        assert case_id not in baseline_errors
        assert _issues(candidate_errors[case_id]) == {"audit_mismatch"}
        output = predictions[case_id]
        assert "amount_mismatch" in output["anomaly_types"]
        assert output["audit_result"] == "manual_review"
        overfit.append(
            {
                "case_id": case_id,
                "kind": "full_gate_overfit",
                "sft_v3_baseline": {"evaluator_error_recorded": False},
                "ground_truth": _decision_view(ground_truth[case_id]),
                "dpo_v3_checkpoint15": _decision_view(output),
                "dpo_v3_issues": sorted(_issues(candidate_errors[case_id])),
            }
        )
    return {"positive_probe_cases": positive, "overfit_full_gate_cases": overfit}


def _error_summary(artifact_root: Path, sft_run: str, full_run: str) -> dict[str, Any]:
    baseline = _read_jsonl(
        artifact_root / sft_run / "outputs/runtime/model_mined_dpo_v3" / sft_run / "eval/sft_v3/errors.jsonl"
    )
    candidate = _read_jsonl(
        artifact_root / full_run / "outputs/runtime/model_mined_dpo_v3" / full_run
        / "eval/dpo_v3_selected/errors.jsonl"
    )
    baseline_misses = {row["case_id"] for row in baseline if "high_risk_miss" in _issues(row)}
    candidate_misses = {row["case_id"] for row in candidate if "high_risk_miss" in _issues(row)}
    assert len(baseline_misses) == 5
    assert len(candidate_misses) == 12
    assert not (baseline_misses - candidate_misses)
    assert len(candidate_misses - baseline_misses) == 7
    transitions = Counter(
        f"{row['truth_audit_result']}->{row['pred_audit_result']}"
        for row in candidate
        if "audit_mismatch" in _issues(row)
    )
    return {
        "baseline_high_risk_misses": sorted(baseline_misses),
        "candidate_high_risk_misses": sorted(candidate_misses),
        "fixed_high_risk_misses": sorted(baseline_misses - candidate_misses),
        "introduced_high_risk_misses": sorted(candidate_misses - baseline_misses),
        "candidate_issue_counts": dict(Counter(code for row in candidate for code in _issues(row))),
        "audit_transition_counts": dict(transitions),
    }


def _plot_metric_groups(rows: list[dict[str, Any]], output: Path) -> None:
    groups = [
        ("sample500", "Historical sample500 (4-split average)"),
        ("train_decode_dev", "Train-only development gate (152 cases)"),
    ]
    metrics = [
        ("audit_accuracy", "Audit accuracy"),
        ("high_risk_miss_rate", "High-risk miss"),
        ("evidence_support_rate", "Evidence support"),
    ]
    colors = ["#2E7D32", "#C62828", "#1565C0", "#6A1B9A", "#EF6C00"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, (group, title) in zip(axes, groups):
        subset = [row for row in rows if row["benchmark_group"] == group]
        x = list(range(len(metrics)))
        width = 0.75 / len(subset)
        for index, row in enumerate(subset):
            values = [float(row[key]) for key, _ in metrics]
            offset = (index - (len(subset) - 1) / 2) * width
            axis.bar([value + offset for value in x], values, width, label=row["model_id"], color=colors[index])
        axis.set_title(title)
        axis.set_xticks(x, [label for _, label in metrics])
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Post-training metrics by evaluation protocol", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_training_curves(
    root: Path,
    weak_records: list[dict[str, Any]],
    strong_records: list[dict[str, Any]],
    output: Path,
) -> None:
    v1 = _read_csv(root / "docs/experiments/phase08_dpo_sample1000/training_history.csv")
    v2 = _read_csv(root / "docs/experiments/phase08_m3v2_sample500/dpo_v2/dpo_v2_training_history.csv")
    v3_weak = [row for row in weak_records if row["stage"] == "dpo_train"]
    v3_strong = [row for row in strong_records if row["stage"] == "dpo_train"]
    datasets = [
        ("DPO v1 (sum log-prob)", v1),
        ("DPO v2 (sum log-prob)", v2),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for row_index, (title, data) in enumerate(datasets):
        steps = [float(row["global_step"]) for row in data]
        loss = [float(row["loss"]) for row in data]
        margin = [float(row["preference_margin"]) for row in data]
        axes[row_index, 0].plot(steps, loss, color="#1565C0", linewidth=1.8)
        axes[row_index, 1].plot(steps, margin, color="#EF6C00", linewidth=1.8)
        axes[row_index, 0].set_title(f"{title}: loss")
        axes[row_index, 1].set_title(f"{title}: preference margin")
        for axis in axes[row_index]:
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)

    weak_steps = [float(row["global_step"]) for row in v3_weak]
    strong_offset = max(weak_steps, default=0.0)
    strong_steps = [strong_offset + float(row["global_step"]) for row in v3_strong]
    for column, key, label in [
        (0, "loss", "loss"),
        (1, "preference_margin", "preference margin"),
    ]:
        axis = axes[2, column]
        axis.plot(
            weak_steps,
            [float(row[key]) for row in v3_weak],
            color="#1565C0",
            linewidth=1.8,
            label="weak v3",
        )
        axis.plot(
            strong_steps,
            [float(row[key]) for row in v3_strong],
            color="#EF6C00",
            linewidth=1.8,
            label="strong continuation",
        )
        if weak_steps and strong_steps:
            axis.axvline(strong_offset, color="#616161", linestyle="--", linewidth=1)
        axis.set_title(f"DPO v3 (mean-token): {label}")
        axis.set_xlabel("effective step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("DPO training curves (raw margins are not cross-version comparable)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_probe(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["checkpoint"] for row in rows]
    reward = [float(row["mean_reward"]) for row in rows]
    pair_rate = [float(row["order_id_pair_rate"]) for row in rows]
    miss_rate = [float(row["high_risk_miss_rate"]) for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, values, title, color in [
        (axes[0], reward, "Mean task reward", "#1565C0"),
        (axes[1], pair_rate, "Order-ID evidence-pair rate", "#2E7D32"),
        (axes[2], miss_rate, "High-risk miss rate", "#C62828"),
    ]:
        axis.plot(x, values, marker="o", linewidth=2, color=color)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("DPO v3 case-disjoint alignment probe", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifact_root", default="outputs/remote_artifacts/model_mined_dpo_v3")
    parser.add_argument("--sft_run", default="20260816_173434")
    parser.add_argument("--probe_run", default="20260820_133149")
    parser.add_argument("--full_run", default="20260820_143221")
    parser.add_argument(
        "--weak_log",
        default=(
            "outputs/model_candidates/repair_sft_r3/outputs/runtime/model_mined_dpo_v3/"
            "20260820_122300/logs/dpo_v3_train.log"
        ),
    )
    parser.add_argument("--output_dir", default="docs/experiments/phase10_model_error_mined_dpo_v3")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifact_root = (root / args.artifact_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _metric_rows(root, artifact_root, args.sft_run, args.full_run)
    metric_fields = list(metrics[0])
    _write_csv(output_dir / "post_training_metrics.csv", metrics, metric_fields)

    probe_root = artifact_root / args.probe_run / "outputs/runtime/model_mined_dpo_v3" / args.probe_run
    probes = _probe_rows(probe_root)
    _write_csv(output_dir / "dpo_v3_probe_summary.csv", probes, list(probes[0]))

    strong_records = _parse_training_log(probe_root / "logs/dpo_v3_train.log")
    history_fields = sorted({key for row in strong_records for key in row})
    _write_csv(output_dir / "dpo_v3_strong_training_history.csv", strong_records, history_fields)

    weak_records = _parse_training_log((root / args.weak_log).resolve())
    weak_history_fields = sorted({key for row in weak_records for key in row})
    _write_csv(output_dir / "dpo_v3_weak_training_history.csv", weak_records, weak_history_fields)

    cases = _case_studies(probe_root, artifact_root, args.sft_run, args.full_run)
    (output_dir / "case_studies.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    errors = _error_summary(artifact_root, args.sft_run, args.full_run)
    (output_dir / "error_attribution_summary.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_metric_groups(metrics, figures / "post_training_metrics_by_benchmark.png")
    _plot_training_curves(
        root,
        weak_records,
        strong_records,
        figures / "dpo_training_curves_v1_v2_v3.png",
    )
    _plot_probe(probes, figures / "dpo_v3_probe_checkpoints.png")
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics_rows": len(metrics),
                "probe_rows": len(probes),
                "weak_history_rows": len(weak_records),
                "strong_history_rows": len(strong_records),
                "positive_cases": [row["case_id"] for row in cases["positive_probe_cases"]],
                "overfit_cases": [row["case_id"] for row in cases["overfit_full_gate_cases"]],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
