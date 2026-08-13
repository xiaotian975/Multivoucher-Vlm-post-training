"""Build a high-risk miss diagnosis and Train-only repair pack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mv_audit.utils import ensure_dir


SAMPLE500_SPLITS = ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]
TWO_CANDIDATE_ARCHIVE = (
    "docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3"
)
DEFAULT_OUTPUT_DIR = "docs/experiments/phase08_high_risk_repair_pack_20260813"
METRIC_KEYS = [
    "json_validity",
    "schema_compliance",
    "audit_accuracy",
    "high_risk_miss_rate",
    "evidence_support_rate",
    "error_cases",
]


@dataclass(frozen=True)
class ErrorSource:
    label: str
    model_id: str
    split: str
    path: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


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


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id", "").split("_full_audit")[0])


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return answer
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        content = messages[-1].get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _anomaly_family(row: dict[str, Any] | None) -> str:
    if not row:
        return "unknown"
    answer = _answer(row) if "messages" in row or "answer" in row else row
    anomalies = {str(item) for item in answer.get("anomaly_types") or row.get("anomaly_types") or []}
    primary = str(row.get("primary_anomaly_type") or answer.get("primary_anomaly_type") or "")
    consistency = answer.get("consistency_check") or {}
    all_tags = set(anomalies)
    if primary and primary != "none":
        all_tags.add(primary)

    if any("amount" in tag for tag in all_tags) or "over_reimbursement" in all_tags or consistency.get("amount_consistent") is False:
        return "amount_abnormal"
    if any("order" in tag for tag in all_tags) or consistency.get("order_id_consistent") is False:
        return "order_id_inconsistent"
    if (
        "missing_document" in all_tags
        or "unreadable_image" in all_tags
        or any("evidence" in tag for tag in all_tags)
        or consistency.get("document_complete") is False
    ):
        return "evidence_or_document_insufficient"
    if "date_mismatch" in all_tags or consistency.get("date_reasonable") is False:
        return "date_mismatch"
    if "merchant_mismatch" in all_tags or consistency.get("merchant_consistent") is False:
        return "merchant_mismatch"
    if (
        "person_mismatch" in all_tags
        or "applicant_mismatch" in all_tags
        or "payer_mismatch" in all_tags
        or "order_user_mismatch" in all_tags
        or consistency.get("person_consistent") is False
    ):
        return "person_mismatch"
    if "duplicate_in_batch" in all_tags or consistency.get("duplicate_in_batch") is True:
        return "duplicate_in_batch"
    if not all_tags or all_tags == {"none"}:
        return "clean_or_low_risk"
    return sorted(all_tags)[0]


def _problem_class(error_row: dict[str, Any], gt_row: dict[str, Any] | None) -> str:
    issues = set(error_row.get("issues") or [])
    if "schema_invalid" in issues or "json_invalid" in issues:
        return "schema_contract_failure"
    if "high_risk_miss" in issues:
        return "model_missed_high_risk"
    if "audit_mismatch" in issues and error_row.get("truth_risk_level") == "high":
        return "recognized_risk_but_decision_released"
    if "unsupported_evidence" in issues or "bbox_strict_error" in issues:
        return "evidence_not_trustworthy"
    if gt_row and _anomaly_family(gt_row) == "evidence_or_document_insufficient":
        return "evidence_or_document_insufficient"
    return "other_business_error"


def _load_main_raw_cases(repo: Path) -> dict[str, dict[str, Any]]:
    raw_by_case: dict[str, dict[str, Any]] = {}
    base = repo / "data/mv_audit/raw_cases/main"
    for name in ["train_cases.jsonl", *(f"{split}_cases.jsonl" for split in SAMPLE500_SPLITS)]:
        path = base / name
        for row in _read_jsonl(path):
            raw_by_case[_case_id(row)] = row
    return raw_by_case


def _error_sources(repo: Path) -> list[ErrorSource]:
    sources: list[ErrorSource] = []
    for split in SAMPLE500_SPLITS:
        sources.append(
            ErrorSource(
                label="M2 sample500",
                model_id="m2_sft",
                split=split,
                path=repo / f"docs/experiments/phase07_sample500/error_cases/m2_sft_{split}_errors.jsonl",
            )
        )
        sources.append(
            ErrorSource(
                label="M3 sample500",
                model_id="m3_dpo",
                split=split,
                path=repo / f"docs/experiments/phase08_m3_sample500/error_cases/m3_dpo_{split}_errors.jsonl",
            )
        )
        sources.append(
            ErrorSource(
                label="M3v2 sample500",
                model_id="m3v2_dpo",
                split=split,
                path=repo / f"docs/experiments/phase08_m3v2_sample500/error_cases/m3v2_dpo_{split}_errors.jsonl",
            )
        )
    archive = repo / TWO_CANDIDATE_ARCHIVE
    for variant in ["dpo_v2_baseline", "auxdpo_v2_strong"]:
        sources.append(
            ErrorSource(
                label=f"{variant} train_decode_dev",
                model_id=variant,
                split="train_decode_dev",
                path=archive / f"train_decode_dev/{variant}/m3v2_dpo_train_decode_dev_errors.jsonl",
            )
        )
    return sources


def _summarize_metrics(repo: Path) -> list[dict[str, Any]]:
    rows = _read_csv(repo / "docs/experiments/phase08_m3v2_sample500/m2_m3_m3v2_split_metrics.csv")
    metric_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row.get("model_id", "")].append(row)
    for model_id, model_rows in by_model.items():
        summary = {"model_id": model_id, "scope": "sample500_mean", "total_cases": 2000}
        for key in METRIC_KEYS:
            summary[key] = sum(_float(row, key) for row in model_rows) / max(1, len(model_rows))
        metric_rows.append(summary)

    two_rows = _read_csv(repo / TWO_CANDIDATE_ARCHIVE / "two_candidate_train_decode_dev_summary.csv")
    for row in two_rows:
        summary = {
            "model_id": row.get("variant", ""),
            "scope": "train_decode_dev",
            "total_cases": int(float(row.get("total_cases", 0) or 0)),
        }
        for key in METRIC_KEYS:
            summary[key] = _float(row, key)
        metric_rows.append(summary)
    return metric_rows


def _diagnose_errors(repo: Path, raw_by_case: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    issue_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    problem_counter: Counter[str] = Counter()

    for source in _error_sources(repo):
        rows = _read_jsonl(source.path)
        local_issues: Counter[str] = Counter()
        local_families: Counter[str] = Counter()
        local_problems: Counter[str] = Counter()
        high_risk_misses = 0
        reject_recommendation_errors = 0

        for row in rows:
            case_id = _case_id(row)
            gt = raw_by_case.get(case_id)
            issues = [str(issue) for issue in row.get("issues") or []]
            family = _anomaly_family(gt)
            problem = _problem_class(row, gt)
            local_issues.update(issues)
            local_families.update([family])
            local_problems.update([problem])
            issue_counter.update(issues)
            family_counter.update([family])
            problem_counter.update([problem])
            if "high_risk_miss" in issues:
                high_risk_misses += 1
            if row.get("truth_audit_result") == "reject_recommendation":
                reject_recommendation_errors += 1
            if len(examples) < 36 and (
                "high_risk_miss" in issues
                or problem in {"recognized_risk_but_decision_released", "evidence_not_trustworthy"}
            ):
                examples.append(
                    {
                        "source": source.label,
                        "model_id": source.model_id,
                        "split": source.split,
                        "case_id": case_id,
                        "issues": "|".join(issues),
                        "problem_class": problem,
                        "anomaly_family": family,
                        "truth_risk_level": row.get("truth_risk_level") or (gt or {}).get("risk_level"),
                        "pred_risk_level": row.get("pred_risk_level"),
                        "truth_audit_result": row.get("truth_audit_result") or (gt or {}).get("audit_result"),
                        "pred_audit_result": row.get("pred_audit_result"),
                    }
                )

        summaries.append(
            {
                "source": source.label,
                "model_id": source.model_id,
                "split": source.split,
                "error_cases": len(rows),
                "high_risk_miss": high_risk_misses,
                "reject_recommendation_errors": reject_recommendation_errors,
                "top_issues": "; ".join(f"{key}:{value}" for key, value in local_issues.most_common(5)),
                "top_anomaly_families": "; ".join(f"{key}:{value}" for key, value in local_families.most_common(5)),
                "top_problem_classes": "; ".join(f"{key}:{value}" for key, value in local_problems.most_common(5)),
            }
        )

    aggregate = {
        "issues": dict(issue_counter.most_common()),
        "anomaly_families": dict(family_counter.most_common()),
        "problem_classes": dict(problem_counter.most_common()),
    }
    return summaries, examples, aggregate


def _load_excluded_case_ids(repo: Path) -> dict[str, list[str]]:
    excluded: dict[str, set[str]] = {
        "dpo_v2_holdout": set(),
        "train_decode_dev": set(),
        "sample500": set(),
    }
    for row in _read_jsonl(repo / "data/mv_audit/dpo_v2/pairs_holdout.jsonl"):
        excluded["dpo_v2_holdout"].add(_case_id(row))
    for row in _read_jsonl(repo / "data/mv_audit/dpo_v2/train_decode_dev.jsonl"):
        excluded["train_decode_dev"].add(_case_id(row))
    archive = repo / TWO_CANDIDATE_ARCHIVE
    for variant in ["dpo_v2_baseline", "auxdpo_v2_strong"]:
        for kind in ["ground_truth", "predictions"]:
            for row in _read_jsonl(archive / f"{kind}/{variant}/train_decode_dev.jsonl"):
                excluded["train_decode_dev"].add(_case_id(row))
    for split in SAMPLE500_SPLITS:
        for row in _read_jsonl(repo / f"data/mv_audit/raw_cases/main/{split}_cases.jsonl"):
            excluded["sample500"].add(_case_id(row))
    return {key: sorted(values) for key, values in excluded.items()}


def _repair_strategy(answer: dict[str, Any], family: str) -> str:
    audit_result = answer.get("audit_result")
    if family == "amount_abnormal":
        return "SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass."
    if family == "order_id_inconsistent":
        return "SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence."
    if family == "evidence_or_document_insufficient":
        return "Rule-constrained SFT: missing/unsupported evidence must not be promoted to pass."
    if audit_result == "reject_recommendation":
        return "SFT reinforce high-risk reject decision with complete evidence and bbox references."
    return "SFT reinforce anomaly-to-risk-to-audit decision consistency."


def _candidate_score(answer: dict[str, Any], family: str, target_families: set[str]) -> int:
    score = 0
    if answer.get("risk_level") == "high":
        score += 5
    if answer.get("audit_result") == "reject_recommendation":
        score += 5
    elif answer.get("audit_result") in {"manual_review", "missing_info"}:
        score += 3
    if family in target_families:
        score += 3
    if family in {"amount_abnormal", "order_id_inconsistent", "evidence_or_document_insufficient"}:
        score += 2
    evidence = answer.get("evidence") or []
    if evidence and all("bbox" in item and "source_doc_type" in item for item in evidence if isinstance(item, dict)):
        score += 2
    return score


def _build_repair_pack(
    repo: Path,
    excluded: dict[str, list[str]],
    aggregate: dict[str, Any],
    *,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    excluded_sets = {key: set(values) for key, values in excluded.items()}
    all_excluded = set().union(*excluded_sets.values())
    target_families = {
        family
        for family, count in Counter(aggregate.get("anomaly_families") or {}).most_common(5)
        if family not in {"unknown", "clean_or_low_risk"}
    }
    target_families.update({"amount_abnormal", "order_id_inconsistent", "evidence_or_document_insufficient"})

    candidates: list[dict[str, Any]] = []
    repair_sft_rows: list[dict[str, Any]] = []
    family_seen: Counter[str] = Counter()
    inspected = 0
    train_only_rejected = 0
    excluded_hits: Counter[str] = Counter()
    sft_path = repo / "data/mv_audit/sft_main/train.jsonl"

    for row in _iter_jsonl(sft_path):
        inspected += 1
        case_id = _case_id(row)
        if case_id in all_excluded:
            for source, ids in excluded_sets.items():
                if case_id in ids:
                    excluded_hits[source] += 1
            continue
        if row.get("source_split") != "MV-Train":
            train_only_rejected += 1
            continue
        answer = _answer(row)
        if answer.get("risk_level") != "high":
            continue
        if answer.get("audit_result") == "pass":
            continue
        family = _anomaly_family(answer)
        evidence = answer.get("evidence") or []
        if not evidence:
            continue
        score = _candidate_score(answer, family, target_families)
        candidate = {
            "case_id": case_id,
            "sft_id": row.get("id"),
            "source_split": row.get("source_split"),
            "risk_level": answer.get("risk_level"),
            "audit_result": answer.get("audit_result"),
            "anomaly_types": answer.get("anomaly_types") or [],
            "anomaly_family": family,
            "score": score,
            "evidence_count": len(evidence),
            "doc_types": sorted({str(image.get("doc_type")) for image in row.get("images") or []}),
            "repair_strategy": _repair_strategy(answer, family),
        }
        candidates.append(candidate)
        repair_row = dict(row)
        repair_row["repair_metadata"] = {
            "pack": "phase08_high_risk_repair_pack_20260813",
            "anomaly_family": family,
            "score": score,
            "repair_strategy": candidate["repair_strategy"],
            "source_policy": "MV-Train high-risk non-pass only; excluded DPO holdout, train_decode_dev, and sample500 cases.",
        }
        repair_sft_rows.append(repair_row)
        family_seen.update([family])

    order = sorted(
        range(len(candidates)),
        key=lambda idx: (-int(candidates[idx]["score"]), str(candidates[idx]["anomaly_family"]), str(candidates[idx]["case_id"])),
    )
    by_family: dict[str, list[int]] = defaultdict(list)
    for idx in order:
        by_family[str(candidates[idx]["anomaly_family"])].append(idx)

    priority_families = [
        "amount_abnormal",
        "order_id_inconsistent",
        "evidence_or_document_insufficient",
        "date_mismatch",
        "person_mismatch",
        "merchant_mismatch",
        "duplicate_in_batch",
    ]
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    per_family_quota = max(1, max_candidates // max(1, len([f for f in priority_families if by_family.get(f)])))
    for family in priority_families:
        for idx in by_family.get(family, [])[:per_family_quota]:
            selected_indices.append(idx)
            selected_set.add(idx)
            if len(selected_indices) >= max_candidates:
                break
        if len(selected_indices) >= max_candidates:
            break
    for idx in order:
        if len(selected_indices) >= max_candidates:
            break
        if idx not in selected_set:
            selected_indices.append(idx)
            selected_set.add(idx)
    selected_candidates = [candidates[idx] for idx in selected_indices]
    selected_sft_rows = [repair_sft_rows[idx] for idx in selected_indices]
    selected_ids = {row["case_id"] for row in selected_candidates}
    overlap = {source: sorted(selected_ids.intersection(ids)) for source, ids in excluded_sets.items()}
    leakage = {
        "inspected_sft_rows": inspected,
        "candidate_pool_size": len(candidates),
        "selected_candidates": len(selected_candidates),
        "train_only_rejected": train_only_rejected,
        "excluded_case_counts": {key: len(value) for key, value in excluded_sets.items()},
        "excluded_hits_while_mining": dict(excluded_hits.most_common()),
        "selected_overlap_counts": {key: len(value) for key, value in overlap.items()},
        "selected_overlaps": overlap,
        "selected_family_counts": dict(Counter(row["anomaly_family"] for row in selected_candidates).most_common()),
        "candidate_pool_family_counts": dict(family_seen.most_common()),
    }
    return selected_candidates, selected_sft_rows, leakage


def _markdown_table(rows: list[dict[str, Any]], fields: list[str], *, limit: int | None = None) -> list[str]:
    rows = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: Path,
    *,
    metric_rows: list[dict[str, Any]],
    error_summaries: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    aggregate: dict[str, Any],
    candidates: list[dict[str, Any]],
    leakage: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Phase08 High-risk Miss Diagnosis and Repair Pack")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "- DPO v2 two-candidate Train decode dev did not move core business metrics: baseline and AuxDPO have the same Audit Accuracy and High-risk Miss Rate."
    )
    lines.append(
        "- The repair direction is therefore data/contract focused: separate schema failures from real business misses, then reinforce high-risk non-pass cases from MV-Train only."
    )
    lines.append(
        "- This pack is a candidate set for a low-cost SFT/rule-constrained validation run; it is not evidence for sample500 generalization yet."
    )
    lines.append("")
    lines.append("## Metric Snapshot")
    lines.extend(_markdown_table(metric_rows, ["model_id", "scope", *METRIC_KEYS]))
    lines.append("")
    lines.append("## Error Source Summary")
    lines.extend(
        _markdown_table(
            error_summaries,
            [
                "source",
                "split",
                "error_cases",
                "high_risk_miss",
                "reject_recommendation_errors",
                "top_issues",
                "top_anomaly_families",
                "top_problem_classes",
            ],
        )
    )
    lines.append("")
    lines.append("## Aggregate Error Mechanisms")
    for title, key in [
        ("Issue tags", "issues"),
        ("Anomaly families", "anomaly_families"),
        ("Problem classes", "problem_classes"),
    ]:
        lines.append(f"### {title}")
        for name, count in (aggregate.get(key) or {}).items():
            lines.append(f"- {name}: {count}")
        lines.append("")
    lines.append("## Representative High-risk Cases")
    lines.extend(
        _markdown_table(
            examples,
            [
                "source",
                "split",
                "case_id",
                "issues",
                "problem_class",
                "anomaly_family",
                "truth_risk_level",
                "pred_risk_level",
                "truth_audit_result",
                "pred_audit_result",
            ],
            limit=24,
        )
    )
    lines.append("")
    lines.append("## Repair Strategy")
    lines.append("")
    lines.append("- Schema contract failures: strengthen output-format constraints before interpreting business metrics.")
    lines.append("- Model missed high risk: add MV-Train high-risk non-pass SFT samples with complete evidence and bbox.")
    lines.append("- Recognized risk but released decision: reinforce anomaly-to-risk-to-audit mapping, especially reject_recommendation.")
    lines.append("- Evidence not trustworthy: keep only candidates with source_doc_type, evidence text, and bbox references.")
    lines.append("")
    lines.append("## Repair Pack")
    lines.append("")
    lines.append(f"- selected_candidates: {len(candidates)}")
    lines.append(f"- candidate_pool_size: {leakage.get('candidate_pool_size')}")
    lines.append(f"- selected_family_counts: {json.dumps(leakage.get('selected_family_counts', {}), ensure_ascii=False)}")
    lines.append("")
    lines.extend(
        _markdown_table(
            candidates,
            ["case_id", "anomaly_family", "score", "evidence_count", "repair_strategy"],
            limit=30,
        )
    )
    lines.append("")
    lines.append("## Leakage Check")
    lines.append("")
    lines.append(f"- selected_overlap_counts: {json.dumps(leakage.get('selected_overlap_counts', {}), ensure_ascii=False)}")
    lines.append("- Source policy: MV-Train high-risk non-pass only; exclude DPO holdout, train_decode_dev, and sample500 case ids.")
    lines.append("")
    lines.append("## Low-cost Validation Gate")
    lines.append("")
    lines.append("- JSON Validity must remain 1.0.")
    lines.append("- Audit Accuracy must not fall below M2, or may drop by at most 0.01.")
    lines.append("- High-risk Miss Rate must improve by at least 0.03 versus M2.")
    lines.append("- Evidence Support Rate may drop by at most 0.01.")
    lines.append("- If error cases improve but High-risk Miss Rate does not, stop the training line and write Phase08 as a DPO negative result.")
    lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build(repo: Path, output_dir: Path, max_candidates: int) -> dict[str, Any]:
    raw_by_case = _load_main_raw_cases(repo)
    metric_rows = _summarize_metrics(repo)
    error_summaries, examples, aggregate = _diagnose_errors(repo, raw_by_case)
    excluded = _load_excluded_case_ids(repo)
    candidates, repair_sft_rows, leakage = _build_repair_pack(
        repo,
        excluded,
        aggregate,
        max_candidates=max_candidates,
    )

    ensure_dir(output_dir)
    _write_csv(
        output_dir / "metric_snapshot.csv",
        metric_rows,
        ["model_id", "scope", "total_cases", *METRIC_KEYS],
    )
    _write_csv(
        output_dir / "error_source_summary.csv",
        error_summaries,
        [
            "source",
            "model_id",
            "split",
            "error_cases",
            "high_risk_miss",
            "reject_recommendation_errors",
            "top_issues",
            "top_anomaly_families",
            "top_problem_classes",
        ],
    )
    _write_jsonl(output_dir / "representative_high_risk_cases.jsonl", examples)
    _write_jsonl(output_dir / "candidate_cases.jsonl", candidates)
    _write_jsonl(output_dir / "repair_pack_sft.jsonl", repair_sft_rows)
    _write_json(output_dir / "leakage_check.json", leakage)
    manifest = {
        "name": "phase08_high_risk_repair_pack_20260813",
        "inputs": {
            "two_candidate_archive": TWO_CANDIDATE_ARCHIVE,
            "sample500_metrics": "docs/experiments/phase08_m3v2_sample500/m2_m3_m3v2_split_metrics.csv",
            "sft_train": "data/mv_audit/sft_main/train.jsonl",
            "raw_cases_main": "data/mv_audit/raw_cases/main",
        },
        "outputs": [
            "high_risk_miss_diagnosis_report.md",
            "metric_snapshot.csv",
            "error_source_summary.csv",
            "representative_high_risk_cases.jsonl",
            "candidate_cases.jsonl",
            "repair_pack_sft.jsonl",
            "leakage_check.json",
        ],
        "selected_candidates": len(candidates),
        "selected_overlap_counts": leakage.get("selected_overlap_counts"),
            "policy": "Train-only high-risk non-pass repair candidates; no sample500/test/holdout/decode overlap.",
    }
    _write_json(output_dir / "repair_pack_manifest.json", manifest)
    _write_report(
        output_dir / "high_risk_miss_diagnosis_report.md",
        metric_rows=metric_rows,
        error_summaries=error_summaries,
        examples=examples,
        aggregate=aggregate,
        candidates=candidates,
        leakage=leakage,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    manifest = build(repo, output_dir, max_candidates=args.max_candidates)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
