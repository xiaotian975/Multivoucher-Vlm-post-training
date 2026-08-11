"""Build conservative DPO v2 pairs from MV-Train only.

The converter intentionally reads only a train cases file and train annotation
file. It creates case-level disjoint DPO train, Train-only holdout, and
Train-only decode-dev splits before constructing preference pairs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.converters.common import (
    build_audit_output,
    build_prompt,
    existing_image_items,
    json_answer,
    output_validator,
    validate_output,
)
from mv_audit.utils import ensure_dir, iter_jsonl, read_jsonl, write_jsonl


PAIR_TYPE_RATIOS = {
    "hard_rejected": 0.35,
    "high_risk_miss": 0.30,
    "protective": 0.20,
    "normal_calibration": 0.15,
}

PROTECTIVE_ANOMALIES = {
    "amount_mismatch",
    "over_reimbursement",
    "order_id_mismatch",
    "merchant_mismatch",
    "applicant_mismatch",
    "missing_document",
    "unreadable_image",
}


def _loads_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _schema_valid(text: str, validator: Draft202012Validator) -> bool:
    output = _loads_json(text)
    if output is None:
        return False
    return not list(validator.iter_errors(output))


def _bbox_range_valid(output: dict[str, Any]) -> bool:
    for evidence in output.get("evidence") or []:
        bbox = evidence.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if any(not isinstance(value, (int, float)) or value < 0 or value > 1000 for value in bbox):
            return False
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            return False
    return True


def _evidence_refs_valid(output: dict[str, Any], image_items: list[dict[str, str]]) -> bool:
    images_by_id = {str(item["image_id"]): str(item["doc_type"]) for item in image_items}
    for evidence in output.get("evidence") or []:
        image_id = str(evidence.get("source_image_id") or "")
        doc_type = str(evidence.get("source_doc_type") or "")
        if image_id not in images_by_id:
            return False
        if doc_type and doc_type != images_by_id[image_id]:
            return False
    return True


def _images_exist(image_items: list[dict[str, str]]) -> bool:
    return all(Path(str(item.get("image_path") or "")).is_file() for item in image_items)


def _base_rejected(chosen: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(chosen, ensure_ascii=False))


def _set_first_evidence_reasonable_but_wrong(rejected: dict[str, Any]) -> None:
    if rejected.get("evidence"):
        rejected["evidence"][0]["evidence_text"] = "该字段已被读取，但结论未充分比较跨凭证差异。"


def _make_rejected(case: dict[str, Any], chosen: dict[str, Any], pair_type: str) -> tuple[dict[str, Any], str]:
    anomaly_types = list(case.get("anomaly_types") or [])
    primary = str(case.get("primary_anomaly_type") or "none")
    rejected = _base_rejected(chosen)

    if pair_type == "high_risk_miss":
        rejected["risk_level"] = "low"
        rejected["audit_result"] = "pass"
        rejected["reason"] = "未发现高风险异常，材料可自动通过。"
        rejected["anomaly_types"] = [name for name in anomaly_types if name not in PROTECTIVE_ANOMALIES]
        _set_first_evidence_reasonable_but_wrong(rejected)
        return rejected, "high_risk_pass"

    if pair_type == "protective":
        if primary in {"amount_mismatch", "over_reimbursement"}:
            rejected["risk_level"] = "medium"
            rejected["audit_result"] = "manual_review"
            rejected["reason"] = "金额差异较小，可先按中风险转人工复核。"
            return rejected, "high_risk_downgrade"
        if primary == "order_id_mismatch":
            rejected["risk_level"] = "low"
            rejected["audit_result"] = "pass"
            rejected["reason"] = "订单号虽然显示不同，但可视为同一订单的展示差异。"
            return rejected, "audit_result_wrong"
        rejected["risk_level"] = "medium" if case.get("risk_level") == "high" else "low"
        rejected["audit_result"] = "manual_review" if rejected["risk_level"] == "medium" else "pass"
        rejected["reason"] = "证据看似完整，暂不建议拒绝。"
        return rejected, "risk_level_wrong"

    if pair_type == "normal_calibration":
        rejected["risk_level"] = "high"
        rejected["audit_result"] = "reject_recommendation"
        rejected["anomaly_types"] = ["amount_mismatch"] if not anomaly_types else anomaly_types
        rejected["reason"] = "正常或低风险样本被错误提升为高风险拒绝。"
        return rejected, "normal_over_reject"

    if pair_type == "hard_rejected":
        if case.get("risk_level") == "high":
            rejected["risk_level"] = "medium"
            rejected["audit_result"] = "manual_review"
            rejected["reason"] = "高风险证据存在，但被错误降级为人工复核。"
            return rejected, "high_risk_downgrade"
        if case.get("audit_result") != "pass":
            rejected["audit_result"] = "pass"
            rejected["risk_level"] = "low"
            rejected["reason"] = "材料看似完整，因此错误建议通过。"
            return rejected, "audit_result_wrong"
        rejected["risk_level"] = "medium"
        rejected["audit_result"] = "manual_review"
        rejected["reason"] = "证据不足，错误转为人工复核。"
        return rejected, "risk_level_wrong"

    raise ValueError(f"Unsupported pair_type: {pair_type}")


def _eligible_pair_types(case: dict[str, Any]) -> list[str]:
    anomaly_types = set(case.get("anomaly_types") or [])
    risk = str(case.get("risk_level") or "")
    audit_result = str(case.get("audit_result") or "")
    eligible = ["hard_rejected"]
    if risk == "high":
        eligible.append("high_risk_miss")
    if risk == "high" and anomaly_types.intersection(PROTECTIVE_ANOMALIES):
        eligible.append("protective")
    if risk in {"low", "medium"} or audit_result in {"pass", "manual_review", "missing_info"}:
        eligible.append("normal_calibration")
    return eligible


def _severity_weight(rejected_error_type: str) -> float:
    return {
        "high_risk_pass": 3.0,
        "high_risk_downgrade": 2.5,
        "audit_result_wrong": 2.0,
        "risk_level_wrong": 1.8,
        "missing_key_anomaly": 1.5,
        "evidence_wrong": 1.2,
        "normal_over_reject": 1.5,
        "schema_minor": 0.8,
    }.get(rejected_error_type, 1.0)


def _hardness_weight(pair_type: str, rejected_error_type: str) -> float:
    if pair_type in {"hard_rejected", "protective"} and rejected_error_type in {
        "audit_result_wrong",
        "risk_level_wrong",
        "high_risk_downgrade",
    }:
        return 1.5
    if pair_type == "high_risk_miss":
        return 1.4
    if pair_type == "normal_calibration":
        return 1.0
    return 1.0


def _quality_checks(
    *,
    chosen_text: str,
    rejected_text: str,
    image_items: list[dict[str, str]],
    validator: Draft202012Validator,
) -> dict[str, bool]:
    chosen = _loads_json(chosen_text)
    rejected = _loads_json(rejected_text)
    return {
        "chosen_json_valid": chosen is not None,
        "rejected_json_valid": rejected is not None,
        "chosen_schema_valid": _schema_valid(chosen_text, validator),
        "rejected_schema_valid": _schema_valid(rejected_text, validator),
        "images_exist": _images_exist(image_items),
        "evidence_refs_valid": bool(chosen)
        and bool(rejected)
        and _evidence_refs_valid(chosen, image_items)
        and _evidence_refs_valid(rejected, image_items),
        "bbox_range_valid": bool(chosen) and bool(rejected) and _bbox_range_valid(chosen) and _bbox_range_valid(rejected),
    }


def _candidate_signature(case: dict[str, Any], pair_type: str) -> tuple[str, str, str, str]:
    metadata = case.get("metadata") or {}
    return (
        pair_type,
        str(case.get("primary_anomaly_type") or "none"),
        str(case.get("risk_level") or ""),
        str(metadata.get("template_group") or metadata.get("split_name") or "train"),
    )


def _quotas(max_pairs: int) -> dict[str, int]:
    remaining = max_pairs
    quotas: dict[str, int] = {}
    ordered = list(PAIR_TYPE_RATIOS.items())
    for index, (name, ratio) in enumerate(ordered):
        if index == len(ordered) - 1:
            quotas[name] = remaining
        else:
            value = int(round(max_pairs * ratio))
            quotas[name] = value
            remaining -= value
    return quotas


def _split_case_ids(
    cases: list[dict[str, Any]],
    *,
    seed: int,
    holdout_ratio: float,
    decode_dev_cases: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed + 802)
    ids = [str(case["case_id"]) for case in cases]
    rng.shuffle(ids)
    holdout_count = max(1, int(len(ids) * holdout_ratio)) if ids else 0
    decode_count = min(max(0, decode_dev_cases), max(0, len(ids) - holdout_count))
    holdout_ids = set(ids[:holdout_count])
    decode_ids = set(ids[holdout_count : holdout_count + decode_count])
    train_ids = set(ids[holdout_count + decode_count :])
    return train_ids, holdout_ids, decode_ids


def _case_records(records_by_case: dict[str, list[dict[str, Any]]], case_id: str) -> list[dict[str, Any]]:
    records = records_by_case.get(case_id)
    if not records:
        raise ValueError(f"Missing annotation records for train case_id={case_id}")
    return records


def _read_cases(path: str | Path, *, max_input_cases: int | None) -> list[dict[str, Any]]:
    if max_input_cases is None:
        return read_jsonl(path)
    cases: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        cases.append(row)
        if len(cases) >= max_input_cases:
            break
    return cases


def _read_records_for_cases(path: str | Path, case_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        case_id = str(row.get("case_id") or "")
        if case_id in case_ids:
            grouped[case_id].append(row)
    return grouped


def _build_pair_for_case(
    *,
    case: dict[str, Any],
    records: list[dict[str, Any]],
    pair_type: str,
    schema_path: str,
    validator: Draft202012Validator,
    rng: random.Random,
    max_weight: float,
    source_role: str,
) -> tuple[dict[str, Any] | None, str | None]:
    output = build_audit_output(case, records)
    validate_output(output, validator)
    image_items = existing_image_items(records, rng=rng)
    prompt = build_prompt(case, image_items, task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。")
    rejected, rejected_error_type = _make_rejected(case, output, pair_type)
    chosen_text = json_answer(output)
    rejected_text = json_answer(rejected)
    checks = _quality_checks(
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        image_items=image_items,
        validator=validator,
    )
    if not checks["images_exist"]:
        return None, "missing_images"
    if not checks["chosen_schema_valid"] or not checks["rejected_schema_valid"]:
        return None, "schema_invalid"
    if not checks["evidence_refs_valid"]:
        return None, "evidence_refs_invalid"
    if not checks["bbox_range_valid"]:
        return None, "bbox_range_invalid"

    severity = _severity_weight(rejected_error_type)
    hardness = _hardness_weight(pair_type, rejected_error_type)
    reliability = 1.0 if case.get("evidence_sufficient", True) else 0.5
    final_weight = min(max_weight, severity * hardness * reliability)
    pair_id = f"{case['case_id']}_{pair_type}_{rejected_error_type}"
    return (
        {
            "id": pair_id,
            "pair_id": pair_id,
            "case_id": case["case_id"],
            "pair_type": pair_type,
            "primary_anomaly_type": case.get("primary_anomaly_type") or "none",
            "risk_level_gt": case.get("risk_level"),
            "audit_result_gt": case.get("audit_result"),
            "rejected_error_type": rejected_error_type,
            "severity_weight": severity,
            "hardness_weight": hardness,
            "reliability_weight": reliability,
            "final_weight": final_weight,
            "sft_loss_weight": 1.0 if pair_type in {"hard_rejected", "high_risk_miss", "protective"} else 0.5,
            "source": "train_rule_generated",
            "source_role": source_role,
            "source_split": "MV-Train",
            "generator_version": "dpo_v2_rule_generated_v1",
            "images": image_items,
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "sft_target": chosen_text,
            "quality_checks": checks,
            "split_guard": {
                "case_level_split": source_role,
                "train_only": True,
                "schema_path": schema_path,
            },
        },
        None,
    )


def _select_pairs(
    candidates: list[dict[str, Any]],
    *,
    max_pairs: int,
    max_anomaly_ratio: float,
) -> list[dict[str, Any]]:
    quotas = _quotas(max_pairs)
    selected: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in candidates:
        by_type[str(pair["pair_type"])].append(pair)
    anomaly_counts: Counter[str] = Counter()
    max_per_anomaly = max(1, int(max_pairs * max_anomaly_ratio))

    for pair_type, quota in quotas.items():
        for pair in by_type.get(pair_type, []):
            anomaly = str(pair.get("primary_anomaly_type") or "none")
            if anomaly_counts[anomaly] >= max_per_anomaly:
                continue
            selected.append(pair)
            anomaly_counts[anomaly] += 1
            if len([item for item in selected if item["pair_type"] == pair_type]) >= quota:
                break

    if len(selected) < max_pairs:
        selected_ids = {str(pair["pair_id"]) for pair in selected}
        for pair in candidates:
            if str(pair["pair_id"]) in selected_ids:
                continue
            anomaly = str(pair.get("primary_anomaly_type") or "none")
            if anomaly_counts[anomaly] >= max_per_anomaly:
                continue
            selected.append(pair)
            selected_ids.add(str(pair["pair_id"]))
            anomaly_counts[anomaly] += 1
            if len(selected) >= max_pairs:
                break
    return selected[:max_pairs]


def _decode_dev_rows(
    *,
    cases: list[dict[str, Any]],
    records_by_case: dict[str, list[dict[str, Any]]],
    decode_ids: set[str],
    validator: Draft202012Validator,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if str(case["case_id"]) not in decode_ids:
            continue
        records = _case_records(records_by_case, str(case["case_id"]))
        output = build_audit_output(case, records)
        validate_output(output, validator)
        image_items = existing_image_items(records, rng=rng)
        prompt = build_prompt(case, image_items, task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。")
        if not _images_exist(image_items):
            continue
        rows.append(
            {
                "case_id": case["case_id"],
                "split": "train_decode_dev",
                "model_id": "ground_truth",
                "images": image_items,
                "prompt": prompt,
                "answer": output,
                "source_split": "MV-Train",
                "split_guard": {"case_level_split": "train_decode_dev", "train_only": True},
            }
        )
    return rows


def _write_report(
    *,
    path: Path,
    train_pairs: list[dict[str, Any]],
    holdout_pairs: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    skipped: Counter[str],
    train_ids: set[str],
    holdout_ids: set[str],
    decode_ids: set[str],
    input_cases: int,
    cases_path: str,
    annotations_path: str,
) -> None:
    all_pairs = train_pairs + holdout_pairs
    pair_types = Counter(str(pair["pair_type"]) for pair in all_pairs)
    anomalies = Counter(str(pair.get("primary_anomaly_type") or "none") for pair in all_pairs)
    rejected_types = Counter(str(pair["rejected_error_type"]) for pair in all_pairs)
    weights = [float(pair["final_weight"]) for pair in all_pairs]
    quality_totals = Counter()
    for pair in all_pairs:
        for key, value in (pair.get("quality_checks") or {}).items():
            if value:
                quality_totals[key] += 1
    overlap_train_holdout = sorted(train_ids & holdout_ids)
    overlap_train_decode = sorted(train_ids & decode_ids)
    overlap_holdout_decode = sorted(holdout_ids & decode_ids)
    payload = {
        "stage": "dpo_v2_pair_build",
        "train_only": True,
        "source_cases": cases_path,
        "source_annotations": annotations_path,
        "input_cases": input_cases,
        "outputs": {
            "train_pairs": len(train_pairs),
            "holdout_pairs": len(holdout_pairs),
            "decode_dev_rows": len(decode_rows),
        },
        "skipped": dict(skipped),
        "pair_type_counts": dict(pair_types),
        "primary_anomaly_type_counts": dict(anomalies),
        "rejected_error_type_counts": dict(rejected_types),
        "weight_distribution": {
            "min": min(weights) if weights else 0.0,
            "max": max(weights) if weights else 0.0,
            "mean": sum(weights) / len(weights) if weights else 0.0,
        },
        "quality_pass_counts": dict(quality_totals),
        "case_level_split": {
            "train_case_ids": len(train_ids),
            "holdout_case_ids": len(holdout_ids),
            "decode_dev_case_ids": len(decode_ids),
            "train_holdout_overlap": len(overlap_train_holdout),
            "train_decode_dev_overlap": len(overlap_train_decode),
            "holdout_decode_dev_overlap": len(overlap_holdout_decode),
            "overlap_examples": {
                "train_holdout": overlap_train_holdout[:5],
                "train_decode_dev": overlap_train_decode[:5],
                "holdout_decode_dev": overlap_holdout_decode[:5],
            },
        },
        "leakage_guard": {
            "val_or_test_inputs_read": False,
            "sample500_inputs_read": False,
            "model_mined_pairs_used": False,
            "model_mined_pairs_note": "No Train-only M2 predictions were present locally; v2 uses train_rule_generated pairs only.",
        },
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dpo_v2(
    *,
    cases_path: str,
    annotations_path: str,
    output_schema: str,
    output_dir: str,
    max_pairs: int,
    holdout_ratio: float,
    decode_dev_cases: int,
    seed: int,
    max_weight: float,
    max_anomaly_ratio: float,
    max_input_cases: int | None,
) -> dict[str, Path]:
    rng = random.Random(seed + 901)
    validator = output_validator(output_schema)
    cases = _read_cases(cases_path, max_input_cases=max_input_cases)
    case_ids = {str(case["case_id"]) for case in cases}
    records_by_case = _read_records_for_cases(annotations_path, case_ids)
    train_ids, holdout_ids, decode_ids = _split_case_ids(
        cases,
        seed=seed,
        holdout_ratio=holdout_ratio,
        decode_dev_cases=decode_dev_cases,
    )

    skipped: Counter[str] = Counter()
    train_candidates: list[dict[str, Any]] = []
    holdout_pairs: list[dict[str, Any]] = []
    shuffled = list(cases)
    rng.shuffle(shuffled)
    holdout_quota = max(1, int(max_pairs * holdout_ratio))

    for case in shuffled:
        case_id = str(case["case_id"])
        if case_id in decode_ids:
            continue
        if case_id not in train_ids and case_id not in holdout_ids:
            continue
        records = _case_records(records_by_case, case_id)
        for pair_type in _eligible_pair_types(case):
            pair, skip_reason = _build_pair_for_case(
                case=case,
                records=records,
                pair_type=pair_type,
                schema_path=output_schema,
                validator=validator,
                rng=rng,
                max_weight=max_weight,
                source_role="holdout" if case_id in holdout_ids else "train",
            )
            if pair is None:
                skipped[str(skip_reason)] += 1
                continue
            if case_id in holdout_ids and len(holdout_pairs) < holdout_quota:
                holdout_pairs.append(pair)
            elif case_id in train_ids:
                train_candidates.append(pair)

    train_pairs = _select_pairs(train_candidates, max_pairs=max_pairs, max_anomaly_ratio=max_anomaly_ratio)
    decode_rows = _decode_dev_rows(
        cases=cases,
        records_by_case=records_by_case,
        decode_ids=decode_ids,
        validator=validator,
        rng=rng,
    )
    output_root = ensure_dir(output_dir)
    paths = {
        "train": output_root / "pairs_train.jsonl",
        "holdout": output_root / "pairs_holdout.jsonl",
        "decode_dev": output_root / "train_decode_dev.jsonl",
        "report": output_root / "pair_report.json",
    }
    write_jsonl(train_pairs, paths["train"])
    write_jsonl(holdout_pairs, paths["holdout"])
    write_jsonl(decode_rows, paths["decode_dev"])
    _write_report(
        path=paths["report"],
        train_pairs=train_pairs,
        holdout_pairs=holdout_pairs,
        decode_rows=decode_rows,
        skipped=skipped,
        train_ids=train_ids,
        holdout_ids=holdout_ids,
        decode_ids=decode_ids,
        input_cases=len(cases),
        cases_path=cases_path,
        annotations_path=annotations_path,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Train-only DPO v2 pairs and reports.")
    parser.add_argument("--cases", default="data/mv_audit/raw_cases/main/train_cases.jsonl")
    parser.add_argument("--annotations", default="data/mv_audit/annotations_main/field_bboxes_train.jsonl")
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--output_dir", default="data/mv_audit/dpo_v2")
    parser.add_argument("--max_pairs", type=int, default=3000)
    parser.add_argument("--holdout_ratio", type=float, default=0.10)
    parser.add_argument("--decode_dev_cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_weight", type=float, default=3.0)
    parser.add_argument("--max_anomaly_ratio", type=float, default=0.25)
    parser.add_argument("--max_input_cases", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_dpo_v2(
        cases_path=args.cases,
        annotations_path=args.annotations,
        output_schema=args.output_schema,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        holdout_ratio=args.holdout_ratio,
        decode_dev_cases=args.decode_dev_cases,
        seed=args.seed,
        max_weight=args.max_weight,
        max_anomaly_ratio=args.max_anomaly_ratio,
        max_input_cases=args.max_input_cases,
    )
    print(f"dpo_v2_train={paths['train']}")
    print(f"dpo_v2_holdout={paths['holdout']}")
    print(f"dpo_v2_decode_dev={paths['decode_dev']}")
    print(f"dpo_v2_pair_report={paths['report']}")


if __name__ == "__main__":
    main()
