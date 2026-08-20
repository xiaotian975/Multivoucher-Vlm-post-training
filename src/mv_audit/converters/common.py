"""Shared helpers for phase 05 training-format converters."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.utils import read_jsonl, read_yaml


FIELD_DOC_TYPES = {
    "invoice_amount": {"invoice"},
    "tax_amount": {"invoice"},
    "invoice_merchant": {"invoice"},
    "invoice_date": {"invoice"},
    "invoice_id": {"invoice"},
    "payment_amount": {"payment"},
    "payment_merchant": {"payment"},
    "payment_date": {"payment"},
    "payment_id": {"payment"},
    "payer": {"payment"},
    "reimbursement_amount": {"reimbursement_form"},
    "applicant": {"reimbursement_form"},
    "application_date": {"reimbursement_form"},
    "order_amount": {"order"},
    "order_merchant": {"order"},
    "order_date": {"order"},
    "order_user": {"order"},
    "order_id": {"reimbursement_form", "order"},
    "expense_type": {"invoice", "reimbursement_form", "order"},
}

FIELD_NAMES = [
    "invoice_amount",
    "payment_amount",
    "reimbursement_amount",
    "order_amount",
    "tax_amount",
    "invoice_merchant",
    "payment_merchant",
    "order_merchant",
    "merchant",
    "applicant",
    "payer",
    "order_user",
    "invoice_date",
    "payment_date",
    "order_date",
    "application_date",
    "invoice_id",
    "order_id",
    "payment_id",
    "expense_type",
]

DOC_LABELS = {
    "invoice": "发票",
    "payment": "支付截图",
    "reimbursement_form": "报销申请单",
    "order": "订单截图",
}


def read_output_schema(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)


def output_validator(schema_path: str | Path) -> Draft202012Validator:
    return Draft202012Validator(read_output_schema(schema_path))


def validate_output(output: dict[str, Any], validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(output), key=lambda err: err.path)
    if errors:
        err = errors[0]
        path = ".".join(str(part) for part in err.path) or "<root>"
        raise ValueError(f"Output schema validation failed at {path}: {err.message}")


def group_records_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)
    return grouped


def existing_image_items(records: list[dict[str, Any]], *, rng: random.Random) -> list[dict[str, str]]:
    images: dict[str, dict[str, str]] = {}
    for record in records:
        image_id = record["image_id"]
        images.setdefault(
            image_id,
            {
                "image_id": image_id,
                "doc_type": record["doc_type"],
                "image_path": record["image_path"],
            },
        )
    values = list(images.values())
    rng.shuffle(values)
    return values


def build_prompt(case: dict[str, Any], image_items: list[dict[str, str]], *, task_instruction: str) -> str:
    image_lines = [
        f"- {item['image_id']}: {DOC_LABELS[item['doc_type']]} ({item['doc_type']}), path={item['image_path']}"
        for item in image_items
    ]
    return (
        "你是企业费用报销一致性审计助手。请只输出一个合法 JSON，不要输出解释性段落。\n"
        "必须使用字段 case_id、field_extraction、consistency_check、anomaly_types、risk_level、"
        "audit_result、reason、evidence、uncertainty。\n"
        "不得输出 primary_anomaly_type 或 evidence_sufficient。\n"
        "检查订单号时必须分别读取订单截图(order)和报销申请单(reimbursement_form)中的 order_id；"
        "如果两者可见且不同，必须设置 consistency_check.order_id_consistent=false，"
        "anomaly_types 包含 order_id_mismatch，risk_level=high，audit_result=reject_recommendation，"
        "并在 evidence 中同时给出这两处 order_id 证据。\n"
        "bbox 必须使用 0 到 1000 的 [x1, y1, x2, y2] 归一化坐标。\n"
        "输入图片如下：\n"
        + "\n".join(image_lines)
        + f"\n\n任务：{task_instruction}\ncase_id={case['case_id']}"
    )


def _record_to_evidence(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_image_id": record["image_id"],
        "source_doc_type": record["doc_type"],
        "field": record["field"],
        "value": str(record["value"]),
        "bbox": record["bbox_norm"],
        "evidence_text": record["evidence_text"],
    }


def _field_value(case: dict[str, Any], field: str) -> str | None:
    if field == "merchant":
        return case.get("merchant_canonical")
    return case.get(field)


def _is_field_readable(field: str, records: list[dict[str, Any]]) -> bool:
    matching = [record for record in records if record["field"] == field]
    return bool(matching) and any(record.get("readable", True) for record in matching)


def build_field_extraction(case: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    unreadable_fields = set((case.get("metadata") or {}).get("unreadable_fields") or [])
    for field in FIELD_NAMES:
        if field == "merchant":
            values[field] = case.get("merchant_canonical")
            continue
        if field in unreadable_fields:
            values[field] = None
            continue
        expected_docs = FIELD_DOC_TYPES.get(field, set())
        if expected_docs and not expected_docs.intersection(set(case.get("documents") or [])):
            values[field] = None
        elif not _is_field_readable(field, records):
            values[field] = None
        else:
            values[field] = _field_value(case, field)
    return values


def build_consistency_check(case: dict[str, Any]) -> dict[str, bool]:
    anomalies = set(case.get("anomaly_types") or [])
    return {
        "amount_consistent": not bool(anomalies & {"amount_mismatch", "over_reimbursement"}),
        "merchant_consistent": "merchant_mismatch" not in anomalies,
        "person_consistent": "applicant_mismatch" not in anomalies,
        "date_reasonable": "date_mismatch" not in anomalies,
        "order_id_consistent": "order_id_mismatch" not in anomalies,
        "payment_id_present": "payment" in set(case.get("documents") or []),
        "document_complete": "missing_document" not in anomalies,
        "duplicate_in_batch": "duplicate_in_batch" in anomalies,
    }


def build_evidence(case: dict[str, Any], records: list[dict[str, Any]], *, max_items: int = 24) -> list[dict[str, Any]]:
    metadata = case.get("metadata") or {}
    unreadable_fields = set(metadata.get("unreadable_fields") or [])
    anomaly_fields = {
        "amount_mismatch": {"invoice_amount", "payment_amount", "reimbursement_amount", "order_amount"},
        "over_reimbursement": {"invoice_amount", "payment_amount", "reimbursement_amount"},
        "date_mismatch": {"order_date", "payment_date", "invoice_date", "application_date"},
        "merchant_mismatch": {"invoice_merchant", "payment_merchant", "order_merchant"},
        "applicant_mismatch": {"applicant", "payer", "order_user"},
        "order_id_mismatch": {"order_id"},
        "duplicate_in_batch": {"invoice_id", "order_id"},
        "unreadable_image": unreadable_fields,
    }
    wanted = set().union(*(anomaly_fields.get(anomaly, set()) for anomaly in case.get("anomaly_types") or []))
    wanted.update({"invoice_amount", "payment_amount", "reimbursement_amount", "order_amount", "applicant", "payment_id"})
    wanted -= unreadable_fields

    readable = [
        record
        for record in records
        if record.get("readable", True) and not record.get("duplicate_of_image_id") and record["field"] in wanted
    ]
    readable.sort(key=lambda record: (record["doc_type"], record["field"], record["image_id"]))
    evidence = [_record_to_evidence(record) for record in readable[:max_items]]
    return evidence


def build_uncertainty(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata") or {}
    uncertain_fields = list(metadata.get("unreadable_fields") or [])
    has_uncertain = bool(uncertain_fields) or not bool(case.get("evidence_sufficient", True))
    return {
        "has_uncertain_fields": has_uncertain,
        "uncertain_fields": uncertain_fields,
        "requires_manual_review": case.get("audit_result") in {"manual_review", "missing_info"},
    }


def build_reason(case: dict[str, Any]) -> str:
    anomaly_types = case.get("anomaly_types") or []
    metadata = case.get("metadata") or {}
    if not anomaly_types:
        return "材料完整且金额、商户、人员、日期与订单号一致，建议通过。"
    if "missing_document" in anomaly_types:
        missing = metadata.get("missing_doc_type", "unknown")
        return f"缺少{DOC_LABELS.get(missing, missing)}，证据不足，需要补充材料。"
    if "unreadable_image" in anomaly_types:
        fields = "、".join(metadata.get("unreadable_fields") or [])
        return f"存在不可读字段：{fields}，无法可靠完成自动审核。"
    labels = "、".join(anomaly_types)
    return f"检测到{labels}，风险等级为{case['risk_level']}，审核建议为{case['audit_result']}。"


def build_audit_output(case: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "field_extraction": build_field_extraction(case, records),
        "consistency_check": build_consistency_check(case),
        "anomaly_types": list(case.get("anomaly_types") or []),
        "risk_level": case["risk_level"],
        "audit_result": case["audit_result"],
        "reason": build_reason(case),
        "evidence": build_evidence(case, records),
        "uncertainty": build_uncertainty(case),
    }


def json_answer(output: dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", default="data/mv_audit/raw_cases/train_cases.jsonl")
    parser.add_argument("--annotations", default="data/mv_audit/annotations/field_bboxes_train.jsonl")
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--seed", type=int, default=42)
