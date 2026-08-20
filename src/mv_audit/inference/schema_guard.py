"""Schema guard helpers for MultiVoucher-Audit model outputs."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.json_parser import parse_json_output
from mv_audit.utils import ensure_dir, read_jsonl, read_yaml, write_jsonl


DEFAULT_TOP_LEVEL_KEYS = {
    "case_id",
    "field_extraction",
    "consistency_check",
    "anomaly_types",
    "risk_level",
    "audit_result",
    "reason",
    "evidence",
    "uncertainty",
}


DEFAULT_FIELD_KEYS = {
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
}


DEFAULT_CONSISTENCY_KEYS = {
    "amount_consistent",
    "merchant_consistent",
    "person_consistent",
    "date_reasonable",
    "order_id_consistent",
    "payment_id_present",
    "document_complete",
    "duplicate_in_batch",
}


def _schema_keys(output_schema: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    properties = output_schema.get("properties") or {}
    field_props = ((properties.get("field_extraction") or {}).get("properties") or {}).keys()
    consistency_props = ((properties.get("consistency_check") or {}).get("properties") or {}).keys()
    return set(properties) or DEFAULT_TOP_LEVEL_KEYS, set(field_props) or DEFAULT_FIELD_KEYS, set(consistency_props) or DEFAULT_CONSISTENCY_KEYS


def _schema_valid(output: dict[str, Any] | None, validator: Draft202012Validator) -> bool:
    return output is not None and not list(validator.iter_errors(output))



def _required_keys(container_schema: dict[str, Any]) -> set[str]:
    required = container_schema.get("required") or []
    return {str(item) for item in required}


def _enum_values(schema_node: dict[str, Any], default: set[str]) -> set[str]:
    values = schema_node.get("enum") or []
    return {str(item) for item in values} or default


def _coerce_nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _coerce_bbox(value: Any) -> list[int]:
    if not isinstance(value, list):
        return [0, 0, 0, 0]
    values = list(value[:4])
    while len(values) < 4:
        values.append(0)
    coerced: list[int] = []
    for item in values:
        try:
            number = int(round(float(item)))
        except (TypeError, ValueError):
            number = 0
        coerced.append(max(0, min(1000, number)))
    return coerced

def _normalize_order_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("：", ":")
    match = re.search(r"\bORD[0-9A-Za-z_-]+\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return re.sub(r"\s+", "", text).upper()


def _order_id_from_evidence_item(item: dict[str, Any]) -> str | None:
    value = _normalize_order_id(item.get("value"))
    if value:
        return value
    return _normalize_order_id(item.get("evidence_text"))


def _evidence_order_ids(output: dict[str, Any]) -> dict[str, str]:
    order_ids: dict[str, str] = {}
    evidence = output.get("evidence")
    if not isinstance(evidence, list):
        return order_ids
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if str(item.get("field") or "") != "order_id":
            continue
        doc_type = str(item.get("source_doc_type") or "")
        if doc_type not in {"order", "reimbursement_form"}:
            continue
        order_id = _order_id_from_evidence_item(item)
        if order_id and doc_type not in order_ids:
            order_ids[doc_type] = order_id
    return order_ids


def apply_order_id_consistency_verifier(output: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Force order_id_mismatch only when evidence shows two different order IDs."""

    verified = copy.deepcopy(output)
    evidence_order_ids = _evidence_order_ids(verified)
    order_doc_order_id = evidence_order_ids.get("order")
    reimbursement_form_order_id = evidence_order_ids.get("reimbursement_form")
    meta: dict[str, Any] = {
        "order_id_verifier_checked": bool(order_doc_order_id and reimbursement_form_order_id),
        "order_id_verifier_changed": False,
        "order_doc_order_id": order_doc_order_id,
        "reimbursement_form_order_id": reimbursement_form_order_id,
    }
    if not order_doc_order_id or not reimbursement_form_order_id:
        return verified, meta
    if order_doc_order_id == reimbursement_form_order_id:
        meta["order_id_verifier_match"] = True
        return verified, meta

    changed = False
    consistency_check = verified.setdefault("consistency_check", {})
    if not isinstance(consistency_check, dict):
        consistency_check = {}
        verified["consistency_check"] = consistency_check
        changed = True
    if consistency_check.get("order_id_consistent") is not False:
        consistency_check["order_id_consistent"] = False
        changed = True

    anomalies = verified.get("anomaly_types")
    if not isinstance(anomalies, list):
        anomalies = []
        verified["anomaly_types"] = anomalies
        changed = True
    if "order_id_mismatch" not in anomalies:
        anomalies.append("order_id_mismatch")
        changed = True

    if verified.get("risk_level") != "high":
        verified["risk_level"] = "high"
        changed = True
    if verified.get("audit_result") != "reject_recommendation":
        verified["audit_result"] = "reject_recommendation"
        changed = True
    reason = (
        "检测到order_id_mismatch：订单截图订单号"
        f"{order_doc_order_id}与报销申请单订单号{reimbursement_form_order_id}不一致，"
        "风险等级为high，审核建议为reject_recommendation。"
    )
    if verified.get("reason") != reason:
        verified["reason"] = reason
        changed = True

    uncertainty = verified.get("uncertainty")
    if isinstance(uncertainty, dict) and uncertainty.get("requires_manual_review") is not False:
        uncertainty["requires_manual_review"] = False
        changed = True

    meta["order_id_verifier_match"] = False
    meta["order_id_verifier_changed"] = changed
    return verified, meta


def _extract_json_value(raw_output: str, key: str) -> Any:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*', raw_output)
    if not match:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(raw_output[match.end() :])
    except json.JSONDecodeError:
        return None
    return value


def _partial_output_from_raw(raw_output: str, parsed_output: dict[str, Any] | None) -> dict[str, Any] | None:
    field_extraction = _extract_json_value(raw_output, "field_extraction")
    if not isinstance(field_extraction, dict) and isinstance(parsed_output, dict):
        if DEFAULT_FIELD_KEYS.intersection(parsed_output):
            field_extraction = {key: parsed_output.get(key) for key in DEFAULT_FIELD_KEYS if key in parsed_output}
    if not isinstance(field_extraction, dict):
        field_extraction = {}

    consistency_check = _extract_json_value(raw_output, "consistency_check")
    if not isinstance(consistency_check, dict):
        consistency_check = {}

    anomaly_types = _extract_json_value(raw_output, "anomaly_types")
    if not isinstance(anomaly_types, list):
        anomaly_types = []

    evidence = _extract_json_value(raw_output, "evidence")
    if not isinstance(evidence, list):
        evidence = []

    uncertainty = _extract_json_value(raw_output, "uncertainty")
    if not isinstance(uncertainty, dict):
        uncertainty = {}

    recovered = {
        "case_id": _extract_json_value(raw_output, "case_id"),
        "field_extraction": field_extraction,
        "consistency_check": consistency_check,
        "anomaly_types": anomaly_types,
        "risk_level": _extract_json_value(raw_output, "risk_level"),
        "audit_result": _extract_json_value(raw_output, "audit_result"),
        "reason": _extract_json_value(raw_output, "reason") or "schema_guard_recovered_partial_output",
        "evidence": evidence,
        "uncertainty": uncertainty,
    }
    if not recovered["case_id"] or not recovered["risk_level"] or not recovered["audit_result"]:
        return None
    return recovered

def normalize_output_to_schema(
    output: dict[str, Any],
    output_schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move known flattened fields back under their schema containers."""

    top_keys, field_keys, consistency_keys = _schema_keys(output_schema)
    normalized = copy.deepcopy(output)
    changed = False
    moved_field_keys: list[str] = []
    moved_consistency_keys: list[str] = []
    dropped_top_level_keys: list[str] = []

    field_extraction = normalized.get("field_extraction")
    if not isinstance(field_extraction, dict):
        field_extraction = {}
        normalized["field_extraction"] = field_extraction
        changed = True
    consistency_check = normalized.get("consistency_check")
    if not isinstance(consistency_check, dict):
        consistency_check = {}
        normalized["consistency_check"] = consistency_check
        changed = True

    for key in list(normalized.keys()):
        if key in field_keys and key not in top_keys:
            if key not in field_extraction:
                field_extraction[key] = normalized[key]
                moved_field_keys.append(key)
            del normalized[key]
            changed = True
        elif key in consistency_keys and key not in top_keys:
            if key not in consistency_check:
                consistency_check[key] = normalized[key]
                moved_consistency_keys.append(key)
            del normalized[key]
            changed = True

    for key in list(normalized.keys()):
        if key not in top_keys:
            del normalized[key]
            dropped_top_level_keys.append(key)
            changed = True

    properties = output_schema.get("properties") or {}
    field_schema = properties.get("field_extraction") or {}
    consistency_schema = properties.get("consistency_check") or {}
    anomaly_schema = properties.get("anomaly_types") or {}
    anomaly_item_schema = anomaly_schema.get("items") or {}
    if isinstance(anomaly_item_schema.get("$ref"), str):
        ref_name = anomaly_item_schema["$ref"].rsplit("/", 1)[-1]
        anomaly_item_schema = ((output_schema.get("$defs") or {}).get(ref_name) or anomaly_item_schema)
    allowed_anomalies = _enum_values(anomaly_item_schema, set())

    for key in list(field_extraction.keys()):
        if key not in field_keys:
            del field_extraction[key]
            changed = True
    for key in _required_keys(field_schema) or field_keys:
        if key not in field_extraction:
            field_extraction[key] = None
            changed = True
        else:
            coerced = _coerce_nullable_string(field_extraction[key])
            if coerced != field_extraction[key]:
                field_extraction[key] = coerced
                changed = True

    for key in list(consistency_check.keys()):
        if key not in consistency_keys:
            del consistency_check[key]
            changed = True
    for key in _required_keys(consistency_schema) or consistency_keys:
        if key not in consistency_check:
            consistency_check[key] = False
            changed = True
        else:
            coerced = _coerce_bool(consistency_check[key])
            if coerced != consistency_check[key]:
                consistency_check[key] = coerced
                changed = True

    anomalies = normalized.get("anomaly_types")
    if not isinstance(anomalies, list):
        normalized["anomaly_types"] = []
        changed = True
    elif allowed_anomalies:
        filtered = []
        seen = set()
        for item in anomalies:
            value = str(item)
            if value in allowed_anomalies and value not in seen:
                filtered.append(value)
                seen.add(value)
        if filtered != anomalies:
            normalized["anomaly_types"] = filtered
            changed = True

    evidence = normalized.get("evidence")
    if not isinstance(evidence, list):
        normalized["evidence"] = []
        changed = True
    else:
        allowed_evidence_keys = {"source_image_id", "source_doc_type", "field", "value", "bbox", "evidence_text"}
        for index, item in enumerate(list(evidence)):
            if not isinstance(item, dict):
                evidence[index] = {
                    "source_image_id": "unknown",
                    "source_doc_type": "invoice",
                    "field": "unknown",
                    "value": "",
                    "bbox": [0, 0, 0, 0],
                    "evidence_text": "",
                }
                changed = True
                continue
            for key in list(item.keys()):
                if key not in allowed_evidence_keys:
                    del item[key]
                    changed = True
            for key in ["source_image_id", "field", "value", "evidence_text"]:
                coerced = _coerce_string(item.get(key))
                if item.get(key) != coerced:
                    item[key] = coerced
                    changed = True
            if item.get("source_doc_type") not in {"invoice", "payment", "reimbursement_form", "order"}:
                item["source_doc_type"] = "invoice"
                changed = True
            bbox = _coerce_bbox(item.get("bbox"))
            if item.get("bbox") != bbox:
                item["bbox"] = bbox
                changed = True

    uncertainty = normalized.get("uncertainty")
    if not isinstance(uncertainty, dict):
        uncertainty = {}
        normalized["uncertainty"] = uncertainty
        changed = True
    for key in list(uncertainty.keys()):
        if key not in {"has_uncertain_fields", "uncertain_fields", "requires_manual_review"}:
            del uncertainty[key]
            changed = True
    if "has_uncertain_fields" not in uncertainty:
        uncertainty["has_uncertain_fields"] = False
        changed = True
    else:
        uncertainty["has_uncertain_fields"] = _coerce_bool(uncertainty["has_uncertain_fields"])
    if not isinstance(uncertainty.get("uncertain_fields"), list):
        uncertainty["uncertain_fields"] = []
        changed = True
    else:
        coerced_fields = [str(item) for item in uncertainty["uncertain_fields"]]
        if coerced_fields != uncertainty["uncertain_fields"]:
            uncertainty["uncertain_fields"] = coerced_fields
            changed = True
    if "requires_manual_review" not in uncertainty:
        uncertainty["requires_manual_review"] = normalized.get("audit_result") in {"manual_review", "missing_info"}
        changed = True
    else:
        uncertainty["requires_manual_review"] = _coerce_bool(uncertainty["requires_manual_review"])

    meta = {
        "changed": changed,
        "moved_field_keys": moved_field_keys,
        "moved_consistency_keys": moved_consistency_keys,
        "dropped_top_level_keys": dropped_top_level_keys,
    }
    return normalized, meta


def guard_raw_output(raw_output: str, output_schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a schema-guarded JSON string when conservative normalization succeeds."""

    parse_result = parse_json_output(raw_output)
    meta: dict[str, Any] = {
        "enabled": True,
        "json_valid_before": bool(parse_result.json_validity),
        "schema_valid_before": False,
        "schema_valid_after": False,
        "changed": False,
    }
    if not parse_result.json_validity or parse_result.output is None:
        partial = _partial_output_from_raw(raw_output, None)
        if partial is None:
            meta["parse_error"] = parse_result.error
            return raw_output, meta
        parse_result_output = partial
        meta["partial_reconstruction"] = True
    else:
        parse_result_output = parse_result.output

    validator = Draft202012Validator(output_schema)
    if _schema_valid(parse_result_output, validator):
        meta["schema_valid_before"] = True
        verified, verifier_meta = apply_order_id_consistency_verifier(parse_result_output)
        meta.update(verifier_meta)
        if verifier_meta.get("order_id_verifier_changed"):
            errors_after = sorted(validator.iter_errors(verified), key=lambda err: err.path)
            meta["schema_valid_after"] = not errors_after
            if errors_after:
                meta["order_id_verifier_schema_errors_after"] = [
                    f"{'.'.join(str(part) for part in err.path) or '<root>'}: {err.message}"
                    for err in errors_after[:3]
                ]
                return raw_output, meta
            meta["changed"] = True
            return json.dumps(verified, ensure_ascii=False, separators=(",", ":")), meta
        meta["schema_valid_after"] = True
        return raw_output, meta

    normalized, normalize_meta = normalize_output_to_schema(parse_result_output, output_schema)
    normalized, verifier_meta = apply_order_id_consistency_verifier(normalized)
    meta["normalization_attempted"] = bool(normalize_meta.get("changed"))
    meta.update(normalize_meta)
    meta.update(verifier_meta)
    if verifier_meta.get("order_id_verifier_changed"):
        meta["changed"] = True
    errors_after = sorted(validator.iter_errors(normalized), key=lambda err: err.path)
    meta["schema_valid_after"] = not errors_after
    if errors_after:
        meta["schema_errors_after"] = [
            f"{'.'.join(str(part) for part in err.path) or '<root>'}: {err.message}"
            for err in errors_after[:3]
        ]
        partial = _partial_output_from_raw(raw_output, parse_result_output)
        if partial is None:
            meta["changed"] = False
            return raw_output, meta
        partial_normalized, partial_meta = normalize_output_to_schema(partial, output_schema)
        partial_normalized, partial_verifier_meta = apply_order_id_consistency_verifier(partial_normalized)
        partial_errors = sorted(validator.iter_errors(partial_normalized), key=lambda err: err.path)
        if partial_errors:
            meta["partial_reconstruction"] = True
            meta["partial_schema_errors_after"] = [
                f"{'.'.join(str(part) for part in err.path) or '<root>'}: {err.message}"
                for err in partial_errors[:3]
            ]
            meta["changed"] = False
            return raw_output, meta
        meta.update(partial_meta)
        meta.update(partial_verifier_meta)
        meta["partial_reconstruction"] = True
        meta["schema_valid_after"] = True
        meta["changed"] = True
        return json.dumps(partial_normalized, ensure_ascii=False, separators=(",", ":")), meta
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), meta


def guard_prediction_row(row: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    raw_output = guarded.get("raw_output")
    if not isinstance(raw_output, str):
        return guarded
    normalized_raw, meta = guard_raw_output(raw_output, output_schema)
    if meta.get("changed"):
        guarded["raw_output_original"] = raw_output
        guarded["raw_output"] = normalized_raw
    guarded["schema_guard"] = meta
    return guarded


def guard_predictions(
    rows: list[dict[str, Any]],
    output_schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guarded_rows = [guard_prediction_row(row, output_schema) for row in rows]
    counters: Counter[str] = Counter()
    for row in guarded_rows:
        meta = row.get("schema_guard") or {}
        counters["total"] += 1
        if meta.get("json_valid_before"):
            counters["json_valid_before"] += 1
        if meta.get("schema_valid_before"):
            counters["schema_valid_before"] += 1
        if meta.get("schema_valid_after"):
            counters["schema_valid_after"] += 1
        if meta.get("changed"):
            counters["changed"] += 1
    report = dict(counters)
    report["schema_repairs"] = report.get("schema_valid_after", 0) - report.get("schema_valid_before", 0)
    return guarded_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a conservative schema guard to prediction JSONL raw outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = read_yaml(args.output_schema)
    guarded_rows, report = guard_predictions(read_jsonl(args.predictions), schema)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_jsonl(guarded_rows, output_path)
    if args.report:
        report_path = Path(args.report)
        ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
