"""Evidence support, value, source, and bbox metrics."""

from __future__ import annotations

from typing import Any

from mv_audit.evaluation.bbox_evaluator import relaxed_match, strict_match


def _key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("source_image_id")),
        str(record.get("source_doc_type")),
        str(record.get("field")),
        str(record.get("value")),
    )


def evidence_counts(truth: dict[str, Any], pred: dict[str, Any] | None, *, schema_ok: bool) -> dict[str, int]:
    """Return per-case evidence matching counts."""

    truth_evidence = truth.get("evidence") or []
    pred_evidence = [] if not schema_ok or pred is None else pred.get("evidence") or []
    if not truth_evidence:
        return {
            "support_correct": 1 if not pred_evidence else 0,
            "support_total": 1,
            "value_correct": 1 if not pred_evidence else 0,
            "value_total": 1,
            "source_correct": 1 if not pred_evidence else 0,
            "source_total": 1,
            "bbox_strict_correct": 1 if not pred_evidence else 0,
            "bbox_relaxed_correct": 1 if not pred_evidence else 0,
            "bbox_total": 1,
        }

    truth_by_full = {_key(item): item for item in truth_evidence}
    truth_by_field_value = {(str(item.get("field")), str(item.get("value"))): item for item in truth_evidence}
    truth_by_source = {
        (str(item.get("source_image_id")), str(item.get("source_doc_type")), str(item.get("field"))): item
        for item in truth_evidence
    }

    support_correct = 0
    value_correct = 0
    source_correct = 0
    bbox_strict_correct = 0
    bbox_relaxed_correct = 0

    for item in truth_evidence:
        matched = next((pred_item for pred_item in pred_evidence if _key(pred_item) == _key(item)), None)
        if matched is not None:
            support_correct += 1
            bbox_strict_correct += int(strict_match(matched.get("bbox", []), item.get("bbox", [])))
            bbox_relaxed_correct += int(relaxed_match(matched.get("bbox", []), item.get("bbox", [])))

        if (str(item.get("field")), str(item.get("value"))) in {
            (str(pred_item.get("field")), str(pred_item.get("value"))) for pred_item in pred_evidence
        }:
            value_correct += 1

        if (
            str(item.get("source_image_id")),
            str(item.get("source_doc_type")),
            str(item.get("field")),
        ) in {
            (str(pred_item.get("source_image_id")), str(pred_item.get("source_doc_type")), str(pred_item.get("field")))
            for pred_item in pred_evidence
        }:
            source_correct += 1

    return {
        "support_correct": support_correct,
        "support_total": len(truth_evidence),
        "value_correct": value_correct,
        "value_total": len(truth_evidence),
        "source_correct": source_correct,
        "source_total": len(truth_evidence),
        "bbox_strict_correct": bbox_strict_correct,
        "bbox_relaxed_correct": bbox_relaxed_correct,
        "bbox_total": len(truth_evidence),
    }
