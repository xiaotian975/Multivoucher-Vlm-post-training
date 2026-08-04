"""Hallucination checks for evidence-grounded audit outputs."""

from __future__ import annotations

from typing import Any


def hallucination_count(
    truth: dict[str, Any],
    pred: dict[str, Any] | None,
    *,
    schema_ok: bool,
    image_items: list[dict[str, Any]],
) -> tuple[int, int]:
    """Return hallucinated evidence count and predicted evidence total."""

    if not schema_ok or pred is None:
        return 0, 0

    valid_images = {item["image_id"]: item["doc_type"] for item in image_items}
    truth_evidence_keys = {
        (
            str(item.get("source_image_id")),
            str(item.get("source_doc_type")),
            str(item.get("field")),
            str(item.get("value")),
        )
        for item in truth.get("evidence") or []
    }
    uncertain_fields = set((truth.get("uncertainty") or {}).get("uncertain_fields") or [])
    hallucinated = 0
    pred_evidence = pred.get("evidence") or []

    for item in pred_evidence:
        image_id = item.get("source_image_id")
        doc_type = item.get("source_doc_type")
        field = str(item.get("field"))
        key = (str(image_id), str(doc_type), field, str(item.get("value")))
        if image_id not in valid_images:
            hallucinated += 1
        elif valid_images[image_id] != doc_type:
            hallucinated += 1
        elif field in uncertain_fields:
            hallucinated += 1
        elif key not in truth_evidence_keys:
            hallucinated += 1
    return hallucinated, len(pred_evidence)
