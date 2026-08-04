"""Field extraction metrics."""

from __future__ import annotations

from typing import Any


def field_exact_counts(truth: dict[str, Any], pred: dict[str, Any] | None, *, schema_ok: bool) -> tuple[int, int]:
    """Return exact-match numerator and denominator for field_extraction."""

    truth_fields = truth.get("field_extraction") or {}
    total = len(truth_fields)
    if not schema_ok or pred is None:
        return 0, total
    pred_fields = pred.get("field_extraction") or {}
    correct = sum(1 for field, value in truth_fields.items() if pred_fields.get(field) == value)
    return correct, total
