"""Consistency-check metrics."""

from __future__ import annotations

from typing import Any


def consistency_exact_counts(truth: dict[str, Any], pred: dict[str, Any] | None, *, schema_ok: bool) -> tuple[int, int]:
    """Return exact-match counts for consistency_check booleans."""

    truth_checks = truth.get("consistency_check") or {}
    total = len(truth_checks)
    if not schema_ok or pred is None:
        return 0, total
    pred_checks = pred.get("consistency_check") or {}
    correct = sum(1 for field, value in truth_checks.items() if pred_checks.get(field) == value)
    return correct, total
