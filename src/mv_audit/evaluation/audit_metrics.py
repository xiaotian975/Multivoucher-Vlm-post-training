"""Risk and audit-result metrics."""

from __future__ import annotations

from typing import Any


RISK_LEVELS = ["low", "medium", "high"]


def macro_f1(truth_labels: list[str], pred_labels: list[str | None], labels: list[str]) -> float:
    """Compute macro F1 with invalid predictions counted as no class."""

    scores: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(truth_labels, pred_labels, strict=True) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(truth_labels, pred_labels, strict=True) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(truth_labels, pred_labels, strict=True) if truth == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def audit_accuracy(truth_outputs: list[dict[str, Any]], pred_outputs: list[dict[str, Any] | None]) -> float:
    correct = 0
    for truth, pred in zip(truth_outputs, pred_outputs, strict=True):
        if pred is not None and pred.get("audit_result") == truth.get("audit_result"):
            correct += 1
    return correct / len(truth_outputs) if truth_outputs else 0.0


def high_risk_miss_rate(truth_outputs: list[dict[str, Any]], pred_outputs: list[dict[str, Any] | None]) -> float:
    high_cases = [index for index, truth in enumerate(truth_outputs) if truth.get("risk_level") == "high"]
    if not high_cases:
        return 0.0
    misses = 0
    for index in high_cases:
        pred = pred_outputs[index]
        if pred is None or pred.get("risk_level") != "high" or pred.get("audit_result") == "pass":
            misses += 1
    return misses / len(high_cases)


def false_manual_review_rate(truth_outputs: list[dict[str, Any]], pred_outputs: list[dict[str, Any] | None]) -> float:
    pass_cases = [index for index, truth in enumerate(truth_outputs) if truth.get("audit_result") == "pass"]
    if not pass_cases:
        return 0.0
    false_manual = sum(
        1
        for index in pass_cases
        if pred_outputs[index] is not None and pred_outputs[index].get("audit_result") == "manual_review"
    )
    return false_manual / len(pass_cases)
