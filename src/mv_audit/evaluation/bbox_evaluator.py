"""BBox matching utilities for normalized 0-1000 coordinates."""

from __future__ import annotations

from typing import Sequence


BBox = Sequence[int | float]


def bbox_iou(pred: BBox, truth: BBox) -> float:
    """Compute IoU for [x1, y1, x2, y2] boxes."""

    px1, py1, px2, py2 = [float(v) for v in pred]
    tx1, ty1, tx2, ty2 = [float(v) for v in truth]
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    truth_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = pred_area + truth_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_inside(pred: BBox, truth: BBox) -> bool:
    """Return whether the predicted center point lies inside the truth box."""

    px1, py1, px2, py2 = [float(v) for v in pred]
    tx1, ty1, tx2, ty2 = [float(v) for v in truth]
    cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def strict_match(pred: BBox, truth: BBox) -> bool:
    return bbox_iou(pred, truth) >= 0.5


def relaxed_match(pred: BBox, truth: BBox) -> bool:
    return bbox_iou(pred, truth) >= 0.3 or center_inside(pred, truth)
