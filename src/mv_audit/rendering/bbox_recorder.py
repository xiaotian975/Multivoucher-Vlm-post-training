"""BBox record helpers for rendered voucher images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageSpec:
    """Rendered image identity used by downstream converters."""

    case_id: str
    image_id: str
    doc_type: str
    image_path: str
    width: int
    height: int


def normalize_bbox(bbox_abs: BBox, width: int, height: int) -> list[int]:
    """Convert an absolute bbox to the global 0-1000 normalized coordinate system."""

    x1, y1, x2, y2 = bbox_abs
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [
        max(0, min(1000, round(x1 * 1000 / width))),
        max(0, min(1000, round(y1 * 1000 / height))),
        max(0, min(1000, round(x2 * 1000 / width))),
        max(0, min(1000, round(y2 * 1000 / height))),
    ]


def make_bbox_record(
    *,
    image: ImageSpec,
    field: str,
    value: Any,
    bbox_abs: BBox,
    evidence_text: str,
    readable: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one field-level annotation record."""

    record: dict[str, Any] = {
        "case_id": image.case_id,
        "image_id": image.image_id,
        "doc_type": image.doc_type,
        "image_path": image.image_path,
        "field": field,
        "value": "" if value is None else str(value),
        "bbox_abs": [int(coord) for coord in bbox_abs],
        "bbox_norm": normalize_bbox(bbox_abs, image.width, image.height),
        "evidence_text": evidence_text,
        "readable": bool(readable),
    }
    if extra:
        record.update(extra)
    return record


def mark_unreadable(records: list[dict[str, Any]], *, image_id: str, fields: set[str]) -> None:
    """Mark selected annotation records unreadable without removing bbox data."""

    for record in records:
        if record.get("image_id") == image_id and record.get("field") in fields:
            record["readable"] = False
