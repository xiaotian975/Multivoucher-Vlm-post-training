"""Generate duplicate-in-batch images without overwriting source images."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def duplicate_image_and_records(
    *,
    source_image_id: str,
    duplicate_image_id: str,
    source_image_path: str,
    duplicate_image_path: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy an image and duplicate its annotation records under a new image_id."""

    destination = Path(duplicate_image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_image_path, destination)

    duplicate_records: list[dict[str, Any]] = []
    for record in records:
        if record.get("image_id") != source_image_id:
            continue
        cloned = dict(record)
        cloned["image_id"] = duplicate_image_id
        cloned["image_path"] = duplicate_image_path
        cloned["duplicate_of_image_id"] = source_image_id
        cloned["duplicate_pair"] = [source_image_id, duplicate_image_id]
        duplicate_records.append(cloned)
    return duplicate_records
