"""Generate unreadable field regions while preserving bbox geometry."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFilter


def obscure_fields(image: Image.Image, records: list[dict], *, image_id: str, fields: set[str], seed: int) -> Image.Image:
    """Obscure selected fields and mark their annotation records unreadable."""

    rng = random.Random(seed)
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for record in records:
        if record.get("image_id") != image_id or record.get("field") not in fields:
            continue
        x1, y1, x2, y2 = [int(v) for v in record["bbox_abs"]]
        pad = 6
        region = (max(0, x1 - pad), max(0, y1 - pad), min(output.width, x2 + pad), min(output.height, y2 + pad))
        patch = output.crop(region).filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 7.0)))
        output.paste(patch, region)
        draw.rectangle(region, outline=(132, 140, 150), width=2)
        record["readable"] = False
        record["unreadable_reason"] = "synthetic_blur_or_occlusion"
    return output
