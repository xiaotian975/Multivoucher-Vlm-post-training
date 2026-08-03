"""Draw bbox annotations on rendered voucher images for human inspection."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mv_audit.rendering.layout import load_font
from mv_audit.utils import read_jsonl


COLORS = {
    True: (34, 139, 74),
    False: (210, 68, 68),
}


def _group_by_image(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["image_id"]].append(record)
    return grouped


def visualize(records: list[dict[str, Any]], *, output_dir: Path, sample_count: int, seed: int) -> int:
    grouped = _group_by_image(records)
    image_ids = sorted(grouped)
    rng = random.Random(seed)
    selected = rng.sample(image_ids, min(sample_count, len(image_ids)))
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(15)

    for image_id in selected:
        group = grouped[image_id]
        image_path = Path(group[0]["image_path"])
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for record in group:
            x1, y1, x2, y2 = [int(v) for v in record["bbox_abs"]]
            color = COLORS[bool(record.get("readable", True))]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = str(record["field"])
            label_left = x2 + 6 if x2 + 140 < image.width else x1
            label_top = y1 if x2 + 140 < image.width else max(0, y1 - 22)
            text_box = draw.textbbox((label_left, label_top), label, font=font)
            draw.rectangle((text_box[0] - 2, text_box[1] - 1, text_box[2] + 2, text_box[3] + 1), fill=color)
            draw.text((label_left, label_top), label, fill=(255, 255, 255), font=font)
        image.save(output_dir / f"{image_id}_bbox.png")
    return len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize rendered bbox annotations.")
    parser.add_argument("--annotations", required=True, nargs="+", help="Annotation JSONL files.")
    parser.add_argument("--output_dir", default="outputs/eval_reports/figures/bbox_samples", help="Output directory for bbox sample images.")
    parser.add_argument("--sample_count", type=int, default=50, help="Total random sample images to save.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for path in args.annotations:
        records.extend(read_jsonl(path))
    saved = visualize(records, output_dir=Path(args.output_dir), sample_count=args.sample_count, seed=args.seed)
    print(f"saved_bbox_samples={saved}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
