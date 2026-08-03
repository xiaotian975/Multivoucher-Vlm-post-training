"""Render phase 04 voucher images and field bbox annotation JSONL."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from mv_audit.perturbation.duplicate_generator import duplicate_image_and_records
from mv_audit.perturbation.robust_augment import apply_robust_augment
from mv_audit.perturbation.unreadable_generator import obscure_fields
from mv_audit.perturbation.visual_augment import apply_light_augment
from mv_audit.rendering.bbox_recorder import ImageSpec
from mv_audit.rendering.layout import CANVAS_SIZE, resolve_font_path
from mv_audit.rendering.render_invoice import render_invoice
from mv_audit.rendering.render_order import render_order
from mv_audit.rendering.render_payment import render_payment
from mv_audit.rendering.render_reimbursement_form import render_reimbursement_form
from mv_audit.utils import read_jsonl, write_jsonl


RENDERERS: dict[str, Callable[..., tuple[Image.Image, list[dict[str, Any]]]]] = {
    "invoice": render_invoice,
    "payment": render_payment,
    "reimbursement_form": render_reimbursement_form,
    "order": render_order,
}

SPLIT_INPUTS = {
    "train": "train_cases.jsonl",
    "val_in_template": "val_in_template_cases.jsonl",
    "val_unseen_template": "val_unseen_template_cases.jsonl",
    "test_clean": "test_clean_cases.jsonl",
    "test_robust": "test_robust_cases.jsonl",
    "test_unseen_template": "test_unseen_template_cases.jsonl",
    "test_hard_negative": "test_hard_negative_cases.jsonl",
}


def _image_id(case_id: str, doc_type: str, copy_index: int = 0) -> str:
    suffix = f"_{copy_index:02d}" if copy_index else ""
    return f"{case_id}_{doc_type}{suffix}"


def _image_path(images_dir: Path, split_name: str, image_id: str) -> Path:
    return images_dir / split_name / f"{image_id}.png"


def _stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def _apply_split_augmentation(image: Image.Image, *, split_name: str, seed: int) -> tuple[Image.Image, str]:
    if split_name == "test_robust":
        return apply_robust_augment(image, seed=seed), "robust"
    if split_name == "train" and seed % 3 == 0:
        return apply_light_augment(image, seed=seed), "light"
    return image, "none"


def render_case(
    case: dict[str, Any],
    *,
    split_name: str,
    images_dir: Path,
    font_path: str | None,
) -> list[dict[str, Any]]:
    """Render all available documents for one case."""

    metadata = case.get("metadata") or {}
    records: list[dict[str, Any]] = []
    rendered_paths: dict[str, Path] = {}
    rendered_image_ids: dict[str, str] = {}

    for doc_type in case["documents"]:
        renderer = RENDERERS[doc_type]
        image_id = _image_id(case["case_id"], doc_type)
        output_path = _image_path(images_dir, split_name, image_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_spec = ImageSpec(
            case_id=case["case_id"],
            image_id=image_id,
            doc_type=doc_type,
            image_path=output_path.as_posix(),
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
        )
        image, image_records = renderer(case, image_spec, font_path=font_path)
        image, augmentation = _apply_split_augmentation(
            image,
            split_name=split_name,
            seed=_stable_seed(image_id, int(metadata.get("seed", 42))),
        )

        if case["primary_anomaly_type"] == "unreadable_image" and metadata.get("unreadable_doc_type") == doc_type:
            unreadable_fields = set(metadata.get("unreadable_fields") or [])
            image = obscure_fields(
                image,
                image_records,
                image_id=image_id,
                fields=unreadable_fields,
                seed=int(metadata.get("seed", 42)) + 404,
            )

        for record in image_records:
            record["augmentation"] = augmentation
            record["source_doc_type"] = doc_type
        image.save(output_path)
        rendered_paths[doc_type] = output_path
        rendered_image_ids[doc_type] = image_id
        records.extend(image_records)

    if case["primary_anomaly_type"] == "duplicate_in_batch":
        duplicate_doc_type = metadata.get("duplicate_doc_type", "invoice")
        if duplicate_doc_type in rendered_paths:
            source_image_id = rendered_image_ids[duplicate_doc_type]
            duplicate_image_id = _image_id(case["case_id"], duplicate_doc_type, copy_index=1)
            duplicate_path = _image_path(images_dir, split_name, duplicate_image_id)
            duplicate_records = duplicate_image_and_records(
                source_image_id=source_image_id,
                duplicate_image_id=duplicate_image_id,
                source_image_path=rendered_paths[duplicate_doc_type].as_posix(),
                duplicate_image_path=duplicate_path.as_posix(),
                records=records,
            )
            records.extend(duplicate_records)
    return records


def render_split(
    *,
    input_path: Path,
    split_name: str,
    images_dir: Path,
    annotation_path: Path,
    font_path: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Render a split JSONL and write its annotation JSONL."""

    cases = read_jsonl(input_path)
    if limit is not None:
        cases = cases[:limit]

    all_records: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    for case in cases:
        case_records = render_case(case, split_name=split_name, images_dir=images_dir, font_path=font_path)
        for record in case_records:
            image_id = record["image_id"]
            if image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
            all_records.append(record)

    write_jsonl(all_records, annotation_path)
    return {
        "split": split_name,
        "cases": len(cases),
        "records": len(all_records),
        "images": len(seen_image_ids),
        "annotation_path": annotation_path.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render phase 04 voucher images and bbox annotations.")
    parser.add_argument("--input", help="Path to one split case JSONL.")
    parser.add_argument("--split", help="Split name for --input.")
    parser.add_argument("--raw_cases_dir", default="data/mv_audit/raw_cases", help="Directory containing phase 03 split JSONL files.")
    parser.add_argument("--images_dir", default="data/mv_audit/images", help="Root output directory for rendered images.")
    parser.add_argument("--annotations_dir", default="data/mv_audit/annotations", help="Root output directory for bbox JSONL files.")
    parser.add_argument("--font_path", default=None, help="Optional local font file path. Defaults to MV_AUDIT_FONT_PATH or common system fonts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split debug limit.")
    parser.add_argument("--all_splits", action="store_true", help="Render all phase 03 split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    font_path = resolve_font_path(args.font_path)
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    if args.all_splits:
        raw_cases_dir = Path(args.raw_cases_dir)
        jobs.extend((split, raw_cases_dir / filename) for split, filename in SPLIT_INPUTS.items())
    else:
        if not args.input or not args.split:
            raise ValueError("--input and --split are required unless --all_splits is used")
        jobs.append((args.split, Path(args.input)))

    summaries = []
    for split_name, input_path in jobs:
        annotation_path = annotations_dir / f"field_bboxes_{split_name}.jsonl"
        summary = render_split(
            input_path=input_path,
            split_name=split_name,
            images_dir=images_dir,
            annotation_path=annotation_path,
            font_path=font_path,
            limit=args.limit,
        )
        summaries.append(summary)
        print(summary)
    print(f"font_path={font_path or 'PIL_default'}")


if __name__ == "__main__":
    main()
