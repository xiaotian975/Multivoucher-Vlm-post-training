"""Multi-image Qwen3-VL smoke test for phase 01."""

from __future__ import annotations

import argparse
from pathlib import Path

from mv_audit.inference.qwen3vl_common import (
    configure_smoke_logger,
    image_to_message_uri,
    load_model_config,
    run_generation,
    validate_image_path,
    write_smoke_log,
)


DEFAULT_PROMPT = (
    "你将看到 2 到 4 张企业费用报销凭证图片。请先按图片编号分别概括每张图片，"
    "再判断这些图片是否可能属于同一个报销 case，并用 JSON 返回。"
)


def build_messages(image_paths: list[Path], prompt: str) -> list[dict]:
    """Build a multi-image chat message with explicit image numbering."""

    content: list[dict[str, str]] = []
    for index, image_path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"图片{index}："})
        content.append({"type": "image", "image": image_to_message_uri(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3-VL multi-image smoke test.")
    parser.add_argument("--config", required=True, help="Path to configs/model/qwen3vl_8b.yaml.")
    parser.add_argument("--images", nargs="+", required=True, help="2 to 4 local image paths.")
    parser.add_argument("--output", required=True, help="Path to write the JSON smoke-test log.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for the multi-image smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_smoke_logger()
    if not 2 <= len(args.images) <= 4:
        raise ValueError(f"Expected 2 to 4 images, got {len(args.images)}.")

    config = load_model_config(args.config)
    image_paths = [validate_image_path(path) for path in args.images]
    messages = build_messages(image_paths, args.prompt)

    logger.info("Starting Qwen3-VL multi-image smoke test with %d images.", len(image_paths))
    output_text, model_path, elapsed, runtime = run_generation(messages, config)
    write_smoke_log(
        output_path=args.output,
        run_type="multi_image",
        model_path=model_path,
        images=image_paths,
        prompt=args.prompt,
        raw_output=output_text,
        elapsed_seconds=elapsed,
        runtime=runtime,
    )
    logger.info("Smoke test completed in %.2f seconds. Output written to %s", elapsed, args.output)


if __name__ == "__main__":
    main()
