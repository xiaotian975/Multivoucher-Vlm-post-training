"""Single-image Qwen3-VL smoke test for phase 01."""

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


DEFAULT_PROMPT = "请读取图片中的主要文字，并用 JSON 返回。"


def build_messages(image_path: Path, prompt: str) -> list[dict]:
    """Build a single-image chat message."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_to_message_uri(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3-VL single-image smoke test.")
    parser.add_argument("--config", required=True, help="Path to configs/model/qwen3vl_8b.yaml.")
    parser.add_argument("--image", required=True, help="Path to one local test image.")
    parser.add_argument("--output", required=True, help="Path to write the JSON smoke-test log.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for the image smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_smoke_logger()
    config = load_model_config(args.config)
    image_path = validate_image_path(args.image)
    messages = build_messages(image_path, args.prompt)

    logger.info("Starting Qwen3-VL single-image smoke test.")
    output_text, model_path, elapsed, runtime = run_generation(messages, config)
    write_smoke_log(
        output_path=args.output,
        run_type="single_image",
        model_path=model_path,
        images=[image_path],
        prompt=args.prompt,
        raw_output=output_text,
        elapsed_seconds=elapsed,
        runtime=runtime,
    )
    logger.info("Smoke test completed in %.2f seconds. Output written to %s", elapsed, args.output)


if __name__ == "__main__":
    main()
