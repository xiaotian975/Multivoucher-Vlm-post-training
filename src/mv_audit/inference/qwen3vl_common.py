"""Shared helpers for Qwen3-VL phase 01 smoke tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from mv_audit.utils import ensure_dir, load_config, setup_logging


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the phase 01 model config."""

    config = load_config(config_path)
    required = ["model_name_or_path", "local_model_dir", "dtype", "device_map"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {missing}")
    return config


def resolve_model_path(config: dict[str, Any]) -> str:
    """Prefer a downloaded local model directory; fall back to model id."""

    local_dir = Path(config["local_model_dir"])
    if local_dir.exists():
        return str(local_dir)
    model_name = str(config["model_name_or_path"])
    raise FileNotFoundError(
        "Local model directory does not exist: "
        f"{local_dir}. Run scripts/00_download_qwen3vl.sh first, or update "
        f"local_model_dir in the config. Remote model id is configured as {model_name!r}."
    )


def validate_image_path(image_path: str | Path) -> Path:
    """Validate that an input image exists and can be opened."""

    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - preserve original image library error
        raise ValueError(f"Image cannot be opened by Pillow: {path}") from exc
    return path


def image_to_message_uri(path: Path) -> str:
    """Convert a local image path to a URI accepted by Qwen VL processors."""

    return path.as_uri()


def resolve_model_class():
    """Resolve the best available transformers model class for Qwen3-VL."""

    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError as exc:
        raise ImportError(
            "Could not import Qwen3VLForConditionalGeneration or "
            "AutoModelForImageTextToText from transformers. Install "
            "transformers>=4.57.0 before running phase 01 smoke tests."
        ) from exc


def load_qwen3vl_model_and_processor(config: dict[str, Any]):
    """Load processor and model for smoke tests."""

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required. Run scripts/00_prepare_env.sh first.") from exc

    model_path = resolve_model_path(config)
    trust_remote_code = bool(config.get("trust_remote_code", True))
    dtype = config.get("dtype", "auto")
    device_map = config.get("device_map", "auto")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_class = resolve_model_class()

    kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    try:
        model = model_class.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("torch_dtype", None)
        kwargs["dtype"] = dtype
        model = model_class.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, processor, model_path


def process_messages(processor, messages: list[dict[str, Any]]):
    """Apply Qwen chat template and process visual inputs."""

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("qwen-vl-utils is required. Run scripts/00_prepare_env.sh first.") from exc

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def move_inputs_to_model(inputs, model):
    """Move processor outputs to the model device when available."""

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return inputs
    return inputs.to(device)


def generate_text(model, processor, inputs, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Run generation and decode only newly generated tokens."""

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    generated_ids = model.generate(**inputs, **generation_kwargs)
    input_token_count = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_count:]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def get_runtime_snapshot(model) -> dict[str, Any]:
    """Collect lightweight runtime information for phase 01 logs."""

    snapshot: dict[str, Any] = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "max_memory_allocated_bytes": None,
        "model_device": None,
    }

    try:
        import torch
    except ImportError:
        return snapshot

    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    snapshot["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        snapshot["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    try:
        snapshot["model_device"] = str(next(model.parameters()).device)
    except StopIteration:
        snapshot["model_device"] = None
    return snapshot


def write_smoke_log(
    *,
    output_path: str | Path,
    run_type: str,
    model_path: str,
    images: list[Path],
    prompt: str,
    raw_output: str,
    elapsed_seconds: float,
    runtime: dict[str, Any],
) -> Path:
    """Write a JSON smoke-test log."""

    output = Path(output_path)
    ensure_dir(output.parent)
    payload = {
        "run_type": run_type,
        "model_path": model_path,
        "image_count": len(images),
        "images": [str(path) for path in images],
        "prompt": prompt,
        "raw_output": raw_output,
        "elapsed_seconds": elapsed_seconds,
        "runtime": runtime,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_generation(messages: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, str, float, dict[str, Any]]:
    """Load the model, generate output, and return runtime metadata."""

    model, processor, model_path = load_qwen3vl_model_and_processor(config)
    inputs = process_messages(processor, messages)
    inputs = move_inputs_to_model(inputs, model)

    started = time.perf_counter()
    output_text = generate_text(
        model,
        processor,
        inputs,
        max_new_tokens=int(config.get("max_new_tokens", 512)),
        temperature=float(config.get("temperature", 0.1)),
        top_p=float(config.get("top_p", 0.9)),
    )
    elapsed = time.perf_counter() - started
    return output_text, model_path, elapsed, get_runtime_snapshot(model)


def configure_smoke_logger():
    """Create a logger for phase 01 smoke scripts."""

    return setup_logging(logger_name="mv_audit.qwen3vl_smoke")
