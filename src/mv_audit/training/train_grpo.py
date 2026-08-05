"""GRPO training entrypoint for Phase 08."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mv_audit.inference.qwen3vl_common import resolve_model_class, resolve_model_path
from mv_audit.training.reward_function import score_output
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, read_yaml, set_random_seed


DEFAULT_CONFIG = "configs/train/grpo_qwen3vl_8b.yaml"


@dataclass(frozen=True)
class GRPOExample:
    case_id: str
    prompt: str
    images: list[dict[str, str]]
    ground_truth: dict[str, Any]


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _read_examples(path: str | Path, *, max_samples: int | None) -> list[GRPOExample]:
    examples: list[GRPOExample] = []
    for row in iter_jsonl(path):
        examples.append(
            GRPOExample(
                case_id=str(row["case_id"]),
                prompt=str(row["prompt"]),
                images=list(row.get("images") or []),
                ground_truth=dict(row["ground_truth"]),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def _validate_examples(examples: list[GRPOExample], *, data_file: str | Path) -> None:
    if not examples:
        raise ValueError(f"No GRPO examples found in {data_file}.")
    for example in examples:
        if not example.prompt:
            raise ValueError(f"GRPO example {example.case_id} is missing prompt.")
        if not isinstance(example.ground_truth.get("output"), dict):
            raise ValueError(f"GRPO example {example.case_id} is missing ground_truth.output.")
        if not example.images:
            raise ValueError(f"GRPO example {example.case_id} has no images.")
        for item in example.images:
            path = Path(str(item.get("image_path") or ""))
            if not path.exists():
                raise FileNotFoundError(f"Image for {example.case_id} does not exist: {path}")


def _dry_run(config: dict[str, Any], *, max_samples: int) -> None:
    data_config = _section(config, "data")
    train_config = _section(config, "training")
    model_config = _section(config, "model")
    data_file = data_config.get("train_file")
    if not data_file:
        raise ValueError("data.train_file is required.")
    examples = _read_examples(data_file, max_samples=max_samples)
    _validate_examples(examples, data_file=data_file)
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    sample = examples[0]
    truth_output = sample.ground_truth["output"]
    reward = score_output(
        raw_output=__import__("json").dumps(truth_output, ensure_ascii=False),
        ground_truth=sample.ground_truth,
        image_items=sample.images,
        output_schema=schema,
    )
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/grpo/qwen3vl_8b_grpo")))
    dpo_checkpoint = Path(str(model_config.get("dpo_checkpoint_dir", "")))
    print("phase08_grpo_dry_run=ok")
    print(f"train_file={data_file}")
    print(f"examples_checked={len(examples)}")
    print(f"sample_reward={reward['reward']:.6f}")
    print(f"output_dir={output_dir}")
    print(f"dpo_checkpoint_exists={dpo_checkpoint.exists()}")


def _to_dataset_rows(examples: list[GRPOExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        images = [Image.open(str(item["image_path"])).convert("RGB") for item in example.images]
        rows.append(
            {
                "prompt": example.prompt,
                "images": images,
                "ground_truth": example.ground_truth,
                "image_items": example.images,
            }
        )
    return rows


def _load_dpo_policy(config: dict[str, Any]):
    try:
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("peft and transformers are required for GRPO training.") from exc

    model_config = _section(config, "model")
    model_path = resolve_model_path(model_config)
    model_class = resolve_model_class()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=bool(model_config.get("trust_remote_code", True)))
    base_kwargs = {
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    dtype = model_config.get("dtype", "auto")
    if dtype:
        base_kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(model_path, **base_kwargs)
    model = PeftModel.from_pretrained(model, str(model_config["dpo_checkpoint_dir"]), is_trainable=True)
    return model, processor


def _train(config: dict[str, Any], *, max_samples: int | None) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("datasets and trl are required for GRPO training.") from exc

    data_config = _section(config, "data")
    train_config = _section(config, "training")
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    set_random_seed(int(config.get("seed", 42)))
    examples = _read_examples(data_config["train_file"], max_samples=max_samples)
    _validate_examples(examples, data_file=data_config["train_file"])
    model, processor = _load_dpo_policy(config)
    dataset = Dataset.from_list(_to_dataset_rows(examples))

    def reward_func(completions: list[str], ground_truth: list[dict[str, Any]], image_items: list[list[dict[str, Any]]], **_kwargs) -> list[float]:
        return [
            float(score_output(completion, truth, images, schema)["reward"])
            for completion, truth, images in zip(completions, ground_truth, image_items, strict=True)
        ]

    args = GRPOConfig(
        output_dir=str(train_config.get("output_dir", "outputs/checkpoints/grpo/qwen3vl_8b_grpo")),
        learning_rate=float(train_config.get("learning_rate", 1e-6)),
        num_train_epochs=float(train_config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 16)),
        num_generations=int(train_config.get("num_generations", 4)),
        max_completion_length=int(train_config.get("max_completion_length", 1536)),
        bf16=bool(train_config.get("bf16", True)),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_steps=int(train_config.get("save_steps", 500)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        remove_unused_columns=False,
        report_to=list(train_config.get("report_to", [])),
    )
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_func],
            args=args,
            train_dataset=dataset,
            processing_class=processor,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_func],
            args=args,
            train_dataset=dataset,
            tokenizer=processor,
        )
    trainer.train()
    trainer.save_model(str(train_config.get("output_dir", "outputs/checkpoints/grpo/qwen3vl_8b_grpo")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08 GRPO training for Qwen3-VL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        _dry_run(config, max_samples=int(args.max_samples or 2))
        return
    _train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
