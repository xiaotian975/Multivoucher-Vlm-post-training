"""DPO training entrypoint for Phase 08."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mv_audit.inference.qwen3vl_common import resolve_model_class, resolve_model_path
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, set_random_seed


DEFAULT_CONFIG = "configs/train/dpo_qwen3vl_8b.yaml"


@dataclass(frozen=True)
class DPOExample:
    case_id: str
    prompt: str
    chosen: str
    rejected: str
    images: list[dict[str, str]]


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _read_examples(path: str | Path, *, max_samples: int | None) -> list[DPOExample]:
    examples: list[DPOExample] = []
    for row in iter_jsonl(path):
        examples.append(
            DPOExample(
                case_id=str(row["case_id"]),
                prompt=str(row["prompt"]),
                chosen=str(row["chosen"]),
                rejected=str(row["rejected"]),
                images=list(row.get("images") or []),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def _validate_examples(examples: list[DPOExample], *, data_file: str | Path) -> None:
    if not examples:
        raise ValueError(f"No DPO examples found in {data_file}.")
    for example in examples:
        if not example.prompt or not example.chosen or not example.rejected:
            raise ValueError(f"DPO example {example.case_id} is missing prompt/chosen/rejected text.")
        if not example.images:
            raise ValueError(f"DPO example {example.case_id} has no images.")
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
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/dpo/qwen3vl_8b_dpo")))
    sft_checkpoint = Path(str(model_config.get("sft_checkpoint_dir", "")))
    print("phase08_dpo_dry_run=ok")
    print(f"train_file={data_file}")
    print(f"examples_checked={len(examples)}")
    print(f"output_dir={output_dir}")
    print(f"sft_checkpoint_exists={sft_checkpoint.exists()}")


def _to_dataset_rows(examples: list[DPOExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        images = [Image.open(str(item["image_path"])).convert("RGB") for item in example.images]
        rows.append(
            {
                "prompt": example.prompt,
                "chosen": example.chosen,
                "rejected": example.rejected,
                "images": images,
            }
        )
    return rows


def _load_sft_policy(config: dict[str, Any]):
    try:
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("peft and transformers are required for DPO training.") from exc

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
    policy = model_class.from_pretrained(model_path, **base_kwargs)
    policy = PeftModel.from_pretrained(policy, str(model_config["sft_checkpoint_dir"]), is_trainable=True)

    ref_policy = model_class.from_pretrained(model_path, **base_kwargs)
    ref_policy = PeftModel.from_pretrained(ref_policy, str(model_config["sft_checkpoint_dir"]), is_trainable=False)
    ref_policy.eval()
    return policy, ref_policy, processor


def _train(config: dict[str, Any], *, max_samples: int | None) -> None:
    try:
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError("datasets and trl are required for DPO training.") from exc

    data_config = _section(config, "data")
    train_config = _section(config, "training")
    set_random_seed(int(config.get("seed", 42)))
    examples = _read_examples(data_config["train_file"], max_samples=max_samples)
    _validate_examples(examples, data_file=data_config["train_file"])
    policy, ref_policy, processor = _load_sft_policy(config)
    dataset = Dataset.from_list(_to_dataset_rows(examples))
    args = DPOConfig(
        output_dir=str(train_config.get("output_dir", "outputs/checkpoints/dpo/qwen3vl_8b_dpo")),
        learning_rate=float(train_config.get("learning_rate", 5e-6)),
        num_train_epochs=float(train_config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 16)),
        beta=float(train_config.get("beta", 0.1)),
        bf16=bool(train_config.get("bf16", True)),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_steps=int(train_config.get("save_steps", 500)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        max_length=None,
        max_prompt_length=None,
        remove_unused_columns=False,
        report_to=list(train_config.get("report_to", [])),
    )
    try:
        trainer = DPOTrainer(
            model=policy,
            ref_model=ref_policy,
            args=args,
            train_dataset=dataset,
            processing_class=processor,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=policy,
            ref_model=ref_policy,
            args=args,
            train_dataset=dataset,
            tokenizer=processor,
        )
    trainer.train()
    trainer.save_model(str(train_config.get("output_dir", "outputs/checkpoints/dpo/qwen3vl_8b_dpo")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08 DPO training for Qwen3-VL.")
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
