"""LoRA-SFT entrypoint for Phase 07.

The dry-run path intentionally avoids loading the base model so this file can be
validated on a local workstation before the server is ready.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mv_audit.inference.qwen3vl_common import image_to_message_uri, resolve_model_path
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, set_random_seed


DEFAULT_CONFIG = "configs/train/sft_lora_qwen3vl_8b.yaml"


@dataclass(frozen=True)
class SFTExample:
    """A single case-level multi-image SFT example."""

    case_id: str
    images: list[dict[str, str]]
    user_prompt: str
    answer: str


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _read_examples(path: str | Path, *, max_samples: int | None = None) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for row in iter_jsonl(path):
        messages = row.get("messages") or []
        if len(messages) < 2:
            raise ValueError(f"SFT row {row.get('id') or row.get('case_id')} must contain user and assistant messages.")
        user_message = messages[0]
        assistant_message = messages[-1]
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            raise ValueError(f"SFT row {row.get('id') or row.get('case_id')} has invalid message roles.")
        examples.append(
            SFTExample(
                case_id=str(row["case_id"]),
                images=list(row.get("images") or []),
                user_prompt=str(user_message.get("content") or ""),
                answer=str(assistant_message.get("content") or ""),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def _validate_examples(examples: list[SFTExample], *, path: str | Path) -> None:
    if not examples:
        raise ValueError(f"No SFT examples found in {path}.")
    for example in examples:
        if not example.case_id:
            raise ValueError("SFT example is missing case_id.")
        if not example.images:
            raise ValueError(f"SFT example {example.case_id} has no images.")
        for item in example.images:
            image_path = Path(str(item.get("image_path") or ""))
            if not image_path.exists():
                raise FileNotFoundError(f"Image for {example.case_id} does not exist: {image_path}")
        try:
            parsed = json.loads(example.answer)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Assistant answer for {example.case_id} is not valid JSON.") from exc
        if parsed.get("case_id") != example.case_id:
            raise ValueError(f"Assistant answer case_id mismatch for {example.case_id}.")


def _conversation(example: SFTExample, *, include_answer: bool) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for item in example.images:
        image_id = str(item.get("image_id") or "")
        doc_type = str(item.get("doc_type") or "")
        image_path = Path(str(item["image_path"])).resolve()
        content.append({"type": "text", "text": f"{image_id}: {doc_type}"})
        content.append({"type": "image", "image": image_to_message_uri(image_path)})
    content.append({"type": "text", "text": example.user_prompt})

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": example.answer})
    return messages


def _dry_run(config: dict[str, Any], *, max_samples: int) -> None:
    data_config = _section(config, "data")
    model_config = _section(config, "model")
    train_config = _section(config, "training")
    lora_config = _section(config, "lora")

    train_path = data_config.get("train_file")
    val_path = data_config.get("val_file")
    if not train_path or not val_path:
        raise ValueError("Config data.train_file and data.val_file are required.")

    train_examples = _read_examples(train_path, max_samples=max_samples)
    val_examples = _read_examples(val_path, max_samples=max(1, min(max_samples, 2)))
    _validate_examples(train_examples, path=train_path)
    _validate_examples(val_examples, path=val_path)

    target_modules = lora_config.get("target_modules") or []
    if not target_modules:
        raise ValueError("Config lora.target_modules must not be empty.")
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/sft")))

    local_model_dir = Path(str(model_config.get("local_model_dir", "")))
    print("phase07_sft_dry_run=ok")
    print(f"train_file={train_path}")
    print(f"train_examples_checked={len(train_examples)}")
    print(f"val_file={val_path}")
    print(f"val_examples_checked={len(val_examples)}")
    print(f"output_dir={output_dir}")
    print(f"local_model_dir_exists={local_model_dir.exists()}")
    print(f"lora_target_modules={','.join(str(item) for item in target_modules)}")


class SFTDataset:
    """Tiny Dataset wrapper that defers multimodal preprocessing to the collator."""

    def __init__(self, examples: list[SFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]


class DataCollatorForQwenVLSFT:
    """Build multimodal batches and mask non-assistant tokens from loss."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    def _encode_one(self, example: SFTExample) -> dict[str, Any]:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("torch and qwen-vl-utils are required for training.") from exc

        full_messages = _conversation(example, include_answer=True)
        prompt_messages = _conversation(example, include_answer=False)

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_images, full_videos = process_vision_info(full_messages)
        prompt_images, prompt_videos = process_vision_info(prompt_messages)
        encoded = self.processor(
            text=[full_text],
            images=full_images,
            videos=full_videos,
            padding=False,
            return_tensors="pt",
        )
        prompt_encoded = self.processor(
            text=[prompt_text],
            images=prompt_images,
            videos=prompt_videos,
            padding=False,
            return_tensors="pt",
        )

        item = {key: value[0] if getattr(value, "ndim", 0) == 2 and key in {"input_ids", "attention_mask"} else value for key, value in encoded.items()}
        labels = item["input_ids"].clone()
        prompt_len = int(prompt_encoded["input_ids"].shape[1])
        labels[: min(prompt_len, labels.shape[0])] = -100
        item["labels"] = labels
        return item

    def __call__(self, features: list[SFTExample]) -> dict[str, Any]:
        import torch

        encoded = [self._encode_one(feature) for feature in features]
        max_len = max(int(item["input_ids"].shape[0]) for item in encoded)
        batch: dict[str, Any] = {}

        input_ids = []
        attention_mask = []
        labels = []
        for item in encoded:
            pad_len = max_len - int(item["input_ids"].shape[0])
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        batch["input_ids"] = torch.stack(input_ids)
        batch["attention_mask"] = torch.stack(attention_mask)
        batch["labels"] = torch.stack(labels)

        for key in encoded[0]:
            if key in {"input_ids", "attention_mask", "labels"}:
                continue
            values = [item[key] for item in encoded if key in item]
            if not values:
                continue
            batch[key] = torch.cat(values, dim=0)
        return batch


def _train(config: dict[str, Any], *, max_samples: int | None) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("transformers, peft, and torch are required for real SFT training.") from exc

    from mv_audit.inference.qwen3vl_common import resolve_model_class

    data_config = _section(config, "data")
    model_config = _section(config, "model")
    train_config = _section(config, "training")
    lora_section = _section(config, "lora")

    set_random_seed(int(config.get("seed", 42)))
    train_examples = _read_examples(data_config["train_file"], max_samples=max_samples)
    val_limit = None if max_samples is None else max(1, min(max_samples, 8))
    val_examples = _read_examples(data_config["val_file"], max_samples=val_limit)
    _validate_examples(train_examples, path=data_config["train_file"])
    _validate_examples(val_examples, path=data_config["val_file"])

    model_path = resolve_model_path(model_config)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=bool(model_config.get("trust_remote_code", True)))
    model_class = resolve_model_class()
    model_kwargs: dict[str, Any] = {
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    dtype = model_config.get("dtype", "auto")
    if dtype and dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, str(dtype))
    elif dtype == "auto":
        model_kwargs["torch_dtype"] = "auto"
    model = model_class.from_pretrained(model_path, **model_kwargs)

    if bool(train_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(lora_section.get("r", 16)),
        lora_alpha=int(lora_section.get("alpha", 32)),
        lora_dropout=float(lora_section.get("dropout", 0.05)),
        target_modules=list(lora_section.get("target_modules") or []),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = str(train_config.get("output_dir", "outputs/checkpoints/sft"))
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(train_config.get("learning_rate", 1e-4)),
        num_train_epochs=float(train_config.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 16)),
        bf16=bool(train_config.get("bf16", True)),
        fp16=bool(train_config.get("fp16", False)),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_steps=int(train_config.get("save_steps", 500)),
        eval_steps=int(train_config.get("eval_steps", 500)),
        eval_strategy=str(train_config.get("eval_strategy", "steps")),
        save_strategy=str(train_config.get("save_strategy", "steps")),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        remove_unused_columns=False,
        report_to=list(train_config.get("report_to", [])),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SFTDataset(train_examples),
        eval_dataset=SFTDataset(val_examples),
        data_collator=DataCollatorForQwenVLSFT(processor),
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 07 LoRA-SFT for Qwen3-VL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry_run", action="store_true", help="Validate config and examples without loading the model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for dry-run or debug training.")
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
