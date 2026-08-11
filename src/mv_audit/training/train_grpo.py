"""GRPO training entrypoint for Phase 08."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mv_audit.inference.qwen3vl_common import move_inputs_to_model, resolve_model_class, resolve_model_path
from mv_audit.training.reward_function import normalize_group_rewards, score_output, summarize_reward_outputs
from mv_audit.training.train_dpo import _model_device, _move_batch_to_device, _sequence_logps
from mv_audit.training.train_sft import DataCollatorForQwenVLSFT, SFTExample, _conversation
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


def _image_paths_exist(example: GRPOExample) -> bool:
    return all(Path(str(item.get("image_path") or "")).exists() for item in example.images)


def _read_existing_image_examples(path: str | Path, *, max_samples: int | None) -> tuple[list[GRPOExample], int]:
    examples: list[GRPOExample] = []
    skipped_missing_images = 0
    for row in iter_jsonl(path):
        example = GRPOExample(
            case_id=str(row["case_id"]),
            prompt=str(row["prompt"]),
            images=list(row.get("images") or []),
            ground_truth=dict(row["ground_truth"]),
        )
        if not _image_paths_exist(example):
            skipped_missing_images += 1
            continue
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples, skipped_missing_images


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


def _checkpoint_has_adapter(path: str | Path) -> bool:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return False
    has_config = (checkpoint / "adapter_config.json").exists()
    has_weights = (checkpoint / "adapter_model.safetensors").exists() or (checkpoint / "adapter_model.bin").exists()
    return has_config and has_weights


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _patch_trl_fsdp_import() -> None:
    """Allow newer TRL imports on torch builds without FSDPModule."""

    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        return
    if hasattr(fsdp, "FSDPModule"):
        return
    fallback = getattr(fsdp, "FullyShardedDataParallel", None)
    if fallback is not None:
        setattr(fsdp, "FSDPModule", fallback)


def _score_ground_truth_examples(
    examples: list[GRPOExample],
    *,
    output_schema: dict[str, Any],
) -> dict[str, Any]:
    scores = [
        score_output(
            json.dumps(example.ground_truth["output"], ensure_ascii=False),
            example.ground_truth,
            example.images,
            output_schema,
        )
        for example in examples
    ]
    rewards = [float(score["reward"]) for score in scores]
    return {
        "examples": len(examples),
        **summarize_reward_outputs(scores),
        "normalized_group_reward_preview": normalize_group_rewards(rewards),
    }


def _dry_run(config: dict[str, Any], *, max_samples: int) -> None:
    data_config = _section(config, "data")
    train_config = _section(config, "training")
    model_config = _section(config, "model")
    data_file = data_config.get("train_file")
    if not data_file:
        raise ValueError("data.train_file is required.")
    if bool(data_config.get("require_existing_images", False)):
        examples, skipped_missing_images = _read_existing_image_examples(data_file, max_samples=max_samples)
    else:
        examples = _read_examples(data_file, max_samples=max_samples)
        skipped_missing_images = 0
    _validate_examples(examples, data_file=data_file)
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    audit = _score_ground_truth_examples(examples, output_schema=schema)
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/grpo/qwen3vl_8b_grpo")))
    dpo_checkpoint = Path(str(model_config.get("dpo_checkpoint_dir", "")))
    metrics_path = train_config.get("metrics_output")
    if metrics_path:
        _write_json(metrics_path, {"stage": "grpo_dry_run", "skipped_missing_images": skipped_missing_images, **audit})
    print("phase08_grpo_dry_run=ok")
    print(f"train_file={data_file}")
    print(f"examples_checked={len(examples)}")
    print(f"skipped_missing_images={skipped_missing_images}")
    print(f"mean_reward={audit['mean_reward']:.6f}")
    print(f"json_valid_rate={audit['json_valid_rate']:.6f}")
    print(f"high_risk_miss_rate={audit['high_risk_miss_rate']:.6f}")
    print(f"hallucination_penalty={audit['hallucination_penalty']:.6f}")
    print(f"output_dir={output_dir}")
    print(f"dpo_checkpoint_exists={dpo_checkpoint.exists()}")
    print(f"dpo_adapter_exists={_checkpoint_has_adapter(dpo_checkpoint)}")


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


def _as_sft_example(example: GRPOExample, answer: str) -> SFTExample:
    return SFTExample(
        case_id=example.case_id,
        images=example.images,
        user_prompt=example.prompt,
        answer=answer,
    )


def _generate_completion(
    model: Any,
    processor: Any,
    example: GRPOExample,
    *,
    max_completion_length: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = _conversation(_as_sft_example(example, ""), include_answer=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model(inputs, model)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_completion_length,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    was_training = bool(model.training)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    if was_training:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    input_len = int(inputs["input_ids"].shape[1])
    return processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _load_dpo_policy(config: dict[str, Any], *, load_reference: bool):
    try:
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("peft and transformers are required for GRPO training.") from exc

    model_config = _section(config, "model")
    if not _checkpoint_has_adapter(model_config["dpo_checkpoint_dir"]):
        raise FileNotFoundError(f"DPO adapter checkpoint is incomplete: {model_config['dpo_checkpoint_dir']}")
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
    ref_model = None
    if load_reference:
        ref_model = model_class.from_pretrained(model_path, **base_kwargs)
        ref_model = PeftModel.from_pretrained(ref_model, str(model_config["dpo_checkpoint_dir"]), is_trainable=False)
        ref_model.eval()
    return model, ref_model, processor


def _train(config: dict[str, Any], *, max_samples: int | None) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for GRPO training.") from exc

    data_config = _section(config, "data")
    train_config = _section(config, "training")
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    set_random_seed(int(config.get("seed", 42)))
    if bool(data_config.get("require_existing_images", False)):
        examples, skipped_missing_images = _read_existing_image_examples(data_config["train_file"], max_samples=max_samples)
    else:
        examples = _read_examples(data_config["train_file"], max_samples=max_samples)
        skipped_missing_images = 0
    _validate_examples(examples, data_file=data_config["train_file"])
    metrics_path = train_config.get("metrics_output")
    audit = _score_ground_truth_examples(examples[: min(len(examples), int(train_config.get("reward_audit_samples", 128)))], output_schema=schema)
    if metrics_path:
        _write_json(metrics_path, {"stage": "grpo_pre_train", "skipped_missing_images": skipped_missing_images, **audit})

    model, ref_model, processor = _load_dpo_policy(config, load_reference=True)
    assert ref_model is not None
    if bool(train_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    ref_model.eval()
    model.train()
    collator = DataCollatorForQwenVLSFT(processor)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config.get("learning_rate", 1e-6)),
    )
    epochs = int(float(train_config.get("num_train_epochs", 1)))
    grad_accum = int(train_config.get("gradient_accumulation_steps", 16))
    num_generations = int(train_config.get("num_generations", 4))
    max_completion_length = int(train_config.get("max_completion_length", 1536))
    temperature = float(train_config.get("temperature", 0.7))
    top_p = float(train_config.get("top_p", 0.9))
    kl_beta = float(train_config.get("kl_beta", 0.02))
    logging_steps = int(train_config.get("logging_steps", 10))
    save_steps = int(train_config.get("save_steps", 500))
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/grpo/qwen3vl_8b_grpo")))
    global_step = 0
    micro_step = 0
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        for example in examples:
            completions = [
                _generate_completion(
                    model,
                    processor,
                    example,
                    max_completion_length=max_completion_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                for _ in range(num_generations)
            ]
            scores = [
                score_output(completion, example.ground_truth, example.images, schema)
                for completion in completions
            ]
            rewards = [float(score["reward"]) for score in scores]
            advantages = torch.tensor(normalize_group_rewards(rewards), device=_model_device(model), dtype=torch.float32)
            sft_items = [_as_sft_example(example, completion) for completion in completions]
            batch = _move_batch_to_device(collator(sft_items), _model_device(model))
            ref_batch = _move_batch_to_device({key: value.detach().clone() if hasattr(value, "detach") else value for key, value in batch.items()}, _model_device(ref_model))
            autocast_enabled = bool(train_config.get("bf16", True)) and torch.cuda.is_available()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                with torch.no_grad():
                    ref_logps = _sequence_logps(ref_model, ref_batch)
                policy_logps = _sequence_logps(model, batch)
                kl_proxy = (policy_logps - ref_logps).pow(2).mean()
                policy_loss = -(advantages.detach() * policy_logps).mean()
                loss = policy_loss + kl_beta * kl_proxy
            (loss / grad_accum).backward()
            micro_step += 1
            reward_summary = summarize_reward_outputs(scores)
            if micro_step % grad_accum == 0 or micro_step == len(examples) * epochs:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "epoch": float(epoch + 1),
                    "global_step": float(global_step),
                    "loss": float(loss.detach().cpu()),
                    "mean_reward": float(reward_summary["mean_reward"]),
                    "json_valid_rate": float(reward_summary["json_valid_rate"]),
                    "high_risk_miss_rate": float(reward_summary["high_risk_miss_rate"]),
                    "hallucination_penalty": float(reward_summary["hallucination_penalty"]),
                    "kl_proxy": float(kl_proxy.detach().cpu()),
                }
                history.append(record)
                if global_step % logging_steps == 0 or global_step == 1:
                    print(json.dumps({"stage": "grpo_train", **record}, ensure_ascii=False))
                if save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_dir = ensure_dir(output_dir / f"checkpoint-{global_step}")
                    model.save_pretrained(str(checkpoint_dir))
                    processor.save_pretrained(str(checkpoint_dir))

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    if metrics_path:
        _write_json(
            metrics_path,
            {
                "stage": "grpo_done",
                "skipped_missing_images": skipped_missing_images,
                **audit,
                "explicit_ref_model": True,
                "training_history": history,
                "global_step": global_step,
            },
        )


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
