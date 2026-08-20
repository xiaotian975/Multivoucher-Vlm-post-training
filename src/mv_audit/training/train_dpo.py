"""DPO training entrypoint for Phase 08."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mv_audit.inference.qwen3vl_common import resolve_model_class, resolve_model_path
from mv_audit.training.reward_function import score_output, summarize_reward_outputs
from mv_audit.training.train_sft import DataCollatorForQwenVLSFT, SFTExample
from mv_audit.utils import ensure_dir, iter_jsonl, load_config, read_yaml, set_random_seed


DEFAULT_CONFIG = "configs/train/dpo_qwen3vl_8b.yaml"


@dataclass(frozen=True)
class DPOExample:
    case_id: str
    prompt: str
    chosen: str
    rejected: str
    images: list[dict[str, str]]
    final_weight: float = 1.0
    sft_loss_weight: float = 0.0
    pair_id: str | None = None
    pair_type: str | None = None
    ground_truth: dict[str, Any] | None = None


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
                final_weight=float(row.get("final_weight", 1.0)),
                sft_loss_weight=float(row.get("sft_loss_weight", 0.0)),
                pair_id=str(row.get("pair_id") or row.get("id") or ""),
                pair_type=str(row.get("pair_type") or ""),
                ground_truth=dict(row.get("ground_truth") or {}),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def _image_paths_exist(example: DPOExample) -> bool:
    return all(Path(str(item.get("image_path") or "")).exists() for item in example.images)


def _read_existing_image_examples(path: str | Path, *, max_samples: int | None) -> tuple[list[DPOExample], int]:
    examples: list[DPOExample] = []
    skipped_missing_images = 0
    for row in iter_jsonl(path):
        example = DPOExample(
            case_id=str(row["case_id"]),
            prompt=str(row["prompt"]),
            chosen=str(row["chosen"]),
            rejected=str(row["rejected"]),
            images=list(row.get("images") or []),
            final_weight=float(row.get("final_weight", 1.0)),
            sft_loss_weight=float(row.get("sft_loss_weight", 0.0)),
            pair_id=str(row.get("pair_id") or row.get("id") or ""),
            pair_type=str(row.get("pair_type") or ""),
            ground_truth=dict(row.get("ground_truth") or {}),
        )
        if not _image_paths_exist(example):
            skipped_missing_images += 1
            continue
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples, skipped_missing_images


def _read_data_file(
    path: str | Path,
    *,
    max_samples: int | None,
    require_existing_images: bool,
) -> tuple[list[DPOExample], int]:
    if require_existing_images:
        return _read_existing_image_examples(path, max_samples=max_samples)
    return _read_examples(path, max_samples=max_samples), 0


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


def _score_preference_examples(
    examples: list[DPOExample],
    *,
    output_schema: dict[str, Any],
) -> dict[str, Any]:
    chosen_scores: list[dict[str, Any]] = []
    rejected_scores: list[dict[str, Any]] = []
    reward_gaps: list[float] = []
    for example in examples:
        truth = (
            example.ground_truth
            if example.ground_truth and isinstance(example.ground_truth.get("output"), dict)
            else {"output": json.loads(example.chosen)}
        )
        chosen = score_output(example.chosen, truth, example.images, output_schema)
        rejected = score_output(example.rejected, truth, example.images, output_schema)
        chosen_scores.append(chosen)
        rejected_scores.append(rejected)
        reward_gaps.append(float(chosen["reward"]) - float(rejected["reward"]))

    positive_gap_count = sum(1 for gap in reward_gaps if gap > 0)
    return {
        "examples": len(examples),
        "chosen": summarize_reward_outputs(chosen_scores),
        "rejected": summarize_reward_outputs(rejected_scores),
        "mean_reward_gap": sum(reward_gaps) / len(reward_gaps) if reward_gaps else 0.0,
        "positive_reward_gap_rate": positive_gap_count / len(reward_gaps) if reward_gaps else 0.0,
    }


def _weight_summary(examples: list[DPOExample]) -> dict[str, Any]:
    weights = [float(example.final_weight) for example in examples]
    pair_types: dict[str, int] = {}
    for example in examples:
        pair_type = example.pair_type or "unknown"
        pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
    return {
        "weight_min": min(weights) if weights else 0.0,
        "weight_max": max(weights) if weights else 0.0,
        "weight_mean": sum(weights) / len(weights) if weights else 0.0,
        "pair_type_counts": pair_types,
    }


def _dry_run(config: dict[str, Any], *, max_samples: int) -> None:
    data_config = _section(config, "data")
    train_config = _section(config, "training")
    model_config = _section(config, "model")
    data_file = data_config.get("train_file")
    if not data_file:
        raise ValueError("data.train_file is required.")
    require_existing_images = bool(data_config.get("require_existing_images", False))
    examples, skipped_missing_images = _read_data_file(
        data_file,
        max_samples=max_samples,
        require_existing_images=require_existing_images,
    )
    _validate_examples(examples, data_file=data_file)
    holdout_examples: list[DPOExample] = []
    holdout_skipped_missing_images = 0
    holdout_file = data_config.get("holdout_file")
    if holdout_file:
        holdout_examples, holdout_skipped_missing_images = _read_data_file(
            holdout_file,
            max_samples=max_samples,
            require_existing_images=require_existing_images,
        )
        _validate_examples(holdout_examples, data_file=holdout_file)
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/dpo/qwen3vl_8b_dpo")))
    sft_checkpoint = Path(str(model_config.get("sft_checkpoint_dir", "")))
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    audit = _score_preference_examples(examples, output_schema=schema)
    holdout_audit = _score_preference_examples(holdout_examples, output_schema=schema) if holdout_examples else {}
    metrics_path = train_config.get("metrics_output")
    if metrics_path:
        _write_json(
            metrics_path,
            {
                "stage": "dpo_dry_run",
                "skipped_missing_images": skipped_missing_images,
                "loss_type": str(train_config.get("loss_type", "dpo")).strip().lower(),
                "beta": float(train_config.get("beta", 0.1)),
                "lambda_sft": float(train_config.get("lambda_sft", 0.0)),
                "max_weight": float(train_config.get("max_weight", 3.0)),
                "logprob_normalization": str(train_config.get("logprob_normalization", "sum")).strip().lower(),
                "early_stop_metric": str(train_config.get("early_stop_metric", "") or ""),
                "weight_summary": _weight_summary(examples),
                **audit,
                "holdout": {
                    "file": holdout_file,
                    "skipped_missing_images": holdout_skipped_missing_images,
                    "weight_summary": _weight_summary(holdout_examples),
                    **holdout_audit,
                }
                if holdout_file
                else None,
                "train_decode_dev_file": data_config.get("decode_dev_file"),
            },
        )
    print("phase08_dpo_dry_run=ok")
    print(f"train_file={data_file}")
    print(f"examples_checked={len(examples)}")
    print(f"skipped_missing_images={skipped_missing_images}")
    if holdout_file:
        print(f"holdout_file={holdout_file}")
        print(f"holdout_examples_checked={len(holdout_examples)}")
        print(f"holdout_skipped_missing_images={holdout_skipped_missing_images}")
    if data_config.get("decode_dev_file"):
        print(f"train_decode_dev_file={data_config['decode_dev_file']}")
    print(f"output_dir={output_dir}")
    print(f"sft_checkpoint_exists={sft_checkpoint.exists()}")
    print(f"sft_adapter_exists={_checkpoint_has_adapter(sft_checkpoint)}")
    print(f"mean_reward_gap={audit['mean_reward_gap']:.6f}")
    print(f"positive_reward_gap_rate={audit['positive_reward_gap_rate']:.6f}")


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


def _as_sft_example(example: DPOExample, answer: str) -> SFTExample:
    return SFTExample(
        case_id=example.case_id,
        images=example.images,
        user_prompt=example.prompt,
        answer=answer,
    )


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _sequence_logps(
    model: Any,
    batch: dict[str, Any],
    *,
    logprob_chunk_size: int = 64,
    normalization: str = "sum",
) -> Any:
    import torch

    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**model_inputs)
    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    label_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~label_mask, 0)
    sequence_logps = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    for start in range(0, logits.shape[1], logprob_chunk_size):
        end = min(start + logprob_chunk_size, logits.shape[1])
        chunk_logits = logits[:, start:end, :]
        chunk_labels = safe_labels[:, start:end]
        chunk_mask = label_mask[:, start:end]
        chunk_logps = torch.nn.functional.log_softmax(chunk_logits, dim=-1).gather(
            dim=-1,
            index=chunk_labels.unsqueeze(-1),
        ).squeeze(-1)
        sequence_logps = sequence_logps + (chunk_logps * chunk_mask).sum(dim=-1)
    normalized = normalization.strip().lower()
    if normalized == "mean_token":
        return sequence_logps / _sequence_token_counts(batch).to(sequence_logps.device)
    if normalized != "sum":
        raise ValueError(f"Unsupported logprob_normalization: {normalization!r}")
    return sequence_logps


def _sequence_token_counts(batch: dict[str, Any]) -> Any:
    labels = batch["labels"]
    return labels[:, 1:].ne(-100).sum(dim=-1).clamp_min(1)


def _example_weights(examples: list[DPOExample], *, device: Any, max_weight: float) -> Any:
    import torch

    weights = torch.tensor([max(0.0, float(example.final_weight)) for example in examples], device=device)
    weights = torch.clamp(weights, max=max_weight)
    mean = weights.mean().clamp_min(1e-6)
    return weights / mean


def _sft_loss_weights(examples: list[DPOExample], *, device: Any) -> Any:
    import torch

    return torch.tensor([max(0.0, float(example.sft_loss_weight)) for example in examples], device=device)


def _weighted_mean(values: Any, weights: Any) -> Any:
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def _preference_loss_values(logits: Any, *, beta: float, loss_type: str) -> Any:
    import torch

    normalized = loss_type.strip().lower()
    if normalized == "dpo":
        return -torch.nn.functional.logsigmoid(beta * logits)
    if normalized == "ipo":
        target_margin = 1.0 / (2.0 * beta)
        return (logits - target_margin).pow(2)
    raise ValueError(f"Unsupported DPO loss_type: {loss_type!r}. Expected 'dpo' or 'ipo'.")


def _ipo_target_margin(*, beta: float, loss_type: str) -> float | None:
    if loss_type.strip().lower() != "ipo":
        return None
    return 1.0 / (2.0 * beta)


def _evaluate_preference_logits(
    *,
    model: Any,
    ref_model: Any,
    processor: Any,
    examples: list[DPOExample],
    beta: float,
    batch_size: int,
    max_examples: int,
    max_weight: float,
    logprob_normalization: str,
    image_max_pixels: int | None,
) -> dict[str, float]:
    import torch

    if not examples:
        return {
            "examples": 0.0,
            "holdout_pair_accuracy": 0.0,
            "holdout_preference_margin": 0.0,
            "holdout_chosen_logp": 0.0,
            "holdout_rejected_logp": 0.0,
            "holdout_chosen_nll": 0.0,
            "holdout_reference_drift": 0.0,
        }
    selected = examples[:max_examples]
    collator = DataCollatorForQwenVLSFT(
        processor,
        image_max_pixels=image_max_pixels,
    )
    margins: list[float] = []
    chosen_logps: list[float] = []
    rejected_logps: list[float] = []
    chosen_nlls: list[float] = []
    drifts: list[float] = []
    correct = 0
    total = 0
    model_was_training = model.training
    model.eval()
    ref_model.eval()
    with torch.no_grad():
        for batch_examples in _iter_batches(selected, batch_size=batch_size):
            chosen_batch = collator([_as_sft_example(example, example.chosen) for example in batch_examples])
            rejected_batch = collator([_as_sft_example(example, example.rejected) for example in batch_examples])
            chosen_batch = _move_batch_to_device(chosen_batch, _model_device(model))
            rejected_batch = _move_batch_to_device(rejected_batch, _model_device(model))
            ref_chosen = _sequence_logps(ref_model, _move_batch_to_device(chosen_batch, _model_device(ref_model)), normalization=logprob_normalization)
            ref_rejected = _sequence_logps(ref_model, _move_batch_to_device(rejected_batch, _model_device(ref_model)), normalization=logprob_normalization)
            policy_chosen = _sequence_logps(model, chosen_batch, normalization=logprob_normalization)
            policy_rejected = _sequence_logps(model, rejected_batch, normalization=logprob_normalization)
            logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            weights = _example_weights(batch_examples, device=logits.device, max_weight=max_weight)
            if logprob_normalization == "mean_token":
                nll = -policy_chosen
            else:
                token_counts = _sequence_token_counts(chosen_batch).to(policy_chosen.device)
                nll = -policy_chosen / token_counts
            margins.append(float(_weighted_mean(logits, weights).detach().cpu()))
            chosen_logps.append(float(_weighted_mean(policy_chosen, weights).detach().cpu()))
            rejected_logps.append(float(_weighted_mean(policy_rejected, weights).detach().cpu()))
            chosen_nlls.append(float(_weighted_mean(nll, weights).detach().cpu()))
            drifts.append(float(_weighted_mean((policy_chosen - ref_chosen).abs(), weights).detach().cpu()))
            correct += int((beta * logits > 0).sum().detach().cpu())
            total += int(logits.numel())
    if model_was_training:
        model.train()
    return {
        "examples": float(total),
        "holdout_pair_accuracy": correct / total if total else 0.0,
        "holdout_preference_margin": sum(margins) / len(margins) if margins else 0.0,
        "holdout_chosen_logp": sum(chosen_logps) / len(chosen_logps) if chosen_logps else 0.0,
        "holdout_rejected_logp": sum(rejected_logps) / len(rejected_logps) if rejected_logps else 0.0,
        "holdout_chosen_nll": sum(chosen_nlls) / len(chosen_nlls) if chosen_nlls else 0.0,
        "holdout_reference_drift": sum(drifts) / len(drifts) if drifts else 0.0,
    }


def _iter_batches(examples: list[DPOExample], *, batch_size: int) -> list[list[DPOExample]]:
    return [examples[index : index + batch_size] for index in range(0, len(examples), batch_size)]


def _load_sft_policy(config: dict[str, Any]):
    try:
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("peft and transformers are required for DPO training.") from exc

    model_config = _section(config, "model")
    sft_checkpoint = Path(str(model_config["sft_checkpoint_dir"]))
    policy_checkpoint = Path(str(model_config.get("policy_checkpoint_dir") or sft_checkpoint))
    if not _checkpoint_has_adapter(sft_checkpoint):
        raise FileNotFoundError(f"SFT reference adapter checkpoint is incomplete: {sft_checkpoint}")
    if not _checkpoint_has_adapter(policy_checkpoint):
        raise FileNotFoundError(f"Policy adapter checkpoint is incomplete: {policy_checkpoint}")
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
    policy = PeftModel.from_pretrained(policy, str(policy_checkpoint), is_trainable=True)

    ref_policy = model_class.from_pretrained(model_path, **base_kwargs)
    ref_policy = PeftModel.from_pretrained(ref_policy, str(model_config["sft_checkpoint_dir"]), is_trainable=False)
    ref_policy.eval()
    return policy, ref_policy, processor


def _train(config: dict[str, Any], *, max_samples: int | None) -> None:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("torch and transformers are required for DPO training.") from exc

    data_config = _section(config, "data")
    train_config = _section(config, "training")
    set_random_seed(int(config.get("seed", 42)))
    require_existing_images = bool(data_config.get("require_existing_images", False))
    examples, skipped_missing_images = _read_data_file(
        data_config["train_file"],
        max_samples=max_samples,
        require_existing_images=require_existing_images,
    )
    _validate_examples(examples, data_file=data_config["train_file"])
    holdout_examples: list[DPOExample] = []
    holdout_skipped_missing_images = 0
    if data_config.get("holdout_file"):
        holdout_examples, holdout_skipped_missing_images = _read_data_file(
            data_config["holdout_file"],
            max_samples=int(train_config.get("max_holdout_examples", 128)),
            require_existing_images=require_existing_images,
        )
        _validate_examples(holdout_examples, data_file=data_config["holdout_file"])
    schema = read_yaml(data_config.get("output_schema", "configs/schema/output_schema.json"))
    preference_audit = _score_preference_examples(examples, output_schema=schema)
    metrics_path = train_config.get("metrics_output")
    if metrics_path:
        _write_json(
            metrics_path,
            {
                "stage": "dpo_pre_train",
                "skipped_missing_images": skipped_missing_images,
                "weight_summary": _weight_summary(examples),
                **preference_audit,
                "holdout": {
                    "file": data_config.get("holdout_file"),
                    "skipped_missing_images": holdout_skipped_missing_images,
                    "weight_summary": _weight_summary(holdout_examples),
                }
                if data_config.get("holdout_file")
                else None,
                "train_decode_dev_file": data_config.get("decode_dev_file"),
            },
        )
    policy, ref_policy, processor = _load_sft_policy(config)
    del AutoProcessor

    if bool(train_config.get("gradient_checkpointing", True)):
        policy.gradient_checkpointing_enable()
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()

    ref_policy.eval()
    policy.train()
    image_max_pixels = train_config.get("image_max_pixels")
    collator = DataCollatorForQwenVLSFT(
        processor,
        image_max_pixels=int(image_max_pixels) if image_max_pixels else None,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(train_config.get("learning_rate", 5e-6)),
    )
    batch_size = int(train_config.get("per_device_train_batch_size", 1))
    grad_accum = int(train_config.get("gradient_accumulation_steps", 16))
    epochs = int(float(train_config.get("num_train_epochs", 1)))
    beta = float(train_config.get("beta", 0.1))
    loss_type = str(train_config.get("loss_type", "dpo")).strip().lower()
    if loss_type not in {"dpo", "ipo"}:
        raise ValueError(f"training.loss_type must be 'dpo' or 'ipo', got {loss_type!r}.")
    ipo_target_margin = _ipo_target_margin(beta=beta, loss_type=loss_type)
    max_weight = float(train_config.get("max_weight", 3.0))
    logprob_normalization = str(train_config.get("logprob_normalization", "sum")).strip().lower()
    if logprob_normalization not in {"sum", "mean_token"}:
        raise ValueError(
            "training.logprob_normalization must be 'sum' or 'mean_token', "
            f"got {logprob_normalization!r}."
        )
    lambda_sft = float(train_config.get("lambda_sft", 0.0))
    early_stop_metric = str(train_config.get("early_stop_metric", "") or "")
    logging_steps = int(train_config.get("logging_steps", 10))
    save_steps = int(train_config.get("save_steps", 500))
    eval_steps = int(train_config.get("eval_steps", 0))
    max_train_steps = int(train_config.get("max_train_steps", 0))
    max_holdout_examples = int(train_config.get("max_holdout_examples", 128))
    output_dir = ensure_dir(str(train_config.get("output_dir", "outputs/checkpoints/dpo/qwen3vl_8b_dpo")))
    batches = _iter_batches(examples, batch_size=batch_size)
    global_step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    holdout_history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(epochs):
        for batch_examples in batches:
            chosen_batch = collator([_as_sft_example(example, example.chosen) for example in batch_examples])
            rejected_batch = collator([_as_sft_example(example, example.rejected) for example in batch_examples])
            chosen_batch = _move_batch_to_device(chosen_batch, _model_device(policy))
            rejected_batch = _move_batch_to_device(rejected_batch, _model_device(policy))
            autocast_enabled = bool(train_config.get("bf16", True)) and torch.cuda.is_available()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                with torch.no_grad():
                    ref_chosen = _sequence_logps(ref_policy, _move_batch_to_device(chosen_batch, _model_device(ref_policy)), normalization=logprob_normalization)
                    ref_rejected = _sequence_logps(ref_policy, _move_batch_to_device(rejected_batch, _model_device(ref_policy)), normalization=logprob_normalization)
                policy_chosen = _sequence_logps(policy, chosen_batch, normalization=logprob_normalization)
                policy_rejected = _sequence_logps(policy, rejected_batch, normalization=logprob_normalization)
                logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
                weights = _example_weights(batch_examples, device=logits.device, max_weight=max_weight)
                preference_loss_values = _preference_loss_values(logits, beta=beta, loss_type=loss_type)
                preference_loss = _weighted_mean(preference_loss_values, weights)
                sft_loss = torch.zeros((), device=logits.device, dtype=preference_loss.dtype)
                if lambda_sft > 0:
                    if logprob_normalization == "mean_token":
                        chosen_nll = -policy_chosen
                    else:
                        token_counts = _sequence_token_counts(chosen_batch).to(policy_chosen.device)
                        chosen_nll = -policy_chosen / token_counts
                    nll_weights = weights * _sft_loss_weights(batch_examples, device=logits.device)
                    if float(nll_weights.detach().sum().cpu()) > 0:
                        sft_loss = _weighted_mean(chosen_nll, nll_weights)
                loss = preference_loss + lambda_sft * sft_loss
            (loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum == 0 or micro_step == len(batches) * epochs:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "epoch": float(epoch + 1),
                    "global_step": float(global_step),
                    "loss_type": loss_type,
                    "logprob_normalization": logprob_normalization,
                    "loss": float(loss.detach().cpu()),
                    "dpo_loss": float(preference_loss.detach().cpu()),
                    "preference_loss": float(preference_loss.detach().cpu()),
                    "sft_nll_loss": float(sft_loss.detach().cpu()),
                    "chosen_logp": float(policy_chosen.detach().mean().cpu()),
                    "rejected_logp": float(policy_rejected.detach().mean().cpu()),
                    "preference_margin": float(logits.detach().mean().cpu()),
                    "ipo_target_margin": float(ipo_target_margin) if ipo_target_margin is not None else 0.0,
                    "reference_drift": float((policy_chosen - ref_chosen).detach().abs().mean().cpu()),
                    "mean_batch_weight": float(weights.detach().mean().cpu()),
                }
                history.append(record)
                if global_step % logging_steps == 0 or global_step == 1:
                    print(json.dumps({"stage": "dpo_train", **record}, ensure_ascii=False))
                if holdout_examples and eval_steps > 0 and global_step % eval_steps == 0:
                    holdout_record = {
                        "global_step": float(global_step),
                        **_evaluate_preference_logits(
                            model=policy,
                            ref_model=ref_policy,
                            processor=processor,
                            examples=holdout_examples,
                            beta=beta,
                            batch_size=batch_size,
                            max_examples=max_holdout_examples,
                            max_weight=max_weight,
                            logprob_normalization=logprob_normalization,
                            image_max_pixels=int(image_max_pixels) if image_max_pixels else None,
                        ),
                    }
                    holdout_history.append(holdout_record)
                    print(json.dumps({"stage": "dpo_holdout", **holdout_record}, ensure_ascii=False))
                if save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_dir = ensure_dir(output_dir / f"checkpoint-{global_step}")
                    policy.save_pretrained(str(checkpoint_dir))
                    processor.save_pretrained(str(checkpoint_dir))
                if max_train_steps > 0 and global_step >= max_train_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    policy.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    if holdout_examples and (not holdout_history or holdout_history[-1].get("global_step") != float(global_step)):
        holdout_history.append(
            {
                "global_step": float(global_step),
                **_evaluate_preference_logits(
                    model=policy,
                    ref_model=ref_policy,
                    processor=processor,
                    examples=holdout_examples,
                    beta=beta,
                    batch_size=batch_size,
                    max_examples=max_holdout_examples,
                    max_weight=max_weight,
                    logprob_normalization=logprob_normalization,
                    image_max_pixels=int(image_max_pixels) if image_max_pixels else None,
                ),
            }
        )
    if metrics_path:
        _write_json(
            metrics_path,
            {
                "stage": "dpo_done",
                "skipped_missing_images": skipped_missing_images,
                "loss_type": loss_type,
                "beta": beta,
                "lambda_sft": lambda_sft,
                "max_weight": max_weight,
                "logprob_normalization": logprob_normalization,
                "early_stop_metric": early_stop_metric,
                "ipo_target_margin": ipo_target_margin,
                **preference_audit,
                "training_history": history,
                "holdout_history": holdout_history,
                "train_decode_dev_file": data_config.get("decode_dev_file"),
                "global_step": global_step,
            },
        )


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
