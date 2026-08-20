"""Sample multiple raw VLM completions for model-mined preference data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mv_audit.inference.qwen3vl_common import (
    generate_text,
    image_to_message_uri,
    load_qwen3vl_model_and_processor,
    move_inputs_to_model,
    process_messages,
)
from mv_audit.utils import read_jsonl, write_jsonl


def _select_shard(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}.")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _messages(row: dict[str, Any], image_max_pixels: int | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in row.get("images") or []:
        image_path = Path(str(item["image_path"])).resolve()
        content.append(
            {
                "type": "text",
                "text": f"{item.get('image_id', '')}: {item.get('doc_type', '')}",
            }
        )
        payload: dict[str, Any] = {"type": "image", "image": image_to_message_uri(image_path)}
        if image_max_pixels is not None:
            payload["max_pixels"] = image_max_pixels
        content.append(payload)
    content.append({"type": "text", "text": str(row["prompt"])})
    return [{"role": "user", "content": content}]


def _seed_for(seed: int, case_id: str, generation_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}:{generation_index}".encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def _generate_batched(
    model: Any,
    processor: Any,
    inputs: Any,
    *,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=count,
    )
    input_token_count = inputs.input_ids.shape[1]
    texts = processor.batch_decode(
        generated_ids[:, input_token_count:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    elapsed = time.perf_counter() - started
    return [
        {
            "generation_index": index,
            "temperature": temperature,
            "raw_output": raw_output,
            "elapsed_seconds": elapsed,
            "batched_generation": True,
        }
        for index, raw_output in enumerate(texts)
    ]

def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    rows = _select_shard(rows, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError("Selected rollout shard is empty.")
    for row in rows:
        if not row.get("prompt") or not row.get("images"):
            raise ValueError(f"Invalid candidate row {row.get('case_id')}")
        for item in row["images"]:
            path = Path(str(item.get("image_path") or ""))
            if not path.exists():
                raise FileNotFoundError(path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "model_mined_sampling_dry_run",
                    "rows": len(rows),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "num_generations": len(args.temperatures),
                    "adapter_exists": Path(args.adapter).exists(),
                }
            )
        )
        return

    try:
        import torch
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("torch and peft are required for model-mined sampling.") from exc

    model_config = {
        "model_name_or_path": args.model_name_or_path,
        "local_model_dir": args.local_model_dir,
        "dtype": args.dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    model, processor, model_path = load_qwen3vl_model_and_processor(model_config)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    output = Path(args.output)
    completed = _existing(output) if args.resume else {}
    ordered_ids = [str(row["case_id"]) for row in rows]
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in completed:
            continue
        inputs = process_messages(processor, _messages(row, args.image_max_pixels))
        inputs = move_inputs_to_model(inputs, model)
        completions = []
        if args.batched_generations or len(args.temperatures) > 1:
            seed = _seed_for(args.seed, case_id, 0)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                completions = _generate_batched(
                    model,
                    processor,
                    inputs,
                    count=len(args.temperatures),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.batched_temperature,
                    top_p=args.top_p,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    json.dumps({"event": "batched_generation_oom_fallback", "case_id": case_id}),
                    flush=True,
                )
        for generation_index, temperature in enumerate(args.temperatures):
            if completions:
                break
            torch.manual_seed(_seed_for(args.seed, case_id, generation_index))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed_for(args.seed, case_id, generation_index))
            started = time.perf_counter()
            raw_output = generate_text(
                model,
                processor,
                inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
            )
            completions.append(
                {
                    "generation_index": generation_index,
                    "temperature": temperature,
                    "raw_output": raw_output,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        result = dict(row)
        result["completions"] = completions
        result["model_path"] = model_path
        result["adapter_path"] = args.adapter
        completed[case_id] = result
        write_jsonl([completed[cid] for cid in ordered_ids if cid in completed], output)
        print(
            json.dumps(
                {
                    "event": "model_mined_case_complete",
                    "case_id": case_id,
                    "completed": len(completed),
                    "total": len(rows),
                    "shard_index": args.shard_index,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local_model_dir", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.6, 0.9, 1.1])
    parser.add_argument("--batched_generations", action="store_true")
    parser.add_argument("--batched_temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()