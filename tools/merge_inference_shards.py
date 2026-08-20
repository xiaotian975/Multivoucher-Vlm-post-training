"""Merge deterministic batch-inference shards into standard evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mv_audit.inference.batch_inference import (
    _ground_truth_path,
    _prediction_path,
    build_eval_rows,
)
from mv_audit.utils import load_config, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--split", default="train_decode_dev")
    parser.add_argument("--num_shards", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    expected_rows = build_eval_rows(
        config=config,
        split=args.split,
        model_id=args.model_id,
        limit=None,
    )
    expected_ids = [str(row["case_id"]) for row in expected_rows]
    predictions: dict[str, dict] = {}
    duplicates: list[str] = []
    for shard_index in range(args.num_shards):
        path = _prediction_path(
            config,
            model_id=args.model_id,
            split=args.split,
            shard_index=shard_index,
            num_shards=args.num_shards,
        )
        for row in read_jsonl(path):
            case_id = str(row["case_id"])
            if case_id in predictions:
                duplicates.append(case_id)
            predictions[case_id] = row
    missing = [case_id for case_id in expected_ids if case_id not in predictions]
    extra = sorted(set(predictions) - set(expected_ids))
    if missing or extra or duplicates:
        raise ValueError(
            f"Shard merge mismatch: missing={len(missing)} extra={len(extra)} duplicates={len(duplicates)}"
        )

    prediction_output = _prediction_path(config, model_id=args.model_id, split=args.split)
    truth_output = _ground_truth_path(config, split=args.split)
    write_jsonl([predictions[case_id] for case_id in expected_ids], prediction_output)
    write_jsonl(expected_rows, truth_output)
    summary = {
        "expected": len(expected_ids),
        "written": len(predictions),
        "prediction_output": str(prediction_output),
        "ground_truth_output": str(truth_output),
        "num_shards": args.num_shards,
    }
    summary_path = prediction_output.with_suffix(".merge.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()