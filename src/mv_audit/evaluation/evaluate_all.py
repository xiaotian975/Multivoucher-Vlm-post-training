"""Evaluate MultiVoucher-Audit predictions against evidence-grounded ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.case_scorer import aggregate_case_scores, error_case_record, score_case
from mv_audit.utils import read_jsonl, read_yaml, write_jsonl


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def evaluate(
    *,
    ground_truth_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validator = Draft202012Validator(output_schema)
    predictions = {row["case_id"]: row for row in prediction_rows}
    scores = [
        score_case(truth_row, predictions.get(truth_row["case_id"]), validator=validator)
        for truth_row in ground_truth_rows
    ]
    metrics = aggregate_case_scores(scores)
    error_cases = [error_case_record(score) for score in scores if score.issue_codes]
    return metrics, error_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MultiVoucher-Audit model outputs.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_schema", default="configs/schema/output_schema.json")
    parser.add_argument("--metrics_output", required=True)
    parser.add_argument("--errors_output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, error_cases = evaluate(
        ground_truth_rows=read_jsonl(args.ground_truth),
        prediction_rows=read_jsonl(args.predictions),
        output_schema=read_yaml(args.output_schema),
    )
    _write_json(metrics, args.metrics_output)
    write_jsonl(error_cases, args.errors_output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
