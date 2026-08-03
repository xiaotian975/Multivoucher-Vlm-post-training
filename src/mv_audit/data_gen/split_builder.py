"""Build phase 03 case-level data splits without image-level leakage."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mv_audit.data_gen.case_validator import load_schema, validate_cases
from mv_audit.data_gen.risk_rule_engine import update_case_labels
from mv_audit.utils import load_config, read_jsonl, write_jsonl


SPLIT_KEYS = [
    "train_cases",
    "val_in_template_cases",
    "val_unseen_template_cases",
    "test_clean_cases",
    "test_robust_cases",
    "test_unseen_template_cases",
    "test_hard_negative_cases",
]

SPLIT_FILENAMES = {
    "train_cases": "train_cases.jsonl",
    "val_in_template_cases": "val_in_template_cases.jsonl",
    "val_unseen_template_cases": "val_unseen_template_cases.jsonl",
    "test_clean_cases": "test_clean_cases.jsonl",
    "test_robust_cases": "test_robust_cases.jsonl",
    "test_unseen_template_cases": "test_unseen_template_cases.jsonl",
    "test_hard_negative_cases": "test_hard_negative_cases.jsonl",
}

SPLIT_NAMES = {
    "train_cases": "train",
    "val_in_template_cases": "val_in_template",
    "val_unseen_template_cases": "val_unseen_template",
    "test_clean_cases": "test_clean",
    "test_robust_cases": "test_robust",
    "test_unseen_template_cases": "test_unseen_template",
    "test_hard_negative_cases": "test_hard_negative",
}

TEMPLATE_GROUPS = {
    "train_cases": "train",
    "val_in_template_cases": "train",
    "val_unseen_template_cases": "val",
    "test_clean_cases": "standard_test",
    "test_robust_cases": "standard_test",
    "test_unseen_template_cases": "strong_generalization_test",
    "test_hard_negative_cases": "standard_test",
}

HARD_NEGATIVE_ANOMALIES = {
    "amount_mismatch",
    "merchant_mismatch",
    "applicant_mismatch",
    "order_id_mismatch",
}


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _split_targets(total_cases: int, config: dict[str, Any]) -> dict[str, int]:
    requested = {key: int(config.get(key, 0)) for key in SPLIT_KEYS}
    requested_total = sum(requested.values())
    if requested_total <= 0:
        raise ValueError("At least one split target must be positive")
    exact = {key: total_cases * requested[key] / requested_total for key in SPLIT_KEYS}
    counts = {key: int(exact[key]) for key in SPLIT_KEYS}
    remainder = total_cases - sum(counts.values())
    fractions = sorted(SPLIT_KEYS, key=lambda key: (exact[key] - counts[key], requested[key]), reverse=True)
    for key in fractions[:remainder]:
        counts[key] += 1
    return counts


def _take_matching(pool: list[dict[str, Any]], count: int, predicate: Any) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for case in pool:
        if len(selected) < count and predicate(case):
            selected.append(case)
        else:
            remaining.append(case)
    pool[:] = remaining
    return selected


def _prepare_split_cases(cases: list[dict[str, Any]], split_key: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source_case in cases:
        case = copy.deepcopy(source_case)
        metadata = case.setdefault("metadata", {})
        metadata["split_name"] = SPLIT_NAMES[split_key]
        metadata["template_group"] = TEMPLATE_GROUPS[split_key]
        update_case_labels(case)
        prepared.append(case)
    return prepared


def build_splits(cases: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build deterministic case-level splits from anomaly-injected cases."""

    seed = int(config.get("seed", 42)) + 301
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)

    targets = _split_targets(len(shuffled), config)
    pool = shuffled
    raw_splits: dict[str, list[dict[str, Any]]] = {}

    hard_count = targets["test_hard_negative_cases"]
    raw_splits["test_hard_negative_cases"] = _take_matching(
        pool,
        hard_count,
        lambda case: case.get("primary_anomaly_type") in HARD_NEGATIVE_ANOMALIES,
    )
    if len(raw_splits["test_hard_negative_cases"]) < hard_count:
        deficit = hard_count - len(raw_splits["test_hard_negative_cases"])
        raw_splits["test_hard_negative_cases"].extend(pool[:deficit])
        del pool[:deficit]

    for key in SPLIT_KEYS:
        if key == "test_hard_negative_cases":
            continue
        count = targets[key]
        raw_splits[key] = pool[:count]
        del pool[:count]

    if pool:
        raw_splits["train_cases"].extend(pool)

    splits = {key: _prepare_split_cases(raw_splits.get(key, []), key) for key in SPLIT_KEYS}
    all_ids = [case["case_id"] for records in splits.values() for case in records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Case-level split leakage detected: duplicate case_id across splits")

    stats = {
        "total_cases": len(all_ids),
        "seed": seed,
        "requested_total": sum(int(config.get(key, 0)) for key in SPLIT_KEYS),
        "normalized_to_input_total": len(cases),
        "splits": {
            key: {
                "file": SPLIT_FILENAMES[key],
                "cases": len(records),
                "primary_anomaly_type": dict(Counter(case["primary_anomaly_type"] for case in records)),
                "risk_level": dict(Counter(case["risk_level"] for case in records)),
                "audit_result": dict(Counter(case["audit_result"] for case in records)),
            }
            for key, records in splits.items()
        },
    }
    return splits, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase 03 case-level splits.")
    parser.add_argument("--input", required=True, help="Path to anomaly-injected case JSONL.")
    parser.add_argument("--config", required=True, help="Path to data generation YAML config.")
    parser.add_argument("--schema", default="configs/schema/case_schema.json", help="Path to case schema.")
    parser.add_argument("--output_dir", default="data/mv_audit/raw_cases", help="Directory for split JSONL files.")
    parser.add_argument("--stats_output", default=None, help="Optional path to JSON statistics report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    schema = load_schema(args.schema)
    cases = read_jsonl(args.input)
    validate_cases(cases, schema=schema, require_normal=False)

    splits, stats = build_splits(cases, config)
    output_dir = Path(args.output_dir)
    for key, records in splits.items():
        validate_cases(records, schema=schema, require_normal=False)
        write_jsonl(records, output_dir / SPLIT_FILENAMES[key])

    stats_output = args.stats_output or output_dir / "split_stats_debug.json"
    _write_json(stats, stats_output)
    print(f"split_cases={stats['total_cases']}")
    print(f"output_dir={output_dir}")
    print(f"stats={stats_output}")


if __name__ == "__main__":
    main()
