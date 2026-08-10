"""Build deterministic Phase 07 sample manifests."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("test_clean", "test_robust", "test_unseen_template", "test_hard_negative")
DEFAULT_SEED = 42


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _stratified_pick(rows: list[dict[str, Any]], *, count: int, seed: int, split: str, anomaly: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("risk_level")), str(row.get("audit_result")))].append(row)

    if count >= len(rows):
        return list(rows)

    quotas: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    remaining = count
    total = len(rows)
    for key, group_rows in groups.items():
        raw_quota = count * len(group_rows) / total
        quota = min(len(group_rows), math.floor(raw_quota))
        quotas[key] = quota
        remaining -= quota
        fractions.append((raw_quota - quota, key))

    for _, key in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1

    while remaining > 0:
        made_progress = False
        for key in sorted(groups):
            if remaining <= 0:
                break
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                made_progress = True
        if not made_progress:
            break

    selected: list[dict[str, Any]] = []
    for key, quota in quotas.items():
        candidates = list(groups[key])
        rng = random.Random(f"{seed}:{split}:{anomaly}:{key[0]}:{key[1]}")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    return selected


def _quota_for_split(rows: list[dict[str, Any]], split: str, sample_size: int) -> dict[str, int]:
    anomalies = sorted({str(row.get("primary_anomaly_type")) for row in rows})
    if split == "test_hard_negative":
        anomalies = [a for a in anomalies if a != "none"]
    base = sample_size // len(anomalies)
    remainder = sample_size % len(anomalies)
    return {anomaly: base + (index < remainder) for index, anomaly in enumerate(anomalies)}


def build_split_manifest(rows: list[dict[str, Any]], *, split: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    by_anomaly: dict[str, list[dict[str, Any]]] = defaultdict(list)
    index_by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        index_by_id[case_id] = index
        by_anomaly[str(row.get("primary_anomaly_type"))].append(row)

    selected: list[dict[str, Any]] = []
    for anomaly, quota in _quota_for_split(rows, split, sample_size).items():
        candidates = by_anomaly[anomaly]
        if len(candidates) < quota:
            raise ValueError(f"{split} has only {len(candidates)} rows for {anomaly}, need {quota}.")
        selected.extend(_stratified_pick(candidates, count=quota, seed=seed, split=split, anomaly=anomaly))

    selected_ids = [str(row["case_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"{split} sample contains duplicate case_id values.")
    if len(selected) != sample_size:
        raise ValueError(f"{split} sample has {len(selected)} rows, expected {sample_size}.")

    selected.sort(key=lambda row: index_by_id[str(row["case_id"])])
    manifest: list[dict[str, Any]] = []
    for sample_index, row in enumerate(selected):
        manifest.append(
            {
                "sample_index": sample_index,
                "split": split,
                "case_id": row["case_id"],
                "primary_anomaly_type": row.get("primary_anomaly_type"),
                "risk_level": row.get("risk_level"),
                "audit_result": row.get("audit_result"),
            }
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 07 deterministic sample manifests.")
    parser.add_argument("--raw_cases_dir", default="data/mv_audit/raw_cases/main")
    parser.add_argument("--output_dir", default="data/mv_audit/eval_sets_phase07_sample500/manifests")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary: dict[str, Any] = {"sample_size": args.sample_size, "seed": args.seed, "splits": {}}
    for split in args.splits:
        rows = _read_jsonl(Path(args.raw_cases_dir) / f"{split}_cases.jsonl")
        manifest = build_split_manifest(rows, split=split, sample_size=args.sample_size, seed=args.seed)
        _write_jsonl(output_dir / f"{split}_case_ids.jsonl", manifest)
        counts: dict[str, dict[str, int]] = {
            "primary_anomaly_type": defaultdict(int),
            "risk_level": defaultdict(int),
            "audit_result": defaultdict(int),
        }
        for row in manifest:
            counts["primary_anomaly_type"][str(row["primary_anomaly_type"])] += 1
            counts["risk_level"][str(row["risk_level"])] += 1
            counts["audit_result"][str(row["audit_result"])] += 1
        summary["splits"][split] = {
            "rows": len(manifest),
            "primary_anomaly_type": dict(sorted(counts["primary_anomaly_type"].items())),
            "risk_level": dict(sorted(counts["risk_level"].items())),
            "audit_result": dict(sorted(counts["audit_result"].items())),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
