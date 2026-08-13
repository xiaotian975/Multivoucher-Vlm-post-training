"""Build the small Train-only SFT mix for Phase08 high-risk repair."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mv_audit.utils import ensure_dir, iter_jsonl, write_jsonl


REPAIR_PACK_DIR = Path("docs/experiments/phase08_high_risk_repair_pack_20260813")
TWO_CANDIDATE_ARCHIVE = Path("docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3")
SAMPLE500_SPLITS = ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id", "").split("_full_audit")[0])


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return answer
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        content = messages[-1].get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _load_excluded_case_ids(repo: Path, repair_case_ids: set[str]) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = {
        "repair_pack": set(repair_case_ids),
        "dpo_v2_holdout": set(),
        "train_decode_dev": set(),
        "sample500": set(),
    }
    for row in _read_jsonl_if_exists(repo / "data/mv_audit/dpo_v2/pairs_holdout.jsonl"):
        excluded["dpo_v2_holdout"].add(_case_id(row))
    for row in _read_jsonl_if_exists(repo / "data/mv_audit/dpo_v2/train_decode_dev.jsonl"):
        excluded["train_decode_dev"].add(_case_id(row))
    for variant in ["dpo_v2_baseline", "auxdpo_v2_strong"]:
        for kind in ["ground_truth", "predictions"]:
            for row in _read_jsonl_if_exists(repo / TWO_CANDIDATE_ARCHIVE / kind / variant / "train_decode_dev.jsonl"):
                excluded["train_decode_dev"].add(_case_id(row))
    for split in SAMPLE500_SPLITS:
        for row in _read_jsonl_if_exists(repo / f"data/mv_audit/raw_cases/main/{split}_cases.jsonl"):
            excluded["sample500"].add(_case_id(row))
    return excluded


def _calibration_bucket(row: dict[str, Any]) -> str | None:
    if row.get("source_split") != "MV-Train":
        return None
    answer = _answer(row)
    risk = answer.get("risk_level")
    audit = answer.get("audit_result")
    if risk == "low" and audit == "pass":
        return "low_pass"
    if risk == "medium" or audit == "manual_review":
        return "medium_or_manual_review"
    return None


def _with_mix_metadata(row: dict[str, Any], *, role: str, bucket: str | None = None) -> dict[str, Any]:
    mixed = dict(row)
    metadata = dict(mixed.get("repair_mix_metadata") or {})
    metadata.update(
        {
            "mix": "phase08_high_risk_repair_sft_r1",
            "role": role,
        }
    )
    if bucket:
        metadata["bucket"] = bucket
    mixed["repair_mix_metadata"] = metadata
    return mixed


def build_mix(
    *,
    repo: Path,
    repair_pack_dir: Path,
    sft_train: Path,
    output_train: Path,
    output_manifest: Path,
    calibration_count: int,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    repair_rows = list(iter_jsonl(repair_pack_dir / "repair_pack_sft.jsonl"))
    repair_case_ids = {_case_id(row) for row in repair_rows}
    excluded = _load_excluded_case_ids(repo, repair_case_ids)
    all_excluded = set().union(*excluded.values())

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {
        "low_pass": [],
        "medium_or_manual_review": [],
    }
    inspected = 0
    for row in iter_jsonl(sft_train):
        inspected += 1
        case_id = _case_id(row)
        if case_id in all_excluded:
            continue
        bucket = _calibration_bucket(row)
        if bucket is None:
            continue
        buckets[bucket].append(row)

    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    low_quota = int(round(calibration_count * 2 / 3))
    medium_quota = calibration_count - low_quota
    selected_calibration = [
        *(_with_mix_metadata(row, role="calibration", bucket="low_pass") for row in buckets["low_pass"][:low_quota]),
        *(
            _with_mix_metadata(row, role="calibration", bucket="medium_or_manual_review")
            for row in buckets["medium_or_manual_review"][:medium_quota]
        ),
    ]
    if len(selected_calibration) < calibration_count:
        selected_ids = {_case_id(row) for row in selected_calibration}
        fallback = [
            (bucket, row)
            for bucket, bucket_rows in buckets.items()
            for row in bucket_rows
            if _case_id(row) not in selected_ids
        ]
        for bucket, row in fallback[: calibration_count - len(selected_calibration)]:
            selected_calibration.append(_with_mix_metadata(row, role="calibration", bucket=bucket))

    repair_mixed = [_with_mix_metadata(row, role="repair") for row in repair_rows]
    output_rows = [*repair_mixed, *selected_calibration]
    output_case_ids = {_case_id(row) for row in output_rows}
    overlap = {
        key: sorted(output_case_ids.intersection(ids))
        for key, ids in excluded.items()
        if key != "repair_pack"
    }
    manifest = {
        "name": "phase08_high_risk_repair_sft_r1_mix",
        "seed": seed,
        "repair_rows": len(repair_rows),
        "calibration_rows": len(selected_calibration),
        "total_rows": len(output_rows),
        "inspected_sft_rows": inspected,
        "calibration_bucket_counts": dict(Counter(row["repair_mix_metadata"]["bucket"] for row in selected_calibration)),
        "excluded_case_counts": {key: len(ids) for key, ids in excluded.items()},
        "overlap_counts": {key: len(ids) for key, ids in overlap.items()},
        "overlaps": overlap,
        "source_policy": "MV-Train only; repair pack plus low/pass and medium/manual-review calibration; no holdout/decode/sample500 overlap.",
        "output_train": str(output_train),
    }
    if not dry_run:
        write_jsonl(output_rows, output_train)
        ensure_dir(output_manifest.parent)
        output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--repair-pack-dir", type=Path, default=REPAIR_PACK_DIR)
    parser.add_argument("--sft-train", type=Path, default=Path("data/mv_audit/sft_main/train.jsonl"))
    parser.add_argument("--output-train", type=Path, default=REPAIR_PACK_DIR / "repair_sft_train_mix.jsonl")
    parser.add_argument("--output-manifest", type=Path, default=REPAIR_PACK_DIR / "repair_sft_train_mix_manifest.json")
    parser.add_argument("--calibration-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    manifest = build_mix(
        repo=repo,
        repair_pack_dir=_resolve(repo, args.repair_pack_dir),
        sft_train=_resolve(repo, args.sft_train),
        output_train=_resolve(repo, args.output_train),
        output_manifest=_resolve(repo, args.output_manifest),
        calibration_count=args.calibration_count,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
