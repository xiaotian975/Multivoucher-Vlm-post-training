"""Build and lock the final holdout manifest for Repair SFT v3."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from mv_audit.converters.common import build_audit_output, build_prompt, existing_image_items, group_records_by_case
from mv_audit.utils import ensure_dir, iter_jsonl, read_jsonl, read_yaml, write_jsonl


DEFAULT_SPLITS = ("test_clean", "test_robust", "test_unseen_template", "test_hard_negative")
DEFAULT_FINAL_HOLDOUT_ROOT = Path("data/mv_audit/final_holdout_v1")
DEFAULT_REGISTRY_PATH = Path("docs/experiments/data_boundary/used_case_registry.json")
DEFAULT_MODEL_SELECTION = Path("docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json")
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}
CASE_RE = re.compile(r"\bMV_MAIN_\d{6}\b")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_ids_sha256(case_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(case_id) for case_id in case_ids)) + "\n"
    return sha256(payload.encode("utf-8")).hexdigest()


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _read_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    if path.suffix.lower() == ".jsonl":
        for row in iter_jsonl(path):
            case_id = _case_id(row)
            if case_id:
                ids.add(case_id)
        return ids
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(CASE_RE.findall(text))


def _iter_text_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _err: None):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


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
        quota = min(len(group_rows), int(raw_quota))
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
        progressed = False
        for key in sorted(groups):
            if remaining <= 0:
                break
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: list[dict[str, Any]] = []
    for key, quota in quotas.items():
        candidates = list(groups[key])
        rng = random.Random(f"{seed}:{split}:{anomaly}:{key[0]}:{key[1]}")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    return selected


def stratified_manifest(rows: list[dict[str, Any]], *, split: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    by_anomaly: dict[str, list[dict[str, Any]]] = defaultdict(list)
    index_by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        index_by_id[case_id] = index
        by_anomaly[str(row.get("primary_anomaly_type"))].append(row)

    anomalies = sorted(by_anomaly)
    if split == "test_hard_negative":
        anomalies = [item for item in anomalies if item != "none"]
    if not anomalies:
        raise ValueError(f"No anomaly groups available for {split}.")

    base = sample_size // len(anomalies)
    remainder = sample_size % len(anomalies)
    selected: list[dict[str, Any]] = []
    for index, anomaly in enumerate(anomalies):
        quota = base + int(index < remainder)
        candidates = by_anomaly[anomaly]
        if len(candidates) < quota:
            raise ValueError(f"{split} has only {len(candidates)} rows for {anomaly}, need {quota}.")
        selected.extend(_stratified_pick(candidates, count=quota, seed=seed, split=split, anomaly=anomaly))

    selected_ids = [str(row["case_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"{split} final holdout contains duplicate case_id values.")
    if len(selected) != sample_size:
        raise ValueError(f"{split} final holdout has {len(selected)} rows, expected {sample_size}.")

    selected.sort(key=lambda row: index_by_id[str(row["case_id"])])
    return [
        {
            "sample_index": index,
            "split": split,
            "case_id": row["case_id"],
            "primary_anomaly_type": row.get("primary_anomaly_type"),
            "risk_level": row.get("risk_level"),
            "audit_result": row.get("audit_result"),
        }
        for index, row in enumerate(selected)
    ]


def _annotation_file(split: str) -> str:
    return f"field_bboxes_{split}.jsonl"


def _images_complete(records: list[dict[str, Any]], repo: Path) -> bool:
    image_paths = {str(record.get("image_path") or "") for record in records}
    if not image_paths:
        return False
    return all((repo / image_path).exists() for image_path in image_paths)


def collect_used_test_registry(
    *,
    repo: Path,
    test_cases_by_split: dict[str, list[dict[str, Any]]],
    seed: int,
    sample500_size: int,
) -> dict[str, Any]:
    test_universe = {str(row["case_id"]) for rows in test_cases_by_split.values() for row in rows}
    sources: dict[str, set[str]] = defaultdict(set)

    manifest_dir = repo / "data/mv_audit/eval_sets_phase07_sample500/manifests"
    for split, rows in test_cases_by_split.items():
        manifest_path = manifest_dir / f"{split}_case_ids.jsonl"
        if manifest_path.exists():
            source_ids = _read_case_ids(manifest_path)
            source_name = str(manifest_path.relative_to(repo))
        else:
            reconstructed = stratified_manifest(rows, split=split, sample_size=sample500_size, seed=seed)
            source_ids = {str(row["case_id"]) for row in reconstructed}
            source_name = f"phase07_sample500_reconstructed/{split}"
        sources[source_name].update(source_ids & test_universe)

    data_boundary_root = (repo / DEFAULT_REGISTRY_PATH).parent.resolve()
    for root in [repo / "docs/experiments", repo / "outputs/predictions", repo / "outputs/eval_reports"]:
        for path in _iter_text_files(root):
            try:
                path.resolve().relative_to(data_boundary_root)
                continue
            except ValueError:
                pass
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            ids = set(CASE_RE.findall(text)) & test_universe
            if ids:
                sources[str(path.relative_to(repo))].update(ids)

    used_ids = set().union(*sources.values()) if sources else set()
    return {
        "created_at": _now(),
        "policy": (
            "Collect historical test case ids from sample500 manifests or deterministic reconstruction, "
            "experiment docs, prediction files, and evaluation reports. Final holdout must exclude them."
        ),
        "test_universe_count": len(test_universe),
        "used_test_case_count": len(used_ids),
        "used_test_case_ids_sha256": case_ids_sha256(used_ids),
        "source_counts": {key: len(value) for key, value in sorted(sources.items())},
        "used_test_case_ids": sorted(used_ids),
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["split", "primary_anomaly_type", "risk_level", "audit_result"]
    payload: dict[str, dict[str, int]] = {}
    for field in fields:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row.get(field))] += 1
        payload[field] = dict(sorted(counts.items()))
    return {"distribution": payload}


def _eval_row(
    *,
    case: dict[str, Any],
    records: list[dict[str, Any]],
    split: str,
    rng: random.Random,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    output = build_audit_output(case, records)
    errors = sorted(validator.iter_errors(output), key=lambda err: err.path)
    if errors:
        raise ValueError(f"Ground truth schema invalid for {case['case_id']}: {errors[0].message}")
    image_items = existing_image_items(records, rng=rng)
    return {
        "case_id": case["case_id"],
        "split": split,
        "model_id": "ground_truth",
        "primary_anomaly_type": case.get("primary_anomaly_type"),
        "risk_level": case.get("risk_level"),
        "audit_result": case.get("audit_result"),
        "images": image_items,
        "prompt": build_prompt(case, image_items, task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。"),
        "answer": output,
        "source_split": split,
        "split_guard": {"case_level_split": "final_holdout_v1", "final_holdout": True},
    }


def _collect_overlap_ids(repo: Path) -> dict[str, set[str]]:
    paths = {
        "dpo_v2_holdout": [Path("data/mv_audit/dpo_v2/pairs_holdout.jsonl"), Path("docs/experiments/phase08_m3v2_sample500/dpo_v2/pair_report.json")],
        "dpo_v2_train_decode_dev": [
            Path("data/mv_audit/dpo_v2/train_decode_dev.jsonl"),
            Path("docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/dpo_v2_baseline/train_decode_dev.jsonl"),
        ],
        "repair_sft_train": [
            Path("docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl"),
            Path("docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl"),
        ],
        "model_mining": [
            Path("data/mv_audit/dpo_v3_model_mined/candidates.jsonl"),
            Path("data/mv_audit/dpo_v3_model_mined/pairs_train.jsonl"),
            Path("data/mv_audit/dpo_v3_model_mined/pairs_holdout.jsonl"),
        ],
    }
    collected: dict[str, set[str]] = {}
    for name, rel_paths in paths.items():
        ids: set[str] = set()
        for rel_path in rel_paths:
            path = repo / rel_path
            if path.exists():
                ids.update(_read_case_ids(path))
        collected[name] = ids
    return collected


def _write_model_lock(*, repo: Path, output_root: Path, model_selection_path: Path) -> dict[str, Any]:
    selection = json.loads((repo / model_selection_path).read_text(encoding="utf-8"))
    candidate = selection["production_candidate"]
    lock = {
        "model_id": candidate["model_id"],
        "role": candidate["role"],
        "adapter_remote_path": candidate.get("adapter_remote_path", ""),
        "adapter_local_path": candidate["adapter_local_path"],
        "adapter_sha256": candidate["adapter_sha256"],
        "adapter_archive_status": candidate["adapter_archive_status"],
        "selection_source": str(model_selection_path),
        "locked_at": _now(),
        "git_commit": _git_commit(repo),
    }
    lock_path = repo / output_root / "FINAL_MODEL_LOCK"
    lock_path.write_text("\n".join(f"{key}={value}" for key, value in lock.items()) + "\n", encoding="utf-8", newline="\n")
    _write_json(repo / output_root / "final_model_lock.json", lock)
    return lock


def build_final_holdout(
    *,
    repo: Path,
    raw_cases_dir: Path,
    annotations_dir: Path,
    output_root: Path,
    registry_path: Path,
    model_selection_path: Path,
    sample_size_per_split: int,
    sample500_size: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    lock_path = repo / output_root / "FINAL_HOLDOUT_LOCKED"
    consumed_path = repo / output_root / "FINAL_HOLDOUT_CONSUMED"
    if consumed_path.exists():
        raise FileExistsError(f"Final holdout was already consumed: {consumed_path}")
    if lock_path.exists() and not force:
        raise FileExistsError(f"Final holdout is already locked: {lock_path}. Use --force only before consumption.")

    test_cases_by_split = {
        split: read_jsonl(repo / raw_cases_dir / f"{split}_cases.jsonl")
        for split in DEFAULT_SPLITS
    }
    registry = collect_used_test_registry(repo=repo, test_cases_by_split=test_cases_by_split, seed=seed, sample500_size=sample500_size)
    _write_json(repo / registry_path, registry)

    used_ids = set(registry["used_test_case_ids"])
    validator = Draft202012Validator(read_yaml(repo / "configs/schema/output_schema.json"))
    rng = random.Random(seed)
    final_rows: list[dict[str, Any]] = []
    manifest_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}

    for split, rows in test_cases_by_split.items():
        records_by_case = group_records_by_case(read_jsonl(repo / annotations_dir / _annotation_file(split)))
        candidates = [
            row
            for row in rows
            if str(row["case_id"]) not in used_ids
            and _images_complete(records_by_case.get(str(row["case_id"]), []), repo)
        ]
        manifest = stratified_manifest(candidates, split=split, sample_size=sample_size_per_split, seed=seed)
        manifest_rows_by_split[split] = manifest
        split_counts[split] = len(manifest)
        cases_by_id = {str(row["case_id"]): row for row in rows}
        for manifest_row in manifest:
            case_id = str(manifest_row["case_id"])
            final_rows.append(
                _eval_row(
                    case=cases_by_id[case_id],
                    records=records_by_case[case_id],
                    split=split,
                    rng=rng,
                    validator=validator,
                )
            )

    final_ids = {str(row["case_id"]) for row in final_rows}
    if len(final_ids) != len(final_rows):
        raise ValueError("Final holdout contains duplicate case_id values.")

    output_dir = ensure_dir(repo / output_root)
    manifest_dir = ensure_dir(output_dir / "manifests")
    for split, rows in manifest_rows_by_split.items():
        write_jsonl(rows, manifest_dir / f"{split}_case_ids.jsonl")
    final_path = output_dir / "final_holdout_v1.jsonl"
    write_jsonl(final_rows, final_path)

    overlap_sources = _collect_overlap_ids(repo)
    overlap_counts = {name: len(final_ids & ids) for name, ids in overlap_sources.items()}
    overlap_counts["used_test"] = len(final_ids & used_ids)
    bad_overlaps = {name: count for name, count in overlap_counts.items() if count}
    if bad_overlaps:
        raise ValueError(f"Final holdout leakage overlaps are non-zero: {bad_overlaps}")

    model_lock = _write_model_lock(repo=repo, output_root=output_root, model_selection_path=model_selection_path)
    manifest_payload = {
        "name": "final_holdout_v1",
        "created_at": _now(),
        "seed": seed,
        "num_cases": len(final_rows),
        "target_per_split": sample_size_per_split,
        "split_counts": split_counts,
        "dataset_sha256": _sha256_file(final_path),
        "case_ids_sha256": case_ids_sha256(final_ids),
        "used_test_case_ids_sha256": registry["used_test_case_ids_sha256"],
        "overlap_counts": overlap_counts,
        "model_lock": model_lock,
        **_distribution(final_rows),
    }
    _write_json(output_dir / "final_holdout_v1_manifest.json", manifest_payload)
    lock_payload = {
        "dataset_sha256": manifest_payload["dataset_sha256"],
        "case_ids_sha256": manifest_payload["case_ids_sha256"],
        "creation_timestamp": manifest_payload["created_at"],
        "git_commit": _git_commit(repo),
        "num_cases": str(len(final_rows)),
        "target_per_split": str(sample_size_per_split),
    }
    lock_path.write_text(
        "\n".join(f"{key}={value}" for key, value in lock_payload.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"registry": registry, "manifest": manifest_payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and lock final_holdout_v1.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--raw_cases_dir", type=Path, default=Path("data/mv_audit/raw_cases/main"))
    parser.add_argument("--annotations_dir", type=Path, default=Path("data/mv_audit/annotations_main"))
    parser.add_argument("--output_root", type=Path, default=DEFAULT_FINAL_HOLDOUT_ROOT)
    parser.add_argument("--registry_path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--model_selection", type=Path, default=DEFAULT_MODEL_SELECTION)
    parser.add_argument("--sample_size_per_split", type=int, default=250)
    parser.add_argument("--sample500_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_final_holdout(
        repo=args.repo.resolve(),
        raw_cases_dir=args.raw_cases_dir,
        annotations_dir=args.annotations_dir,
        output_root=args.output_root,
        registry_path=args.registry_path,
        model_selection_path=args.model_selection,
        sample_size_per_split=args.sample_size_per_split,
        sample500_size=args.sample500_size,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps({"registry": result["registry"], "manifest": result["manifest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
