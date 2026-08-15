"""Prepare Phase A-G artifacts for the post-training v4 repair loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from mv_audit.converters.common import build_audit_output, build_prompt, existing_image_items, group_records_by_case
from mv_audit.utils import ensure_dir, iter_jsonl, read_jsonl, read_yaml, write_jsonl


SEED = 20260815
PHASE_DIR = Path("docs/experiments/phase09_repair_v4")
DATA_BOUNDARY_DIR = Path("docs/experiments/data_boundary")
DEV_DIR = Path("data/mv_audit/dev")
RAW_DIR = Path("data/mv_audit/raw_cases/main")
ANNOTATIONS_DIR = Path("data/mv_audit/annotations_main")
SAMPLE500_SPLITS = ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]
VALIDATION_SPLITS = ["val_in_template", "val_unseen_template"]
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
CASE_RE = re.compile(r"MV_MAIN_\d+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_ids_sha256(case_ids: Iterable[str]) -> str:
    return _sha256_text("\n".join(sorted(case_ids)) + "\n")


def _read_cases(split: str) -> list[dict[str, Any]]:
    return read_jsonl(RAW_DIR / f"{split}_cases.jsonl")


def _annotation_path(split: str) -> Path:
    return ANNOTATIONS_DIR / f"field_bboxes_{split}.jsonl"


def _load_records_by_case(split: str) -> dict[str, list[dict[str, Any]]]:
    return group_records_by_case(read_jsonl(_annotation_path(split)))


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id", "").split("_full_audit")[0])


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return answer
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        try:
            parsed = json.loads(str(messages[-1].get("content") or ""))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _split_key(case: dict[str, Any], image_count: int) -> tuple[str, str, str, str, int]:
    metadata = case.get("metadata") or {}
    anomalies = ",".join(sorted(str(item) for item in case.get("anomaly_types") or []))
    return (
        str(case.get("risk_level")),
        str(case.get("audit_result")),
        anomalies,
        str(metadata.get("template_group") or metadata.get("template") or "unknown"),
        image_count,
    )


def _image_status(case: dict[str, Any], records: list[dict[str, Any]], repo: Path) -> dict[str, Any]:
    image_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        image_by_id.setdefault(str(record["image_id"]), record)
    doc_types = {str(record.get("doc_type")) for record in image_by_id.values()}
    expected_docs = set(str(doc) for doc in case.get("documents") or [])
    missing_doc_types = sorted(expected_docs - doc_types)
    missing_paths = []
    for record in image_by_id.values():
        image_path = repo / str(record.get("image_path") or "")
        if not image_path.exists():
            missing_paths.append(str(record.get("image_path")))
    return {
        "complete": not missing_doc_types and not missing_paths and bool(image_by_id),
        "image_count": len(image_by_id),
        "missing_doc_types": missing_doc_types,
        "missing_paths": missing_paths,
    }


def _build_eval_row(case: dict[str, Any], records: list[dict[str, Any]], split: str, rng: random.Random) -> dict[str, Any]:
    image_items = existing_image_items(records, rng=rng)
    output = build_audit_output(case, records)
    prompt = build_prompt(
        case,
        image_items,
        task_instruction="完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。",
    )
    return {
        "case_id": case["case_id"],
        "split": split,
        "images": image_items,
        "prompt": prompt,
        "answer": output,
        "source_split": split,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {
        "source_split": Counter(),
        "risk_level": Counter(),
        "audit_result": Counter(),
        "anomaly_types": Counter(),
        "template": Counter(),
        "number_of_images": Counter(),
    }
    for row in rows:
        answer = _answer(row)
        counters["source_split"][str(row.get("source_split") or row.get("split") or "unknown")] += 1
        counters["risk_level"][str(answer.get("risk_level") or row.get("risk_level") or "unknown")] += 1
        counters["audit_result"][str(answer.get("audit_result") or row.get("audit_result") or "unknown")] += 1
        anomalies = answer.get("anomaly_types") or row.get("anomaly_types") or []
        counters["anomaly_types"][",".join(sorted(str(item) for item in anomalies)) or "none"] += 1
        metadata = row.get("metadata") or {}
        counters["template"][str(metadata.get("template_group") or metadata.get("template") or "unknown")] += 1
        counters["number_of_images"][str(len(row.get("images") or row.get("documents") or []))] += 1
    return {key: dict(counter.most_common()) for key, counter in counters.items()}


def _write_manifest(path: Path, *, dataset_path: Path, name: str, rows: list[dict[str, Any]], seed: int, source_splits: dict[str, int]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "name": name,
        "created_at": _now(),
        "seed": seed,
        "num_cases": len(rows),
        "case_ids_sha256": _case_ids_sha256(row["case_id"] for row in rows),
        "dataset_sha256": _sha256_file(dataset_path) if dataset_path.exists() else None,
        "source_splits": source_splits,
        **_distribution(rows),
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp_path.replace(path)


def repository_audit(repo: Path) -> dict[str, Any]:
    phase_dir = repo / PHASE_DIR
    ensure_dir(phase_dir)
    m2_adapter = Path("outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1")
    findings = {
        "m2_checkpoint": str(m2_adapter),
        "m2_checkpoint_exists": (repo / m2_adapter).exists(),
        "adapter_loading": "src/mv_audit/inference/batch_inference.py::_load_model_for_inference uses PeftModel.from_pretrained(model, adapter_dir); train_dpo.py::_load_sft_policy uses is_trainable=True for continued adapter training.",
        "prompt": "src/mv_audit/converters/common.py::build_prompt; batch_inference.build_eval_rows passes task_instruction='完成多凭证一致性审核，输出完整 Evidence-Grounded JSON。'",
        "generation_config": "src/mv_audit/inference/qwen3vl_common.py::generate_text reads max_new_tokens/temperature/top_p from config inference section; config examples pin do_sample through temperature=0.0.",
        "high_risk_miss": "src/mv_audit/evaluation/audit_metrics.py::high_risk_miss_rate and src/mv_audit/evaluation/case_scorer.py::CaseScore.high_risk_miss.",
        "evidence": "src/mv_audit/evaluation/evidence_metrics.py::evidence_counts.",
        "hallucination": "src/mv_audit/evaluation/hallucination_metrics.py::hallucination_count.",
        "false_manual_review": "src/mv_audit/evaluation/audit_metrics.py::false_manual_review_rate.",
        "repair_data": "docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl.",
        "train_decode_dev": "docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/dpo_v2_baseline/train_decode_dev.jsonl.",
        "conflicts": [],
    }
    if not findings["m2_checkpoint_exists"]:
        findings["conflicts"].append("LOCAL_M2_ADAPTER_MISSING: M2 adapter is referenced by config/README but not present in local outputs/checkpoints.")
    lines = [
        "# Phase09 Repository Audit",
        "",
        f"- generated_at: `{_now()}`",
        f"- M2 checkpoint: `{findings['m2_checkpoint']}`",
        f"- M2 checkpoint exists locally: `{findings['m2_checkpoint_exists']}`",
        "",
        "## Implementation Map",
        "",
    ]
    for key in [
        "adapter_loading",
        "prompt",
        "generation_config",
        "high_risk_miss",
        "evidence",
        "hallucination",
        "false_manual_review",
        "repair_data",
        "train_decode_dev",
    ]:
        lines.append(f"- **{key}**: {findings[key]}")
    lines.append("")
    lines.append("## Conflicts")
    lines.extend(f"- {item}" for item in findings["conflicts"]) if findings["conflicts"] else lines.append("- none")
    (phase_dir / "repository_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return findings


def env_snapshot(repo: Path) -> None:
    commands = [
        [sys.executable, "--version"],
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        [sys.executable, "-c", "import peft; print(peft.__version__)"],
        [sys.executable, "-c", "import trl; print(trl.__version__)"],
        [sys.executable, "-c", "import accelerate; print(accelerate.__version__)"],
        ["nvidia-smi"],
    ]
    lines = [f"generated_at={_now()}", ""]
    for command in commands:
        lines.append("$ " + " ".join(command))
        try:
            result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=20, check=False)
        except Exception as exc:  # noqa: BLE001 - snapshot should record failures, not abort.
            lines.append(f"ERROR: {exc}")
            lines.append("")
            continue
        if result.stdout.strip():
            lines.append(result.stdout.rstrip())
        if result.stderr.strip():
            lines.append("[stderr]")
            lines.append(result.stderr.rstrip())
        lines.append(f"exit_code={result.returncode}")
        lines.append("")
    ensure_dir(repo / PHASE_DIR)
    (repo / PHASE_DIR / "env_snapshot.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def validation_inventory_and_splits(repo: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {"created_at": _now(), "splits": {}, "existing_sft_val_cases": 0}
    sft_val = repo / "data/mv_audit/sft_main/val_existing_images.jsonl"
    if sft_val.exists():
        inventory["existing_sft_val_cases"] = sum(1 for _ in iter_jsonl(sft_val))
    complete_by_split: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], int]]] = {}
    for split in VALIDATION_SPLITS:
        cases = _read_cases(split)
        records_by_case = _load_records_by_case(split)
        complete: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
        missing_images = 0
        missing_annotations = 0
        for case in cases:
            records = records_by_case.get(case["case_id"], [])
            if not records:
                missing_annotations += 1
            status = _image_status(case, records, repo)
            if status["complete"]:
                complete.append((case, records, int(status["image_count"])))
            else:
                missing_images += 1
        complete_by_split[split] = complete
        inventory["splits"][split] = {
            "total_cases": len(cases),
            "complete_images": len(complete),
            "missing_images": missing_images,
            "missing_annotations": missing_annotations,
        }

    total_complete = sum(len(items) for items in complete_by_split.values())
    if total_complete >= 1600:
        target = 800
    elif total_complete >= 1000:
        target = 500
    elif total_complete < 600:
        target = 0
        inventory["status"] = "INSUFFICIENT_VALIDATION_DATA"
    else:
        target = total_complete // 2
    inventory["target_per_dev"] = target
    _write_json(repo / PHASE_DIR / "validation_inventory.json", inventory)
    if target <= 0:
        return {"inventory": inventory, "repair_rows": [], "rl_rows": []}

    quotas = {
        "val_in_template": min(len(complete_by_split["val_in_template"]) // 2, round(target * 2 / 3)),
        "val_unseen_template": min(len(complete_by_split["val_unseen_template"]) // 2, target - round(target * 2 / 3)),
    }
    while sum(quotas.values()) < target:
        for split in VALIDATION_SPLITS:
            if quotas[split] < len(complete_by_split[split]) // 2:
                quotas[split] += 1
                break
        else:
            break

    repair_items: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    rl_items: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    for split, quota in quotas.items():
        grouped: dict[tuple[str, str, str, str, int], list[tuple[dict[str, Any], list[dict[str, Any]], int]]] = defaultdict(list)
        for item in complete_by_split[split]:
            grouped[_split_key(item[0], item[2])].append(item)
        ordered_groups = sorted(grouped)
        rng = random.Random(f"{SEED}:{split}")
        for rows in grouped.values():
            rng.shuffle(rows)
        repair_split: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
        rl_split: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
        progress = True
        while progress and (len(repair_split) < quota or len(rl_split) < quota):
            progress = False
            for key in ordered_groups:
                rows = grouped[key]
                if rows and len(repair_split) < quota:
                    repair_split.append(rows.pop())
                    progress = True
                if rows and len(rl_split) < quota:
                    rl_split.append(rows.pop())
                    progress = True
        repair_items.extend((case, records, split) for case, records, _ in repair_split[:quota])
        rl_items.extend((case, records, split) for case, records, _ in rl_split[:quota])

    rng = random.Random(SEED)
    repair_rows = [_build_eval_row(case, records, split, rng) for case, records, split in repair_items]
    rl_rows = [_build_eval_row(case, records, split, rng) for case, records, split in rl_items]
    ensure_dir(repo / DEV_DIR)
    write_jsonl(repair_rows, repo / DEV_DIR / "repair_dev_v1.jsonl")
    write_jsonl(rl_rows, repo / DEV_DIR / "rl_dev_v1.jsonl")
    _write_manifest(
        repo / DEV_DIR / "repair_dev_v1_manifest.json",
        name="repair_dev_v1",
        rows=repair_rows,
        seed=SEED,
        source_splits=dict(Counter(row["source_split"] for row in repair_rows)),
    )
    _write_manifest(
        repo / DEV_DIR / "rl_dev_v1_manifest.json",
        name="rl_dev_v1",
        rows=rl_rows,
        seed=SEED,
        source_splits=dict(Counter(row["source_split"] for row in rl_rows)),
    )
    return {"inventory": inventory, "repair_rows": repair_rows, "rl_rows": rl_rows}


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
    selected: list[dict[str, Any]] = []
    for key, quota in quotas.items():
        candidates = list(groups[key])
        rng = random.Random(f"{seed}:{split}:{anomaly}:{key[0]}:{key[1]}")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    return selected


def _sample500_manifest(rows: list[dict[str, Any]], *, split: str, sample_size: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    by_anomaly: dict[str, list[dict[str, Any]]] = defaultdict(list)
    index_by_id = {str(row["case_id"]): index for index, row in enumerate(rows)}
    for row in rows:
        by_anomaly[str(row.get("primary_anomaly_type"))].append(row)
    anomalies = sorted(by_anomaly)
    if split == "test_hard_negative":
        anomalies = [item for item in anomalies if item != "none"]
    base = sample_size // len(anomalies)
    remainder = sample_size % len(anomalies)
    selected: list[dict[str, Any]] = []
    for index, anomaly in enumerate(anomalies):
        quota = base + int(index < remainder)
        selected.extend(_stratified_pick(by_anomaly[anomaly], count=quota, seed=seed, split=split, anomaly=anomaly))
    selected.sort(key=lambda row: index_by_id[str(row["case_id"])])
    return selected


def used_case_registry_and_final_holdout(repo: Path) -> dict[str, Any]:
    test_cases = {split: _read_cases(split) for split in SAMPLE500_SPLITS}
    test_universe = {case["case_id"] for rows in test_cases.values() for case in rows}
    sources: dict[str, set[str]] = defaultdict(set)
    for split, rows in test_cases.items():
        reconstructed = _sample500_manifest(rows, split=split)
        sources[f"phase07_sample500_reconstructed/{split}"].update(str(row["case_id"]) for row in reconstructed)
    for root in [repo / "docs/experiments", repo / "outputs/predictions", repo / "outputs/eval_reports"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            ids = set(CASE_RE.findall(text)).intersection(test_universe)
            if ids:
                sources[str(path.relative_to(repo))].update(ids)
    used_ids = set().union(*sources.values()) if sources else set()
    registry = {
        "created_at": _now(),
        "policy": "Collect historical test case ids from experiment text artifacts and reconstruct Phase07 sample500 ids from scripts/build_phase07_sample_manifest.py logic.",
        "used_test_case_count": len(used_ids),
        "used_test_case_ids_sha256": _case_ids_sha256(used_ids),
        "source_counts": {key: len(value) for key, value in sorted(sources.items())},
        "used_test_case_ids": sorted(used_ids),
    }
    _write_json(repo / DATA_BOUNDARY_DIR / "used_case_registry.json", registry)

    final_rows: list[dict[str, Any]] = []
    rng = random.Random(SEED)
    final_counts: dict[str, int] = {}
    for split, rows in test_cases.items():
        records_by_case = _load_records_by_case(split)
        candidates = [row for row in rows if row["case_id"] not in used_ids]
        grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            records = records_by_case.get(row["case_id"], [])
            status = _image_status(row, records, repo)
            if status["complete"]:
                grouped[_split_key(row, int(status["image_count"]))].append(row)
        for group_rows in grouped.values():
            rng.shuffle(group_rows)
        selected: list[dict[str, Any]] = []
        for key in sorted(grouped):
            if len(selected) >= 250:
                break
            if grouped[key]:
                selected.append(grouped[key].pop())
        while len(selected) < 250:
            added = False
            for key in sorted(grouped):
                if len(selected) >= 250:
                    break
                if grouped[key]:
                    selected.append(grouped[key].pop())
                    added = True
            if not added:
                break
        final_counts[split] = len(selected)
        final_rows.extend(_build_eval_row(row, records_by_case[row["case_id"]], split, rng) for row in selected)
    ensure_dir(repo / DEV_DIR)
    final_path = repo / DEV_DIR / "final_holdout_v1.jsonl"
    write_jsonl(final_rows, final_path)
    final_manifest = {
        "name": "final_holdout_v1",
        "created_at": _now(),
        "seed": SEED,
        "num_cases": len(final_rows),
        "target_per_split": 250,
        "split_counts": final_counts,
        "dataset_sha256": _sha256_file(final_path),
        "case_ids_sha256": _case_ids_sha256(row["case_id"] for row in final_rows),
        "used_test_case_ids_sha256": registry["used_test_case_ids_sha256"],
        **_distribution(final_rows),
    }
    _write_json(repo / DEV_DIR / "final_holdout_v1_manifest.json", final_manifest)
    lock = {
        "dataset_sha256": final_manifest["dataset_sha256"],
        "case_ids_sha256": final_manifest["case_ids_sha256"],
        "creation_timestamp": final_manifest["created_at"],
        "git_commit": _git(repo, "rev-parse", "HEAD"),
    }
    (repo / DEV_DIR / "FINAL_HOLDOUT_LOCKED").write_text(
        "\n".join(f"{key}={value}" for key, value in lock.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"registry": registry, "final_rows": final_rows, "final_manifest": final_manifest}


def repair_data_audit(repo: Path, *, repair_dev_ids: set[str], rl_dev_ids: set[str], final_ids: set[str], used_test_ids: set[str]) -> dict[str, Any]:
    path = repo / "docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl"
    rows = read_jsonl(path)
    validator = Draft202012Validator(read_yaml(repo / "configs/schema/output_schema.json"))
    ids = [_case_id(row) for row in rows]
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    missing_images: list[dict[str, str]] = []
    invalid_json: list[str] = []
    schema_invalid: list[dict[str, str]] = []
    role_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        role = str((row.get("repair_mix_metadata") or {}).get("role") or "unknown")
        role_rows[role].append(row)
        answer = _answer(row)
        if not answer:
            invalid_json.append(_case_id(row))
        else:
            errors = sorted(validator.iter_errors(answer), key=lambda err: err.path)
            if errors:
                err = errors[0]
                schema_invalid.append({"case_id": _case_id(row), "error": err.message})
        for image in row.get("images") or []:
            image_path = repo / str(image.get("image_path") or "")
            if not image_path.exists():
                missing_images.append({"case_id": _case_id(row), "image_path": str(image.get("image_path"))})
    train_case_ids = {_case_id(row) for row in read_jsonl(repo / RAW_DIR / "train_cases.jsonl")}
    test_case_ids = set()
    for split in SAMPLE500_SPLITS:
        test_case_ids.update(_case_id(row) for row in _read_cases(split))
    overlap = {
        "repair_dev": sorted(set(ids).intersection(repair_dev_ids)),
        "rl_dev": sorted(set(ids).intersection(rl_dev_ids)),
        "final_holdout": sorted(set(ids).intersection(final_ids)),
        "used_test": sorted(set(ids).intersection(used_test_ids)),
        "all_test": sorted(set(ids).intersection(test_case_ids)),
        "not_train": sorted(set(ids) - train_case_ids),
    }
    report = {
        "created_at": _now(),
        "path": str(path.relative_to(repo)),
        "total_rows": len(rows),
        "unique_case_ids": len(set(ids)),
        "duplicate_case_ids": duplicate_ids,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:50],
        "invalid_json_answers": invalid_json,
        "schema_invalid_answers": schema_invalid[:50],
        "overlap_counts": {key: len(value) for key, value in overlap.items()},
        "overlaps": overlap,
        "roles": {},
    }
    for role, selected in sorted(role_rows.items()):
        report["roles"][role] = {"rows": len(selected), **_distribution(selected)}
    _write_json(repo / PHASE_DIR / "repair_distribution_report.json", report)
    lines = ["# Repair SFT R1 Data Audit", "", f"- total_rows: {len(rows)}", f"- unique_case_ids: {len(set(ids))}"]
    lines.append(f"- missing_image_count: {len(missing_images)}")
    lines.append(f"- duplicate_case_ids: {len(duplicate_ids)}")
    lines.append(f"- invalid_json_answers: {len(invalid_json)}")
    lines.append(f"- schema_invalid_answers: {len(schema_invalid)}")
    lines.append("")
    lines.append("## Overlap Counts")
    for key, value in report["overlap_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Role Distribution")
    for role, payload in report["roles"].items():
        lines.append(f"### {role}")
        lines.append(f"- rows: {payload['rows']}")
        lines.append(f"- risk_level: `{json.dumps(payload['risk_level'], ensure_ascii=False)}`")
        lines.append(f"- audit_result: `{json.dumps(payload['audit_result'], ensure_ascii=False)}`")
        lines.append(f"- anomaly_types: `{json.dumps(payload['anomaly_types'], ensure_ascii=False)}`")
    (repo / PHASE_DIR / "repair_distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def repair_preflight(repo: Path, repair_report: dict[str, Any]) -> dict[str, Any]:
    sft_cfg = read_yaml(repo / "configs/train/sft_lora_qwen3vl_8b_phase07_sample500_server.yaml")
    repair_cfg = read_yaml(repo / "configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml")
    m2_adapter = repo / str(sft_cfg["training"]["output_dir"])
    output_dir = repo / str(repair_cfg["training"]["output_dir"])
    train_file = repo / str(repair_cfg["data"]["train_file"])
    rows = read_jsonl(train_file) if train_file.exists() else []
    training = repair_cfg["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    batch = int(training.get("per_device_train_batch_size", 1))
    grad_acc = int(training.get("gradient_accumulation_steps", 1))
    epochs = float(training.get("num_train_epochs", 1))
    updates = math.ceil(len(rows) / max(1, world_size * batch * grad_acc)) * epochs if rows else 0
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    m2_files = {name: (m2_adapter / name).exists() for name in adapter_files}
    hard_failures = []
    if not m2_adapter.exists() or not all(m2_files.values()):
        hard_failures.append("M2_ADAPTER_MISSING")
    if output_dir == m2_adapter:
        hard_failures.append("OUTPUT_DIR_WOULD_OVERWRITE_M2")
    if repair_report["missing_image_count"] > 0:
        hard_failures.append("REPAIR_DATA_MISSING_IMAGES")
    if repair_report["duplicate_case_ids"]:
        hard_failures.append("REPAIR_DATA_DUPLICATE_IDS")
    if repair_report["invalid_json_answers"] or repair_report["schema_invalid_answers"]:
        hard_failures.append("REPAIR_DATA_INVALID_JSON_OR_SCHEMA")
    if any(value > 0 for key, value in repair_report["overlap_counts"].items() if key != "not_train"):
        hard_failures.append("REPAIR_DATA_LEAKAGE")
    if repair_report["overlap_counts"].get("not_train", 0) > 0:
        hard_failures.append("REPAIR_DATA_NOT_TRAIN_ONLY")
    payload = {
        "created_at": _now(),
        "status": "READY_FOR_REPAIR_R1" if not hard_failures else "NOT_READY_FOR_REPAIR_R1",
        "hard_failures": hard_failures,
        "m2_adapter": str(m2_adapter.relative_to(repo)),
        "m2_adapter_exists": m2_adapter.exists(),
        "m2_adapter_files": m2_files,
        "m2_adapter_sha256": _sha256_file(m2_adapter / "adapter_model.safetensors") if (m2_adapter / "adapter_model.safetensors").exists() else None,
        "trainable_load_semantics": "Use PeftModel.from_pretrained(base_model, M2_ADAPTER_PATH, is_trainable=True); local full load was not attempted in preparation-only mode.",
        "repair_output_dir": str(output_dir.relative_to(repo)),
        "output_dir_exists": output_dir.exists(),
        "will_overwrite_m2": output_dir == m2_adapter,
        "optimizer_updates": {
            "rows": len(rows),
            "world_size": world_size,
            "per_device_train_batch_size": batch,
            "gradient_accumulation_steps": grad_acc,
            "epochs": epochs,
            "estimated_optimizer_updates": updates,
            "target_range": "30-60",
            "in_target_range": 30 <= updates <= 60,
        },
        "commands": {
            "m2_baseline_repair_dev": "python -m mv_audit.inference.batch_inference --config configs/eval/audit_eval_frozen_v1.yaml --model_id m2_sft --split repair_dev_v1 --dry_run",
            "repair_r1_dry_run": "DRY_RUN=1 MAX_SAMPLES=16 CONFIG=configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml bash scripts/04_train_sft.sh",
            "repair_r1_formal": "ALLOW_TRAINING=1 CONFIG=configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml bash scripts/11_run_high_risk_repair_sft_r1_server.sh",
        },
    }
    _write_json(repo / PHASE_DIR / "repair_r1_preflight.json", payload)
    return payload


def overlaps(repair_rows: list[dict[str, Any]], rl_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], repair_data_ids: set[str], used_ids: set[str]) -> dict[str, int]:
    repair_dev_ids = {row["case_id"] for row in repair_rows}
    rl_dev_ids = {row["case_id"] for row in rl_rows}
    final_ids = {row["case_id"] for row in final_rows}
    return {
        "repair_dev_vs_rl_dev": len(repair_dev_ids.intersection(rl_dev_ids)),
        "repair_train_vs_repair_dev": len(repair_data_ids.intersection(repair_dev_ids)),
        "repair_train_vs_rl_dev": len(repair_data_ids.intersection(rl_dev_ids)),
        "repair_train_vs_final_holdout": len(repair_data_ids.intersection(final_ids)),
        "used_test_vs_final_holdout": len(used_ids.intersection(final_ids)),
        "repair_dev_vs_final_holdout": len(repair_dev_ids.intersection(final_ids)),
        "rl_dev_vs_final_holdout": len(rl_dev_ids.intersection(final_ids)),
    }


def write_state_and_manifest(repo: Path, preflight: dict[str, Any], overlap_counts: dict[str, int], output_files: list[str]) -> None:
    state = {
        "phase": "REPAIR_PREP",
        "status": preflight["status"],
        "last_completed_step": "REPAIR_R1_PREFLIGHT",
        "next_allowed_step": "RUN_M2_REPAIR_DEV" if preflight["status"] == "READY_FOR_REPAIR_R1" else "FIX_PREFLIGHT_FAILURES",
        "final_holdout_locked": True,
        "final_holdout_consumed": False,
    }
    _write_json(repo / "outputs/runtime/post_training_v4/state.json", state)
    manifest = {
        "run_id": "phase09_repair_v4_prepare",
        "phase": "Phase A-G Preparation",
        "start_time": None,
        "end_time": _now(),
        "git_commit_before": _git(repo, "rev-parse", "HEAD"),
        "git_commit_after": _git(repo, "rev-parse", "HEAD"),
        "input_files": [
            "README.md",
            "docs/code_inventory.md",
            "docs/post_training_v4_requirements.md",
            "configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml",
        ],
        "output_files": output_files,
        "dataset_hash": _sha256_file(repo / DEV_DIR / "repair_dev_v1.jsonl") if (repo / DEV_DIR / "repair_dev_v1.jsonl").exists() else "",
        "config_hash": _sha256_file(repo / "configs/analysis/error_attribution_v1.yaml"),
        "checkpoint_hash": preflight.get("m2_adapter_sha256") or "",
        "commands": ["python -m mv_audit.analysis.post_training_v4_prepare"],
        "dry_run": True,
        "success": True,
        "gate_result": preflight["status"],
        "failure_reason": ";".join(preflight["hard_failures"]),
        "overlap_counts": overlap_counts,
    }
    _write_json(repo / PHASE_DIR / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    ensure_dir(repo / PHASE_DIR)
    ensure_dir(repo / DATA_BOUNDARY_DIR)

    audit = repository_audit(repo)
    env_snapshot(repo)
    split_payload = validation_inventory_and_splits(repo)
    used_payload = used_case_registry_and_final_holdout(repo)
    repair_rows = split_payload["repair_rows"]
    rl_rows = split_payload["rl_rows"]
    final_rows = used_payload["final_rows"]
    used_ids = set(used_payload["registry"]["used_test_case_ids"])
    repair_data_ids = {_case_id(row) for row in read_jsonl(repo / "docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl")}
    repair_report = repair_data_audit(
        repo,
        repair_dev_ids={row["case_id"] for row in repair_rows},
        rl_dev_ids={row["case_id"] for row in rl_rows},
        final_ids={row["case_id"] for row in final_rows},
        used_test_ids=used_ids,
    )
    preflight = repair_preflight(repo, repair_report)
    overlap_counts = overlaps(repair_rows, rl_rows, final_rows, repair_data_ids, used_ids)
    _write_json(repo / PHASE_DIR / "leakage_report.json", {"created_at": _now(), "overlap_counts": overlap_counts})
    output_files = [
        "docs/experiments/phase09_repair_v4/repository_audit.md",
        "docs/experiments/phase09_repair_v4/env_snapshot.txt",
        "docs/experiments/phase09_repair_v4/validation_inventory.json",
        "data/mv_audit/dev/repair_dev_v1.jsonl",
        "data/mv_audit/dev/repair_dev_v1_manifest.json",
        "data/mv_audit/dev/rl_dev_v1.jsonl",
        "data/mv_audit/dev/rl_dev_v1_manifest.json",
        "docs/experiments/data_boundary/used_case_registry.json",
        "data/mv_audit/dev/final_holdout_v1.jsonl",
        "data/mv_audit/dev/final_holdout_v1_manifest.json",
        "data/mv_audit/dev/FINAL_HOLDOUT_LOCKED",
        "docs/experiments/phase09_repair_v4/repair_distribution_report.json",
        "docs/experiments/phase09_repair_v4/repair_distribution_report.md",
        "docs/experiments/phase09_repair_v4/repair_r1_preflight.json",
        "docs/experiments/phase09_repair_v4/leakage_report.json",
        "docs/experiments/phase09_repair_v4/run_manifest.json",
        "outputs/runtime/post_training_v4/state.json",
    ]
    write_state_and_manifest(repo, preflight, overlap_counts, output_files)
    print(
        json.dumps(
            {
                "repository_conflicts": audit["conflicts"],
                "repair_dev_v1": len(repair_rows),
                "rl_dev_v1": len(rl_rows),
                "final_holdout_v1": len(final_rows),
                "overlap_counts": overlap_counts,
                "preflight_status": preflight["status"],
                "hard_failures": preflight["hard_failures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
