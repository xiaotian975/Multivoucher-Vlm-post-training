"""Build Order-ID Structured Repair SFT v3 mix and config.

This is intentionally Train-only: it excludes DPO holdout/decode-dev and
sample500/Test cases. The v3 target keeps the public output schema unchanged,
but rewrites order_id_mismatch answers so the two order-id evidence items are
first and the reason explicitly compares order document vs reimbursement form.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mv_audit.utils import read_yaml, write_yaml


SEED = 44
ORDER_ID_REPAIR_COUNT = 120
CALIBRATION_COUNT = 120
CARRYOVER_LIMIT = 240
PACK_DIR = Path("docs/experiments/phase09_order_id_structured_repair_v3")
R1_MIX = Path("docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl")
SFT_TRAIN = Path("data/mv_audit/sft_main/train.jsonl")
BASE_CONFIG = Path("configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml")
OUTPUT_MIX = PACK_DIR / "repair_sft_v3_order_id_structured_mix.jsonl"
OUTPUT_MANIFEST = PACK_DIR / "repair_sft_v3_order_id_structured_mix_manifest.json"
OUTPUT_CONFIG = Path("configs/train/high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml")

def repo_path(path: Path | str) -> str:
    """Use Linux-friendly repo-relative paths in generated server configs."""
    return Path(path).as_posix()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or str(row.get("id", "")).split("_", 1)[0])


def answer(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    messages = row.get("messages") or []
    if messages:
        content = messages[-1].get("content")
        if isinstance(content, str):
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
    return {}


def with_answer(row: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated["answer"] = output
    messages = list(updated.get("messages") or [])
    if messages:
        messages[-1] = dict(messages[-1])
        messages[-1]["content"] = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        updated["messages"] = messages
    return updated


def images_exist(row: dict[str, Any]) -> bool:
    return all(Path(str(item.get("image_path", ""))).exists() for item in row.get("images") or [])


def excluded_case_ids() -> set[str]:
    excluded: set[str] = set()
    for path in [
        Path("data/mv_audit/dpo_v2/pairs_holdout.jsonl"),
        Path("data/mv_audit/dpo_v2/train_decode_dev.jsonl"),
        Path("docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/dpo_v2_baseline/train_decode_dev.jsonl"),
        Path("docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/auxdpo_v2_strong/train_decode_dev.jsonl"),
    ]:
        excluded.update(case_id(row) for row in iter_jsonl(path))
    for split in ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]:
        excluded.update(case_id(row) for row in iter_jsonl(Path(f"data/mv_audit/raw_cases/main/{split}_cases.jsonl")))
    return excluded


def role_row(row: dict[str, Any], *, role: str, bucket: str | None = None) -> dict[str, Any]:
    out = dict(row)
    meta = dict(out.get("repair_mix_metadata") or {})
    meta.update({"mix": "phase09_repair_sft_v3_order_id_structured", "role": role})
    if bucket:
        meta["bucket"] = bucket
    out["repair_mix_metadata"] = meta
    return out


def order_id_evidence(output: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    order_item = None
    reimbursement_item = None
    for item in output.get("evidence") or []:
        if not isinstance(item, dict) or item.get("field") != "order_id":
            continue
        if item.get("source_doc_type") == "order" and order_item is None:
            order_item = copy.deepcopy(item)
        elif item.get("source_doc_type") == "reimbursement_form" and reimbursement_item is None:
            reimbursement_item = copy.deepcopy(item)
    return order_item, reimbursement_item


def structured_order_id_output(output: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    structured = copy.deepcopy(output)
    order_item, reimbursement_item = order_id_evidence(structured)
    if not order_item or not reimbursement_item:
        return structured, False

    order_value = str(order_item.get("value") or "")
    reimbursement_value = str(reimbursement_item.get("value") or "")
    if not order_value or not reimbursement_value or order_value == reimbursement_value:
        return structured, False

    consistency = dict(structured.get("consistency_check") or {})
    consistency["order_id_consistent"] = False
    structured["consistency_check"] = consistency
    anomalies = list(structured.get("anomaly_types") or [])
    if "order_id_mismatch" not in anomalies:
        anomalies.append("order_id_mismatch")
    structured["anomaly_types"] = anomalies
    structured["risk_level"] = "high"
    structured["audit_result"] = "reject_recommendation"
    structured["reason"] = (
        "检测到order_id_mismatch：订单截图订单号"
        f"{order_value}与报销申请单订单号{reimbursement_value}不一致，"
        "风险等级为high，审核建议为reject_recommendation。"
    )

    remaining = [
        copy.deepcopy(item)
        for item in structured.get("evidence") or []
        if not (
            isinstance(item, dict)
            and item.get("field") == "order_id"
            and item.get("source_doc_type") in {"order", "reimbursement_form"}
        )
    ]
    structured["evidence"] = [order_item, reimbursement_item, *remaining]
    return structured, True


def calibration_bucket(row: dict[str, Any]) -> str | None:
    output = answer(row)
    if row.get("source_split") != "MV-Train":
        return None
    if output.get("risk_level") == "low" and output.get("audit_result") == "pass":
        return "low_pass"
    if output.get("risk_level") == "medium" or output.get("audit_result") in {"manual_review", "missing_info"}:
        return "medium_or_manual_review"
    return None


def build() -> dict[str, Any]:
    rng = random.Random(SEED)
    excluded = excluded_case_ids()
    selected: set[str] = set()

    carryover_rows = []
    carryover_dropped = 0
    for row in iter_jsonl(R1_MIX):
        cid = case_id(row)
        if cid in excluded or cid in selected or not images_exist(row):
            carryover_dropped += 1
            continue
        carryover_rows.append(role_row(row, role="r1_carryover"))
        selected.add(cid)
        if len(carryover_rows) >= CARRYOVER_LIMIT:
            break

    order_candidates = []
    carryover_fill_candidates = []
    calibration = {"low_pass": [], "medium_or_manual_review": []}
    missing_order_id_pair = 0
    for row in iter_jsonl(SFT_TRAIN):
        cid = case_id(row)
        if cid in excluded or cid in selected or not images_exist(row):
            continue
        output = answer(row)
        if (
            row.get("source_split") == "MV-Train"
            and "order_id_mismatch" in set(output.get("anomaly_types") or [])
            and output.get("risk_level") == "high"
            and output.get("audit_result") == "reject_recommendation"
        ):
            structured_output, changed = structured_order_id_output(output)
            if not changed:
                missing_order_id_pair += 1
                continue
            order_candidates.append(with_answer(row, structured_output))
            continue
        if output.get("risk_level") == "high" and output.get("audit_result") == "reject_recommendation":
            carryover_fill_candidates.append(row)
            continue
        bucket = calibration_bucket(row)
        if bucket:
            calibration[bucket].append(row)

    rng.shuffle(order_candidates)
    rng.shuffle(carryover_fill_candidates)
    for rows in calibration.values():
        rng.shuffle(rows)

    carryover_fill_rows = []
    for row in carryover_fill_candidates:
        if len(carryover_rows) + len(carryover_fill_rows) >= CARRYOVER_LIMIT:
            break
        cid = case_id(row)
        if cid not in selected:
            carryover_fill_rows.append(role_row(row, role="high_risk_carryover_fill"))
            selected.add(cid)
    carryover_rows.extend(carryover_fill_rows)

    order_rows = []
    for row in order_candidates:
        if len(order_rows) >= ORDER_ID_REPAIR_COUNT:
            break
        cid = case_id(row)
        if cid not in selected:
            order_rows.append(role_row(row, role="order_id_structured_repair"))
            selected.add(cid)

    low_quota = int(round(CALIBRATION_COUNT * 2 / 3))
    medium_quota = CALIBRATION_COUNT - low_quota
    calibration_rows = []
    for bucket, quota in [("low_pass", low_quota), ("medium_or_manual_review", medium_quota)]:
        used = 0
        for row in calibration[bucket]:
            if used >= quota:
                break
            cid = case_id(row)
            if cid not in selected:
                calibration_rows.append(role_row(row, role="calibration", bucket=bucket))
                selected.add(cid)
                used += 1

    output_rows = [*carryover_rows, *order_rows, *calibration_rows]
    write_jsonl(output_rows, OUTPUT_MIX)

    config = read_yaml(BASE_CONFIG)
    config["data"]["train_file"] = repo_path(OUTPUT_MIX)
    config["data"]["val_file"] = "data/mv_audit/sft_main/val.jsonl"
    config["training"]["base_adapter_dir"] = "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r2_order_id_from_r1_existing_images"
    config["training"]["output_dir"] = "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r3_order_id_structured_from_r2"
    config["training"]["learning_rate"] = 2.0e-5
    config["inference"]["sft_adapter_dir"] = config["training"]["output_dir"]
    config["inference"]["predictions_dir"] = "outputs/predictions/phase09_repair_sft_r3_order_id_structured_train_decode_dev"
    config["inference"]["ground_truth_dir"] = "outputs/eval_sets/phase09_repair_sft_r3_order_id_structured"
    config["inference"]["sample_manifest_dir"] = ""
    config["inference"]["train_decode_dev_ground_truth_dir"] = "outputs/eval_sets/phase09_repair_sft_r3_order_id_structured_train_decode_dev"
    config["inference"]["schema_guard"] = True
    config["inference"]["max_new_tokens"] = 1536
    write_yaml(config, OUTPUT_CONFIG)

    manifest = {
        "name": "phase09_repair_sft_v3_order_id_structured_mix",
        "seed": SEED,
        "carryover_source": repo_path(R1_MIX),
        "carryover_rows": len(carryover_rows),
        "carryover_limit": CARRYOVER_LIMIT,
        "carryover_fill_rows": len(carryover_fill_rows),
        "carryover_dropped_before_limit": carryover_dropped,
        "order_id_structured_repair_rows": len(order_rows),
        "order_id_candidates_available": len(order_candidates),
        "order_id_candidates_missing_pair": missing_order_id_pair,
        "calibration_rows": len(calibration_rows),
        "calibration_bucket_counts": dict(Counter(row["repair_mix_metadata"].get("bucket") for row in calibration_rows)),
        "total_rows": len(output_rows),
        "excluded_case_count": len(excluded),
        "overlap_with_excluded_count": len(selected & excluded),
        "source_policy": "MV-Train only; excludes DPO holdout, train_decode_dev and sample500/Test raw cases.",
        "target_policy": "Public output schema unchanged; order_id mismatch answers put order/reimbursement order_id evidence first and use explicit A-vs-B reason.",
        "output_train": repo_path(OUTPUT_MIX),
        "output_config": repo_path(OUTPUT_CONFIG),
    }
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Order-ID Structured Repair SFT v3 mix.")
    parser.add_argument("--dry_run", action="store_true", help="Build artifacts and print manifest; no training is run.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    manifest = build()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
