import json
import random
from collections import Counter
from pathlib import Path

from mv_audit.utils import read_yaml, write_yaml


SEED = 43
ORDER_ID_REPAIR_COUNT = 120
CALIBRATION_COUNT = 120
REPO = Path(".")
PACK_DIR = Path("docs/experiments/phase08_high_risk_repair_pack_20260813")
R1_MIX = PACK_DIR / "repair_sft_train_mix_existing_images.jsonl"
SFT_TRAIN = Path("data/mv_audit/sft_main/train.jsonl")
OUTPUT_MIX = PACK_DIR / "repair_sft_v2_order_id_mix_existing_images.jsonl"
OUTPUT_MANIFEST = PACK_DIR / "repair_sft_v2_order_id_mix_manifest.json"
BASE_CONFIG = Path("configs/train/high_risk_repair_sft_r1_from_m2_existing_images_qwen3vl_8b_server.yaml")
OUTPUT_CONFIG = Path("configs/train/high_risk_repair_sft_r2_order_id_from_r1_existing_images_qwen3vl_8b_server.yaml")


def iter_jsonl(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def case_id(row):
    return str(row.get("case_id") or row.get("id", "").split("_full_audit")[0])


def answer(row):
    if isinstance(row.get("answer"), dict):
        return row["answer"]
    messages = row.get("messages") or []
    if messages:
        content = messages[-1].get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def images_exist(row):
    return all(Path(str(item.get("image_path", ""))).exists() for item in row.get("images") or [])


def excluded_case_ids():
    excluded = set()
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


def with_metadata(row, role, bucket=None):
    out = dict(row)
    meta = dict(out.get("repair_mix_metadata") or {})
    meta.update({"mix": "phase08_repair_sft_v2_order_id", "role": role})
    if bucket:
        meta["bucket"] = bucket
    out["repair_mix_metadata"] = meta
    return out


def calibration_bucket(row):
    ans = answer(row)
    if row.get("source_split") != "MV-Train":
        return None
    if ans.get("risk_level") == "low" and ans.get("audit_result") == "pass":
        return "low_pass"
    if ans.get("risk_level") == "medium" or ans.get("audit_result") in {"manual_review", "missing_info"}:
        return "medium_or_manual_review"
    return None


def main():
    rng = random.Random(SEED)
    excluded = excluded_case_ids()
    raw_r1_rows = list(iter_jsonl(R1_MIX))
    r1_rows = [row for row in raw_r1_rows if case_id(row) not in excluded and images_exist(row)]
    r1_dropped = [case_id(row) for row in raw_r1_rows if case_id(row) in excluded or not images_exist(row)]
    r1_ids = {case_id(row) for row in r1_rows}
    selected_ids = set(r1_ids)

    sft_rows = list(iter_jsonl(SFT_TRAIN))
    order_candidates = []
    calibration = {"low_pass": [], "medium_or_manual_review": []}
    for row in sft_rows:
        cid = case_id(row)
        if cid in excluded or cid in selected_ids or not images_exist(row):
            continue
        ans = answer(row)
        if (
            row.get("source_split") == "MV-Train"
            and "order_id_mismatch" in set(ans.get("anomaly_types") or [])
            and ans.get("risk_level") == "high"
            and ans.get("audit_result") == "reject_recommendation"
        ):
            order_candidates.append(row)
            continue
        bucket = calibration_bucket(row)
        if bucket:
            calibration[bucket].append(row)

    rng.shuffle(order_candidates)
    for rows in calibration.values():
        rng.shuffle(rows)

    order_rows = []
    for row in order_candidates:
        if len(order_rows) >= ORDER_ID_REPAIR_COUNT:
            break
        cid = case_id(row)
        if cid not in selected_ids:
            order_rows.append(with_metadata(row, "order_id_repair"))
            selected_ids.add(cid)

    low_quota = int(round(CALIBRATION_COUNT * 2 / 3))
    medium_quota = CALIBRATION_COUNT - low_quota
    calibration_rows = []
    for bucket, quota in [("low_pass", low_quota), ("medium_or_manual_review", medium_quota)]:
        for row in calibration[bucket]:
            if len([item for item in calibration_rows if item["repair_mix_metadata"]["bucket"] == bucket]) >= quota:
                break
            cid = case_id(row)
            if cid not in selected_ids:
                calibration_rows.append(with_metadata(row, "calibration", bucket=bucket))
                selected_ids.add(cid)

    r1_carryover = [with_metadata(row, "r1_carryover") for row in r1_rows]
    output_rows = [*r1_carryover, *order_rows, *calibration_rows]
    write_jsonl(output_rows, OUTPUT_MIX)

    manifest = {
        "name": "phase08_repair_sft_v2_order_id_mix",
        "seed": SEED,
        "r1_carryover_rows": len(r1_carryover),
        "r1_dropped_rows": len(r1_dropped),
        "r1_dropped_preview": r1_dropped[:20],
        "order_id_repair_rows": len(order_rows),
        "calibration_rows": len(calibration_rows),
        "total_rows": len(output_rows),
        "order_id_candidates_available": len(order_candidates),
        "excluded_case_count": len(excluded),
        "calibration_bucket_counts": dict(Counter(row["repair_mix_metadata"].get("bucket") for row in calibration_rows)),
        "overlap_with_excluded": sorted(selected_ids & excluded)[:20],
        "overlap_with_excluded_count": len(selected_ids & excluded),
        "source_policy": "MV-Train only; excludes DPO holdout, train_decode_dev and sample500/Test raw cases.",
        "output_train": str(OUTPUT_MIX),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    config = read_yaml(BASE_CONFIG)
    config["data"]["train_file"] = str(OUTPUT_MIX)
    config["data"]["val_file"] = "data/mv_audit/sft_main/val_existing_images.jsonl"
    config["training"]["base_adapter_dir"] = "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r1_from_m2_existing_images"
    config["training"]["output_dir"] = "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r2_order_id_from_r1_existing_images"
    config["training"]["learning_rate"] = 2.0e-5
    config["inference"]["sft_adapter_dir"] = config["training"]["output_dir"]
    config["inference"]["predictions_dir"] = "outputs/predictions/phase08_high_risk_repair_r2_order_id_train_decode_dev"
    config["inference"]["train_decode_dev_ground_truth_dir"] = "outputs/eval_sets/phase08_high_risk_repair_r2_order_id_train_decode_dev"
    config["inference"]["schema_guard"] = True
    config["inference"]["max_new_tokens"] = 1536
    write_yaml(config, OUTPUT_CONFIG)

    print(json.dumps({"manifest": manifest, "config": str(OUTPUT_CONFIG)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
