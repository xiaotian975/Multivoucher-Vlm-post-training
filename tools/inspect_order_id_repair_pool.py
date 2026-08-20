import json
from collections import Counter
from pathlib import Path


SFT_TRAIN = Path("data/mv_audit/sft_main/train.jsonl")
EXISTING_MIX = Path("docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix_existing_images.jsonl")
TRAIN_DECODE_DEV = Path("data/mv_audit/dpo_v2/train_decode_dev.jsonl")
SAMPLE_SPLITS = [
    "test_clean",
    "test_robust",
    "test_unseen_template",
    "test_hard_negative",
]


def iter_jsonl(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def main():
    excluded = {case_id(row) for row in iter_jsonl(TRAIN_DECODE_DEV)}
    for split in SAMPLE_SPLITS:
        excluded.update(case_id(row) for row in iter_jsonl(Path(f"data/mv_audit/raw_cases/main/{split}_cases.jsonl")))

    existing_mix_ids = {case_id(row) for row in iter_jsonl(EXISTING_MIX)}
    rows = list(iter_jsonl(SFT_TRAIN))
    order_rows = []
    for row in rows:
        ans = answer(row)
        if row.get("source_split") != "MV-Train":
            continue
        if "order_id_mismatch" not in set(ans.get("anomaly_types") or []):
            continue
        if ans.get("risk_level") != "high" or ans.get("audit_result") != "reject_recommendation":
            continue
        order_rows.append(row)

    usable = [row for row in order_rows if case_id(row) not in excluded and images_exist(row)]
    in_existing_mix = [row for row in usable if case_id(row) in existing_mix_ids]
    not_in_existing_mix = [row for row in usable if case_id(row) not in existing_mix_ids]
    report = {
        "sft_train_rows": len(rows),
        "order_id_mismatch_high_reject_mv_train": len(order_rows),
        "usable_after_exclusions_and_existing_images": len(usable),
        "usable_already_in_r1_existing_mix": len(in_existing_mix),
        "usable_not_in_r1_existing_mix": len(not_in_existing_mix),
        "excluded_case_count": len(excluded),
        "usable_case_preview": [case_id(row) for row in usable[:20]],
        "not_in_r1_preview": [case_id(row) for row in not_in_existing_mix[:20]],
        "risk_audit_counts": {f"{key[0]}|{key[1]}": value for key, value in Counter((answer(row).get("risk_level"), answer(row).get("audit_result")) for row in order_rows).items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
