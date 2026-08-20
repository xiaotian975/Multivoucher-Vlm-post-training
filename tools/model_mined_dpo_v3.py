"""Build, audit, and select artifacts for model-mined DPO v3."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.evaluation.json_parser import parse_json_output
from mv_audit.training.reward_function import score_output
from mv_audit.utils import iter_jsonl, read_jsonl, read_yaml, write_jsonl


DEFAULT_CANDIDATES = "data/mv_audit/dpo_v3_model_mined/candidates.jsonl"
DEFAULT_TRAIN = "data/mv_audit/dpo_v3_model_mined/pairs_train.jsonl"
DEFAULT_HOLDOUT = "data/mv_audit/dpo_v3_model_mined/pairs_holdout.jsonl"
DEFAULT_PROBE = "data/mv_audit/dpo_v3_model_mined/alignment_probe.jsonl"
DEFAULT_SCHEMA = "configs/schema/output_schema.json"
V3_MIX = "docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl"
SFT_TRAIN = "data/mv_audit/sft_main/train.jsonl"


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("ground_truth"), dict):
        ground_truth = row["ground_truth"]
        if isinstance(ground_truth.get("output"), dict):
            return copy.deepcopy(ground_truth["output"])
    if isinstance(row.get("answer"), dict):
        return copy.deepcopy(row["answer"])
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1].get("content"), str):
        parsed = json.loads(messages[-1]["content"])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Missing answer for {_case_id(row)}")


def _prompt(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    messages = row.get("messages") or []
    if messages and isinstance(messages[0].get("content"), str):
        return messages[0]["content"]
    raise ValueError(f"Missing prompt for {_case_id(row)}")


def _images_exist(row: dict[str, Any]) -> bool:
    images = row.get("images") or []
    return bool(images) and all(Path(str(item.get("image_path") or "")).exists() for item in images)


def _structured_order_id_output(output: dict[str, Any]) -> dict[str, Any] | None:
    structured = copy.deepcopy(output)
    pair: dict[str, dict[str, Any]] = {}
    for item in structured.get("evidence") or []:
        if not isinstance(item, dict) or item.get("field") != "order_id":
            continue
        doc_type = str(item.get("source_doc_type") or "")
        if doc_type in {"order", "reimbursement_form"}:
            pair.setdefault(doc_type, copy.deepcopy(item))
    if set(pair) != {"order", "reimbursement_form"}:
        return None
    order_value = str(pair["order"].get("value") or "")
    reimbursement_value = str(pair["reimbursement_form"].get("value") or "")
    if not order_value or not reimbursement_value or order_value == reimbursement_value:
        return None
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
        f"检测到order_id_mismatch：订单截图订单号{order_value}与"
        f"报销申请单订单号{reimbursement_value}不一致，建议拒绝报销。"
    )
    remaining = [
        copy.deepcopy(item)
        for item in structured.get("evidence") or []
        if not (
            isinstance(item, dict)
            and item.get("field") == "order_id"
            and item.get("source_doc_type") in pair
        )
    ]
    structured["evidence"] = [pair["order"], pair["reimbursement_form"], *remaining]
    return structured


def _excluded_case_ids() -> set[str]:
    excluded: set[str] = set()
    paths = [
        V3_MIX,
        "data/mv_audit/dpo_v2/pairs_holdout.jsonl",
        "data/mv_audit/dpo_v2/train_decode_dev.jsonl",
        "docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/dpo_v2_baseline/train_decode_dev.jsonl",
    ]
    for path in paths:
        if Path(path).exists():
            excluded.update(_case_id(row) for row in iter_jsonl(path))
    for split in ["test_clean", "test_robust", "test_unseen_template", "test_hard_negative"]:
        path = Path(f"data/mv_audit/raw_cases/main/{split}_cases.jsonl")
        if path.exists():
            excluded.update(_case_id(row) for row in iter_jsonl(path))
    return {item for item in excluded if item}


def _candidate_row(row: dict[str, Any], output: dict[str, Any], bucket: str) -> dict[str, Any]:
    return {
        "case_id": _case_id(row),
        "prompt": _prompt(row),
        "images": copy.deepcopy(row.get("images") or []),
        "ground_truth": {"output": output},
        "bucket": bucket,
        "source_split": "MV-Train",
    }


def build_candidates(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    excluded = _excluded_case_ids()
    buckets: dict[str, list[dict[str, Any]]] = {
        "order_id_mismatch": [],
        "other_high_risk": [],
        "low_pass": [],
    }
    for row in iter_jsonl(args.sft_train):
        cid = _case_id(row)
        if not cid or cid in excluded or row.get("source_split") != "MV-Train" or not _images_exist(row):
            continue
        output = _answer(row)
        anomalies = set(output.get("anomaly_types") or [])
        risk = output.get("risk_level")
        audit = output.get("audit_result")
        if "order_id_mismatch" in anomalies and risk == "high" and audit == "reject_recommendation":
            structured = _structured_order_id_output(output)
            if structured is not None:
                buckets["order_id_mismatch"].append(_candidate_row(row, structured, "order_id_mismatch"))
        elif risk == "high" and audit == "reject_recommendation":
            buckets["other_high_risk"].append(_candidate_row(row, output, "other_high_risk"))
        elif risk == "low" and audit == "pass":
            buckets["low_pass"].append(_candidate_row(row, output, "low_pass"))

    quotas = {
        "order_id_mismatch": args.order_id_count,
        "other_high_risk": args.other_high_count,
        "low_pass": args.low_pass_count,
    }
    selected: list[dict[str, Any]] = []
    available = {bucket: len(rows) for bucket, rows in buckets.items()}
    for bucket, rows in buckets.items():
        rng.shuffle(rows)
        if len(rows) < quotas[bucket]:
            raise ValueError(f"Not enough {bucket} candidates: {len(rows)} < {quotas[bucket]}")
        selected.extend(rows[: quotas[bucket]])
    rng.shuffle(selected)
    write_jsonl(selected, args.output)
    manifest = {
        "stage": "model_mined_dpo_v3_candidates",
        "seed": args.seed,
        "counts": dict(Counter(row["bucket"] for row in selected)),
        "available": available,
        "total": len(selected),
        "excluded_case_count": len(excluded),
        "overlap_with_excluded": len({_case_id(row) for row in selected} & excluded),
        "source_policy": "MV-Train only; excludes v3 mix, DPO holdout/decode-dev and all Test/sample500 cases.",
        "output": args.output,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


def _schema_valid(raw_output: str, validator: Draft202012Validator) -> tuple[bool, dict[str, Any] | None]:
    parsed = parse_json_output(raw_output)
    if not parsed.json_validity or parsed.output is None:
        return False, None
    return not list(validator.iter_errors(parsed.output)), parsed.output


def _core_ok(bucket: str, details: dict[str, float]) -> bool:
    if bucket == "order_id_mismatch":
        return (
            details.get("r_order_id_pair") == 1.0
            and details.get("r_audit") == 1.0
            and details.get("r_anomaly") == 1.0
        )
    if bucket == "other_high_risk":
        return (
            details.get("r_audit") == 1.0
            and details.get("r_anomaly") == 1.0
            and details.get("r_evidence", 0.0) >= 0.70
        )
    return details.get("r_audit") == 1.0 and details.get("p_false_escalation", 0.0) == 0.0


def _preference_quality(bucket: str, details: dict[str, float]) -> float:
    """Return a non-saturating 0..1 score used only to mine preference pairs."""
    if bucket == "order_id_mismatch":
        value = (
            0.40 * details.get("r_json", 0.0)
            + 0.15 * details.get("r_order_id_pair", 0.0)
            + 0.15 * details.get("r_audit", 0.0)
            + 0.10 * details.get("r_anomaly", 0.0)
            + 0.10 * details.get("r_evidence", 0.0)
            + 0.05 * details.get("r_field", 0.0)
            + 0.05 * details.get("r_consistency", 0.0)
        )
    elif bucket == "low_pass":
        value = (
            0.40 * details.get("r_json", 0.0)
            + 0.25 * details.get("r_audit", 0.0)
            + 0.20 * (1.0 - details.get("p_false_escalation", 0.0))
            + 0.10 * details.get("r_anomaly", 0.0)
            + 0.05 * details.get("r_evidence", 0.0)
        )
    else:
        value = (
            0.40 * details.get("r_json", 0.0)
            + 0.25 * details.get("r_audit", 0.0)
            + 0.15 * details.get("r_anomaly", 0.0)
            + 0.10 * details.get("r_evidence", 0.0)
            + 0.05 * details.get("r_field", 0.0)
            + 0.05 * details.get("r_consistency", 0.0)
        )
    return max(0.0, min(1.0, float(value)))


def _is_order_failure(details: dict[str, float]) -> bool:
    return any(
        (
            details.get("r_order_id_pair", 0.0) < 1.0,
            details.get("r_audit", 0.0) < 1.0,
            details.get("r_anomaly", 0.0) < 1.0,
            details.get("p_high_risk_miss", 0.0) > 0.0,
        )
    )


def _make_low_pass_rejected(output: dict[str, Any]) -> dict[str, Any]:
    rejected = copy.deepcopy(output)
    rejected["risk_level"] = "high"
    rejected["audit_result"] = "reject_recommendation"
    anomalies = [item for item in rejected.get("anomaly_types") or [] if item != "missing_document"]
    rejected["anomaly_types"] = [*anomalies, "missing_document"]
    rejected["consistency_check"]["document_complete"] = False
    rejected["reason"] = "Calibration negative: incorrectly escalates a low-risk case."
    return rejected

def _load_rollouts(directory: str) -> list[dict[str, Any]]:
    root = Path(directory)
    paths = sorted(root.glob("rollouts.shard-*.jsonl"))
    if not paths and (root / "rollouts.jsonl").exists():
        paths = [root / "rollouts.jsonl"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    if not rows:
        raise ValueError(f"No rollout rows found in {root}")
    case_ids = [_case_id(row) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id found across rollout shards.")
    return rows


def _make_pair(
    row: dict[str, Any],
    *,
    validator: Draft202012Validator,
    schema: dict[str, Any],
    min_gap: float,
    max_gap: float,
    allow_ground_truth_chosen: bool,
) -> dict[str, Any] | None:
    truth = row["ground_truth"]
    bucket = str(row["bucket"])
    scored: list[dict[str, Any]] = []
    for completion in row.get("completions") or []:
        raw = str(completion.get("raw_output") if isinstance(completion, dict) else completion)
        valid, parsed = _schema_valid(raw, validator)
        if not valid or parsed is None:
            continue
        score = score_output(raw, truth, row["images"], schema)
        scored.append(
            {
                "raw": raw,
                "parsed": parsed,
                "task_reward": float(score["reward"]),
                "details": score["details"],
                "preference_score": _preference_quality(bucket, score["details"]),
            }
        )
    if not scored or bucket == "other_high_risk":
        return None

    chosen_candidates = [item for item in scored if _core_ok(bucket, item["details"])]
    chosen_source = "model_generated"
    rejected_source = "model_generated"
    if bucket == "order_id_mismatch":
        if chosen_candidates:
            chosen = max(chosen_candidates, key=lambda item: item["preference_score"])
        elif allow_ground_truth_chosen:
            raw = json.dumps(truth["output"], ensure_ascii=False, separators=(",", ":"))
            score = score_output(raw, truth, row["images"], schema)
            chosen = {
                "raw": raw,
                "parsed": truth["output"],
                "task_reward": float(score["reward"]),
                "details": score["details"],
                "preference_score": _preference_quality(bucket, score["details"]),
            }
            chosen_source = "expert_corrected_target"
        else:
            return None
        rejected_candidates = [
            item
            for item in scored
            if item["raw"] != chosen["raw"] and _is_order_failure(item["details"])
        ]
        if not rejected_candidates:
            return None
        rejected = max(rejected_candidates, key=lambda item: item["preference_score"])
    else:
        if not chosen_candidates:
            return None
        chosen = max(chosen_candidates, key=lambda item: item["preference_score"])
        rejected_output = _make_low_pass_rejected(chosen["parsed"])
        rejected_raw = json.dumps(rejected_output, ensure_ascii=False, separators=(",", ":"))
        valid, parsed = _schema_valid(rejected_raw, validator)
        if not valid or parsed is None:
            raise ValueError(f"Synthetic calibration output is schema-invalid for {_case_id(row)}")
        score = score_output(rejected_raw, truth, row["images"], schema)
        rejected = {
            "raw": rejected_raw,
            "parsed": parsed,
            "task_reward": float(score["reward"]),
            "details": score["details"],
            "preference_score": _preference_quality(bucket, score["details"]),
        }
        rejected_source = "synthetic_false_escalation_calibration"

    gap = float(chosen["preference_score"] - rejected["preference_score"])
    if gap + 1e-9 < min_gap or gap - 1e-9 > max_gap:
        return None
    weights = {"order_id_mismatch": 2.0, "other_high_risk": 1.5, "low_pass": 1.0}
    cid = _case_id(row)
    return {
        "id": f"{cid}_model_error_mined_dpo_v3",
        "pair_id": f"{cid}_model_error_mined_dpo_v3",
        "case_id": cid,
        "pair_type": bucket,
        "pair_strategy": "model_error_mined" if bucket == "order_id_mismatch" else "false_escalation_calibration",
        "source": "model_error_mined_dpo_v3",
        "chosen_source": chosen_source,
        "rejected_source": rejected_source,
        "source_split": "MV-Train",
        "prompt": row["prompt"],
        "images": row["images"],
        "ground_truth": truth,
        "chosen": chosen["raw"],
        "rejected": rejected["raw"],
        "chosen_reward": chosen["preference_score"],
        "rejected_reward": rejected["preference_score"],
        "reward_gap": gap,
        "chosen_task_reward": chosen["task_reward"],
        "rejected_task_reward": rejected["task_reward"],
        "final_weight": weights[bucket],
        "sft_loss_weight": 1.0,
    }

def _take_holdout(pairs: list[dict[str, Any]], seed: int, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 1000)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_bucket.setdefault(str(pair["pair_type"]), []).append(pair)
    order_quota = (count * 3) // 4
    quotas = {
        "order_id_mismatch": order_quota,
        "low_pass": count - order_quota,
    }
    holdout: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket, quota in quotas.items():
        rows = by_bucket.get(bucket, [])
        rng.shuffle(rows)
        chosen = rows[:quota]
        holdout.extend(chosen)
        selected_ids.update(_case_id(row) for row in chosen)
    train = [row for row in pairs if _case_id(row) not in selected_ids]
    train.sort(key=lambda row: 0 if row["pair_type"] == "order_id_mismatch" else 1)
    rng.shuffle(holdout)
    return train, holdout


def build_pairs(args: argparse.Namespace) -> None:
    schema = read_yaml(args.schema)
    validator = Draft202012Validator(schema)
    rows = _load_rollouts(args.rollout_dir)
    pairs: list[dict[str, Any]] = []
    for row in rows:
        pair = _make_pair(
            row,
            validator=validator,
            schema=schema,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            allow_ground_truth_chosen=True,
        )
        if pair is None:
            continue
        pairs.append(pair)

    train, holdout = _take_holdout(pairs, args.seed, args.holdout_count)
    train = train[: args.train_count]
    random.Random(args.seed + 2000).shuffle(train)
    holdout = holdout[: args.holdout_count]
    if len(train) < args.train_count or len(holdout) < args.holdout_count:
        raise ValueError(
            f"Insufficient hard pairs: train={len(train)}/{args.train_count}, "
            f"holdout={len(holdout)}/{args.holdout_count}."
        )
    write_jsonl(train, args.train_output)
    write_jsonl(holdout, args.holdout_output)
    probes = [
        {
            "case_id": pair["case_id"],
            "prompt": pair["prompt"],
            "images": pair["images"],
            "ground_truth": pair["ground_truth"],
            "bucket": pair["pair_type"],
            "source_split": "MV-Train",
        }
        for pair in holdout
    ]
    write_jsonl(probes, args.probe_output)
    all_pairs = [*train, *holdout]
    manifest = {
        "stage": "model_mined_dpo_v3_pairs",
        "rollout_cases": len(rows),
        "eligible_pairs": len(pairs),
        "train_pairs": len(train),
        "holdout_pairs": len(holdout),
        "train_bucket_counts": dict(Counter(row["pair_type"] for row in train)),
        "holdout_bucket_counts": dict(Counter(row["pair_type"] for row in holdout)),
        "chosen_source_counts": dict(Counter(row["chosen_source"] for row in all_pairs)),
        "expert_correction_rate": sum(row["chosen_source"] == "expert_corrected_target" for row in all_pairs) / len(all_pairs),
        "rejected_source_counts": dict(Counter(row["rejected_source"] for row in all_pairs)),
        "reward_gap_min": min(row["reward_gap"] for row in all_pairs),
        "reward_gap_max": max(row["reward_gap"] for row in all_pairs),
        "reward_gap_mean": sum(row["reward_gap"] for row in all_pairs) / len(all_pairs),
        "train_holdout_overlap": len({_case_id(row) for row in train} & {_case_id(row) for row in holdout}),
        "outputs": {
            "train": args.train_output,
            "holdout": args.holdout_output,
            "probe": args.probe_output,
        },
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


def audit_pairs(args: argparse.Namespace) -> None:
    schema = read_yaml(args.schema)
    validator = Draft202012Validator(schema)
    train = read_jsonl(args.train)
    holdout = read_jsonl(args.holdout)
    if len(train) != args.train_count or len(holdout) != args.holdout_count:
        raise ValueError(
            f"Unexpected pair counts: train={len(train)}/{args.train_count}, "
            f"holdout={len(holdout)}/{args.holdout_count}"
        )
    train_ids = {_case_id(row) for row in train}
    holdout_ids = {_case_id(row) for row in holdout}
    overlap = train_ids & holdout_ids
    if overlap:
        raise ValueError(f"Train/holdout case overlap: {sorted(overlap)[:5]}")

    invalid: list[str] = []
    nonpositive_task_gap: list[str] = []
    for row in [*train, *holdout]:
        cid = _case_id(row)
        if row.get("source_split") != "MV-Train":
            raise ValueError(f"Non-train source in pair {cid}: {row.get('source_split')}")
        chosen_valid, _ = _schema_valid(str(row["chosen"]), validator)
        rejected_valid, _ = _schema_valid(str(row["rejected"]), validator)
        if not chosen_valid or not rejected_valid:
            invalid.append(cid)
            continue
        chosen = score_output(str(row["chosen"]), row["ground_truth"], row["images"], schema)
        rejected = score_output(str(row["rejected"]), row["ground_truth"], row["images"], schema)
        if float(chosen["reward"]) <= float(rejected["reward"]):
            nonpositive_task_gap.append(cid)
        gap = float(row["reward_gap"])
        if gap + 1e-9 < args.min_gap or gap - 1e-9 > args.max_gap:
            raise ValueError(f"Preference gap out of range for {cid}: {gap}")
    if invalid:
        raise ValueError(f"Schema-invalid pair outputs: {invalid[:5]}")
    if nonpositive_task_gap:
        raise ValueError(f"Non-positive task-reward pairs: {nonpositive_task_gap[:5]}")

    all_pairs = [*train, *holdout]
    payload = {
        "status": "PASS",
        "train_pairs": len(train),
        "holdout_pairs": len(holdout),
        "train_holdout_overlap": 0,
        "schema_valid_pairs": len(all_pairs),
        "positive_task_reward_gap_rate": 1.0,
        "train_bucket_counts": dict(Counter(row["pair_type"] for row in train)),
        "holdout_bucket_counts": dict(Counter(row["pair_type"] for row in holdout)),
        "chosen_source_counts": dict(Counter(row["chosen_source"] for row in all_pairs)),
        "rejected_source_counts": dict(Counter(row["rejected_source"] for row in all_pairs)),
        "preference_gap_min": min(float(row["reward_gap"]) for row in all_pairs),
        "preference_gap_max": max(float(row["reward_gap"]) for row in all_pairs),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))

def score_probe(args: argparse.Namespace) -> None:
    schema = read_yaml(args.schema)
    rows = read_jsonl(args.rollouts)
    scores: list[dict[str, Any]] = []
    order_scores: list[float] = []
    for row in rows:
        completions = row.get("completions") or []
        if not completions:
            continue
        completion = completions[0]
        raw = str(completion.get("raw_output") if isinstance(completion, dict) else completion)
        score = score_output(raw, row["ground_truth"], row["images"], schema)
        scores.append(score)
        if row.get("bucket") == "order_id_mismatch":
            order_scores.append(float(score["details"].get("r_order_id_pair", 0.0)))
    count = len(scores)
    details = [score["details"] for score in scores]
    metrics = {
        "cases": count,
        "mean_reward": sum(float(score["reward"]) for score in scores) / count if count else 0.0,
        "json_valid_rate": sum(float(item.get("r_json", 0.0)) for item in details) / count if count else 0.0,
        "high_risk_miss_rate": sum(float(item.get("p_high_risk_miss", 0.0)) for item in details) / count if count else 0.0,
        "false_escalation_rate": sum(float(item.get("p_false_escalation", 0.0)) for item in details) / count if count else 0.0,
        "order_id_pair_rate": sum(order_scores) / len(order_scores) if order_scores else 0.0,
        "order_id_cases": len(order_scores),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


def select_checkpoint(args: argparse.Namespace) -> None:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidates = []
    for path in sorted(Path().glob(args.candidates)):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        match = re.search(r"checkpoint-(\d+)", path.as_posix())
        if not match:
            continue
        step = int(match.group(1))
        reward_delta = float(metrics["mean_reward"]) - float(baseline["mean_reward"])
        order_delta = float(metrics["order_id_pair_rate"]) - float(baseline["order_id_pair_rate"])
        eligible = (
            reward_delta >= args.min_reward_delta
            and order_delta >= args.min_order_delta
            and float(metrics["json_valid_rate"]) >= float(baseline["json_valid_rate"]) - 0.01
            and float(metrics["false_escalation_rate"]) <= float(baseline["false_escalation_rate"]) + 0.01
        )
        candidates.append(
            {
                "checkpoint": str(Path(args.checkpoint_root) / f"checkpoint-{step}") if args.checkpoint_root else str(path.parent),
                "step": step,
                "eligible": eligible,
                "reward_delta": reward_delta,
                "order_id_pair_delta": order_delta,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise ValueError("No checkpoint probe metrics found.")
    eligible = sorted((row for row in candidates if row["eligible"]), key=lambda row: row["step"])
    selected = eligible[0] if eligible else sorted(
        candidates,
        key=lambda row: (row["reward_delta"], row["order_id_pair_delta"], -row["step"]),
        reverse=True,
    )[0]
    payload = {
        "status": "ELIGIBLE" if selected["eligible"] else "NO_CHECKPOINT_MET_FULL_GATE",
        "baseline": baseline,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["step"]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-candidates")
    build.add_argument("--sft_train", default=SFT_TRAIN)
    build.add_argument("--output", default=DEFAULT_CANDIDATES)
    build.add_argument("--manifest", default="data/mv_audit/dpo_v3_model_mined/candidate_manifest.json")
    build.add_argument("--seed", type=int, default=45)
    build.add_argument("--order_id_count", type=int, default=120)
    build.add_argument("--other_high_count", type=int, default=60)
    build.add_argument("--low_pass_count", type=int, default=60)
    build.set_defaults(func=build_candidates)

    pairs = subparsers.add_parser("build-pairs")
    pairs.add_argument("--rollout_dir", required=True)
    pairs.add_argument("--schema", default=DEFAULT_SCHEMA)
    pairs.add_argument("--train_output", default=DEFAULT_TRAIN)
    pairs.add_argument("--holdout_output", default=DEFAULT_HOLDOUT)
    pairs.add_argument("--probe_output", default=DEFAULT_PROBE)
    pairs.add_argument("--manifest", default="data/mv_audit/dpo_v3_model_mined/pair_manifest.json")
    pairs.add_argument("--seed", type=int, default=45)
    pairs.add_argument("--min_gap", type=float, default=0.15)
    pairs.add_argument("--max_gap", type=float, default=0.60)
    pairs.add_argument("--train_count", type=int, default=120)
    pairs.add_argument("--holdout_count", type=int, default=24)
    pairs.add_argument("--max_fallback_rate", type=float, default=0.20)
    pairs.set_defaults(func=build_pairs)

    audit = subparsers.add_parser("audit-pairs")
    audit.add_argument("--train", default=DEFAULT_TRAIN)
    audit.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    audit.add_argument("--schema", default=DEFAULT_SCHEMA)
    audit.add_argument("--output", required=True)
    audit.add_argument("--train_count", type=int, default=120)
    audit.add_argument("--holdout_count", type=int, default=24)
    audit.add_argument("--min_gap", type=float, default=0.15)
    audit.add_argument("--max_gap", type=float, default=0.60)
    audit.set_defaults(func=audit_pairs)
    probe = subparsers.add_parser("score-probe")
    probe.add_argument("--rollouts", required=True)
    probe.add_argument("--schema", default=DEFAULT_SCHEMA)
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=score_probe)

    select = subparsers.add_parser("select-checkpoint")
    select.add_argument("--baseline", required=True)
    select.add_argument("--candidates", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--checkpoint_root")
    select.add_argument("--min_reward_delta", type=float, default=0.05)
    select.add_argument("--min_order_delta", type=float, default=0.10)
    select.set_defaults(func=select_checkpoint)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()