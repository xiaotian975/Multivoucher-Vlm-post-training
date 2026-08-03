"""Inject phase 03 anomalies into normal MultiVoucher-Audit base cases."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from mv_audit.data_gen.case_validator import load_schema, validate_cases
from mv_audit.data_gen.risk_rule_engine import DATE_FORMAT, money, money_string, update_case_labels
from mv_audit.utils import load_config, read_jsonl, write_jsonl


DEFAULT_ANOMALY_DISTRIBUTION = {
    "none": 0.30,
    "amount_mismatch": 0.12,
    "over_reimbursement": 0.08,
    "date_mismatch": 0.08,
    "merchant_mismatch": 0.08,
    "applicant_mismatch": 0.08,
    "order_id_mismatch": 0.08,
    "missing_document": 0.08,
    "duplicate_in_batch": 0.05,
    "unreadable_image": 0.05,
}

ANOMALY_TYPES = tuple(DEFAULT_ANOMALY_DISTRIBUTION)
NON_MATCHING_PEOPLE = ["赵明远", "孙雅宁", "周启航", "吴思雨", "郑子涵"]
MISSING_DOCUMENT_CHOICES = ["invoice", "payment", "reimbursement_form", "order"]


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _allocate_labels(total: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    if total <= 0:
        return []
    unknown = set(weights) - set(ANOMALY_TYPES)
    if unknown:
        raise ValueError(f"Unknown anomaly types in config: {sorted(unknown)}")
    ordered = list(ANOMALY_TYPES)
    raw_weights = {name: float(weights.get(name, 0.0)) for name in ordered}
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        raise ValueError("anomaly_distribution must contain at least one positive weight")

    exact = {name: total * raw_weights[name] / weight_sum for name in ordered}
    counts = {name: int(exact[name]) for name in ordered}
    remainder = total - sum(counts.values())
    fractions = sorted(ordered, key=lambda name: (exact[name] - counts[name], raw_weights[name]), reverse=True)
    for name in fractions[:remainder]:
        counts[name] += 1

    labels: list[str] = []
    for name in ordered:
        labels.extend([name] * counts[name])
    rng.shuffle(labels)
    return labels


def _set_anomaly(case: dict[str, Any], anomaly_type: str, *, evidence_sufficient: bool = True) -> None:
    case["primary_anomaly_type"] = anomaly_type
    case["anomaly_types"] = [] if anomaly_type == "none" else [anomaly_type]
    case["evidence_sufficient"] = evidence_sufficient
    metadata = case.setdefault("metadata", {})
    metadata["source"] = "phase_03_anomaly_injector"
    metadata["primary_anomaly_source"] = anomaly_type


def _small_delta(base_amount: Decimal) -> Decimal:
    return max(Decimal("1.00"), min(Decimal("80.00"), base_amount * Decimal("0.03")))


def _large_delta(base_amount: Decimal) -> Decimal:
    return max(Decimal("150.00"), base_amount * Decimal("0.12"))


def _inject_none(case: dict[str, Any], _rng: random.Random) -> None:
    _set_anomaly(case, "none", evidence_sufficient=True)


def _inject_amount_mismatch(case: dict[str, Any], rng: random.Random) -> None:
    base_amount = money(case["invoice_amount"])
    delta = _small_delta(base_amount) if rng.random() < 0.45 else _large_delta(base_amount)
    case["order_amount"] = money_string(base_amount + delta)
    _set_anomaly(case, "amount_mismatch", evidence_sufficient=True)
    case["metadata"]["amount_mismatch_field"] = "order_amount"


def _inject_over_reimbursement(case: dict[str, Any], rng: random.Random) -> None:
    base_amount = money(case["invoice_amount"])
    delta = _small_delta(base_amount) if rng.random() < 0.35 else _large_delta(base_amount)
    case["reimbursement_amount"] = money_string(base_amount + delta)
    _set_anomaly(case, "over_reimbursement", evidence_sufficient=True)
    case["metadata"]["over_reimbursement_delta"] = money_string(delta)


def _shift_date(date_value: str, days: int) -> str:
    return (datetime.strptime(date_value, DATE_FORMAT) + timedelta(days=days)).strftime(DATE_FORMAT)


def _inject_date_mismatch(case: dict[str, Any], rng: random.Random) -> None:
    mismatch_type = rng.choice(["application_before_payment", "payment_invoice_gap", "invoice_before_order"])
    if mismatch_type == "application_before_payment":
        case["application_date"] = _shift_date(case["payment_date"], -rng.randint(1, 5))
    elif mismatch_type == "payment_invoice_gap":
        case["invoice_date"] = _shift_date(case["payment_date"], rng.randint(91, 140))
    else:
        case["invoice_date"] = _shift_date(case["order_date"], -rng.randint(31, 70))
    _set_anomaly(case, "date_mismatch", evidence_sufficient=True)
    case["metadata"]["date_mismatch_type"] = mismatch_type


def _inject_merchant_mismatch(case: dict[str, Any], rng: random.Random) -> None:
    case["payment_merchant"] = f"外部供应商{rng.randint(1000, 9999)}有限公司"
    _set_anomaly(case, "merchant_mismatch", evidence_sufficient=True)
    case["metadata"]["merchant_mismatch_field"] = "payment_merchant"


def _pick_different_person(current: str) -> str:
    for person in NON_MATCHING_PEOPLE:
        if person != current:
            return person
    return "陈临川"


def _inject_applicant_mismatch(case: dict[str, Any], rng: random.Random) -> None:
    target = rng.choice(["payer", "order_user"])
    case[target] = _pick_different_person(case["applicant"])
    _set_anomaly(case, "applicant_mismatch", evidence_sufficient=True)
    case["metadata"]["applicant_mismatch_field"] = target


def _inject_order_id_mismatch(case: dict[str, Any], rng: random.Random) -> None:
    base_order_id = str(case["order_id"])
    suffix_digit = str((int(base_order_id[-1]) + rng.randint(1, 8)) % 10)
    mismatched_order_id = f"{base_order_id[:-1]}{suffix_digit}"
    _set_anomaly(case, "order_id_mismatch", evidence_sufficient=True)
    case["metadata"]["reimbursement_form_order_id"] = base_order_id
    case["metadata"]["order_screenshot_order_id"] = mismatched_order_id


def _inject_missing_document(case: dict[str, Any], rng: random.Random) -> None:
    missing_doc = rng.choice(MISSING_DOCUMENT_CHOICES)
    case["documents"] = [doc for doc in case["documents"] if doc != missing_doc]
    _set_anomaly(case, "missing_document", evidence_sufficient=False)
    case["metadata"]["missing_doc_type"] = missing_doc


def _inject_duplicate_in_batch(case: dict[str, Any], rng: random.Random) -> None:
    duplicate_doc_type = rng.choice(["invoice", "order"])
    _set_anomaly(case, "duplicate_in_batch", evidence_sufficient=True)
    case["metadata"]["duplicate_doc_type"] = duplicate_doc_type
    case["metadata"]["duplicate_source_doc_type"] = duplicate_doc_type
    case["metadata"]["duplicate_pair_index"] = rng.randint(1, 3)


def _inject_unreadable_image(case: dict[str, Any], rng: random.Random) -> None:
    core = rng.random() < 0.7
    if core:
        unreadable_doc, unreadable_fields = rng.choice(
            [
                ("invoice", ["invoice_amount", "invoice_merchant"]),
                ("payment", ["payment_amount", "payer"]),
                ("reimbursement_form", ["reimbursement_amount", "applicant"]),
                ("order", ["order_id", "order_merchant"]),
            ]
        )
    else:
        unreadable_doc, unreadable_fields = rng.choice(
            [
                ("invoice", ["tax_amount"]),
                ("payment", ["payment_date"]),
                ("reimbursement_form", ["expense_type"]),
                ("order", ["order_date"]),
            ]
        )
    _set_anomaly(case, "unreadable_image", evidence_sufficient=not core)
    case["metadata"]["unreadable_doc_type"] = unreadable_doc
    case["metadata"]["unreadable_fields"] = unreadable_fields
    case["metadata"]["core_field_unreadable"] = core


INJECTORS = {
    "none": _inject_none,
    "amount_mismatch": _inject_amount_mismatch,
    "over_reimbursement": _inject_over_reimbursement,
    "date_mismatch": _inject_date_mismatch,
    "merchant_mismatch": _inject_merchant_mismatch,
    "applicant_mismatch": _inject_applicant_mismatch,
    "order_id_mismatch": _inject_order_id_mismatch,
    "missing_document": _inject_missing_document,
    "duplicate_in_batch": _inject_duplicate_in_batch,
    "unreadable_image": _inject_unreadable_image,
}


def inject_anomalies(cases: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inject configured anomalies and return cases with a statistics report."""

    seed = int(config.get("seed", 42)) + 300
    rng = random.Random(seed)
    distribution = config.get("anomaly_distribution") or DEFAULT_ANOMALY_DISTRIBUTION
    labels = _allocate_labels(len(cases), distribution, rng)

    output_cases: list[dict[str, Any]] = []
    for source_case, anomaly_type in zip(cases, labels, strict=True):
        case = copy.deepcopy(source_case)
        INJECTORS[anomaly_type](case, rng)
        update_case_labels(case)
        output_cases.append(case)

    stats = {
        "total_cases": len(output_cases),
        "seed": seed,
        "primary_anomaly_type": dict(Counter(case["primary_anomaly_type"] for case in output_cases)),
        "risk_level": dict(Counter(case["risk_level"] for case in output_cases)),
        "audit_result": dict(Counter(case["audit_result"] for case in output_cases)),
        "evidence_sufficient": dict(Counter(str(case["evidence_sufficient"]).lower() for case in output_cases)),
    }
    return output_cases, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject phase 03 anomalies into base cases.")
    parser.add_argument("--input", required=True, help="Path to phase 02 base case JSONL.")
    parser.add_argument("--output", required=True, help="Path to anomaly-injected JSONL.")
    parser.add_argument("--config", required=True, help="Path to data generation YAML config.")
    parser.add_argument("--schema", default="configs/schema/case_schema.json", help="Path to case schema.")
    parser.add_argument("--stats_output", default=None, help="Optional path to JSON statistics report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cases = read_jsonl(args.input)
    schema = load_schema(args.schema)
    validate_cases(cases, schema=schema, require_normal=True)

    injected_cases, stats = inject_anomalies(cases, config)
    validate_cases(injected_cases, schema=schema, require_normal=False)
    write_jsonl(injected_cases, args.output)

    stats_output = args.stats_output or str(Path(args.output).with_suffix(".stats.json"))
    _write_json(stats, stats_output)
    print(f"injected_cases={len(injected_cases)}")
    print(f"output={args.output}")
    print(f"stats={stats_output}")


if __name__ == "__main__":
    main()
