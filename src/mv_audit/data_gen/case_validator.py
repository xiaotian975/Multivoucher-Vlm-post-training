"""Validation helpers for MultiVoucher-Audit base cases.

Phase 02 validates normal structured cases only. It does not inject anomalies,
split datasets, render images, or implement risk rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mv_audit.utils import read_jsonl, read_yaml


ALLOWED_DOCUMENTS = {"invoice", "payment", "reimbursement_form", "order"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
ALLOWED_AUDIT_RESULTS = {"pass", "manual_review", "missing_info", "reject_recommendation"}
DATE_FORMAT = "%Y-%m-%d"


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    """Load the JSON schema file."""

    return read_yaml(schema_path)


def _require_money(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not a valid decimal amount: {value!r}") from exc
    if amount < 0:
        raise ValueError(f"{field} must be non-negative")
    if amount.quantize(Decimal("0.01")) != amount:
        raise ValueError(f"{field} must have exactly two decimal places: {value!r}")


def _parse_date(case: dict[str, Any], field: str) -> datetime:
    value = case.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format: {value!r}") from exc


def validate_case(
    case: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    require_normal: bool = False,
) -> None:
    """Validate one case and raise ValueError on the first semantic failure."""

    if schema is not None:
        errors = sorted(Draft202012Validator(schema).iter_errors(case), key=lambda err: err.path)
        if errors:
            err = errors[0]
            path = ".".join(str(part) for part in err.path) or "<root>"
            raise ValueError(f"Schema validation failed at {path}: {err.message}")

    for field in [
        "invoice_amount",
        "payment_amount",
        "reimbursement_amount",
        "order_amount",
        "tax_amount",
    ]:
        _require_money(case.get(field), field)

    order_date = _parse_date(case, "order_date")
    payment_date = _parse_date(case, "payment_date")
    invoice_date = _parse_date(case, "invoice_date")
    application_date = _parse_date(case, "application_date")

    documents = case.get("documents")
    if not isinstance(documents, list) or not set(documents).issubset(ALLOWED_DOCUMENTS):
        raise ValueError(f"documents must be a list drawn from {sorted(ALLOWED_DOCUMENTS)}")

    if case.get("risk_level") not in ALLOWED_RISK_LEVELS:
        raise ValueError(f"risk_level is not allowed: {case.get('risk_level')!r}")
    if case.get("audit_result") not in ALLOWED_AUDIT_RESULTS:
        raise ValueError(f"audit_result is not allowed: {case.get('audit_result')!r}")

    if require_normal:
        amounts = {
            case["invoice_amount"],
            case["payment_amount"],
            case["reimbursement_amount"],
            case["order_amount"],
        }
        if len(amounts) != 1:
            raise ValueError("normal base case amounts must be equal")
        if not (case["applicant"] == case["payer"] == case["order_user"]):
            raise ValueError("normal base case applicant, payer, and order_user must be equal")
        if not (order_date <= payment_date <= invoice_date <= application_date):
            raise ValueError("normal base case dates must satisfy order <= payment <= invoice <= application")
        if case.get("primary_anomaly_type") != "none":
            raise ValueError("normal base case must use primary_anomaly_type='none'")
        if case.get("anomaly_types") != []:
            raise ValueError("normal base case must have no anomaly_types")
        if case.get("risk_level") != "low" or case.get("audit_result") != "pass":
            raise ValueError("normal base case must be low + pass")
        if case.get("evidence_sufficient") is not True:
            raise ValueError("normal base case must have evidence_sufficient=true")


def validate_cases(
    cases: list[dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    require_normal: bool = False,
) -> None:
    """Validate a list of cases, including case_id uniqueness."""

    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id at row {index}: {case_id}")
        seen.add(case_id)
        try:
            validate_case(case, schema=schema, require_normal=require_normal)
        except ValueError as exc:
            raise ValueError(f"Invalid case at row {index} ({case_id}): {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MultiVoucher-Audit case JSONL files.")
    parser.add_argument("--input", required=True, help="Path to case JSONL file.")
    parser.add_argument("--schema", required=True, help="Path to case_schema.json.")
    parser.add_argument("--require_normal", action="store_true", help="Require phase 02 normal base-case invariants.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.input)
    schema = load_schema(args.schema)
    validate_cases(cases, schema=schema, require_normal=args.require_normal)
    print(f"validated_cases={len(cases)}")


if __name__ == "__main__":
    main()
