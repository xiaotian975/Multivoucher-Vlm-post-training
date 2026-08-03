"""Central risk and audit-result rules for phase 03 case generation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


DATE_FORMAT = "%Y-%m-%d"
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
REQUIRED_DOCUMENTS = {"invoice", "payment", "reimbursement_form", "order"}
CORE_UNREADABLE_FIELDS = {
    "invoice_amount",
    "payment_amount",
    "reimbursement_amount",
    "order_amount",
    "invoice_merchant",
    "payment_merchant",
    "order_merchant",
    "applicant",
    "payer",
    "order_user",
    "order_id",
    "payment_id",
}


def money(value: Any) -> Decimal:
    """Parse a schema-compatible money string into Decimal."""

    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid money value: {value!r}") from exc


def money_string(value: Decimal) -> str:
    """Format Decimal as a two-decimal money string."""

    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def parse_date(value: Any) -> datetime:
    """Parse a schema-compatible date string."""

    if not isinstance(value, str):
        raise ValueError(f"Invalid date value: {value!r}")
    return datetime.strptime(value, DATE_FORMAT)


def amount_delta(case: dict[str, Any]) -> tuple[Decimal, Decimal]:
    """Return absolute amount spread and spread ratio for the four audited amounts."""

    amounts = [
        money(case["invoice_amount"]),
        money(case["payment_amount"]),
        money(case["reimbursement_amount"]),
        money(case["order_amount"]),
    ]
    positive_amounts = [amount for amount in amounts if amount > 0]
    min_positive = min(positive_amounts) if positive_amounts else Decimal("1.00")
    delta = max(amounts) - min(amounts)
    ratio = delta / max(min_positive, Decimal("1.00"))
    return delta, ratio


def _amount_risk(delta: Decimal, ratio: Decimal) -> str:
    if delta <= Decimal("100.00") and ratio <= Decimal("0.05"):
        return "medium"
    return "high"


def _max_risk(risks: list[str]) -> str:
    return max(risks or ["low"], key=lambda risk: RISK_ORDER[risk])


def _missing_documents(case: dict[str, Any]) -> set[str]:
    documents = set(case.get("documents") or [])
    return REQUIRED_DOCUMENTS - documents


def _merchant_matches(case: dict[str, Any]) -> bool:
    accepted = {case.get("merchant_canonical"), *case.get("merchant_aliases", [])}
    return case.get("invoice_merchant") in accepted and case.get("payment_merchant") in accepted and case.get("order_merchant") in accepted


def _date_risks(case: dict[str, Any]) -> list[str]:
    order_date = parse_date(case["order_date"])
    payment_date = parse_date(case["payment_date"])
    invoice_date = parse_date(case["invoice_date"])
    application_date = parse_date(case["application_date"])

    risks: list[str] = []
    if (invoice_date - payment_date).days > 90 or (payment_date - invoice_date).days > 90:
        risks.append("medium")
    if (order_date - invoice_date).days > 30:
        risks.append("medium")
    if application_date < payment_date:
        risks.append("high")
    return risks


def risk_reasons(case: dict[str, Any]) -> list[tuple[str, str]]:
    """Return rule-level risk decisions as (risk_level, reason_code)."""

    anomaly_types = set(case.get("anomaly_types") or [])
    metadata = case.get("metadata") or {}
    risks: list[tuple[str, str]] = []

    missing = _missing_documents(case)
    if "order" in missing:
        risks.append(("medium", "missing_order_document"))
    if missing & {"invoice", "payment", "reimbursement_form"}:
        risks.append(("high", "missing_core_document"))

    delta, ratio = amount_delta(case)
    reimbursement = money(case["reimbursement_amount"])
    invoice = money(case["invoice_amount"])
    payment = money(case["payment_amount"])
    if "over_reimbursement" in anomaly_types or reimbursement > min(invoice, payment):
        risks.append((_amount_risk(reimbursement - min(invoice, payment), (reimbursement - min(invoice, payment)) / max(min(invoice, payment), Decimal("1.00"))), "over_reimbursement"))
    elif "amount_mismatch" in anomaly_types or delta > 0:
        risks.append((_amount_risk(delta, ratio), "amount_mismatch"))

    if "date_mismatch" in anomaly_types:
        date_risks = _date_risks(case)
        risks.extend((risk, "date_mismatch") for risk in date_risks)

    if "merchant_mismatch" in anomaly_types or not _merchant_matches(case):
        risks.append(("high", "merchant_mismatch"))
    if "applicant_mismatch" in anomaly_types or not (case.get("applicant") == case.get("payer") == case.get("order_user")):
        risks.append(("high", "applicant_mismatch"))
    if "order_id_mismatch" in anomaly_types:
        risks.append(("high", "order_id_mismatch"))
    if "duplicate_in_batch" in anomaly_types:
        risks.append(("high", "duplicate_in_batch"))
    if "unreadable_image" in anomaly_types:
        unreadable_fields = set(metadata.get("unreadable_fields") or [])
        core_unreadable = bool(metadata.get("core_field_unreadable")) or bool(unreadable_fields & CORE_UNREADABLE_FIELDS)
        risks.append(("high" if core_unreadable else "medium", "unreadable_image"))

    return risks


def assign_risk_level(case: dict[str, Any]) -> str:
    """Assign the unique case-level risk level from global rules."""

    if not case.get("anomaly_types") and not _missing_documents(case):
        return "low"
    return _max_risk([risk for risk, _reason in risk_reasons(case)])


def assign_audit_result(case: dict[str, Any], risk_level: str | None = None) -> str:
    """Assign the unique audit result with the documented priority order."""

    risk = risk_level or assign_risk_level(case)
    if _missing_documents(case) or "missing_document" in set(case.get("anomaly_types") or []):
        return "missing_info"
    if risk == "high" and bool(case.get("evidence_sufficient")):
        return "reject_recommendation"
    if risk == "high" and not bool(case.get("evidence_sufficient")):
        return "manual_review"
    if risk == "medium":
        return "manual_review"
    return "pass"


def update_case_labels(case: dict[str, Any]) -> dict[str, Any]:
    """Mutate and return case with risk_level and audit_result generated centrally."""

    risk_level = assign_risk_level(case)
    case["risk_level"] = risk_level
    case["audit_result"] = assign_audit_result(case, risk_level)
    return case
