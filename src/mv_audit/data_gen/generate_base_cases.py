"""Generate normal structured base cases for MultiVoucher-Audit phase 02."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from mv_audit.data_gen.case_validator import load_schema, validate_cases
from mv_audit.utils import load_config, read_yaml, write_jsonl


DATE_FORMAT = "%Y-%m-%d"
DEFAULT_TEMPLATE_GROUP = "train"


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _random_money(rng: random.Random, minimum: int | float, maximum: int | float) -> str:
    cents_min = int(Decimal(str(minimum)) * 100)
    cents_max = int(Decimal(str(maximum)) * 100)
    cents = rng.randint(cents_min, cents_max)
    return _money(Decimal(cents) / Decimal(100))


def _build_names(names_config: Any) -> list[str]:
    if isinstance(names_config, list):
        return [str(name) for name in names_config]
    if isinstance(names_config, dict):
        surnames = names_config.get("surnames", [])
        given_names = names_config.get("given_names", [])
        names = [f"{surname}{given}" for surname in surnames for given in given_names]
        if names:
            return names
    raise ValueError("names.json must be a list or contain surnames/given_names")


def _build_merchants(merchants_config: Any, cities: list[str]) -> list[dict[str, Any]]:
    if isinstance(merchants_config, list):
        return merchants_config
    if not isinstance(merchants_config, dict):
        raise ValueError("merchants.json must be a list or template object")

    brands = merchants_config.get("brands", [])
    templates = merchants_config.get("merchant_templates", [])
    merchants: list[dict[str, Any]] = []
    for city in cities:
        for brand in brands:
            for template in templates:
                categories = template.get("categories", [])
                for category in categories:
                    values = {"city": city, "brand": brand, "category": category}
                    canonical = template["canonical"].format(**values)
                    aliases = [alias.format(**values) for alias in template.get("aliases", [])]
                    merchants.append(
                        {
                            "canonical": canonical,
                            "aliases": aliases,
                            "city": city,
                            "category": category,
                        }
                    )
    if not merchants:
        raise ValueError("merchants.json produced no merchants")
    return merchants


def load_dictionaries(config: dict[str, Any]) -> dict[str, Any]:
    """Load and normalize dictionary files."""

    dictionary_paths = config["dictionaries"]
    cities = _read_json(dictionary_paths["cities"])
    if not isinstance(cities, list) or not cities:
        raise ValueError("cities dictionary must be a non-empty list")

    expense_types = _read_json(dictionary_paths["expense_types"])
    if not isinstance(expense_types, list) or not expense_types:
        raise ValueError("expense_types dictionary must be a non-empty list")

    return {
        "names": _build_names(_read_json(dictionary_paths["names"])),
        "merchants": _build_merchants(_read_json(dictionary_paths["merchants"]), [str(city) for city in cities]),
        "expense_types": expense_types,
        "cities": [str(city) for city in cities],
    }


def _sample_dates(rng: random.Random, start_date: str, end_date: str) -> tuple[str, str, str, str]:
    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)
    max_offset = max((end - start).days - 21, 0)
    order_date = start + timedelta(days=rng.randint(0, max_offset))
    payment_date = order_date + timedelta(days=rng.randint(0, 3))
    invoice_date = payment_date + timedelta(days=rng.randint(0, 5))
    application_date = invoice_date + timedelta(days=rng.randint(0, 14))
    return (
        order_date.strftime(DATE_FORMAT),
        payment_date.strftime(DATE_FORMAT),
        invoice_date.strftime(DATE_FORMAT),
        application_date.strftime(DATE_FORMAT),
    )


def _make_id(prefix: str, date_value: str, sequence: int) -> str:
    return f"{prefix}{date_value.replace('-', '')}{sequence:06d}"


def generate_case(
    *,
    sequence: int,
    split_name: str,
    config: dict[str, Any],
    dictionaries: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one normal base case."""

    base_config = config["base_case"]
    applicant = rng.choice(dictionaries["names"])
    expense_type = rng.choice(dictionaries["expense_types"])
    amount = _random_money(rng, expense_type["min_amount"], expense_type["max_amount"])
    tax_amount = _money(Decimal(amount) * Decimal(str(base_config.get("tax_rate", 0.06))))

    merchants = [m for m in dictionaries["merchants"] if m.get("category") == expense_type["name"]]
    merchant = rng.choice(merchants or dictionaries["merchants"])
    merchant_aliases = list(dict.fromkeys(str(alias) for alias in merchant.get("aliases", [])))
    merchant_text_options = [merchant["canonical"], *merchant_aliases]

    order_date, payment_date, invoice_date, application_date = _sample_dates(
        rng,
        base_config["start_date"],
        base_config["end_date"],
    )

    case_prefix = str(base_config.get("id_prefix", "MV_DEBUG"))
    template_group = "train"
    if split_name.startswith("val"):
        template_group = "val"
    elif split_name.startswith("test_unseen"):
        template_group = "strong_generalization_test"
    elif split_name.startswith("test"):
        template_group = "standard_test"

    return {
        "case_id": f"{case_prefix}_{sequence:06d}",
        "applicant": applicant,
        "payer": applicant,
        "order_user": applicant,
        "merchant_canonical": merchant["canonical"],
        "invoice_merchant": merchant["canonical"],
        "payment_merchant": rng.choice(merchant_text_options),
        "order_merchant": rng.choice(merchant_text_options),
        "merchant_aliases": merchant_aliases,
        "expense_type": expense_type["name"],
        "invoice_amount": amount,
        "payment_amount": amount,
        "reimbursement_amount": amount,
        "order_amount": amount,
        "tax_amount": tax_amount,
        "order_date": order_date,
        "payment_date": payment_date,
        "invoice_date": invoice_date,
        "application_date": application_date,
        "invoice_id": _make_id("INV", invoice_date, sequence),
        "order_id": _make_id("ORD", order_date, sequence),
        "payment_id": _make_id("PAY", payment_date, sequence),
        "documents": list(base_config.get("default_documents", ["invoice", "payment", "reimbursement_form", "order"])),
        "primary_anomaly_type": "none",
        "anomaly_types": [],
        "evidence_sufficient": True,
        "risk_level": "low",
        "audit_result": "pass",
        "metadata": {
            "split_name": split_name,
            "source": "phase_02_base_generator",
            "template_group": template_group,
            "seed": int(config.get("seed", 42)),
            "city": merchant.get("city"),
            "merchant_category": merchant.get("category"),
        },
    }


def generate_cases(
    *,
    config: dict[str, Any],
    dictionaries: dict[str, Any],
    num_cases: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Generate a deterministic list of normal base cases."""

    rng = random.Random(int(config.get("seed", 42)))
    return [
        generate_case(
            sequence=index,
            split_name=split_name,
            config=config,
            dictionaries=dictionaries,
            rng=rng,
        )
        for index in range(1, num_cases + 1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate phase 02 normal base cases.")
    parser.add_argument("--config", required=True, help="Path to data generation YAML config.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    parser.add_argument("--num_cases", type=int, default=None, help="Override number of cases to generate.")
    parser.add_argument("--split_name", default="debug", help="Logical split name stored in metadata.")
    parser.add_argument("--schema", default="configs/schema/case_schema.json", help="Path to case schema.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    num_cases = args.num_cases if args.num_cases is not None else int(config.get("train_cases", 2000))
    if num_cases <= 0:
        raise ValueError("--num_cases must be positive")

    dictionaries = load_dictionaries(config)
    cases = generate_cases(config=config, dictionaries=dictionaries, num_cases=num_cases, split_name=args.split_name)
    schema = load_schema(args.schema)
    validate_cases(cases, schema=schema, require_normal=True)
    write_jsonl(cases, args.output)
    print(f"generated_cases={len(cases)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
