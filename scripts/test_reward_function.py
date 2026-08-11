"""Lightweight Phase 08 reward-function tests."""

from __future__ import annotations

import json
from copy import deepcopy

from mv_audit.training.reward_function import normalize_group_rewards, score_output, summarize_reward_outputs
from mv_audit.utils import read_yaml


SCHEMA = read_yaml("configs/schema/output_schema.json")


def _truth() -> dict:
    return {
        "case_id": "CASE_001",
        "field_extraction": {
            "invoice_amount": "120.00",
            "payment_amount": "120.00",
            "reimbursement_amount": "120.00",
            "order_amount": "120.00",
            "tax_amount": "7.20",
            "invoice_merchant": "福州示例有限公司",
            "payment_merchant": "福州示例有限公司",
            "order_merchant": "福州示例有限公司",
            "merchant": "福州示例有限公司",
            "applicant": "张三",
            "payer": "张三",
            "order_user": "张三",
            "invoice_date": "2026-01-02",
            "payment_date": "2026-01-02",
            "order_date": "2026-01-01",
            "application_date": "2026-01-03",
            "invoice_id": "INV001",
            "order_id": "ORD001",
            "payment_id": "PAY001",
            "expense_type": "办公用品",
        },
        "consistency_check": {
            "amount_consistent": True,
            "merchant_consistent": True,
            "person_consistent": True,
            "date_reasonable": True,
            "order_id_consistent": True,
            "payment_id_present": True,
            "document_complete": True,
            "duplicate_in_batch": False,
        },
        "anomaly_types": [],
        "risk_level": "low",
        "audit_result": "pass",
        "reason": "材料完整且一致，建议通过。",
        "evidence": [
            {
                "source_image_id": "CASE_001_invoice",
                "source_doc_type": "invoice",
                "field": "invoice_amount",
                "value": "120.00",
                "bbox": [100, 100, 200, 140],
                "evidence_text": "价税合计120.00",
            }
        ],
        "uncertainty": {
            "has_uncertain_fields": False,
            "uncertain_fields": [],
            "requires_manual_review": False,
        },
    }


def _images() -> list[dict]:
    return [{"image_id": "CASE_001_invoice", "doc_type": "invoice", "image_path": "unused.png"}]


def _score(output: dict | str, truth: dict | None = None) -> dict:
    raw = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    return score_output(raw, truth or _truth(), _images(), SCHEMA)


def test_perfect_answer() -> None:
    result = _score(_truth())
    assert result["reward"] >= 0.99, result


def test_invalid_json() -> None:
    result = _score('{"case_id": "broken", "audit_result": pass')
    assert result["reward"] == -1.0, result


def test_high_risk_pass() -> None:
    truth = _truth()
    truth["risk_level"] = "high"
    truth["audit_result"] = "reject_recommendation"
    truth["anomaly_types"] = ["merchant_mismatch"]
    pred = deepcopy(truth)
    pred["risk_level"] = "low"
    pred["audit_result"] = "pass"
    result = _score(pred, truth)
    assert result["reward"] == -1.0, result


def test_missing_document_pass() -> None:
    truth = _truth()
    truth["risk_level"] = "high"
    truth["audit_result"] = "missing_info"
    truth["anomaly_types"] = ["missing_document"]
    truth["consistency_check"]["document_complete"] = False
    pred = deepcopy(truth)
    pred["risk_level"] = "low"
    pred["audit_result"] = "pass"
    pred["anomaly_types"] = []
    pred["consistency_check"]["document_complete"] = True
    result = _score(pred, truth)
    assert result["reward"] == -1.0, result


def test_unreadable_guess_penalty() -> None:
    truth = _truth()
    truth["risk_level"] = "high"
    truth["audit_result"] = "manual_review"
    truth["anomaly_types"] = ["unreadable_image"]
    truth["field_extraction"]["payment_amount"] = None
    truth["uncertainty"] = {
        "has_uncertain_fields": True,
        "uncertain_fields": ["payment_amount"],
        "requires_manual_review": True,
    }
    pred = deepcopy(truth)
    pred["field_extraction"]["payment_amount"] = "120.00"
    pred["uncertainty"] = {
        "has_uncertain_fields": False,
        "uncertain_fields": [],
        "requires_manual_review": False,
    }
    pred["evidence"].append(
        {
            "source_image_id": "CASE_001_invoice",
            "source_doc_type": "invoice",
            "field": "payment_amount",
            "value": "120.00",
            "bbox": [100, 100, 200, 140],
            "evidence_text": "猜测支付金额120.00",
        }
    )
    result = _score(pred, truth)
    assert result["details"]["p_hallucination"] > 0, result
    assert result["details"]["r_uncertainty"] < 1, result


def test_wrong_evidence_source_penalty() -> None:
    pred = deepcopy(_truth())
    pred["evidence"][0]["source_image_id"] = "wrong_image"
    result = _score(pred)
    assert result["details"]["r_evidence"] < 1, result
    assert result["reward"] < 1, result


def test_reward_summary_and_group_normalization() -> None:
    perfect = _score(_truth())
    invalid = _score('{"case_id": "broken", "audit_result": pass')
    summary = summarize_reward_outputs([perfect, invalid])
    assert summary["count"] == 2.0, summary
    assert summary["json_valid_rate"] == 0.5, summary
    normalized = normalize_group_rewards([1.0, 0.0, -1.0])
    assert len(normalized) == 3, normalized
    assert abs(sum(normalized)) < 1e-6, normalized
    assert normalize_group_rewards([0.5, 0.5]) == [0.0, 0.0]


def main() -> None:
    tests = [
        test_perfect_answer,
        test_invalid_json,
        test_high_risk_pass,
        test_missing_document_pass,
        test_unreadable_guess_penalty,
        test_wrong_evidence_source_penalty,
        test_reward_summary_and_group_normalization,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}=ok")
    print("reward_function_tests=ok")


if __name__ == "__main__":
    main()
