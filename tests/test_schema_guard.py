import json

from jsonschema import Draft202012Validator

from mv_audit.inference.schema_guard import guard_raw_output
from mv_audit.utils import read_yaml


def _valid_output() -> dict:
    return {
        "case_id": "MV_MAIN_TEST",
        "field_extraction": {
            "invoice_amount": "100.00",
            "payment_amount": "100.00",
            "reimbursement_amount": "100.00",
            "order_amount": "100.00",
            "tax_amount": "6.00",
            "invoice_merchant": "A",
            "payment_merchant": "A",
            "order_merchant": "A",
            "merchant": "A",
            "applicant": "张三",
            "payer": "张三",
            "order_user": "张三",
            "invoice_date": "2026-01-02",
            "payment_date": "2026-01-02",
            "order_date": "2026-01-01",
            "application_date": "2026-01-03",
            "invoice_id": "INV1",
            "order_id": "ORD1",
            "payment_id": "PAY1",
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
        "reason": "未检测到异常。",
        "evidence": [
            {
                "source_image_id": "MV_MAIN_TEST_invoice",
                "source_doc_type": "invoice",
                "field": "invoice_amount",
                "value": "100.00",
                "bbox": [1, 2, 3, 4],
                "evidence_text": "价税合计100.00",
            }
        ],
        "uncertainty": {
            "has_uncertain_fields": False,
            "uncertain_fields": [],
            "requires_manual_review": False,
        },
    }


def test_schema_guard_moves_flattened_field_keys_back_under_field_extraction():
    schema = read_yaml("configs/schema/output_schema.json")
    flattened = _valid_output()
    for key, value in list(flattened["field_extraction"].items()):
        flattened[key] = value
    del flattened["field_extraction"]

    guarded, meta = guard_raw_output(json.dumps(flattened, ensure_ascii=False), schema)
    parsed = json.loads(guarded)

    assert meta["changed"] is True
    assert meta["schema_valid_before"] is False
    assert meta["schema_valid_after"] is True
    assert "invoice_amount" not in parsed
    assert parsed["field_extraction"]["invoice_amount"] == "100.00"
    assert not list(Draft202012Validator(schema).iter_errors(parsed))


def test_schema_guard_does_not_claim_failed_normalization_changed_output():
    schema = read_yaml("configs/schema/output_schema.json")
    broken = {"case_id": "MV_MAIN_TEST", "invoice_amount": "100.00"}
    raw = json.dumps(broken)

    guarded, meta = guard_raw_output(raw, schema)

    assert guarded == raw
    assert meta["changed"] is False
    assert meta["schema_valid_after"] is False



def test_schema_guard_recovers_truncated_outer_json_prefix():
    schema = read_yaml("configs/schema/output_schema.json")
    valid = _valid_output()
    raw = (
        "{"
        + f'"case_id":{json.dumps(valid["case_id"])},'
        + f'"field_extraction":{json.dumps(valid["field_extraction"], ensure_ascii=False)},'
        + f'"consistency_check":{json.dumps(valid["consistency_check"])},'
        + f'"anomaly_types":{json.dumps(valid["anomaly_types"])},'
        + f'"risk_level":{json.dumps(valid["risk_level"])},'
        + f'"audit_result":{json.dumps(valid["audit_result"])},'
        + f'"reason":{json.dumps(valid["reason"], ensure_ascii=False)},'
        + '"evidence":[{"source_image_id":"MV_MAIN_TEST_invoice","field":"invoice'
    )

    guarded, meta = guard_raw_output(raw, schema)
    parsed = json.loads(guarded)

    assert meta["changed"] is True
    assert meta["partial_reconstruction"] is True
    assert meta["schema_valid_after"] is True
    assert parsed["case_id"] == "MV_MAIN_TEST"
    assert parsed["risk_level"] == "low"
    assert parsed["audit_result"] == "pass"
    assert parsed["evidence"] == []
    assert not list(Draft202012Validator(schema).iter_errors(parsed))


def test_order_id_verifier_forces_reject_when_two_evidence_values_disagree():
    schema = read_yaml("configs/schema/output_schema.json")
    output = _valid_output()
    output["evidence"] = [
        {
            "source_image_id": "MV_MAIN_TEST_order",
            "source_doc_type": "order",
            "field": "order_id",
            "value": "ORD202601010001",
            "bbox": [1, 2, 3, 4],
            "evidence_text": "订单号：ORD202601010001",
        },
        {
            "source_image_id": "MV_MAIN_TEST_reimbursement_form",
            "source_doc_type": "reimbursement_form",
            "field": "order_id",
            "value": "ORD202601010009",
            "bbox": [5, 6, 7, 8],
            "evidence_text": "订单号：ORD202601010009",
        },
    ]

    guarded, meta = guard_raw_output(json.dumps(output, ensure_ascii=False), schema)
    parsed = json.loads(guarded)

    assert meta["changed"] is True
    assert meta["order_id_verifier_checked"] is True
    assert meta["order_id_verifier_changed"] is True
    assert parsed["consistency_check"]["order_id_consistent"] is False
    assert "order_id_mismatch" in parsed["anomaly_types"]
    assert parsed["risk_level"] == "high"
    assert parsed["audit_result"] == "reject_recommendation"
    assert not list(Draft202012Validator(schema).iter_errors(parsed))


def test_order_id_verifier_leaves_matching_evidence_unchanged():
    schema = read_yaml("configs/schema/output_schema.json")
    output = _valid_output()
    output["evidence"] = [
        {
            "source_image_id": "MV_MAIN_TEST_order",
            "source_doc_type": "order",
            "field": "order_id",
            "value": "ORD202601010001",
            "bbox": [1, 2, 3, 4],
            "evidence_text": "订单号：ORD202601010001",
        },
        {
            "source_image_id": "MV_MAIN_TEST_reimbursement_form",
            "source_doc_type": "reimbursement_form",
            "field": "order_id",
            "value": "ORD202601010001",
            "bbox": [5, 6, 7, 8],
            "evidence_text": "订单号：ORD202601010001",
        },
    ]
    raw = json.dumps(output, ensure_ascii=False)

    guarded, meta = guard_raw_output(raw, schema)

    assert guarded == raw
    assert meta["changed"] is False
    assert meta["order_id_verifier_checked"] is True
    assert meta["order_id_verifier_changed"] is False


def test_order_id_verifier_requires_both_order_and_reimbursement_evidence():
    schema = read_yaml("configs/schema/output_schema.json")
    output = _valid_output()
    output["evidence"] = [
        {
            "source_image_id": "MV_MAIN_TEST_order",
            "source_doc_type": "order",
            "field": "order_id",
            "value": "ORD202601010001",
            "bbox": [1, 2, 3, 4],
            "evidence_text": "订单号：ORD202601010001",
        }
    ]
    raw = json.dumps(output, ensure_ascii=False)

    guarded, meta = guard_raw_output(raw, schema)

    assert guarded == raw
    assert meta["changed"] is False
    assert meta["order_id_verifier_checked"] is False
    assert meta["order_id_verifier_changed"] is False


def test_order_id_verifier_does_not_infer_mismatch_from_single_flattened_order_id():
    schema = read_yaml("configs/schema/output_schema.json")
    output = _valid_output()
    output["field_extraction"]["order_id"] = "ORD202601010001"
    output["evidence"] = [
        {
            "source_image_id": "MV_MAIN_TEST_invoice",
            "source_doc_type": "invoice",
            "field": "invoice_amount",
            "value": "100.00",
            "bbox": [1, 2, 3, 4],
            "evidence_text": "价税合计100.00",
        }
    ]
    raw = json.dumps(output, ensure_ascii=False)

    guarded, meta = guard_raw_output(raw, schema)

    assert guarded == raw
    assert meta["changed"] is False
    assert meta["order_id_verifier_checked"] is False
    assert meta["order_id_verifier_changed"] is False