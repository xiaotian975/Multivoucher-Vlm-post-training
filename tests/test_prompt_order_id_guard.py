from mv_audit.converters.common import build_prompt


def test_build_prompt_requires_cross_document_order_id_evidence():
    case = {"case_id": "MV_MAIN_TEST"}
    image_items = [
        {"image_id": "MV_MAIN_TEST_order", "doc_type": "order", "image_path": "order.png"},
        {
            "image_id": "MV_MAIN_TEST_reimbursement_form",
            "doc_type": "reimbursement_form",
            "image_path": "form.png",
        },
    ]

    prompt = build_prompt(case, image_items, task_instruction="完成多凭证一致性审核。")

    assert "分别读取订单截图(order)和报销申请单(reimbursement_form)中的 order_id" in prompt
    assert "consistency_check.order_id_consistent=false" in prompt
    assert "order_id_mismatch" in prompt
    assert "risk_level=high" in prompt
    assert "audit_result=reject_recommendation" in prompt
    assert "同时给出这两处 order_id 证据" in prompt