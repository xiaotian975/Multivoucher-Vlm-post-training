"""Render reimbursement application form images."""

from __future__ import annotations

from typing import Any

from mv_audit.rendering.bbox_recorder import ImageSpec
from mv_audit.rendering.layout import VoucherCanvas


def render_reimbursement_form(
    case: dict[str, Any],
    image: ImageSpec,
    *,
    font_path: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Render one reimbursement form image and return the Pillow image plus bbox records."""

    metadata = case.get("metadata") or {}
    form_order_id = metadata.get("reimbursement_form_order_id", case["order_id"])

    canvas = VoucherCanvas(case=case, image=image, title="费用报销申请单", font_path=font_path)
    canvas.section("申请信息", x=64, y=122)
    canvas.value_row(label="申请人：", field="applicant", value=case["applicant"], x=84, y=176, value_x=240)
    canvas.value_row(label="费用类型：", field="expense_type", value=case["expense_type"], x=84, y=222, value_x=240)
    canvas.value_row(label="申请日期：", field="application_date", value=case["application_date"], x=84, y=268, value_x=240)
    canvas.value_row(label="订单号：", field="order_id", value=form_order_id, x=84, y=314, value_x=240)
    canvas.value_row(label="事由：", field="reason", value=f"{case['expense_type']}费用报销", x=84, y=360, value_x=240)
    canvas.amount_box(label="报销金额", field="reimbursement_amount", value=case["reimbursement_amount"], x=84, y=452)
    canvas.stamp("待审核")
    return canvas.canvas, canvas.records
