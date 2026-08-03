"""Render invoice voucher images."""

from __future__ import annotations

from typing import Any

from mv_audit.rendering.bbox_recorder import ImageSpec
from mv_audit.rendering.layout import VoucherCanvas


def render_invoice(case: dict[str, Any], image: ImageSpec, *, font_path: str | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Render one invoice image and return the Pillow image plus bbox records."""

    canvas = VoucherCanvas(case=case, image=image, title="增值税电子普通发票", font_path=font_path)
    canvas.section("发票信息", x=64, y=122)
    canvas.value_row(label="发票号码：", field="invoice_id", value=case["invoice_id"], x=84, y=176, value_x=220)
    canvas.value_row(label="开票日期：", field="invoice_date", value=case["invoice_date"], x=84, y=222, value_x=220)
    canvas.value_row(label="销售方：", field="invoice_merchant", value=case["invoice_merchant"], x=84, y=268, value_x=220)
    canvas.value_row(label="项目名称：", field="expense_type", value=case["expense_type"], x=84, y=314, value_x=220)
    canvas.line(372)
    canvas.amount_box(label="价税合计", field="invoice_amount", value=case["invoice_amount"], x=84, y=410)
    canvas.amount_box(label="税额", field="tax_amount", value=case["tax_amount"], x=380, y=410)
    canvas.stamp("发票专用章")
    return canvas.canvas, canvas.records
