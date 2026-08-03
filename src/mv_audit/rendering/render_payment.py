"""Render payment screenshot voucher images."""

from __future__ import annotations

from typing import Any

from mv_audit.rendering.bbox_recorder import ImageSpec
from mv_audit.rendering.layout import VoucherCanvas


def render_payment(case: dict[str, Any], image: ImageSpec, *, font_path: str | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Render one payment screenshot image and return the Pillow image plus bbox records."""

    canvas = VoucherCanvas(case=case, image=image, title="企业支付凭证", font_path=font_path)
    canvas.section("支付明细", x=64, y=122)
    canvas.value_row(label="支付流水号：", field="payment_id", value=case["payment_id"], x=84, y=176, value_x=240)
    canvas.value_row(label="支付日期：", field="payment_date", value=case["payment_date"], x=84, y=222, value_x=240)
    canvas.value_row(label="收款方：", field="payment_merchant", value=case["payment_merchant"], x=84, y=268, value_x=240)
    canvas.value_row(label="付款人：", field="payer", value=case["payer"], x=84, y=314, value_x=240)
    canvas.amount_box(label="支付金额", field="payment_amount", value=case["payment_amount"], x=84, y=390)
    canvas.stamp("支付成功")
    return canvas.canvas, canvas.records
