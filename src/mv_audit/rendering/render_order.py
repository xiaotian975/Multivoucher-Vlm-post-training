"""Render order screenshot voucher images."""

from __future__ import annotations

from typing import Any

from mv_audit.rendering.bbox_recorder import ImageSpec
from mv_audit.rendering.layout import VoucherCanvas


def render_order(case: dict[str, Any], image: ImageSpec, *, font_path: str | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Render one order screenshot image and return the Pillow image plus bbox records."""

    metadata = case.get("metadata") or {}
    order_screenshot_id = metadata.get("order_screenshot_order_id", case["order_id"])

    canvas = VoucherCanvas(case=case, image=image, title="订单截图", font_path=font_path)
    canvas.section("订单详情", x=64, y=122)
    canvas.value_row(label="订单号：", field="order_id", value=order_screenshot_id, x=84, y=176, value_x=240)
    canvas.value_row(label="下单日期：", field="order_date", value=case["order_date"], x=84, y=222, value_x=240)
    canvas.value_row(label="商户：", field="order_merchant", value=case["order_merchant"], x=84, y=268, value_x=240)
    canvas.value_row(label="订单用户：", field="order_user", value=case["order_user"], x=84, y=314, value_x=240)
    canvas.value_row(label="商品/服务：", field="expense_type", value=case["expense_type"], x=84, y=360, value_x=240)
    canvas.amount_box(label="订单金额", field="order_amount", value=case["order_amount"], x=84, y=452)
    canvas.stamp("订单完成")
    return canvas.canvas, canvas.records
