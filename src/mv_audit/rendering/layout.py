"""Shared Pillow layout primitives for phase 04 voucher rendering."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mv_audit.rendering.bbox_recorder import BBox, ImageSpec, make_bbox_record


CANVAS_SIZE = (1000, 700)
BACKGROUND = (250, 252, 255)
INK = (24, 32, 43)
MUTED = (92, 101, 116)
LINE = (182, 194, 210)
ACCENT = (24, 113, 158)
PANEL = (255, 255, 255)


FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def resolve_font_path(font_path: str | None = None) -> str | None:
    """Resolve a local font path without bundling font assets."""

    if font_path:
        candidate = Path(font_path)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Configured font file does not exist: {font_path}")

    env_path = os.environ.get("MV_AUDIT_FONT_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"MV_AUDIT_FONT_PATH does not exist: {env_path}")

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def load_font(size: int, *, font_path: str | None = None) -> ImageFont.ImageFont:
    """Load a TrueType font with fallback to PIL's default bitmap font."""

    resolved = resolve_font_path(font_path)
    if resolved:
        return ImageFont.truetype(resolved, size=size)
    return ImageFont.load_default()


class VoucherCanvas:
    """Small helper that draws labels and records value bounding boxes."""

    def __init__(self, *, case: dict[str, Any], image: ImageSpec, title: str, font_path: str | None = None) -> None:
        self.case = case
        self.image = image
        self.font_path = font_path
        self.canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
        self.draw = ImageDraw.Draw(self.canvas)
        self.title_font = load_font(34, font_path=font_path)
        self.section_font = load_font(22, font_path=font_path)
        self.label_font = load_font(20, font_path=font_path)
        self.value_font = load_font(22, font_path=font_path)
        self.small_font = load_font(15, font_path=font_path)
        self.records: list[dict[str, Any]] = []
        self._draw_shell(title)

    def _draw_shell(self, title: str) -> None:
        self.draw.rounded_rectangle((32, 28, 968, 672), radius=10, fill=PANEL, outline=LINE, width=2)
        self.draw.rectangle((32, 28, 968, 92), fill=(230, 244, 250), outline=LINE)
        self.draw.text((64, 44), title, fill=INK, font=self.title_font)
        self.draw.text((720, 52), self.case["case_id"], fill=MUTED, font=self.small_font)

    def line(self, y: int) -> None:
        self.draw.line((64, y, 936, y), fill=LINE, width=1)

    def stamp(self, text: str, *, x: int = 780, y: int = 590) -> None:
        self.draw.rounded_rectangle((x, y, x + 132, y + 48), radius=6, outline=(198, 47, 47), width=3)
        self.draw.text((x + 18, y + 13), text, fill=(198, 47, 47), font=self.label_font)

    def section(self, text: str, *, x: int, y: int) -> None:
        self.draw.text((x, y), text, fill=ACCENT, font=self.section_font)
        self.draw.line((x, y + 34, x + 840, y + 34), fill=(202, 220, 230), width=1)

    def value_row(self, *, label: str, field: str, value: Any, x: int, y: int, value_x: int = 220) -> None:
        value_text = "" if value is None else str(value)
        self.draw.text((x, y), label, fill=MUTED, font=self.label_font)
        bbox = self._draw_value(value_text, value_x, y - 2)
        self.records.append(
            make_bbox_record(
                image=self.image,
                field=field,
                value=value_text,
                bbox_abs=bbox,
                evidence_text=f"{label}{value_text}",
            )
        )

    def amount_box(self, *, label: str, field: str, value: Any, x: int, y: int) -> None:
        self.draw.rounded_rectangle((x, y, x + 250, y + 74), radius=8, fill=(246, 250, 252), outline=LINE)
        self.draw.text((x + 18, y + 12), label, fill=MUTED, font=self.label_font)
        bbox = self._draw_value(str(value), x + 18, y + 38)
        self.records.append(
            make_bbox_record(
                image=self.image,
                field=field,
                value=value,
                bbox_abs=bbox,
                evidence_text=f"{label}{value}",
            )
        )

    def _draw_value(self, text: str, x: int, y: int) -> BBox:
        self.draw.text((x, y), text, fill=INK, font=self.value_font)
        left, top, right, bottom = self.draw.textbbox((x, y), text, font=self.value_font)
        return (int(left), int(top), int(right), int(bottom))
