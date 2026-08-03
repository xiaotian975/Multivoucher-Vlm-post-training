"""Minimal local build backend for offline editable installs.

This backend exists so phase 00 can pass `pip install -e .` in restricted
environments where build isolation cannot download setuptools. It implements
only the small PEP 660 surface needed by this skeleton project.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NAME = "multivoucher-audit"
NORMALIZED_NAME = "multivoucher_audit"
VERSION = "0.0.0"
DIST_INFO = f"{NORMALIZED_NAME}-{VERSION}.dist-info"
WHEEL_NAME = f"{NORMALIZED_NAME}-{VERSION}-py3-none-any.whl"


def _metadata_text() -> str:
    return "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            "Summary: Multi-image enterprise reimbursement audit post-training project skeleton.",
            "Requires-Python: >=3.10",
            "",
        ]
    )


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: multivoucher-audit-local-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _write_dist_info(metadata_dir: Path) -> str:
    dist_info = metadata_dir / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    return DIST_INFO


def _write_wheel(wheel_path: Path, editable: bool) -> None:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    records: list[tuple[str, str, str]] = []

    def add_bytes(archive_name: str, data: bytes) -> None:
        wheel.writestr(archive_name, data)
        records.append((archive_name, _hash_bytes(data), str(len(data))))

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as wheel:
        add_bytes(f"{DIST_INFO}/METADATA", _metadata_text().encode("utf-8"))
        add_bytes(f"{DIST_INFO}/WHEEL", _wheel_text().encode("utf-8"))

        if editable:
            pth_name = f"_{NORMALIZED_NAME}_editable.pth"
            pth_data = (os.fspath(src_path) + "\n").encode("utf-8")
            add_bytes(pth_name, pth_data)
        else:
            for file_path in (src_path / "mv_audit").rglob("*"):
                if file_path.is_file():
                    archive_name = file_path.relative_to(src_path).as_posix()
                    add_bytes(archive_name, file_path.read_bytes())

        record_name = f"{DIST_INFO}/RECORD"
        records.append((record_name, "", ""))
        record_lines = []
        for row in records:
            output = io.StringIO()
            writer = csv.writer(output, lineterminator="")
            writer.writerow(row)
            record_lines.append(output.getvalue())
        wheel.writestr(record_name, "\n".join(record_lines) + "\n")


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None) -> str:
    return _write_dist_info(Path(metadata_directory))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None) -> str:
    return _write_dist_info(Path(metadata_directory))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    wheel_path = Path(wheel_directory) / WHEEL_NAME
    _write_wheel(wheel_path, editable=False)
    return wheel_path.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    wheel_path = Path(wheel_directory) / WHEEL_NAME
    _write_wheel(wheel_path, editable=True)
    return wheel_path.name
