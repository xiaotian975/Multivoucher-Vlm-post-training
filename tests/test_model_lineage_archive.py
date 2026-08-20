from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from tools.audit_model_lineage_archive import audit_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter(root: Path) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    header = json.dumps({"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    return adapter


def _manifest(root: Path, adapter: Path) -> dict:
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    return {
        "manifest_id": "test-lineage",
        "models": [
            {
                "model_id": "m2",
                "role": "BASELINE",
                "parent_model_id": None,
                "remote_path": "remote/m2",
                "local_path": adapter.relative_to(root).as_posix(),
                "adapter_config_sha256": _sha256(config),
                "adapter_sha256": _sha256(weights),
                "adapter_bytes": weights.stat().st_size,
            }
        ],
    }


def test_archive_audit_accepts_valid_adapter(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    report = audit_manifest(tmp_path, _manifest(tmp_path, adapter))
    assert report["all_verified"] is True
    assert report["results"][0]["status"] == "VERIFIED"


def test_archive_audit_reports_hash_mismatch(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter)
    manifest["models"][0]["adapter_sha256"] = "0" * 64
    report = audit_manifest(tmp_path, manifest)
    assert report["all_verified"] is False
    assert report["results"][0]["status"] == "HASH_MISMATCH"


def test_archive_audit_reports_missing_adapter(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter)
    manifest["models"][0]["local_path"] = "missing"
    report = audit_manifest(tmp_path, manifest)
    assert report["results"][0]["status"] == "LOCAL_MISSING"
