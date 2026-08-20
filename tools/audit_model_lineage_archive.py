"""Verify the local MultiVoucher-Audit adapter lineage against a hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_safetensors(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                return False, "missing_header_size"
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size <= 0 or header_size > path.stat().st_size - 8:
                return False, "invalid_header_size"
            header = json.loads(handle.read(header_size).decode("utf-8"))
            if not isinstance(header, dict) or not header:
                return False, "empty_header"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _audit_entry(root: Path, entry: dict[str, Any], known_ids: set[str]) -> dict[str, Any]:
    result = {
        "model_id": entry["model_id"],
        "role": entry["role"],
        "parent_model_id": entry.get("parent_model_id"),
        "remote_path": entry["remote_path"],
        "local_path": entry["local_path"],
        "status": "VERIFIED",
        "checks": {},
    }
    parent = entry.get("parent_model_id")
    if parent is not None and parent not in known_ids:
        result["status"] = "INVALID_LINEAGE"
        result["checks"]["parent"] = f"unknown parent: {parent}"

    adapter_dir = root / entry["local_path"]
    config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        result["status"] = "LOCAL_MISSING"
        result["checks"]["config_exists"] = config_path.is_file()
        result["checks"]["weight_exists"] = weight_path.is_file()
        return result

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result["checks"]["config_readable"] = isinstance(config, dict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["status"] = "INVALID_CONFIG"
        result["checks"]["config_error"] = f"{type(exc).__name__}: {exc}"
        return result

    config_hash = _sha256(config_path)
    weight_hash = _sha256(weight_path)
    result["checks"].update(
        {
            "config_sha256": config_hash,
            "weight_sha256": weight_hash,
            "weight_bytes": weight_path.stat().st_size,
        }
    )
    if config_hash != entry["adapter_config_sha256"] or weight_hash != entry["adapter_sha256"]:
        result["status"] = "HASH_MISMATCH"
    if weight_path.stat().st_size != int(entry["adapter_bytes"]):
        result["status"] = "SIZE_MISMATCH"

    safetensors_ok, safetensors_error = _check_safetensors(weight_path)
    result["checks"]["safetensors_header_valid"] = safetensors_ok
    if safetensors_error:
        result["checks"]["safetensors_error"] = safetensors_error
        result["status"] = "INVALID_SAFETENSORS"

    archive_path_value = entry.get("local_archive")
    archive_hash_expected = entry.get("archive_sha256")
    if archive_path_value and archive_hash_expected:
        archive_path = root / archive_path_value
        result["checks"]["archive_exists"] = archive_path.is_file()
        if not archive_path.is_file():
            result["status"] = "ARCHIVE_MISSING"
        else:
            archive_hash = _sha256(archive_path)
            result["checks"]["archive_sha256"] = archive_hash
            if archive_hash != archive_hash_expected:
                result["status"] = "ARCHIVE_HASH_MISMATCH"
    return result


def audit_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    entries = list(manifest.get("models") or [])
    known_ids = {str(entry["model_id"]) for entry in entries}
    results = [_audit_entry(root, entry, known_ids) for entry in entries]
    verified = sum(result["status"] == "VERIFIED" for result in results)
    return {
        "schema_version": "1.0",
        "source_manifest": manifest.get("manifest_id"),
        "models_total": len(results),
        "models_verified": verified,
        "all_verified": verified == len(results) and bool(results),
        "results": results,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Lineage Archive Audit",
        "",
        f"- Source manifest: {report.get('source_manifest')}",
        f"- Verified: {report['models_verified']}/{report['models_total']}",
        f"- All verified: {str(report['all_verified']).lower()}",
        "",
        "| Model | Role | Parent | Status | Weight bytes |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for result in report["results"]:
        lines.append(
            "| {model_id} | {role} | {parent} | {status} | {size} |".format(
                model_id=result["model_id"],
                role=result["role"],
                parent=result.get("parent_model_id") or "-",
                status=result["status"],
                size=result.get("checks", {}).get("weight_bytes", 0),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive.json",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_manifest(root, manifest)
    if args.output_json:
        output = (root / args.output_json).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output = (root / args.output_md).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["all_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
