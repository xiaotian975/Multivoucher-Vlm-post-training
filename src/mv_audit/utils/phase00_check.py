"""Phase 00 skeleton validation entrypoint.

This module is intentionally limited to package, directory, and utility checks.
It must not call model, data generation, rendering, evaluation, or training
logic.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mv_audit
from mv_audit.utils import (
    ensure_dir,
    read_jsonl,
    read_yaml,
    set_random_seed,
    setup_logging,
    write_jsonl,
    write_yaml,
)


REQUIRED_DIRS = [
    "configs/data_gen",
    "configs/train",
    "configs/eval",
    "configs/schema",
    "configs/model",
    "data/mv_audit/dictionaries",
    "data/mv_audit/templates",
    "data/mv_audit/raw_cases",
    "data/mv_audit/images",
    "data/mv_audit/annotations",
    "data/mv_audit/sft",
    "data/mv_audit/dpo",
    "data/mv_audit/grpo",
    "data/mv_audit/eval_sets",
    "src/mv_audit/data_gen",
    "src/mv_audit/rendering",
    "src/mv_audit/perturbation",
    "src/mv_audit/converters",
    "src/mv_audit/training",
    "src/mv_audit/inference",
    "src/mv_audit/evaluation",
    "src/mv_audit/utils",
    "scripts",
    "outputs/checkpoints",
    "outputs/predictions",
    "outputs/eval_reports",
    "outputs/logs",
    "notebooks",
]


def check_required_dirs() -> None:
    """Fail if any phase 00 directory is missing."""

    missing = [path for path in REQUIRED_DIRS if not Path(path).is_dir()]
    if missing:
        raise RuntimeError(f"Missing phase 00 directories: {missing}")


def check_utilities() -> None:
    """Validate basic JSONL, YAML, directory, seed, and logging helpers."""

    set_random_seed(42)
    logger = setup_logging(logger_name="mv_audit.debug")
    logger.info("MultiVoucher-Audit %s phase 00 skeleton is importable.", mv_audit.__version__)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ensure_dir(tmp_path / "nested")
        jsonl_path = tmp_path / "sample.jsonl"
        yaml_path = tmp_path / "sample.yaml"

        write_jsonl([{"case_id": "MV_DEBUG_000001", "ok": True}], jsonl_path)
        write_yaml({"seed": 42, "phase": "00"}, yaml_path)

        if read_jsonl(jsonl_path)[0]["case_id"] != "MV_DEBUG_000001":
            raise AssertionError("JSONL utility roundtrip failed")
        if read_yaml(yaml_path)["phase"] != "00":
            raise AssertionError("YAML utility roundtrip failed")

    logger.info("Phase 00 debug utility check passed.")


def main() -> None:
    """Run phase 00 validation checks."""

    check_required_dirs()
    check_utilities()


if __name__ == "__main__":
    main()
