"""Input/output helpers used across project phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ensure_parent(path: Path) -> None:
    if path.parent and path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {jsonl_path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {jsonl_path}")
            yield obj


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""

    return list(iter_jsonl(path))


def write_jsonl(records: Iterable[dict[str, Any]], path: PathLike, *, append: bool = False) -> Path:
    """Write records to JSONL with UTF-8 encoding."""

    jsonl_path = Path(path)
    _ensure_parent(jsonl_path)
    mode = "a" if append else "w"
    with jsonl_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("write_jsonl expects an iterable of dictionaries")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
    return jsonl_path


def read_yaml(path: PathLike) -> dict[str, Any]:
    """Read a YAML mapping from disk."""

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {yaml_path}")
    return data


def write_yaml(data: dict[str, Any], path: PathLike) -> Path:
    """Write a YAML mapping to disk."""

    if not isinstance(data, dict):
        raise TypeError("write_yaml expects a dictionary")
    yaml_path = Path(path)
    _ensure_parent(yaml_path)
    with yaml_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return yaml_path
