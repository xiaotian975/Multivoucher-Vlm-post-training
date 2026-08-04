"""Parse model JSON output without over-repairing invalid generations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ParseResult:
    """Parsed model output and parse status."""

    json_validity: int
    output: dict[str, Any] | None
    error: str | None


def _json_candidates(raw_output: str) -> list[str]:
    text = raw_output.strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in FENCE_RE.finditer(raw_output))

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_output):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(raw_output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(raw_output[index : index + end])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def parse_json_output(raw_output: str) -> ParseResult:
    """Parse direct JSON, fenced JSON, or natural-language-wrapped JSON."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        return ParseResult(json_validity=0, output=None, error="empty_output")

    last_error = "no_json_object_found"
    for candidate in _json_candidates(raw_output):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if not isinstance(parsed, dict):
            last_error = "json_root_not_object"
            continue
        return ParseResult(json_validity=1, output=parsed, error=None)
    return ParseResult(json_validity=0, output=None, error=last_error)
