"""
Common utilities for LLM adapters: response parsing, validation, retries, and limits.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n|\n```$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences from a string."""
    return CODE_FENCE_RE.sub("", text).strip()


def balance_braces(text: str) -> str:
    """Best-effort brace balancing for JSON-like outputs."""
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)
    return text


def try_parse_json(text: str) -> tuple[dict[str, Any] | None, Exception | None]:
    cleaned = strip_code_fences(balance_braces(text))
    try:
        return json.loads(cleaned), None
    except Exception as e:
        return None, e


def with_retries(
    func: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    jitter: float = 0.2,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:  # pragma: no cover - timing variability
            last_err = e
            sleep_s = base_delay * (2**attempt)
            sleep_s += jitter * (attempt + 1)
            logger.warning(
                "LLM call failed (attempt %s/%s): %s", attempt + 1, retries, e
            )
            time.sleep(min(5.0, sleep_s))
    assert last_err is not None
    raise last_err


@dataclass
class ParsedResponse:
    success: bool
    data: dict[str, Any] | None
    raw: str
    error: str | None = None


def normalize_output(text: str) -> str:
    text = strip_code_fences(text)
    # Replace HTML entities and common escapes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.strip()


def parse_json_response(text: str) -> ParsedResponse:
    norm = normalize_output(text)
    data, err = try_parse_json(norm)
    if data is not None:
        return ParsedResponse(success=True, data=data, raw=text)
    # Minimal repair: take substring from first '{' and attempt balanced parse
    brace_start = norm.find("{")
    if brace_start != -1:
        candidate = norm[brace_start:]
        data2, err2 = try_parse_json(candidate)
        if data2 is not None:
            return ParsedResponse(success=True, data=data2, raw=text, error=str(err))
        err = err2
    return ParsedResponse(success=False, data=None, raw=text, error=str(err))


def enforce_timeout(start_time: float, timeout_s: float) -> None:
    if timeout_s <= 0:
        return
    if time.time() - start_time > timeout_s:
        raise TimeoutError("LLM operation timed out")
