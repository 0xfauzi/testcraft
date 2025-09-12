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


def _fix_escape_sequences(text: str) -> str:
    """Fix common invalid escape sequences in JSON strings."""
    # Fix unescaped backslashes that aren't part of valid JSON escape sequences
    # Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX

    import re

    def fix_string_content(match):
        quote, content, end_quote = match.groups()
        if not content:
            return match.group(0)

        # Fix common invalid escape sequences
        # Replace single backslashes with double backslashes, but preserve valid JSON escapes
        result = ""
        i = 0
        while i < len(content):
            if content[i] == "\\" and i + 1 < len(content):
                next_char = content[i + 1]
                # Check if this is a valid JSON escape sequence
                if next_char in '"\\/bfnrt':
                    # Valid escape sequence, keep as is
                    result += content[i : i + 2]
                    i += 2
                elif next_char == "u" and i + 5 < len(content):
                    # Unicode escape sequence, keep as is
                    result += content[i : i + 6]
                    i += 6
                else:
                    # Invalid escape sequence, escape the backslash
                    result += "\\\\" + next_char
                    i += 2
            else:
                result += content[i]
                i += 1

        return quote + result + end_quote

    # Apply fix to strings in the JSON
    return re.sub(r'(")((?:[^"\\]|\\.)*)(")', fix_string_content, text)


def _remove_control_chars(text: str) -> str:
    """Remove control characters that can cause JSON parsing issues."""
    import re

    # Remove control characters except for valid JSON whitespace
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas that cause JSON parsing errors."""
    import re

    # Remove trailing commas before closing brackets/braces
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


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

    # Try parsing directly first
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}")

        # Try multiple repair strategies for common JSON errors
        repair_strategies = [
            # Strategy 1: Fix common escape sequence issues
            lambda x: _fix_escape_sequences(x),
            # Strategy 2: Remove invalid control characters
            lambda x: _remove_control_chars(x),
            # Strategy 3: Fix trailing commas
            lambda x: _fix_trailing_commas(x),
        ]

        for i, strategy in enumerate(repair_strategies):
            try:
                repaired = strategy(cleaned)
                result = json.loads(repaired)
                logger.debug(f"JSON repair strategy {i+1} succeeded")
                return result, None
            except Exception as repair_err:
                logger.debug(f"JSON repair strategy {i+1} failed: {repair_err}")
                continue

        # If all strategies fail, return the original error
        return None, e
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


@dataclass
class SchemaValidationResult:
    """Result of schema validation and repair attempt."""
    is_valid: bool
    data: dict[str, Any] | None
    error: str | None
    repaired: bool = False
    repair_type: str | None = None


def validate_and_repair_schema(
    data: dict[str, Any],
    required_fields: list[str],
    optional_fields: list[str] | None = None,
    field_types: dict[str, type] | None = None,
    attempt_repair: bool = True
) -> SchemaValidationResult:
    """
    Validate and optionally repair LLM response schema.
    
    Args:
        data: Parsed JSON data from LLM response
        required_fields: List of required field names
        optional_fields: List of optional field names (defaults added if missing)
        field_types: Expected types for fields (for validation/coercion)
        attempt_repair: Whether to attempt single-shot repairs
        
    Returns:
        SchemaValidationResult with validation status and potentially repaired data
    """
    optional_fields = optional_fields or []
    field_types = field_types or {}
    
    if not isinstance(data, dict):
        return SchemaValidationResult(
            is_valid=False, 
            data=None, 
            error=f"Expected dict, got {type(data).__name__}"
        )
    
    # Check for missing required fields
    missing_required = [field for field in required_fields if field not in data]
    if missing_required and not attempt_repair:
        return SchemaValidationResult(
            is_valid=False,
            data=None,
            error=f"Missing required fields: {missing_required}"
        )
    
    # Attempt repairs if enabled
    repaired_data = data.copy()
    repair_performed = False
    repair_type = None
    
    if attempt_repair:
        # Add missing required fields with defaults
        for field in missing_required:
            if field == "refined_content":
                # This is critical - can't provide default
                return SchemaValidationResult(
                    is_valid=False,
                    data=None,
                    error=f"Critical field '{field}' missing and cannot be defaulted"
                )
            elif field == "changes_made":
                repaired_data[field] = "Changes made but not specified"
                repair_performed = True
                repair_type = "added_missing_fields"
            elif field == "confidence":
                repaired_data[field] = 0.5  # Default medium confidence
                repair_performed = True
                repair_type = "added_missing_fields"
            elif field == "improvement_areas":
                repaired_data[field] = ["general"]  # Generic default
                repair_performed = True
                repair_type = "added_missing_fields"
        
        # Add missing optional fields with defaults
        for field in optional_fields:
            if field not in repaired_data:
                if field == "suspected_prod_bug":
                    repaired_data[field] = None  # Explicit null for consistency
                    repair_performed = True
                    repair_type = "added_optional_defaults"
    
    # Type validation and coercion
    for field, expected_type in field_types.items():
        if field in repaired_data:
            current_value = repaired_data[field]
            
            # Skip None values for optional fields
            if current_value is None and field in optional_fields:
                continue
                
            if not isinstance(current_value, expected_type):
                # Attempt type coercion
                try:
                    if expected_type == str:
                        repaired_data[field] = str(current_value)
                        repair_performed = True
                        repair_type = "type_coercion"
                    elif expected_type == float:
                        repaired_data[field] = float(current_value)
                        repair_performed = True
                        repair_type = "type_coercion"
                    elif expected_type == list:
                        if isinstance(current_value, str):
                            # Try to convert string to list
                            repaired_data[field] = [current_value]
                            repair_performed = True
                            repair_type = "type_coercion"
                        else:
                            repaired_data[field] = list(current_value)
                            repair_performed = True
                            repair_type = "type_coercion"
                except (ValueError, TypeError):
                    return SchemaValidationResult(
                        is_valid=False,
                        data=repaired_data if repair_performed else data,
                        error=f"Field '{field}' has invalid type: expected {expected_type.__name__}, got {type(current_value).__name__}",
                        repaired=repair_performed,
                        repair_type=repair_type
                    )
    
    # Final validation after repairs
    final_missing = [field for field in required_fields if field not in repaired_data]
    if final_missing:
        return SchemaValidationResult(
            is_valid=False,
            data=repaired_data if repair_performed else data,
            error=f"Still missing required fields after repair: {final_missing}",
            repaired=repair_performed,
            repair_type=repair_type
        )
    
    return SchemaValidationResult(
        is_valid=True,
        data=repaired_data,
        error=None,
        repaired=repair_performed,
        repair_type=repair_type
    )


def create_repair_prompt(original_error: str, missing_fields: list[str]) -> str:
    """
    Create a minimal repair prompt for schema issues.
    
    Args:
        original_error: The original validation error
        missing_fields: List of missing required fields
        
    Returns:
        Short repair instruction prompt
    """
    return f"""
Your previous response had a schema issue: {original_error}

Please provide a valid JSON response with these required fields:
{', '.join(missing_fields)}

Return ONLY the corrected JSON object, no additional text.
"""


def normalize_refinement_response(response: dict[str, Any]) -> SchemaValidationResult:
    """
    Normalize and validate refinement response from any LLM provider.
    
    This is the main entry point for refinement schema enforcement.
    
    Args:
        response: Raw response dict from LLM adapter
        
    Returns:
        SchemaValidationResult with normalized, validated response
    """
    required_fields = ["refined_content", "changes_made", "confidence", "improvement_areas"]
    optional_fields = ["suspected_prod_bug"]
    field_types = {
        "refined_content": str,
        "changes_made": str, 
        "confidence": float,
        "improvement_areas": list,
        "suspected_prod_bug": str  # Note: None values handled separately
    }
    
    return validate_and_repair_schema(
        data=response,
        required_fields=required_fields,
        optional_fields=optional_fields,
        field_types=field_types,
        attempt_repair=True
    )
