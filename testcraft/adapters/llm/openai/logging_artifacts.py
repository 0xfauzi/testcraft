"""OpenAI logging and artifact management."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenAILoggingArtifacts:
    """Handles debug logging and artifact persistence for OpenAI requests/responses."""

    def __init__(self):
        """Initialize logging artifacts manager."""
        pass

    def debug_log_request(
        self, endpoint: str, payload: dict[str, Any], max_debug_chars: int = 2000
    ) -> None:
        """Log request details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            # Avoid logging API keys/headers (we do not place them here anyway)
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(payload)

        # Truncate for console if too long
        console_content = pretty
        if len(pretty) > max_debug_chars:
            console_content = pretty[:max_debug_chars] + "\n... (truncated)"

        logger.debug(
            "\n===== LLM REQUEST (%s) =====\n%s\n===== END REQUEST =====",
            endpoint,
            console_content,
        )

        # Always artifact full payload when verbose or configured to persist
        self.artifact_request(endpoint, payload)

    def debug_log_response(
        self,
        endpoint: str,
        *,
        content: str | None = None,
        usage: dict[str, Any] | None = None,
        raw_obj: Any | None = None,
        max_debug_chars: int = 2000,
    ) -> None:
        """Log response details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        # Raw content (as-is) - truncate if too long
        if content is not None:
            console_content = content
            if len(content) > max_debug_chars:
                console_content = content[:max_debug_chars] + "\n... (truncated)"
            logger.debug(
                "\n----- LLM RAW CONTENT (%s) -----\n%s\n----- END RAW CONTENT -----",
                endpoint,
                console_content,
            )

        # Usage
        if usage:
            try:
                usage_pretty = json.dumps(usage, indent=2, ensure_ascii=False)
            except Exception:
                usage_pretty = str(usage)
            logger.debug(
                "\n----- LLM USAGE (%s) -----\n%s\n----- END USAGE -----",
                endpoint,
                usage_pretty,
            )

        # Full raw SDK response if available - truncate if too long
        if raw_obj is not None:
            try:
                if hasattr(raw_obj, "model_dump"):
                    raw_pretty = json.dumps(raw_obj.model_dump(), indent=2, ensure_ascii=False)  # type: ignore[attr-defined]
                elif hasattr(raw_obj, "dict"):
                    raw_pretty = json.dumps(raw_obj.dict(), indent=2, ensure_ascii=False)  # type: ignore[call-arg]
                else:
                    raw_pretty = str(raw_obj)
            except Exception:
                raw_pretty = str(raw_obj)

            console_raw = raw_pretty
            if len(raw_pretty) > max_debug_chars:
                console_raw = raw_pretty[:max_debug_chars] + "\n... (truncated)"
            logger.debug(
                "\n===== LLM RAW RESPONSE (%s) =====\n%s\n===== END RESPONSE =====",
                endpoint,
                console_raw,
            )

        # Always artifact full response when verbose or configured to persist
        response_data = {
            "endpoint": endpoint,
            "content": content,
            "usage": usage,
            "raw_obj": raw_obj,
        }
        self.artifact_response(endpoint, response_data)

    def should_persist_artifacts(self) -> bool:
        """Check if artifacts should be persisted to disk."""
        # Always persist in verbose mode
        if logger.isEnabledFor(logging.DEBUG):
            return True

        # Check for configuration setting - for now, default to True when verbose
        # In a full implementation, this would check config.logging.persist_verbose_artifacts
        return False

    def artifact_request(
        self, operation: str, payload: dict[str, Any], pretty_json: str | None = None
    ) -> None:
        """Write full request payload to artifacts directory.

        Backward compatible signature: accepts optional pretty_json third arg
        (ignored if provided). Tests may call with this parameter.
        """
        if not self.should_persist_artifacts():
            return

        try:
            # Convert payload to pretty JSON (ignore provided pretty_json to ensure consistency)
            try:
                pretty = json.dumps(payload, indent=2, ensure_ascii=False)
            except Exception:
                pretty = str(payload)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            artifacts_dir = Path(".artifacts/llm") / timestamp
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            request_file = artifacts_dir / "request.json"
            with open(request_file, "w", encoding="utf-8") as f:
                f.write(pretty)

        except Exception as e:
            logger.debug(f"Failed to artifact request: {e}")

    def artifact_response(
        self,
        operation: str,
        content: str | dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        raw_obj: Any | None = None,
    ) -> None:
        """Write full response data to artifacts directory.

        Backward compatible signature: tests may call with (op, content, usage, raw_obj).
        Internal callers may pass a single response dict as the second positional arg.
        """
        if not self.should_persist_artifacts():
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            artifacts_dir = Path(".artifacts/llm") / timestamp
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Support both calling conventions
            response_data: dict[str, Any]
            if isinstance(content, dict) and usage is None and raw_obj is None:
                # content actually holds the full response dict from internal callers
                payload = content
                response_data = {
                    "operation": operation,
                    "endpoint": payload.get("endpoint"),
                    "content": payload.get("content"),
                    "usage": payload.get("usage"),
                }
                raw_obj = payload.get("raw_obj")  # type: ignore[assignment]
            else:
                response_data = {
                    "operation": operation,
                    "endpoint": None,
                    "content": content,
                    "usage": usage,
                }

            if raw_obj is not None:
                try:
                    if hasattr(raw_obj, "model_dump"):
                        response_data["raw_response"] = raw_obj.model_dump()  # type: ignore[attr-defined]
                    elif hasattr(raw_obj, "dict"):
                        response_data["raw_response"] = raw_obj.dict()  # type: ignore[call-arg]
                    else:
                        response_data["raw_response"] = str(raw_obj)
                except Exception:
                    response_data["raw_response"] = str(raw_obj)

            response_file = artifacts_dir / "response.json"
            with open(response_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.debug(f"Failed to artifact response: {e}")
