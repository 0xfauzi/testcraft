from __future__ import annotations

from typing import Any, Dict, Optional

from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries, enforce_timeout


class OpenAIAdapter(LLMPort):
    """
    Minimal OpenAI adapter skeleton.

    Note: This implementation intentionally avoids real network calls and is
    designed to be mocked in tests. It demonstrates common plumbing: retries,
    timeouts, and JSON response parsing.
    """

    def __init__(self, model: str = "gpt-4o-mini", timeout_s: float = 30.0):
        self.model = model
        self.timeout_s = timeout_s

    # --- public API ---
    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_chat_completion(
                prompt=f"Generate {test_framework} tests for given code",
                code=code_content,
                context=context or "",
                **kwargs,
            )

        result = with_retries(call)
        resp = parse_json_response(result["content"])  # type: ignore[index]
        return {
            "tests": result.get("content", ""),
            "coverage_focus": result.get("focus", []),
            "confidence": 0.5 if not resp.success else 0.8,
            "metadata": {"model": self.model, "parsed": resp.success},
        }

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_chat_completion(
                prompt=f"Analyze code: {analysis_type}", code=code_content, **kwargs
            )

        result = with_retries(call)
        return {"analysis": result.get("content", ""), "metadata": {"model": self.model}}

    def refine_content(
        self, original_content: str, refinement_instructions: str, **kwargs: Any
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_chat_completion(
                prompt=refinement_instructions, code=original_content, **kwargs
            )

        result = with_retries(call)
        return {"refined_content": result.get("content", ""), "metadata": {"model": self.model}}

    # --- internal helpers ---
    def _fake_chat_completion(self, *, prompt: str, code: str, context: str = "", **_: Any) -> Dict[str, Any]:
        # Emulate time-bound behavior
        import time as _time

        start = _time.time()
        enforce_timeout(start, self.timeout_s)
        # Minimal deterministic content
        content = f"{{\"prompt\": \"{prompt}\", \"summary\": \"ok\"}}"
        return {"content": content, "focus": ["functions", "branches"], "usage": {"tokens": 128}}
