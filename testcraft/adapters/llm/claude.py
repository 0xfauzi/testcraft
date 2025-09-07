from __future__ import annotations

from typing import Any, Dict, Optional

from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries, enforce_timeout


class ClaudeAdapter(LLMPort):
    def __init__(self, model: str = "claude-3-haiku", timeout_s: float = 30.0):
        self.model = model
        self.timeout_s = timeout_s

    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_completion(
                instruction=f"Generate {test_framework} tests",
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

    def analyze_code(self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_completion(instruction=f"Analyze: {analysis_type}", code=code_content, **kwargs)

        result = with_retries(call)
        return {"analysis": result.get("content", ""), "metadata": {"model": self.model}}

    def refine_content(self, original_content: str, refinement_instructions: str, **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake_completion(instruction=refinement_instructions, code=original_content, **kwargs)

        result = with_retries(call)
        return {"refined_content": result.get("content", ""), "metadata": {"model": self.model}}

    def _fake_completion(self, *, instruction: str, code: str, context: str = "", **_: Any) -> Dict[str, Any]:
        import time as _time

        start = _time.time()
        enforce_timeout(start, self.timeout_s)
        content = f"{{\"instruction\": \"{instruction}\", \"status\": \"ok\"}}"
        return {"content": content, "focus": ["paths", "edges"], "usage": {"tokens": 96}}
