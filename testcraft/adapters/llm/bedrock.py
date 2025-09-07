from __future__ import annotations

from typing import Any, Dict, Optional

from ...ports.llm_port import LLMPort
from .common import with_retries, enforce_timeout


class BedrockAdapter(LLMPort):
    def __init__(self, model_id: str = "anthropic.claude-3-haiku", timeout_s: float = 30.0):
        self.model_id = model_id
        self.timeout_s = timeout_s

    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(model_id=self.model_id, code=code_content)

        result = with_retries(call)
        return {"tests": result.get("content", ""), "metadata": {"model_id": self.model_id}}

    def analyze_code(self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(model_id=self.model_id, code=code_content)

        result = with_retries(call)
        return {"analysis": result.get("content", ""), "metadata": {"model_id": self.model_id}}

    def refine_content(self, original_content: str, refinement_instructions: str, **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(model_id=self.model_id, code=original_content)

        result = with_retries(call)
        return {"refined_content": result.get("content", ""), "metadata": {"model_id": self.model_id}}

    def _fake(self, *, model_id: str, code: str) -> Dict[str, Any]:
        import time as _time

        start = _time.time()
        enforce_timeout(start, self.timeout_s)
        return {"content": f"{model_id}: ok", "usage": {"tokens": 72}}
