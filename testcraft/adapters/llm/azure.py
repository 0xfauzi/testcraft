from __future__ import annotations

from typing import Any, Dict, Optional

from ...ports.llm_port import LLMPort
from .common import with_retries, enforce_timeout


class AzureOpenAIAdapter(LLMPort):
    def __init__(self, deployment: str = "gpt-4o-mini", timeout_s: float = 30.0):
        self.deployment = deployment
        self.timeout_s = timeout_s

    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(deployment=self.deployment, code=code_content)

        result = with_retries(call)
        return {"tests": result.get("content", ""), "metadata": {"deployment": self.deployment}}

    def analyze_code(self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(deployment=self.deployment, code=code_content)

        result = with_retries(call)
        return {"analysis": result.get("content", ""), "metadata": {"deployment": self.deployment}}

    def refine_content(self, original_content: str, refinement_instructions: str, **kwargs: Any) -> Dict[str, Any]:
        def call() -> Dict[str, Any]:
            return self._fake(deployment=self.deployment, code=original_content)

        result = with_retries(call)
        return {"refined_content": result.get("content", ""), "metadata": {"deployment": self.deployment}}

    def _fake(self, *, deployment: str, code: str) -> Dict[str, Any]:
        import time as _time

        start = _time.time()
        enforce_timeout(start, self.timeout_s)
        return {"content": f"{deployment}: ok", "usage": {"tokens": 64}}
