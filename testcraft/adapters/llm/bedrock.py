"""Real AWS Bedrock adapter implementation using LangChain ChatBedrock."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from ...config.credentials import CredentialManager, CredentialError
from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries

logger = logging.getLogger(__name__)


class BedrockError(Exception):
    """Bedrock adapter specific errors."""
    pass


class BedrockAdapter(LLMPort):
    """
    Production AWS Bedrock adapter using LangChain ChatBedrock.

    Features:
    - Secure credential management via AWS credentials
    - Proper error handling and retries with exponential backoff
    - Support for Anthropic Claude models on Bedrock
    - LangChain ChatBedrock integration for consistency
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region_name: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: Optional[CredentialManager] = None,
        **kwargs: Any,
    ):
        """Initialize Bedrock adapter.
        
        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
            region_name: AWS region name (defaults to credentials or environment)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0-1.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            **kwargs: Additional ChatBedrock parameters
        """
        self.model_id = model_id
        self.region_name = region_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize credential manager
        self.credential_manager = credential_manager or CredentialManager()
        
        # Initialize ChatBedrock client
        self._client: Optional[ChatBedrock] = None
        self._initialize_client(**kwargs)

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize the ChatBedrock client with credentials."""
        try:
            credentials = self.credential_manager.get_provider_credentials('bedrock')
            
            client_kwargs = {
                'model_id': self.model_id,
                'model_kwargs': {
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                },
                'region_name': self.region_name or credentials['region_name'],
                'aws_access_key_id': credentials['aws_access_key_id'],
                'aws_secret_access_key': credentials['aws_secret_access_key'],
                **kwargs
            }
            
            self._client = ChatBedrock(**client_kwargs)
            
            logger.info(f"ChatBedrock client initialized with model: {self.model_id}")
            
        except CredentialError as e:
            raise BedrockError(f"Failed to initialize ChatBedrock client: {e}") from e
        except Exception as e:
            raise BedrockError(f"Unexpected error initializing ChatBedrock client: {e}") from e

    @property
    def client(self) -> ChatBedrock:
        """Get the ChatBedrock client, initializing if needed."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate test cases for the provided code content."""
        
        system_message = f"""You are an expert Python test generator specializing in {test_framework} tests. Your task is to generate comprehensive, production-ready test cases for the provided Python code.

Requirements:
- Use {test_framework} testing framework
- Generate tests covering normal usage, edge cases, and error conditions
- Include appropriate fixtures, mocks, and test data as needed
- Focus on achieving high code coverage and testing all logical paths
- Write clean, readable, and maintainable test code
- Include descriptive test method names and docstrings where helpful

Please return your response as valid JSON in this exact format:
{{
  "tests": "# Your complete test code here",
  "coverage_focus": ["list", "of", "specific", "areas", "to", "test"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of your test strategy and approach"
}}

Generate thorough, professional-quality test code."""

        user_content = f"Code to generate tests for:\n\n```python\n{code_content}\n```"
        
        if context:
            user_content += f"\n\nAdditional context:\n{context}"

        def call() -> Dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")
            
            # Parse JSON response
            parsed = parse_json_response(content)
            
            if parsed.success and parsed.data:
                return {
                    "tests": parsed.data.get("tests", content),
                    "coverage_focus": parsed.data.get("coverage_focus", []),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": {
                        "model_id": self.model_id,
                        "parsed": True,
                        "reasoning": parsed.data.get("reasoning", ""),
                        **result.get("usage", {})
                    }
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": {
                        "model_id": self.model_id,
                        "parsed": False,
                        "parse_error": parsed.error,
                        **result.get("usage", {})
                    }
                }
                
        except Exception as e:
            logger.error(f"Bedrock test generation failed: {e}")
            raise BedrockError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> Dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""
        
        system_message = f"""You are an expert Python code analyst. Perform a {analysis_type} analysis of the provided code to assess its quality, testability, and potential issues.

Your analysis should cover:
- Testability assessment (score from 0-10, where 10 is most testable)
- Code complexity metrics (cyclomatic complexity, nesting depth, function count, etc.)
- Specific recommendations for improving testability and code quality
- Identification of potential issues, code smells, or anti-patterns
- Assessment of dependencies, coupling, and overall architecture

Please return your analysis as valid JSON in this exact format:
{{
  "testability_score": 8.5,
  "complexity_metrics": {{
    "cyclomatic_complexity": 5,
    "nesting_depth": 3,
    "function_count": 10,
    "lines_of_code": 150
  }},
  "recommendations": ["specific suggestion 1", "specific suggestion 2"],
  "potential_issues": ["identified issue 1", "identified issue 2"],
  "analysis_summary": "Brief overall summary of findings and key insights"
}}

Provide actionable, specific recommendations."""

        user_content = f"Code to analyze:\n\n```python\n{code_content}\n```"

        def call() -> Dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")
            
            # Parse JSON response
            parsed = parse_json_response(content)
            
            if parsed.success and parsed.data:
                return {
                    "testability_score": parsed.data.get("testability_score", 5.0),
                    "complexity_metrics": parsed.data.get("complexity_metrics", {}),
                    "recommendations": parsed.data.get("recommendations", []),
                    "potential_issues": parsed.data.get("potential_issues", []),
                    "metadata": {
                        "model_id": self.model_id,
                        "analysis_type": analysis_type,
                        "parsed": True,
                        "summary": parsed.data.get("analysis_summary", ""),
                        **result.get("usage", {})
                    }
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    "testability_score": 5.0,
                    "complexity_metrics": {},
                    "recommendations": [],
                    "potential_issues": [],
                    "metadata": {
                        "model_id": self.model_id,
                        "analysis_type": analysis_type,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {})
                    }
                }
                
        except Exception as e:
            logger.error(f"Bedrock code analysis failed: {e}")
            raise BedrockError(f"Code analysis failed: {e}") from e

    def refine_content(
        self, original_content: str, refinement_instructions: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Refine existing content based on specific instructions."""
        
        system_message = """You are an expert Python developer with deep knowledge of testing best practices, code quality, and software engineering principles. Your task is to refine the provided content according to the specific instructions given.

Focus on:
- Improving code quality, readability, and maintainability
- Fixing any bugs, issues, or potential problems
- Enhancing test coverage and test quality
- Following Python best practices and modern conventions
- Maintaining or improving functionality while enhancing structure
- Ensuring code is production-ready and robust

Please return your refined content as valid JSON in this exact format:
{{
  "refined_content": "# Your improved content here",
  "changes_made": "Detailed summary of all changes and improvements applied",
  "confidence": 0.9,
  "improvement_areas": ["area1", "area2", "area3"]
}}

Provide clear explanations of your improvements."""

        user_content = f"""Original content to refine:
```python
{original_content}
```

Specific refinement instructions:
{refinement_instructions}

Please improve the content according to these instructions while maintaining functionality."""

        def call() -> Dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")
            
            # Parse JSON response
            parsed = parse_json_response(content)
            
            if parsed.success and parsed.data:
                return {
                    "refined_content": parsed.data.get("refined_content", original_content),
                    "changes_made": parsed.data.get("changes_made", "No changes made"),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": {
                        "model_id": self.model_id,
                        "parsed": True,
                        "improvement_areas": parsed.data.get("improvement_areas", []),
                        **result.get("usage", {})
                    }
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    "refined_content": content or original_content,
                    "changes_made": "Refinement applied (JSON parse failed)",
                    "confidence": 0.3,
                    "metadata": {
                        "model_id": self.model_id,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {})
                    }
                }
                
        except Exception as e:
            logger.error(f"Bedrock content refinement failed: {e}")
            raise BedrockError(f"Content refinement failed: {e}") from e

    def _invoke_chat(
        self, 
        system_message: str, 
        user_content: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Invoke ChatBedrock with system and user messages."""
        
        # Create LangChain messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_content)
        ]
        
        try:
            # Invoke the ChatBedrock client
            response = self.client.invoke(messages, **kwargs)
            
            # Extract content from LangChain AIMessage response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract usage information if available
            usage_info = {}
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                # Try to extract token usage from various possible locations
                if 'usage' in metadata:
                    usage_data = metadata['usage']
                    usage_info = {
                        "input_tokens": usage_data.get('input_tokens', 0),
                        "output_tokens": usage_data.get('output_tokens', 0),
                        "total_tokens": usage_data.get('total_tokens', 0)
                    }
                elif 'token_usage' in metadata:
                    usage_data = metadata['token_usage']
                    usage_info = {
                        "input_tokens": usage_data.get('prompt_tokens', 0),
                        "output_tokens": usage_data.get('completion_tokens', 0),
                        "total_tokens": usage_data.get('total_tokens', 0)
                    }
            
            return {
                "content": content,
                "usage": usage_info,
                "model_id": self.model_id,
                "response_metadata": getattr(response, 'response_metadata', {})
            }
            
        except Exception as e:
            logger.error(f"ChatBedrock invocation error: {e}")
            raise BedrockError(f"ChatBedrock invocation failed: {e}") from e
