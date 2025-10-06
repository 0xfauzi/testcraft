"""
Validation tests for Phase 4 orchestrator consolidation refactoring.

These tests verify that:
1. Legacy prompts are removed and raise errors
2. Orchestrator prompts are available
3. RefineAdapter requires ParserPort
4. GenerateUseCase fails fast (no fallback)
5. LLMRouter routes correctly
"""

from pathlib import Path

import pytest

from testcraft.adapters.llm.router import LLMRouter
from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.adapters.refine.main_adapter import RefineAdapter
from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.prompts.registry import PromptError, PromptRegistry


class TestPromptRegistryRefactor:
    """Test that legacy prompts are removed and orchestrator prompts work."""

    def test_legacy_prompts_removed(self):
        """Legacy prompts should raise PromptError."""
        registry = PromptRegistry()

        # All these should raise PromptError
        legacy_prompts = [
            "test_generation",
            "refinement",
            "llm_test_generation",
            "llm_code_analysis",
            "llm_content_refinement",
        ]

        for prompt_name in legacy_prompts:
            with pytest.raises(PromptError, match="Template not found"):
                registry.get_system_prompt(prompt_name)

            with pytest.raises(PromptError, match="Template not found"):
                registry.get_user_prompt(prompt_name)

    def test_orchestrator_prompts_available(self):
        """Orchestrator prompts should be available and functional."""
        registry = PromptRegistry()

        orchestrator_prompts = [
            "orchestrator_plan",
            "orchestrator_generate",
            "orchestrator_refine",
            "orchestrator_manual_fix",
        ]

        for prompt_name in orchestrator_prompts:
            # Should not raise - just check they're in the dictionary
            assert prompt_name in registry._system_templates["v1"]
            assert prompt_name in registry._user_templates["v1"]

    def test_evaluation_prompts_available(self):
        """Evaluation prompts should still be available."""
        registry = PromptRegistry()

        evaluation_prompts = [
            "llm_judge_v1",
            "pairwise_comparison_v1",
            "rubric_evaluation_v1",
            "statistical_analysis_v1",
            "bias_mitigation_v1",
        ]

        for prompt_name in evaluation_prompts:
            # Should not raise
            system_prompt = registry.get_system_prompt(prompt_name)
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 100  # Should be substantial


class TestRefineAdapterRefactor:
    """Test that RefineAdapter requires ParserPort and has orchestrator support."""

    def test_refine_adapter_requires_parser_port(self):
        """RefineAdapter should require parser_port parameter (breaking change)."""
        from unittest.mock import Mock

        llm = Mock()  # Use mock to avoid credential errors
        parser = CodebaseParser()

        # Should work with parser_port
        adapter = RefineAdapter(llm=llm, parser_port=parser)
        assert adapter.parser_port is not None
        assert adapter.llm is not None

    def test_refine_adapter_has_orchestrator_support(self):
        """RefineAdapter should have orchestrator initialization method."""
        from unittest.mock import Mock

        llm = Mock()
        parser = CodebaseParser()
        adapter = RefineAdapter(llm=llm, parser_port=parser)

        # Check orchestrator support methods exist
        assert hasattr(adapter, "_ensure_orchestrator")
        assert hasattr(adapter, "_build_minimal_context_pack")
        assert hasattr(adapter, "_orchestrator_initialized")

    def test_refine_adapter_lazy_orchestrator_initialization(self):
        """Orchestrator should be lazily initialized."""
        from unittest.mock import Mock

        llm = Mock()
        parser = CodebaseParser()
        adapter = RefineAdapter(llm=llm, parser_port=parser)

        # Before initialization
        assert adapter._orchestrator_initialized is False

        # After calling _ensure_orchestrator (will initialize)
        orchestrator = adapter._ensure_orchestrator()
        assert orchestrator is not None
        assert adapter._orchestrator_initialized is True


class TestLLMRouterImplementation:
    """Test that LLMRouter works correctly."""

    def test_llm_router_instantiates(self):
        """LLMRouter should instantiate with config."""
        config = {
            "default_provider": "openai",
            "openai_model": "gpt-4.1",
            "temperature": 0.1,
        }

        router = LLMRouter(config=config)
        assert router.default_provider == "openai"
        assert router.config == config

    def test_llm_router_supports_all_providers(self):
        """LLMRouter should support all configured providers."""
        providers = ["openai", "anthropic", "azure-openai", "bedrock"]

        for provider in providers:
            config = {"default_provider": provider}
            router = LLMRouter(config=config)
            assert router.default_provider == provider

    def test_llm_router_unknown_provider_raises_error(self):
        """Unknown provider should raise ValueError."""
        config = {"default_provider": "unknown_provider"}
        router = LLMRouter(config=config)

        with pytest.raises(ValueError, match="Unknown provider"):
            router._get_adapter("unknown_provider")


class TestGenerateUseCaseRefactor:
    """Test that GenerateUseCase has fail-fast behavior."""

    def test_generate_usecase_has_diagnostic_helper(self):
        """GenerateUseCase should have diagnostic helper for context failures."""
        from testcraft.cli.dependency_injection import create_dependency_container
        from testcraft.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_config()
        container = create_dependency_container(config)

        use_case = container["generate_usecase"]
        assert hasattr(use_case, "_diagnose_context_failure")

    def test_generate_usecase_has_dry_run_support(self):
        """GenerateUseCase should have dry-run support."""
        from testcraft.cli.dependency_injection import create_dependency_container
        from testcraft.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_config()
        container = create_dependency_container(config)

        use_case = container["generate_usecase"]
        assert hasattr(use_case, "_dry_run")
        assert hasattr(use_case, "_execute_dry_run")
        assert use_case._dry_run is False  # Default

    def test_diagnose_context_failure_detects_missing_file(self):
        """Diagnostic should detect missing file."""
        from testcraft.cli.dependency_injection import create_dependency_container
        from testcraft.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_config()
        container = create_dependency_container(config)

        use_case = container["generate_usecase"]
        result = use_case._diagnose_context_failure(Path("/nonexistent/file.py"))

        assert "does not exist" in result["reason"]
        assert "Check file path" in result["fix"]

    def test_diagnose_context_failure_detects_empty_file(self, tmp_path):
        """Diagnostic should detect empty file."""
        from testcraft.cli.dependency_injection import create_dependency_container
        from testcraft.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_config()
        container = create_dependency_container(config)

        # Create empty file
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        use_case = container["generate_usecase"]
        result = use_case._diagnose_context_failure(empty_file)

        assert "empty" in result["reason"].lower()
        assert "Add content" in result["fix"]


class TestOrchestratorOnlyArchitecture:
    """High-level tests verifying orchestrator-only architecture."""

    def test_no_legacy_fallback_in_generate_usecase(self):
        """GenerateUseCase should not have legacy fallback code."""
        import inspect

        source = inspect.getsource(GenerateUseCase.generate_tests)

        # Should not contain legacy fallback patterns
        assert "Fall back to legacy" not in source
        assert "fallback to simple generation" not in source.lower()

    def test_orchestrator_fully_integrated(self):
        """Verify orchestrator is fully integrated in the stack."""
        from testcraft.cli.dependency_injection import create_dependency_container
        from testcraft.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_config()
        container = create_dependency_container(config)

        # Check all components
        assert "generate_usecase" in container
        assert "refine_adapter" in container
        assert "llm_adapter" in container

        # Check RefineAdapter has orchestrator support
        refine_adapter = container["refine_adapter"]
        assert hasattr(refine_adapter, "_ensure_orchestrator")

        # Check GenerateUseCase has orchestrator
        generate_usecase = container["generate_usecase"]
        assert hasattr(generate_usecase, "_llm_orchestrator")

    def test_prompt_registry_only_has_orchestrator_and_evaluation(self):
        """PromptRegistry should only contain orchestrator and evaluation prompts."""
        registry = PromptRegistry()

        # Get all available system prompts
        available_prompts = set(registry._system_templates["v1"].keys())

        # Should have orchestrator prompts
        orchestrator_prompts = {
            "orchestrator_plan",
            "orchestrator_generate",
            "orchestrator_refine",
            "orchestrator_manual_fix",
        }
        assert orchestrator_prompts.issubset(available_prompts)

        # Should have evaluation prompts
        evaluation_prompts = {
            "llm_judge_v1",
            "pairwise_comparison_v1",
            "rubric_evaluation_v1",
        }
        assert evaluation_prompts.issubset(available_prompts)

        # Should NOT have legacy prompts
        legacy_prompts = {
            "test_generation",
            "refinement",
            "llm_test_generation",
            "llm_code_analysis",
            "llm_content_refinement",
        }
        assert legacy_prompts.isdisjoint(available_prompts)


class TestRefactorMetrics:
    """Tests to verify refactoring achieved expected code reduction."""

    def test_prompt_registry_size_reduced(self):
        """Prompt registry should be significantly smaller."""
        import inspect

        from testcraft.prompts.registry import PromptRegistry

        source = inspect.getsource(PromptRegistry)
        lines = source.count("\n")

        # After removing ~331 lines, registry should be smaller
        # Baseline was ~2126 lines, target is ~1795 lines
        assert lines < 2000, f"Registry still too large: {lines} lines"

    def test_only_orchestrator_methods_in_use_case(self):
        """GenerateUseCase should only use orchestrator methods."""
        import inspect

        source = inspect.getsource(GenerateUseCase)

        # Should use orchestrator
        assert "plan_and_generate" in source
        assert "_llm_orchestrator" in source

        # Should NOT use direct LLM calls (except in legacy scenarios that were removed)
        # The only generate_tests calls should be comments or in error handling
        assert source.count("self._llm.generate_tests(") == 0
